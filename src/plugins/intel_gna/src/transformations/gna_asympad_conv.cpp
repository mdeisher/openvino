// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_asympad_conv.hpp"

#include "backend/gna_limitations.hpp"
#include "memory"
#include "ngraph/rt_info.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/transformation_helper.hpp"

using namespace ov::intel_gna::pass;

namespace ov {
namespace intel_gna {
namespace pass {

static std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> get_input_dimensions(ov::Shape input_shape) {
    uint64_t d0 = input_shape[0];
    uint64_t d1 = input_shape[1];
    uint64_t d2 = input_shape[2];
    uint64_t d3 = input_shape[3];
    return std::make_tuple(d0, d1, d2, d3);
}

static std::tuple<int64_t, int64_t, int64_t> extract_height_padding(ov::CoordinateDiff pads_begin,
                                                                    ov::CoordinateDiff pads_end) {
    auto height_begin = pads_begin[0];
    auto height_end = pads_end[0];
    return std::make_tuple(height_begin, height_end, std::abs(height_begin - height_end));
}

static std::tuple<int64_t, int64_t, int64_t> extract_width_padding(ov::CoordinateDiff pads_begin,
                                                                   ov::CoordinateDiff pads_end) {
    auto width_begin = pads_begin[1];
    auto width_end = pads_end[1];
    return std::make_tuple(width_begin, width_end, std::abs(width_begin - width_end));
}

static std::shared_ptr<ov::opset11::Reshape> create_reshape(const ov::Output<ov::Node>& input,
                                                            uint64_t ndims,
                                                            ov::Shape shape) {
    return std::make_shared<ov::opset11::Reshape>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{ndims}, shape)->output(0),
        false);
}

static std::shared_ptr<ov::opset11::Constant> create_zero_const(ov::Shape shape) {
    return ov::opset11::Constant::create(ov::element::f32, shape, std::vector<float>(shape[0] * shape[1], 0.0f));
}

static std::shared_ptr<ov::op::v0::Concat> concatenate_zeros(uint64_t pad_begin,
                                                             uint64_t pad_end,
                                                             std::shared_ptr<ov::Node> padding_const,
                                                             std::shared_ptr<ov::Node> input_node) {
    ov::OutputVector concat_vector;
    if (pad_begin > pad_end) {
        concat_vector.push_back(padding_const->output(0));
        concat_vector.push_back(input_node->output(0));
    } else {
        concat_vector.push_back(input_node->output(0));
        concat_vector.push_back(padding_const->output(0));
    }
    return std::make_shared<ov::opset11::Concat>(concat_vector, 0);
}

static void trimm_padding(ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end) {
    if (pads_begin[0] > pads_end[0]) {
        pads_begin[0] = pads_end[0];
    } else {
        pads_end[0] = pads_begin[0];
    }
    if (pads_begin[1] > pads_end[1]) {
        pads_begin[1] = pads_end[1];
    } else {
        pads_end[1] = pads_begin[1];
    }
}

static std::shared_ptr<ov::Node> modify_padding(std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv,
                                                const ov::Output<ov::Node>& input,
                                                ov::CoordinateDiff pads_begin,
                                                ov::CoordinateDiff pads_end) {
    trimm_padding(pads_begin, pads_end);

    if (nullptr != conv) {
        return std::make_shared<ov::intel_gna::op::GNAConvolution>(input,
                                                                   conv->input_value(1),
                                                                   conv->get_strides(),
                                                                   pads_begin,
                                                                   pads_end,
                                                                   conv->get_dilations(),
                                                                   conv->get_auto_pad());
    }

    return nullptr;
}

static std::shared_ptr<ov::opset11::Transpose> create_2d_transpose(const ov::Output<ov::Node>& input) {
    return std::make_shared<ov::opset11::Transpose>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
}

static ov::Output<ov::Node> pad_validate_transpose(ov::Output<ov::Node> input,
                                                   ov::Shape& trsp_shape,
                                                   size_t transpose_validate_factor) {
    if (trsp_shape[0] % transpose_validate_factor == 0)
        return input;

    auto n_to_pad = transpose_validate_factor - trsp_shape[0] % transpose_validate_factor;

    auto padding_const = create_zero_const(ov::Shape{n_to_pad, trsp_shape[1]});
    ov::OutputVector concat_vector;
    concat_vector.push_back(input);
    concat_vector.push_back(padding_const->output(0));

    return std::make_shared<ov::opset11::Concat>(concat_vector, 0)->output(0);
}

static ov::Output<ov::Node> postcrop_validate_transpose(ov::Output<ov::Node> input,
                                                        ov::Shape& trsp_shape,
                                                        size_t transpose_validate_factor) {
    if (trsp_shape[0] % transpose_validate_factor == 0)
        return input;

    auto slice_start = ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {0ull, 0ull});
    auto slice_stop = ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {trsp_shape[0], trsp_shape[1]});
    auto slice_step = ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {1ull, 1ull});
    auto new_slice =
        std::make_shared<ov::opset11::Slice>(input, slice_start, slice_stop, slice_step);  // try variadic split

    return new_slice->output(0);
}

static ov::Output<ov::Node> decompose_height(ov::Output<ov::Node> input,
                                             ov::CoordinateDiff pads_begin,
                                             ov::CoordinateDiff pads_end,
                                             ov::Shape conv_input_shape) {
    uint64_t height_begin, height_end, height_padding, width_padding;
    std::tie(height_begin, height_end, height_padding) = extract_height_padding(pads_begin, pads_end);
    width_padding = std::abs(pads_end[1] - pads_begin[1]);
    uint64_t N, C, H, W;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);

    if (0 == height_padding)
        return input;

    auto new_reshape = create_reshape(input, 2, ov::Shape{H, W * C});
    auto padding_const = create_zero_const(ov::Shape{height_padding, W * C});
    auto new_concat = concatenate_zeros(height_begin, height_end, padding_const, new_reshape);

    if (0 == width_padding)
        return create_reshape(new_concat->output(0), 4, ov::Shape{N, H + height_padding, W, C})->output(0);
    return (new_concat->output(0));
}

static ov::Output<ov::Node> decompose_width(ov::Output<ov::Node> input,
                                            ov::CoordinateDiff pads_begin,
                                            ov::CoordinateDiff pads_end,
                                            ov::Shape conv_input_shape) {
    uint64_t width_begin, width_end, width_padding, height_padding;
    std::tie(width_begin, width_end, width_padding) = extract_width_padding(pads_begin, pads_end);
    height_padding = std::abs(pads_end[0] - pads_begin[0]);
    uint64_t N, H, W, C;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);
    if (0 == width_padding) {
        return input;
    }

    auto new_reshape = create_reshape(input, 2, ov::Shape{(H + height_padding), W * C});

    size_t transpose_validate_factor;
    if ((W + width_padding) * C % 8 == 0 && W * C % 8 == 0)
        transpose_validate_factor = 8;
    else if ((W + width_padding) * C % 4 == 0 && W * C % 4 == 0)
        transpose_validate_factor = 16;
    else if ((W + width_padding) * C % 2 == 0 && W * C % 2 == 0)
        transpose_validate_factor = 32;
    else
        transpose_validate_factor = 64;

    auto trsp_input = pad_validate_transpose(new_reshape->output(0),
                                             ov::Shape{(H + height_padding), W * C},
                                             transpose_validate_factor);
    auto new_transpose = create_2d_transpose(trsp_input);
    auto second_dim = H + height_padding;
    auto padding_const = create_zero_const(
        ov::Shape{width_padding * C,
                  second_dim + (second_dim % transpose_validate_factor
                                    ? transpose_validate_factor - second_dim % transpose_validate_factor
                                    : 0)});
    auto new_concat = concatenate_zeros(width_begin, width_end, padding_const, new_transpose);
    auto new_untranspose = create_2d_transpose(new_concat->output(0));
    auto trsp_output = postcrop_validate_transpose(new_untranspose->output(0),
                                                   ov::Shape{(H + height_padding), (W + width_padding) * C},
                                                   transpose_validate_factor);

    auto new_unshape = create_reshape(trsp_output, 4, {N, H + height_padding, W + width_padding, C});
    return new_unshape->output(0);
}

static bool decompose(std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv) {
    if (conv == nullptr)
        return false;

    auto pads_begin = conv->get_pads_begin();
    auto pads_end = conv->get_pads_end();
    if (pads_begin.size() < 2 || pads_end.size() < 2)
        return false;
    if (pads_begin[0] == pads_end[0] && pads_begin[1] == pads_end[1])
        return false;

    Output<Node> input = conv->input_value(0);
    auto input_shape = input.get_shape();
    if (input_shape.size() != 4 || input_shape[0] != 1)
        return false;

    Output<Node> skip_input_H_const = decompose_height(input, pads_begin, pads_end, input_shape);
    Output<Node> skip_input_W_const = decompose_width(skip_input_H_const, pads_begin, pads_end, input_shape);

    auto new_conv = modify_padding(conv, skip_input_W_const, pads_begin, pads_end);
    if (new_conv == nullptr)
        return false;

    new_conv->set_friendly_name(conv->get_friendly_name());
    ov::copy_runtime_info(conv, new_conv);
    ov::replace_node_update_name(conv, new_conv);
    return true;
}

GnaAsymPadConvDecomposition::GnaAsymPadConvDecomposition() {
    MATCHER_SCOPE(GnaAsymPadConvDecomposition);
    auto conv = ov::pass::pattern::wrap_type<ov::intel_gna::op::GNAConvolution>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(m.get_match_root());
        return decompose(conv);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
