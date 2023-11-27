// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_mha.hpp"
#include <layers/gna_layer_info.hpp>
#include <layers/gna_layer_type.hpp>

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "gna_helper.hpp"

using namespace ov::opset12;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

void GnaGetTransposeOrder(std::vector<int64_t> &order, std::shared_ptr<ov::Node> node) {
    auto input_shape = node->input(0).get_shape();
    auto output_shape = node->output(0).get_shape();
    auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(node);
    const ov::Output<ov::Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();
    auto const_with_order_values = std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_order.get_node_shared_ptr());
    if (const_with_order_values->get_output_element_type(0) == ov::element::i8) {
        const int8_t* ptr_order = const_with_order_values->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_shape.size(); i++) {
            order.push_back(*(ptr_order + i));
        }
    } else if (const_with_order_values->get_output_element_type(0) == ov::element::i32) {
        const int32_t* ptr_order = const_with_order_values->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_shape.size(); i++) {
            order.push_back(*(ptr_order + i));
        }
    } else {
        const int64_t* ptr_order = const_with_order_values->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_shape.size(); i++) {
            order.push_back(*(ptr_order + i));
        }
    }
}

std::shared_ptr<ov::op::v0::Constant> GnaNewConvWeights(std::shared_ptr<ov::Node> node) {
    std::shared_ptr<ov::op::v0::Constant> new_weights_const = nullptr;

    auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
    if (matmul) {
        std::shared_ptr<ov::op::v0::Constant> weights_const = nullptr;
        const ov::Output<ov::Node>& input1 = matmul->input_value(1);
        auto weights_fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(input1.get_node()->shared_from_this());
        if (weights_fq) {
            const ov::Output<ov::Node>& inputfq = weights_fq->input_value(0);
            weights_const = std::dynamic_pointer_cast<ngraph::op::Constant>(inputfq.get_node()->shared_from_this());
        } else {
            weights_const = std::dynamic_pointer_cast<ngraph::op::Constant>(input1.get_node()->shared_from_this());
        }
        if (weights_const) {
            auto weights_shape = weights_const->get_output_shape(0);
            const float* weight_ptr = weights_const->get_data_ptr<float>();
            std::vector<float> new_weights(weights_shape[0] * weights_shape[1], 0.0f);
            float* new_weight_ptr = new_weights.data();
            ov::Shape new_weights_shape;
            if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(matmul)->get_transpose_b()) {
                // leave weights alone since transpose for MatMul and transpose for convolution cancel each other
                new_weights_shape.push_back(weights_shape[0]);
                new_weights_shape.push_back(1);
                new_weights_shape.push_back(1);
                new_weights_shape.push_back(weights_shape[1]);
                memcpy(new_weight_ptr, weight_ptr, new_weights.size() * sizeof(float));
            } else {
                // transpose weight matrix
                new_weights_shape.push_back(weights_shape[1]);
                new_weights_shape.push_back(1);
                new_weights_shape.push_back(1);
                new_weights_shape.push_back(weights_shape[0]);
                for (auto i = 0; i < weights_shape[0]; i++) {
                    for (auto j = 0; j < weights_shape[1]; j++) {
                        new_weight_ptr[j * weights_shape[0] + i] = weight_ptr[i * weights_shape[1] + j];
                    }
                }
            }
            new_weights_const = ov::op::v0::Constant::create(ngraph::element::f32, new_weights_shape, new_weights);
        }
    }

    return new_weights_const;
}

std::shared_ptr<ov::op::v0::Constant> GnaNewConvBias(std::shared_ptr<ov::Node> node) {
    std::shared_ptr<ov::op::v0::Constant> new_bias_const = nullptr;

    auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(node);
    if (add) {
        std::shared_ptr<ov::op::v0::FakeQuantize> bias_fq = nullptr;
        std::shared_ptr<ov::op::v0::Constant> bias_const = nullptr;
        const ov::Output<ov::Node>& input0 = add->input_value(0);
        const ov::Output<ov::Node>& input1 = add->input_value(1);
        auto fq0 = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(input0.get_node()->shared_from_this());
        auto fq1 = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(input1.get_node()->shared_from_this());
        auto bias0 = std::dynamic_pointer_cast<ngraph::op::Constant>(input0.get_node()->shared_from_this());
        auto bias1 = std::dynamic_pointer_cast<ngraph::op::Constant>(input1.get_node()->shared_from_this());
        if (fq0) {
            bias_const = std::dynamic_pointer_cast<ngraph::op::Constant>(fq0->input_value(0).get_node()->shared_from_this());
            if (bias_const) {
                bias_fq = fq0;
            } else if (fq1) {
                bias_const = std::dynamic_pointer_cast<ngraph::op::Constant>(fq1->input_value(0).get_node()->shared_from_this());
                if (bias_const) {
                    bias_fq = fq1;
                }
            }
        } else if (fq1) {
            bias_const = std::dynamic_pointer_cast<ngraph::op::Constant>(fq1->input_value(0).get_node()->shared_from_this());
            if (bias_const) {
                bias_fq = fq1;
            }
        } else if (bias0) {
            bias_const = bias0;
        } else if (bias1) {
            bias_const = bias1;
        }
        if (bias_const) {
            auto bias_shape = bias_const->get_output_shape(0);
            const float* bias_ptr = bias_const->get_data_ptr<float>();
            size_t len = 1;
            for (size_t i = 0; i < bias_shape.size(); i++) {
                len *= bias_shape[i];
            }
            std::vector<float> new_bias(len, 0.0f);
            float* new_bias_ptr = new_bias.data();
            ov::Shape new_bias_shape;
            new_bias_shape.push_back(1);
            new_bias_shape.push_back(1);
            new_bias_shape.push_back(1);
            new_bias_shape.push_back(len);
            memcpy(new_bias_ptr, bias_ptr, new_bias.size() * sizeof(float));
            new_bias_const = ov::op::v0::Constant::create(ngraph::element::f32, new_bias_shape, new_bias);
        }
    }

    return new_bias_const;
}

ov::OutputVector GnaTransposeSplit(ov::Output<ov::Node>& prev, size_t H, size_t W, size_t C, bool transpose_input, bool transpose_output, bool zero_pad, bool insert_fq) {
    auto reshape = std::make_shared<Reshape>(prev, Constant::create(ov::element::i64, ov::Shape{2}, {H, C * W})->output(0), false);
    ov::OutputVector upstream;
    upstream.push_back(reshape->output(0));
    if (transpose_input) {
        auto transpose = std::make_shared<Transpose>(reshape->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        upstream[0] = transpose->output(0);
    }
    reshape = std::make_shared<Reshape>(upstream[0], Constant::create(ov::element::i64, ov::Shape{2}, {1ull, H * W * C})->output(0), false);
    std::vector<size_t> split_lengths(C, H * W);
    const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
    auto split = std::make_shared<VariadicSplit>(reshape->output(0), Constant::create(ov::element::i64, ov::Shape{}, {1}), split_lengths_const);
    ov::OutputVector output;
    for (auto i = 0; i < split->get_output_size(); i++) {
        ov::OutputVector upstream;
        auto reshape = std::make_shared<Reshape>(split->output(i), Constant::create(ov::element::i64, ov::Shape{2}, {W, H})->output(0), false);
        upstream.push_back(reshape->output(0));
        if (zero_pad) {
            auto new_shape = reshape->output(0).get_shape();
            auto num_pad_rows = (new_shape[0] % 8 == 0) ? 0 : 8 - (new_shape[0] % 8);
            if (num_pad_rows > 0) {
                std::vector<float> padding(num_pad_rows * new_shape[1], 0.0f);
                auto pad_const = Constant::create(ngraph::element::f32, ov::Shape{num_pad_rows, new_shape[1]}, padding);
                ov::OutputVector parts;
                if (insert_fq) {
                    size_t levels = 65535;  // value used by POT 2023.0.1 when quantizing a zero const
                    auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;
                    auto fq_dim = pad_const->output(0).get_shape().size();
                    auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
                    auto fq_type = pad_const->output(0).get_element_type();
                    auto input_low_data = -0.00007999999797903001f;  // value used by POT 2023.0.1 when quantizing a zero const
                    auto input_high_data = 0.00007999999797903001f;  // value used by POT 2023.0.1 when quantizing a zero const
                    auto output_low_data = -0.00007999999797903001f;  // value used by POT 2023.0.1 when quantizing a zero const
                    auto output_high_data = 0.00007999999797903001f;  // value used by POT 2023.0.1 when quantizing a zero const
                    auto input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
                    auto input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
                    auto output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
                    auto output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
                    auto fq = std::make_shared<FakeQuantize>(pad_const->output(0), input_low->output(0), input_high->output(0), output_low->output(0), output_high->output(0), levels, auto_broadcast);
                    parts.push_back(fq->output(0));
                } else {
                    parts.push_back(pad_const->output(0));
                }
                parts.push_back(reshape->output(0));
                auto concat = std::make_shared<Concat>(parts, 0);
                upstream[0] = concat->output(0);
            }
        }
        if (transpose_output) {
            auto transpose = std::make_shared<Transpose>(upstream[0], Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            output.push_back(transpose->output(0));
        } else {
            output.push_back(upstream[0]);
        }
    }
    return output;
}

std::shared_ptr<ov::intel_gna::op::GNAConvolution> InsertGnaConvolution(const ov::Output<ov::Node>& input, std::shared_ptr<ov::Node> matmul,
                                                                        std::shared_ptr<ov::Node> add, size_t H, size_t W, size_t C) {
    std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv = nullptr;
    auto reshape = std::make_shared<Reshape>(input, Constant::create(ov::element::i64, ov::Shape{4}, {1ull, H, 1ull, C * W})->output(0), false);
    std::shared_ptr<ov::op::v0::Constant> weights_const = GnaNewConvWeights(matmul);
    auto matmulfq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(matmul->input_value(1).get_node()->shared_from_this());
    ov::OutputVector weights;
    weights.push_back(weights_const->output(0));
    if (matmulfq) {
        auto weightsfq = CopyFQ(weights_const->output(0), matmulfq);
        weights[0] = weightsfq->output(0);
    }
    if (add) {
        auto bias_const = GnaNewConvBias(add);
        conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0), weights[0], bias_const->output(0),
            ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID );
    } else {
        conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0), weights[0],
            ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID );
    }
    return conv;
}

//#define PRE_LAYOUT

// GNA Multihead Attention factorization

GnaMhaTransformation::GnaMhaTransformation() {
    MATCHER_SCOPE(GnaMhaTransformation);

#ifdef PRE_LAYOUT
    auto transpose_pattern_2a = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<MatMul>({transpose_pattern_2a, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<MatMul>({transpose_pattern_2b, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<MatMul>({transpose_pattern_2c, pattern::any_input()});
#else
    auto transpose_pattern_1a = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_1b = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_1c = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_2a = pattern::wrap_type<Reshape>({transpose_pattern_1a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Reshape>({transpose_pattern_1b, pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Reshape>({transpose_pattern_1c, pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2a, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2b, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2c, pattern::any_input()});
#endif
    auto add_pattern_1a = pattern::wrap_type<Add>({matmul_pattern_1a, pattern::any_input()});
    auto add_pattern_1b = pattern::wrap_type<Add>({matmul_pattern_1b, pattern::any_input()});
    auto add_pattern_1c = pattern::wrap_type<Add>({matmul_pattern_1c, pattern::any_input()});
    auto addrev_pattern_1a = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1a});
    auto addrev_pattern_1b = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1b});
    auto addrev_pattern_1c = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1c});
    auto matmuladd_pattern_1a = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1a, add_pattern_1a, addrev_pattern_1a});
    auto matmuladd_pattern_1b = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1b, add_pattern_1b, addrev_pattern_1b});
    auto matmuladd_pattern_1c = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1c, add_pattern_1c, addrev_pattern_1c});
    auto reshape_pattern_1a = pattern::wrap_type<Reshape>({matmuladd_pattern_1a, pattern::any_input()});
    auto reshape_pattern_1b = pattern::wrap_type<Reshape>({matmuladd_pattern_1b, pattern::any_input()});
    auto reshape_pattern_1c = pattern::wrap_type<Reshape>({matmuladd_pattern_1c, pattern::any_input()});
    auto transpose_pattern_3a = pattern::wrap_type<Transpose>({reshape_pattern_1a, pattern::any_input()});
    auto transpose_pattern_3b = pattern::wrap_type<Transpose>({reshape_pattern_1b, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto transpose_pattern_3c = pattern::wrap_type<Transpose>({reshape_pattern_1c, pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({transpose_pattern_3c, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({multiply_pattern_1, transpose_pattern_3b});
#else
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_1c, pattern::any_input()});
    auto transpose_pattern_3c = pattern::wrap_type<Transpose>({multiply_pattern_1, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({transpose_pattern_3c, transpose_pattern_3b});
#endif
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({matmul_pattern_2bc});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({softmax_pattern, transpose_pattern_3a});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({matmul_pattern_2abc, pattern::any_input()});
    auto reshape_pattern_2 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto matmul_pattern_3 = pattern::wrap_type<MatMul>({reshape_pattern_2, pattern::any_input()});
    auto reshape_pattern_4a = pattern::wrap_type<Reshape>({matmul_pattern_3, pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({reshape_pattern_4a, pattern::any_input()});
#else
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({reshape_pattern_2, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input()});
    auto matmuladd_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_3 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_3, matmuladd_pattern_3});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({matmulor_pattern_3, pattern::any_input()});
#endif

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

#ifdef PRE_LAYOUT
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
#else
        auto transpose1a = pattern_map.at(transpose_pattern_1a).get_node_shared_ptr();
        auto transpose1b = pattern_map.at(transpose_pattern_1b).get_node_shared_ptr();
        auto transpose1c = pattern_map.at(transpose_pattern_1c).get_node_shared_ptr();
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
#endif
        auto matmul1a = pattern_map.at(matmul_pattern_1a).get_node_shared_ptr();
        auto matmul1b = pattern_map.at(matmul_pattern_1b).get_node_shared_ptr();
        auto matmul1c = pattern_map.at(matmul_pattern_1c).get_node_shared_ptr();
        auto reshape1a = pattern_map.at(reshape_pattern_1a).get_node_shared_ptr();
        auto reshape1b = pattern_map.at(reshape_pattern_1b).get_node_shared_ptr();
        auto reshape1c = pattern_map.at(reshape_pattern_1c).get_node_shared_ptr();
        auto transpose3a = pattern_map.at(transpose_pattern_3a).get_node_shared_ptr();
        auto transpose3b = pattern_map.at(transpose_pattern_3b).get_node_shared_ptr();
        auto transpose3c = pattern_map.at(transpose_pattern_3c).get_node_shared_ptr();
        auto multiply1 = pattern_map.at(multiply_pattern_1).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape2 = pattern_map.at(reshape_pattern_2).get_node_shared_ptr();

        auto children = matmul1a->output(0).get_target_inputs();
        auto add1a = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
        children = matmul1b->output(0).get_target_inputs();
        auto add1b = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
        children = matmul1c->output(0).get_target_inputs();
        auto add1c = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());

        children = softmax->output(0).get_target_inputs();
        if (children.size() != 2) {
            return false;
        }
#ifdef PRE_LAYOUT
        auto reducesum = std::dynamic_pointer_cast<ReduceSum>(children.begin()->get_node()->shared_from_this());
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
#else
        std::shared_ptr<Reshape> reshape5 = nullptr;
        for (auto child : children) {
            reshape5 = std::dynamic_pointer_cast<Reshape>(child.get_node()->shared_from_this());
            if (reshape5 != nullptr) {
                break;
            }
        }
        if (reshape5 == nullptr) {
            return false;
        }
        children = reshape5->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto avgpool = std::dynamic_pointer_cast<AvgPool>(children.begin()->get_node()->shared_from_this());
        if (avgpool == nullptr) {
            return false;
        }
        children = avgpool->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply2 == nullptr) {
            return false;
        }
        children = multiply2->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto reducesum = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (reducesum == nullptr) {
            return false;
        }
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
#endif

#ifdef PRE_LAYOUT
        const Output<Node>& input1a = transpose2a->input_value(0);
        const Output<Node>& input1b = transpose2b->input_value(0);
        const Output<Node>& input1c = transpose2c->input_value(0);
        auto transpose2a_input_shape = transpose2a->input(0).get_shape();
        auto transpose2b_input_shape = transpose2b->input(0).get_shape();
        auto transpose2c_input_shape = transpose2c->input(0).get_shape();
        auto transpose2a_output_shape = transpose2a->output(0).get_shape();
        auto transpose2b_output_shape = transpose2b->output(0).get_shape();
        auto transpose2c_output_shape = transpose2c->output(0).get_shape();
        std::vector<int64_t> order2a, order2b, order2c;
        std::vector<int64_t> expected_order = {1, 0, 2};
        GnaGetTransposeOrder(order2a, transpose2a);
        GnaGetTransposeOrder(order2b, transpose2b);
        GnaGetTransposeOrder(order2c, transpose2c);
        if ((order2a != expected_order) || (order2b != expected_order) || (order2c != expected_order)) {
            return false;
        }
#else
        const Output<Node>& input1a = transpose1a->input_value(0);
        const Output<Node>& input1b = transpose1b->input_value(0);
        const Output<Node>& input1c = transpose1c->input_value(0);
        auto transpose1a_input_shape = transpose1a->input(0).get_shape();
        auto transpose1b_input_shape = transpose1b->input(0).get_shape();
        auto transpose1c_input_shape = transpose1c->input(0).get_shape();
        auto transpose1a_output_shape = transpose1a->output(0).get_shape();
        auto transpose1b_output_shape = transpose1b->output(0).get_shape();
        auto transpose1c_output_shape = transpose1c->output(0).get_shape();
#endif
        if ((transpose1a_input_shape != transpose1b_input_shape) || (transpose1b_input_shape != transpose1c_input_shape)) {
            return false;
        }
        auto transpose3c_output_shape = transpose3c->output(0).get_shape();
        auto C = transpose3c_output_shape[0];
        auto H = transpose3c_output_shape[1];
        auto W = transpose3c_output_shape[2];
        OutputVector part_a = GnaTransposeSplit(reshape1a->output(0), H, W, C, true, true, false, false);
        OutputVector part_b = GnaTransposeSplit(reshape1b->output(0), H, W, C, true, false, true, false);
        OutputVector part_c = GnaTransposeSplit(reshape1c->output(0), H, W, C, true, true, true, false);
        OutputVector out_a, out_b;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiply1->input_value(1).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_mul_weight->output(0));
            auto new_matmul2bc = std::make_shared<MatMul>(new_multiply1->output(0), part_b[i], false, false);
            auto new_softmax = std::make_shared<Softmax>(new_matmul2bc->output(0), 1);
            out_b.push_back(new_softmax->output(0));
            auto new_matmul2abc = std::make_shared<MatMul>(new_softmax->output(0), part_a[i], false, false);
            auto new_transpose3 = std::make_shared<Transpose>(new_matmul2abc->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            out_a.push_back(new_transpose3->output(0));
        }
        auto new_concat3 = std::make_shared<Concat>(out_a, 0);
        auto new_transpose3 = std::make_shared<Transpose>(new_concat3->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        auto new_reshape2 = std::make_shared<Reshape>(new_transpose3->output(0),
            Constant::create(element::i64, Shape{3}, {1ull, H, C * W})->output(0),false);
        replace_output_update_name(reshape2, new_reshape2);

        // replace reducesum
        //   Note that the optimal way to implement this is with sum pooling (not yet merged) which takes C MAC/clock.
        //   Also note that problems where the width is greater than about 3K will cause GNA cpkg to fail due to insufficient internal buffer space.
        //   So for now, we just transpose the problem which takes an extra round trip to memory and then 1 MAC/clock.
        //auto new_concat4 = std::make_shared<Concat>(out_b, 0);
        //auto new_reshape5 = std::make_shared<Reshape>(new_concat4->output(0),
        //    Constant::create(element::i64, Shape{4}, {1ull, C, H * H, 1ull})->output(0),false);
        //std::vector<float> weight(C, 1.0f);
        //auto weight_const = Constant::create(ngraph::element::f32, Shape{1, C, 1, 1}, weight);
        //auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0),weight_const->output(0), 
        //    Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        //auto new_reshape5 = std::make_shared<Reshape>(new_conv4->output(0), 
        //    Constant::create(element::i64, Shape{3}, {1ull, H, H})->output(0), false);
        auto new_concat4 = std::make_shared<Concat>(out_b, 0);
        auto new_reshape5 = std::make_shared<Reshape>(new_concat4->output(0),
            Constant::create(element::i64, Shape{2}, {C, H * H})->output(0),false);
        // this transpose is to work around GNA HW limitation where large W causes run out of internal buffer memory
        auto new_transpose4 = std::make_shared<Transpose>(new_reshape5->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
#define PAR_CONV
#define NUM_K 16
#ifdef PAR_CONV
        size_t num_K = NUM_K;
        while (H * H % num_K != 0) {  // reduce parallelism if sizes don't match
            num_K = num_K / 2;
        }
        if (num_K != NUM_K) {
            printf("Warning:  failed to optimize reducesum\n");
        }
        // replace reducesum + multiply-by-1/C with multi-channel convolution
        new_reshape5 = std::make_shared<Reshape>(
            new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H / num_K, C * num_K, 1ull})->output(0),false);
        std::vector<float> weight(num_K * C * num_K, 0.0);
        for (size_t i = 0; i < num_K; i++) {
            for (size_t j = 0; j < C; j++) {
                weight[i * num_K * C + i * C + j] = 1.0f / C;
            }
        }
        auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, C * num_K, 1}, weight);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0),weight_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#else
        // replace reducesum + multiply-by-1/C with pointwise convolution and sum pool
        new_reshape5 = std::make_shared<Reshape>(new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H, C, 1ull})->output(0),false);
        std::vector<float> weight(C, 1.0f / C);
        auto weight_const = Constant::create(ngraph::element::f32, Shape{1, 1, C, 1}, weight);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0),weight_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#endif
        auto new_reshape6 = std::make_shared<Reshape>(new_conv4->output(0), 
            Constant::create(element::i64, Shape{2}, {1ull, H * H})->output(0), false);
        replace_output_update_name(multiply3, new_reshape6);

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(reshape_pattern_4, matcher_name);
    this->register_matcher(m, callback);
}

//#define PRE_LAYOUT
// GNA Multihead Attention factorization with FakeQuantize layers

GnaMhaFqTransformation::GnaMhaFqTransformation() {
    MATCHER_SCOPE(GnaMhaFqTransformation);

    auto inputfq_pattern_1a = pattern::wrap_type<FakeQuantize>({pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto inputfq_pattern_1b = pattern::wrap_type<FakeQuantize>({pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto inputfq_pattern_1c = pattern::wrap_type<FakeQuantize>({pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
#ifdef PRE_LAYOUT
    auto transpose_pattern_2a = pattern::wrap_type<Transpose>({inputfq_pattern_1a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Transpose>({inputfq_pattern_1b, pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({inputfq_pattern_1c, pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<MatMul>({transpose_pattern_2a, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<MatMul>({transpose_pattern_2b, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<MatMul>({transpose_pattern_2c, pattern::any_input()});
#else
    auto transpose_pattern_1a = pattern::wrap_type<Reshape>({inputfq_pattern_1a, pattern::any_input()});
    auto transpose_pattern_1b = pattern::wrap_type<Reshape>({inputfq_pattern_1b, pattern::any_input()});
    auto transpose_pattern_1c = pattern::wrap_type<Reshape>({inputfq_pattern_1c, pattern::any_input()});
    auto transpose_pattern_2a = pattern::wrap_type<Reshape>({transpose_pattern_1a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Reshape>({transpose_pattern_1b, pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Reshape>({transpose_pattern_1c, pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2a, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2b, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_2c, pattern::any_input()});
#endif
    auto add_pattern_1a = pattern::wrap_type<Add>({matmul_pattern_1a, pattern::any_input()});
    auto add_pattern_1b = pattern::wrap_type<Add>({matmul_pattern_1b, pattern::any_input()});
    auto add_pattern_1c = pattern::wrap_type<Add>({matmul_pattern_1c, pattern::any_input()});
    auto addrev_pattern_1a = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1a});
    auto addrev_pattern_1b = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1b});
    auto addrev_pattern_1c = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1c});
    auto matmuladd_pattern_1a = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1a, add_pattern_1a, addrev_pattern_1a});
    auto matmuladd_pattern_1b = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1b, add_pattern_1b, addrev_pattern_1b});
    auto matmuladd_pattern_1c = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1c, add_pattern_1c, addrev_pattern_1c});
    auto reshapefq_pattern_1a = pattern::wrap_type<FakeQuantize>({matmuladd_pattern_1a,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto reshapefq_pattern_1b = pattern::wrap_type<FakeQuantize>({matmuladd_pattern_1b,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto reshapefq_pattern_1c = pattern::wrap_type<FakeQuantize>({matmuladd_pattern_1c,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto missingfq_pattern_1a = std::make_shared<pattern::op::Or>(OutputVector{reshapefq_pattern_1a, matmuladd_pattern_1a});
    auto reshape_pattern_1a = pattern::wrap_type<Reshape>({missingfq_pattern_1a, pattern::any_input()});
    auto reshape_pattern_1b = pattern::wrap_type<Reshape>({reshapefq_pattern_1b, pattern::any_input()});
    auto reshape_pattern_1c = pattern::wrap_type<Reshape>({reshapefq_pattern_1c, pattern::any_input()});
    auto transpose_pattern_3a = pattern::wrap_type<Transpose>({reshape_pattern_1a, pattern::any_input()});
    auto transpose_pattern_3b = pattern::wrap_type<Transpose>({reshape_pattern_1b, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto transpose_pattern_3c = pattern::wrap_type<Transpose>({reshape_pattern_1c, pattern::any_input()});
    auto multiplyfq_pattern_1 = pattern::wrap_type<FakeQuantize>({transpose_pattern_3c,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({multiplyfq_pattern_1, pattern::any_input()});
    auto matmulfq_pattern_2bc = pattern::wrap_type<FakeQuantize>({multiply_pattern_1,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({matmulfq_pattern_2bc, transpose_pattern_3b});
#else
    auto multiplyfq_pattern_1 = pattern::wrap_type<FakeQuantize>({pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_1c, multiplyfq_pattern_1});
    auto transposefq_pattern_3c = pattern::wrap_type<FakeQuantize>({multiply_pattern_1,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto transpose_pattern_3c = pattern::wrap_type<Transpose>({transposefq_pattern_3c, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({transpose_pattern_3c, transpose_pattern_3b});
#endif
    auto softmaxfq_pattern = pattern::wrap_type<FakeQuantize>({matmul_pattern_2bc,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto missingfq_pattern = std::make_shared<pattern::op::Or>(OutputVector{softmaxfq_pattern, matmul_pattern_2bc});
#define POST_SOFTMAX
#ifdef POST_SOFTMAX
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({missingfq_pattern});
#else
    auto smreshape1_pattern = pattern::wrap_type<Reshape>({softmaxfq_pattern, pattern::any_input()});
    auto smreshape2_pattern = pattern::wrap_type<Reshape>({smreshape1_pattern, pattern::any_input()});
    auto smconv1_pattern = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({smreshape2_pattern, pattern::any_input()});
    auto smmaxpool_pattern = pattern::wrap_type<ov::intel_gna::op::GNAMaxPool>({smconv1_pattern});
    auto smfq1_pattern = pattern::wrap_type<FakeQuantize>({smmaxpool_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smconv2_pattern = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({smfq1_pattern, pattern::any_input()});
    auto smfq2_pattern = pattern::wrap_type<FakeQuantize>({smconv2_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smreshape5_pattern = pattern::wrap_type<Reshape>({smfq2_pattern, pattern::any_input()});
    auto smmultiply1_pattern = pattern::wrap_type<Multiply>({smreshape5_pattern, pattern::any_input()});
    auto smadd1_pattern = pattern::wrap_type<Add>({smreshape1_pattern, smmultiply1_pattern});
    auto smfq3_pattern = pattern::wrap_type<FakeQuantize>({smadd1_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smexp1_pattern = pattern::wrap_type<Exp>({smfq3_pattern});
    auto smfq4_pattern = pattern::wrap_type<FakeQuantize>({smexp1_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smreshape6_pattern = pattern::wrap_type<Reshape>({smfq4_pattern, pattern::any_input()});
    auto smconv3_pattern = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({smreshape6_pattern, pattern::any_input()});
    auto smfq5_pattern = pattern::wrap_type<FakeQuantize>({smconv3_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smreshape7_pattern = pattern::wrap_type<Reshape>({smfq5_pattern, pattern::any_input()});
    auto smlog_pattern = pattern::wrap_type<Exp>({smreshape7_pattern});
    auto smfq6_pattern = pattern::wrap_type<FakeQuantize>({smlog_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto smmultiply2_pattern = pattern::wrap_type<Multiply>({smfq6_pattern, pattern::any_input()});
    auto smadd2_pattern = pattern::wrap_type<Add>({smfq3_pattern, pattern::any_input()});
    auto smfq7_pattern = pattern::wrap_type<FakeQuantize>({smadd2_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    /////////////////////// FOR SOME UNKNOWN REASON MATCHER CANNOT RECOGNIZE THIS ADD -- NEEDS MORE DETAILED DEBUGGING ///////////////////////
    auto smadd3_pattern = pattern::wrap_type<Add>({smfq7_pattern, smmultiply2_pattern});
    auto smfq8_pattern = pattern::wrap_type<FakeQuantize>({smadd3_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto softmax_pattern = pattern::wrap_type<Exp>({smfq8_pattern});
#endif
    auto matmulfq_pattern_2abc = pattern::wrap_type<FakeQuantize>({softmax_pattern,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto missingfq_pattern_2abc = std::make_shared<pattern::op::Or>(OutputVector{matmulfq_pattern_2abc, softmax_pattern});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({missingfq_pattern_2abc, transpose_pattern_3a});
    auto transposefq_pattern_4 = pattern::wrap_type<FakeQuantize>({matmul_pattern_2abc,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({transposefq_pattern_4, pattern::any_input()});
    auto reshape_pattern_2 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto matmulfq_pattern_3 = pattern::wrap_type<FakeQuantize>({reshape_pattern_2,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<MatMul>({matmulfq_pattern_3, pattern::any_input()});
    auto reshapefq_pattern_4a = pattern::wrap_type<FakeQuantize>({matmul_pattern_3,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto reshape_pattern_4a = pattern::wrap_type<Reshape>({reshapefq_pattern_4a, pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({reshape_pattern_4a, pattern::any_input()});
#else
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({reshape_pattern_2, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input()});
    auto matmuladd_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_3 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_3, matmuladd_pattern_3});
    auto reshapefq_pattern_4a = pattern::wrap_type<FakeQuantize>({matmulor_pattern_3,pattern::any_input(),pattern::any_input(),pattern::any_input(),pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({reshapefq_pattern_4a, pattern::any_input()});
#endif

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

#ifdef PRE_LAYOUT
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
#else
        auto transpose1a = pattern_map.at(transpose_pattern_1a).get_node_shared_ptr();
        auto transpose1b = pattern_map.at(transpose_pattern_1b).get_node_shared_ptr();
        auto transpose1c = pattern_map.at(transpose_pattern_1c).get_node_shared_ptr();
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
#endif
        auto matmul1a = pattern_map.at(matmul_pattern_1a).get_node_shared_ptr();
        auto matmul1b = pattern_map.at(matmul_pattern_1b).get_node_shared_ptr();
        auto matmul1c = pattern_map.at(matmul_pattern_1c).get_node_shared_ptr();
        //auto reshapefq1a = pattern_map.at(reshapefq_pattern_1a).get_node_shared_ptr();
        auto reshapefq1b = pattern_map.at(reshapefq_pattern_1b).get_node_shared_ptr();
        auto reshapefq1c = pattern_map.at(reshapefq_pattern_1c).get_node_shared_ptr();
        auto reshape1a = pattern_map.at(reshape_pattern_1a).get_node_shared_ptr();
        auto reshape1b = pattern_map.at(reshape_pattern_1b).get_node_shared_ptr();
        auto reshape1c = pattern_map.at(reshape_pattern_1c).get_node_shared_ptr();
        auto transpose3a = pattern_map.at(transpose_pattern_3a).get_node_shared_ptr();
        auto transpose3b = pattern_map.at(transpose_pattern_3b).get_node_shared_ptr();
        auto transpose3c = pattern_map.at(transpose_pattern_3c).get_node_shared_ptr();
#ifdef PRE_LAYOUT
        std::shared_ptr<ov::op::v0::FakeQuantize> transposefq3 = nullptr;
#else
        auto transposefq3c = pattern_map.at(transposefq_pattern_3c).get_node_shared_ptr();
#endif
        auto multiplyfq1 = pattern_map.at(multiplyfq_pattern_1).get_node_shared_ptr();
        auto multiply1 = pattern_map.at(multiply_pattern_1).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        //auto softmaxfq = pattern_map.at(softmaxfq_pattern).get_node_shared_ptr();
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        //auto matmulfq2abc = pattern_map.at(matmulfq_pattern_2abc).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transposefq4 = pattern_map.at(transposefq_pattern_4).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape2 = pattern_map.at(reshape_pattern_2).get_node_shared_ptr();

        auto children = matmul1a->output(0).get_target_inputs();
        auto add1a = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
        children = matmul1b->output(0).get_target_inputs();
        auto add1b = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
        children = matmul1c->output(0).get_target_inputs();
        auto add1c = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());

        children = matmul1a->output(0).get_target_inputs();
        auto reshapefq1a = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());

        children = matmul2bc->output(0).get_target_inputs();
        auto softmaxfq = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());

        children = softmax->output(0).get_target_inputs();
        if (children.size() != 2) {
            return false;
        }
        auto matmulfq2abc = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
#ifdef PRE_LAYOUT
        auto reducesum = std::dynamic_pointer_cast<ReduceSum>(children.begin()->get_node()->shared_from_this());
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiplyfq3 = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        children = multiplyfq3->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
#else
        std::shared_ptr<Reshape> reshape5 = nullptr;
        for (auto child : children) {
            reshape5 = std::dynamic_pointer_cast<Reshape>(child.get_node()->shared_from_this());
            if (reshape5 != nullptr) {
                break;
            }
        }
        if (reshape5 == nullptr) {
            return false;
        }
        children = reshape5->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto avgpool = std::dynamic_pointer_cast<AvgPool>(children.begin()->get_node()->shared_from_this());
        if (avgpool == nullptr) {
            return false;
        }
        children = avgpool->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply2 == nullptr) {
            return false;
        }
        children = multiply2->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto reducesum = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (reducesum == nullptr) {
            return false;
        }
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiplyfq3 = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        children = multiplyfq3->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
        auto multiplyconstfq3 = std::dynamic_pointer_cast<FakeQuantize>(multiply3->get_input_node_shared_ptr(1));
        if (multiplyconstfq3 == nullptr) {
            return false;
        }
#endif

#ifdef PRE_LAYOUT
        const Output<Node>& input1a = transpose2a->input_value(0);
        const Output<Node>& input1b = transpose2b->input_value(0);
        const Output<Node>& input1c = transpose2c->input_value(0);
        auto transpose2a_input_shape = transpose2a->input(0).get_shape();
        auto transpose2b_input_shape = transpose2b->input(0).get_shape();
        auto transpose2c_input_shape = transpose2c->input(0).get_shape();
        auto transpose2a_output_shape = transpose2a->output(0).get_shape();
        auto transpose2b_output_shape = transpose2b->output(0).get_shape();
        auto transpose2c_output_shape = transpose2c->output(0).get_shape();
        std::vector<int64_t> order2a, order2b, order2c;
        std::vector<int64_t> expected_order = {1, 0, 2};
        GnaGetTransposeOrder(order2a, transpose2a);
        GnaGetTransposeOrder(order2b, transpose2b);
        GnaGetTransposeOrder(order2c, transpose2c);
        if ((order2a != expected_order) || (order2b != expected_order) || (order2c != expected_order)) {
            return false;
        }
#else
        const Output<Node>& input1a = transpose1a->input_value(0);
        const Output<Node>& input1b = transpose1b->input_value(0);
        const Output<Node>& input1c = transpose1c->input_value(0);
        auto transpose1a_input_shape = transpose1a->input(0).get_shape();
        auto transpose1b_input_shape = transpose1b->input(0).get_shape();
        auto transpose1c_input_shape = transpose1c->input(0).get_shape();
        auto transpose1a_output_shape = transpose1a->output(0).get_shape();
        auto transpose1b_output_shape = transpose1b->output(0).get_shape();
        auto transpose1c_output_shape = transpose1c->output(0).get_shape();
#endif
        if ((transpose1a_input_shape != transpose1b_input_shape) || (transpose1b_input_shape != transpose1c_input_shape)) {
            return false;
        }
        auto transpose3c_output_shape = transpose3c->output(0).get_shape();
        auto C = transpose3c_output_shape[0];
        auto H = transpose3c_output_shape[1];
        auto W = transpose3c_output_shape[2];
        OutputVector part_a = GnaTransposeSplit(reshape1a->output(0), H, W, C, true, true, false, true);
        OutputVector part_b = GnaTransposeSplit(reshape1b->output(0), H, W, C, true, false, true, true);
        OutputVector part_c = GnaTransposeSplit(reshape1c->output(0), H, W, C, true, true, true, true);
        OutputVector out_a, out_b;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiplyfq1->input_value(0).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiplyfq1 = CopyFQ(new_mul_weight->output(0), multiplyfq1);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_multiplyfq1->output(0));
            auto new_transposefq3c = CopyFQ(new_multiply1->output(0), transposefq3c);
            auto new_matmul2bc = std::make_shared<MatMul>(new_transposefq3c->output(0), part_b[i], false, false);
            auto new_softmaxfq = CopyFQ(new_matmul2bc->output(0), softmaxfq);
            std::shared_ptr<ov::op::v8::Softmax> new_softmax;
            if (new_softmaxfq) {
                new_softmax = std::make_shared<Softmax>(new_softmaxfq->output(0), 1);
            } else {
                new_softmax = std::make_shared<Softmax>(new_matmul2bc->output(0), 1);
            }
            out_b.push_back(new_softmax->output(0));
            auto new_matmulfq2abc = CopyFQ(new_softmax->output(0), matmulfq2abc);
            std::shared_ptr<ov::op::v0::MatMul> new_matmul2abc;
            if (new_matmulfq2abc) {
                new_matmul2abc = std::make_shared<MatMul>(new_matmulfq2abc->output(0), part_a[i], false, false);
            } else {
                new_matmul2abc = std::make_shared<MatMul>(new_softmax->output(0), part_a[i], false, false);
            }
            auto new_transposefq4 = CopyFQ(new_matmul2abc->output(0), transposefq4);
            auto new_transpose3 = std::make_shared<Transpose>(new_transposefq4->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            out_a.push_back(new_transpose3->output(0));
        }
        auto new_concat3 = std::make_shared<Concat>(out_a, 0);
        auto new_transpose3 = std::make_shared<Transpose>(new_concat3->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        auto new_reshape2 = std::make_shared<Reshape>(new_transpose3->output(0),
            Constant::create(element::i64, Shape{3}, {1ull, H, C * W})->output(0),false);
        replace_output_update_name(reshape2, new_reshape2);

        auto new_concat4 = std::make_shared<Concat>(out_b, 0);
        auto new_reshape5 = std::make_shared<Reshape>(new_concat4->output(0),
            Constant::create(element::i64, Shape{2}, {C, H * H})->output(0),false);
        auto new_transpose4 = std::make_shared<Transpose>(new_reshape5->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));

#define PAR_CONV
#define NUM_K 16
#ifdef PAR_CONV
        size_t num_K = NUM_K;
        while (H * H % num_K != 0) {  // reduce parallelism if sizes don't match
            num_K = num_K / 2;
        }
        if (num_K != NUM_K) {
            printf("Warning:  failed to optimize reducesum\n");
        }
        // replace reducesum + multiply-by-1/C with multi-channel convolution
        new_reshape5 = std::make_shared<Reshape>(
            new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H / num_K, C * num_K, 1ull})->output(0),false);
        std::vector<float> weight(num_K * C * num_K, 0.0);
        for (size_t i = 0; i < num_K; i++) {
            for (size_t j = 0; j < C; j++) {
                weight[i * num_K * C + i * C + j] = 1.0f / C;
            }
        }
        auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, C * num_K, 1}, weight);
        auto weightfq = CopyFQ(weight_const->output(0), multiplyconstfq3);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0), weightfq->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#else
        // replace reducesum + multiply-by-1/C with pointwise convolution and sum pool
        new_reshape5 = std::make_shared<Reshape>(new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H, C, 1ull})->output(0),false);
        std::vector<float> weight(C, 1.0f / C);
        auto weight_const = Constant::create(ngraph::element::f32, Shape{1, 1, C, 1}, weight);
        auto weightfq = CopyFQ(weight_const->output(0), multiplyconstfq3);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0), weightfq->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        //auto new_sumpool = std::make_shared<ov::intel_gna::op::GNASumPool>(new_conv4->output(0), Strides{1, 1}, 
        //    Shape{0, 0}, Shape{0, 0}, Shape{1ull, C}, ov::op::RoundingType::FLOOR, ov::op::PadType::EXPLICIT);
#endif

        auto new_reshape6 = std::make_shared<Reshape>(new_conv4->output(0), 
            Constant::create(element::i64, Shape{2}, {1ull, H * H})->output(0), false);
        replace_output_update_name(multiply3, new_reshape6);

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(reshape_pattern_4, matcher_name);
    this->register_matcher(m, callback);
}


//#define PRE_LAYOUT
// GNA Multihead Attention factorization for case where Q=K=V

GnaMhaQKVTransformation::GnaMhaQKVTransformation() {
    MATCHER_SCOPE(GnaMhaQKVTransformation);

#ifdef PRE_LAYOUT
    auto transpose_pattern_1 = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1 = pattern::wrap_type<MatMul>({transpose_pattern_1, pattern::any_input()});
    auto matmuladd_pattern_1 = pattern::wrap_type<Add>({matmul_pattern_1, pattern::any_input()});
    auto matmulor_pattern_1 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1, matmuladd_pattern_1});
#else
    auto transpose_pattern_1 = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_1, pattern::any_input()});
    auto matmuladd_pattern_1 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_1, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_1 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1, matmuladd_pattern_1});
#endif
    auto reshape_pattern_1 = pattern::wrap_type<Reshape>({matmulor_pattern_1, pattern::any_input()});
    auto variadicsplit_pattern = pattern::wrap_type<VariadicSplit>({reshape_pattern_1, pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_2a = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto reshape_pattern_2b = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto reshape_pattern_2c = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto transpose_pattern_2a = pattern::wrap_type<Transpose>({reshape_pattern_2a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Transpose>({reshape_pattern_2b, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({reshape_pattern_2c, pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({transpose_pattern_2c, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({multiply_pattern_1, transpose_pattern_3b});
#else
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_2c, pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({multiply_pattern_1, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({transpose_pattern_2c, transpose_pattern_2b});
#endif
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({matmul_pattern_2bc, pattern::any_input()});
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({reshape_pattern_3});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({softmax_pattern, pattern::any_input()});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({reshape_pattern_4, transpose_pattern_2a});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({matmul_pattern_2abc, pattern::any_input()});
    auto reshape_pattern_7 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
#ifdef PRE_LAYOUT
    auto matmul_pattern_3 = pattern::wrap_type<MatMul>({reshape_pattern_2, pattern::any_input()});
    auto reshape_pattern_4a = pattern::wrap_type<Reshape>({matmul_pattern_3, pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({reshape_pattern_4a, pattern::any_input()});
#else
    auto reshape_pattern_8 = pattern::wrap_type<Reshape>({reshape_pattern_7, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_8, pattern::any_input()});
    auto matmuladd_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_8, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_3 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_3, matmuladd_pattern_3});
    auto reshape_pattern_9 = pattern::wrap_type<Reshape>({matmulor_pattern_3, pattern::any_input()});
#endif

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape1 = pattern_map.at(reshape_pattern_1).get_node_shared_ptr();
        auto reshape2a = pattern_map.at(reshape_pattern_2a).get_node_shared_ptr();
        auto reshape2b = pattern_map.at(reshape_pattern_2b).get_node_shared_ptr();
        auto reshape2c = pattern_map.at(reshape_pattern_2c).get_node_shared_ptr();
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
        auto multiply1 = pattern_map.at(multiply_pattern_1).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        auto reshape3 = pattern_map.at(reshape_pattern_3).get_node_shared_ptr();
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        auto reshape4 = pattern_map.at(reshape_pattern_4).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape7 = pattern_map.at(reshape_pattern_7).get_node_shared_ptr();

        auto children = reshape4->output(0).get_target_inputs();
        if (children.size() != 2) {
            return false;
        }
#ifdef PRE_LAYOUT
        auto reducesum = std::dynamic_pointer_cast<ReduceSum>(children.begin()->get_node()->shared_from_this());
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
#else
        std::shared_ptr<Reshape> reshape5 = nullptr;
        for (auto child : children) {
            reshape5 = std::dynamic_pointer_cast<Reshape>(child.get_node()->shared_from_this());
            if (reshape5 != nullptr) {
                break;
            }
        }
        if (reshape5 == nullptr) {
            return false;
        }
        children = reshape5->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto avgpool = std::dynamic_pointer_cast<AvgPool>(children.begin()->get_node()->shared_from_this());
        if (avgpool == nullptr) {
            return false;
        }
        children = avgpool->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply2 == nullptr) {
            return false;
        }
        children = multiply2->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto reducesum = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (reducesum == nullptr) {
            return false;
        }
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
#endif

        Output<Node>& input = reshape1->output(0);
        auto input_shape = input.get_shape();
        auto transpose2c_output_shape = transpose2c->output(0).get_shape();
        auto C = transpose2c_output_shape[0];
        auto H = transpose2c_output_shape[1];
        auto W = transpose2c_output_shape[2];
        OutputVector parts = GnaTransposeSplit(input, H, C * W, 3, true, false, false, false);
        OutputVector part_a = GnaTransposeSplit(parts[2], H, W, C, false, true, false, false);
        OutputVector part_b = GnaTransposeSplit(parts[1], H, W, C, false, false, true, false);
        OutputVector part_c = GnaTransposeSplit(parts[0], H, W, C, false, true, true, false);
        OutputVector out_a, out_b;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiply1->input_value(1).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_mul_weight->output(0));
            auto new_matmul2bc = std::make_shared<MatMul>(new_multiply1->output(0), part_b[i], false, false);
            auto new_softmax = std::make_shared<Softmax>(new_matmul2bc->output(0), 1);
            out_b.push_back(new_softmax->output(0));
            auto new_matmul2abc = std::make_shared<MatMul>(new_softmax->output(0), part_a[i], false, false);
            auto new_transpose4 = std::make_shared<Transpose>(new_matmul2abc->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            out_a.push_back(new_transpose4->output(0));
        }
        auto new_concat3 = std::make_shared<Concat>(out_a, 0);
        auto new_transpose5 = std::make_shared<Transpose>(new_concat3->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        auto new_reshape7 = std::make_shared<Reshape>(new_transpose5->output(0),
            Constant::create(element::i64, Shape{3}, {1ull, H, C * W})->output(0),false);
        // reshape to avoid 3D output bug while testing
        //auto new_reshape7b = std::make_shared<Reshape>(new_transpose5->output(0),
        //    Constant::create(element::i64, Shape{2}, {1ull, H * C * W})->output(0),false);
        replace_output_update_name(reshape7, new_reshape7);

        auto new_concat4 = std::make_shared<Concat>(out_b, 0);
        auto new_reshape5 = std::make_shared<Reshape>(new_concat4->output(0),
            Constant::create(element::i64, Shape{2}, {C, H * H})->output(0),false);
        auto new_transpose6 = std::make_shared<Transpose>(new_reshape5->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        
#define PAR_CONV
#define NUM_K 16
#ifdef PAR_CONV
        size_t num_K = NUM_K;
        while (H * H % num_K != 0) {  // reduce parallelism if sizes don't match
            num_K = num_K / 2;
        }
        if (num_K != NUM_K) {
            printf("Warning:  failed to optimize reducesum\n");
        }
        // replace reducesum + multiply-by-1/C with multi-channel convolution
        new_reshape5 = std::make_shared<Reshape>(new_transpose6->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H / num_K, C * num_K, 1ull})->output(0),false);
        std::vector<float> weight(num_K * C * num_K, 0.0);
        for (size_t i = 0; i < num_K; i++) {
            for (size_t j = 0; j < C; j++) {
                weight[i * num_K * C + i * C + j] = 1.0f / C;
            }
        }
        auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, C * num_K, 1}, weight);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0),weight_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#else
        // replace reducesum + multiply-by-1/C with pointwise convolution and sum pool
        new_reshape5 = std::make_shared<Reshape>(new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H, C, 1ull})->output(0),false);
        std::vector<float> weight(C, 1.0f / C);
        auto weight_const = Constant::create(ngraph::element::f32, Shape{1, 1, C, 1}, weight);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0),weight_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#endif
         
        auto new_reshape6 = std::make_shared<Reshape>(new_conv4->output(0), 
            Constant::create(element::i64, Shape{3}, {1ull, H, H})->output(0), false);
        // reshape to avoid 3D output bug while testing
        //auto new_reshape6b = std::make_shared<Reshape>(new_reshape6->output(0), 
        //    Constant::create(element::i64, Shape{2}, {1ull, H * H})->output(0), false);
        replace_output_update_name(multiply3, new_reshape6);

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(reshape_pattern_9, matcher_name);
    this->register_matcher(m, callback);
}

//#define PRE_LAYOUT
// GNA Multihead Attention factorization with FakeQuantize layers where Q=K=V

GnaMhaQKVFqTransformation::GnaMhaQKVFqTransformation() {
    MATCHER_SCOPE(GnaMhaQKVFqTransformation);

    auto inputfq_pattern = pattern::wrap_type<FakeQuantize>({pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_1 = pattern::wrap_type<Reshape>({inputfq_pattern, pattern::any_input()});
    auto matmul_pattern_1 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_1, pattern::any_input()});
    auto matmuladd_pattern_1 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transpose_pattern_1, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_1 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1, matmuladd_pattern_1});
    auto reshapefq_pattern_1 = pattern::wrap_type<FakeQuantize>({matmulor_pattern_1, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_1 = pattern::wrap_type<Reshape>({reshapefq_pattern_1, pattern::any_input()});
    auto variadicsplit_pattern = pattern::wrap_type<VariadicSplit>({reshape_pattern_1, pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_2a = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto reshape_pattern_2b = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto reshape_pattern_2c = pattern::wrap_type<Reshape>({variadicsplit_pattern, pattern::any_input()});
    auto transpose_pattern_2a = pattern::wrap_type<Transpose>({reshape_pattern_2a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Transpose>({reshape_pattern_2b, pattern::any_input()});
    auto multiplyfq_pattern_1 = pattern::wrap_type<FakeQuantize>({pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_2c, multiplyfq_pattern_1});
    auto transposefq_pattern_2c = pattern::wrap_type<FakeQuantize>({multiply_pattern_1, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({transposefq_pattern_2c, pattern::any_input()});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({transpose_pattern_2c, transpose_pattern_2b});
    auto reshapefq_pattern_3 = pattern::wrap_type<FakeQuantize>({matmul_pattern_2bc, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({reshapefq_pattern_3, pattern::any_input()});
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({reshape_pattern_3});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({softmax_pattern, pattern::any_input()});
    auto matmulfq_pattern_2abc = pattern::wrap_type<FakeQuantize>({reshape_pattern_4, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({matmulfq_pattern_2abc, transpose_pattern_2a});
    auto transposefq_pattern_4 = pattern::wrap_type<FakeQuantize>({matmul_pattern_2abc, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({transposefq_pattern_4, pattern::any_input()});
    auto reshape_pattern_7 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
    auto reshape_pattern_8 = pattern::wrap_type<Reshape>({reshape_pattern_7, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_8, pattern::any_input()});
    auto matmuladd_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_8, pattern::any_input(), pattern::any_input()});
    auto matmulor_pattern_3 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_3, matmuladd_pattern_3});
    auto reshapefq_pattern_9 = pattern::wrap_type<FakeQuantize>({matmulor_pattern_3, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_9 = pattern::wrap_type<Reshape>({reshapefq_pattern_9, pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape1 = pattern_map.at(reshape_pattern_1).get_node_shared_ptr();
        auto reshape2a = pattern_map.at(reshape_pattern_2a).get_node_shared_ptr();
        auto reshape2b = pattern_map.at(reshape_pattern_2b).get_node_shared_ptr();
        auto reshape2c = pattern_map.at(reshape_pattern_2c).get_node_shared_ptr();
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto transpose2c = pattern_map.at(transpose_pattern_2c).get_node_shared_ptr();
        auto multiplyfq1 = pattern_map.at(multiplyfq_pattern_1).get_node_shared_ptr();
        auto multiply1 = pattern_map.at(multiply_pattern_1).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        auto reshapefq3 = pattern_map.at(reshapefq_pattern_3).get_node_shared_ptr();
        auto reshape3 = pattern_map.at(reshape_pattern_3).get_node_shared_ptr();
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        auto reshape4 = pattern_map.at(reshape_pattern_4).get_node_shared_ptr();
        auto matmulfq2abc = pattern_map.at(matmulfq_pattern_2abc).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transposefq4 = pattern_map.at(transposefq_pattern_4).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape7 = pattern_map.at(reshape_pattern_7).get_node_shared_ptr();

        auto children = reshape4->output(0).get_target_inputs();
        if (children.size() != 2) {
            return false;
        }
        std::shared_ptr<Reshape> reshape5 = nullptr;
        for (auto child : children) {
            reshape5 = std::dynamic_pointer_cast<Reshape>(child.get_node()->shared_from_this());
            if (reshape5 != nullptr) {
                break;
            }
        }
        if (reshape5 == nullptr) {
            return false;
        }
        children = reshape5->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto avgpool = std::dynamic_pointer_cast<AvgPool>(children.begin()->get_node()->shared_from_this());
        if (avgpool == nullptr) {
            return false;
        }
        children = avgpool->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply2 == nullptr) {
            return false;
        }
        children = multiply2->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto reducesum = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (reducesum == nullptr) {
            return false;
        }
        children = reducesum->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiplyfq3 = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        if (multiplyfq3 == nullptr) {
            return false;
        }
        children = multiplyfq3->output(0).get_target_inputs();
        if (children.size() != 1) {
            return false;
        }
        auto multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (multiply3 == nullptr) {
            return false;
        }
        auto multiplyconstfq3 = std::dynamic_pointer_cast<FakeQuantize>(multiply3->get_input_node_shared_ptr(1));
        if (multiplyconstfq3 == nullptr) {
            return false;
        }

        Output<Node>& input = reshape1->output(0);
        auto input_shape = input.get_shape();
        auto transpose2c_output_shape = transpose2c->output(0).get_shape();
        auto C = transpose2c_output_shape[0];
        auto H = transpose2c_output_shape[1];
        auto W = transpose2c_output_shape[2];
        OutputVector parts = GnaTransposeSplit(input, H, C * W, 3, true, false, false, true);
        OutputVector part_a = GnaTransposeSplit(parts[2], H, W, C, false, true, false, true);
        OutputVector part_b = GnaTransposeSplit(parts[1], H, W, C, false, false, true, true);
        OutputVector part_c = GnaTransposeSplit(parts[0], H, W, C, false, true, true, true);
        OutputVector out_a, out_b;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiplyfq1->input_value(0).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiplyfq1 = CopyFQ(new_mul_weight->output(0), multiplyfq1);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_multiplyfq1->output(0));
            auto new_matmul2bc = std::make_shared<MatMul>(new_multiply1->output(0), part_b[i], false, false);
            auto new_reshapefq3 = CopyFQ(new_matmul2bc->output(0), reshapefq3);
            auto new_softmax = std::make_shared<Softmax>(new_reshapefq3->output(0), 1);
            auto new_matmulfq2abc = CopyFQ(new_softmax->output(0), matmulfq2abc);
            out_b.push_back(new_softmax->output(0));
            auto new_matmul2abc = std::make_shared<MatMul>(new_matmulfq2abc->output(0), part_a[i], false, false);
            auto new_transposefq4 = CopyFQ(new_matmul2abc->output(0), transposefq4);
            auto new_transpose4 = std::make_shared<Transpose>(new_transposefq4->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            out_a.push_back(new_transpose4->output(0));
        }
        auto new_concat3 = std::make_shared<Concat>(out_a, 0);
        auto new_transpose4 = std::make_shared<Transpose>(new_concat3->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        auto new_reshape7 = std::make_shared<Reshape>(new_transpose4->output(0),
            Constant::create(element::i64, Shape{3}, {1ull, H, C * W})->output(0),false);
        replace_output_update_name(reshape7, new_reshape7);

        auto new_concat4 = std::make_shared<Concat>(out_b, 0);
        auto new_reshape5 = std::make_shared<Reshape>(new_concat4->output(0),
            Constant::create(element::i64, Shape{2}, {C, H * H})->output(0),false);
        auto new_transpose6 = std::make_shared<Transpose>(new_reshape5->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));

#define PAR_CONV
#define NUM_K 16
#ifdef PAR_CONV
        size_t num_K = NUM_K;
        while (H * H % num_K != 0) {  // reduce parallelism if sizes don't match
            num_K = num_K / 2;
        }
        if (num_K != NUM_K) {
            printf("Warning:  failed to optimize reducesum\n");
        }
        // replace reducesum + multiply-by-1/C with multi-channel convolution
        new_reshape5 = std::make_shared<Reshape>(new_transpose6->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H / num_K, C * num_K, 1ull})->output(0),false);
        std::vector<float> weight(num_K * C * num_K, 0.0);
        for (size_t i = 0; i < num_K; i++) {
            for (size_t j = 0; j < C; j++) {
                weight[i * num_K * C + i * C + j] = 1.0f / C;
            }
        }
        auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, C * num_K, 1}, weight);
        auto weightfq = CopyFQ(weight_const->output(0), multiplyconstfq3);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0), weightfq->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
#else
        // replace reducesum + multiply-by-1/C with pointwise convolution and sum pool
        new_reshape5 = std::make_shared<Reshape>(new_transpose4->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, H * H, C, 1ull})->output(0),false);
        std::vector<float> weight(C, 1.0f / C);
        auto weight_const = Constant::create(ngraph::element::f32, Shape{1, 1, C, 1}, weight);
        auto weightfq = CopyFQ(weight_const->output(0), multiplyconstfq3);
        auto new_conv4 = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape5->output(0), weightfq->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        //auto new_sumpool = std::make_shared<ov::intel_gna::op::GNASumPool>(new_conv4->output(0), Strides{1, 1}, 
        //    Shape{0, 0}, Shape{0, 0}, Shape{1ull, C}, ov::op::RoundingType::FLOOR, ov::op::PadType::EXPLICIT);
#endif

        auto new_reshape6 = std::make_shared<Reshape>(new_conv4->output(0), 
            Constant::create(element::i64, Shape{2}, {1ull, H * H})->output(0), false);
        replace_output_update_name(multiply3, new_reshape6);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_pattern_9, matcher_name);
    this->register_matcher(m, callback);
}
