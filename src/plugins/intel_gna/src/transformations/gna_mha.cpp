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

//#define PRE_LAYOUT

// GNA Multihead Attention factorization

GnaMhaTransformation::GnaMhaTransformation() {
    MATCHER_SCOPE(GnaMhaTransformation);

    auto matmul_pattern_1a = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input()});
    auto matmuladd_pattern_1a = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto matmuladd_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto matmuladd_pattern_1c = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto or_pattern_1a = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1a, matmuladd_pattern_1a});
    auto or_pattern_1b = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1b, matmuladd_pattern_1b});
    auto or_pattern_1c = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1c, matmuladd_pattern_1c});
    auto reshape_pattern_1a = pattern::wrap_type<Reshape>({or_pattern_1a, pattern::any_input()});
    auto reshape_pattern_1b = pattern::wrap_type<Reshape>({or_pattern_1b, pattern::any_input()});
    auto reshape_pattern_1c = pattern::wrap_type<Reshape>({or_pattern_1c, pattern::any_input()});
    auto transpose_pattern_3a = pattern::wrap_type<Transpose>({reshape_pattern_1a, pattern::any_input()});
    auto transpose_pattern_3b = pattern::wrap_type<Transpose>({reshape_pattern_1b, pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_1c, pattern::any_input()});
    auto transpose_pattern_3c = pattern::wrap_type<Transpose>({multiply_pattern_1, pattern::any_input()});
    auto transpose_pattern_3c_ = pattern::wrap_type<Transpose>({reshape_pattern_1c, pattern::any_input()});
    auto multiply_pattern_1_ = pattern::wrap_type<Multiply>({transpose_pattern_3c_, pattern::any_input()});
    auto or_pattern_3c = std::make_shared<pattern::op::Or>(OutputVector{transpose_pattern_3c, multiply_pattern_1_});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({or_pattern_3c, transpose_pattern_3b});
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({matmul_pattern_2bc});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({softmax_pattern, transpose_pattern_3a});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({matmul_pattern_2abc, pattern::any_input()});
    auto reshape_pattern_2 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({reshape_pattern_2, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input()});
    auto matmuladd_pattern_3 = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_3, pattern::any_input(), pattern::any_input()});
    auto transpose_pattern_5 = pattern::wrap_type<Transpose>({matmul_pattern_3, pattern::any_input()});
    auto add_pattern_3 = pattern::wrap_type<Add>({transpose_pattern_5, pattern::any_input()});
    auto matmulor_pattern_3 = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_3, matmuladd_pattern_3, add_pattern_3});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({matmulor_pattern_3, pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape1a = pattern_map.at(reshape_pattern_1a).get_node_shared_ptr();
        auto reshape1b = pattern_map.at(reshape_pattern_1b).get_node_shared_ptr();
        auto reshape1c = pattern_map.at(reshape_pattern_1c).get_node_shared_ptr();
        auto matmul1a = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(reshape1a->input_value(0).get_node_shared_ptr());
        auto matmul1b = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(reshape1b->input_value(0).get_node_shared_ptr());
        auto matmul1c = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(reshape1c->input_value(0).get_node_shared_ptr());
        auto transpose3a = pattern_map.at(transpose_pattern_3a).get_node_shared_ptr();
        auto transpose3b = pattern_map.at(transpose_pattern_3b).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        auto multiply1 = std::dynamic_pointer_cast<Multiply>(matmul2bc->input_value(0).get_node_shared_ptr());
        auto transpose3c = std::dynamic_pointer_cast<Transpose>(matmul2bc->input_value(0).get_node_shared_ptr());
        if (multiply1 == nullptr) {
            multiply1 = std::dynamic_pointer_cast<Multiply>(transpose3c->input_value(0).get_node_shared_ptr());
        } else {
            transpose3c = std::dynamic_pointer_cast<Transpose>(multiply1->input_value(0).get_node_shared_ptr());
        }
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape2 = pattern_map.at(reshape_pattern_2).get_node_shared_ptr();

        auto children = softmax->output(0).get_target_inputs();
        if (children.size() > 2) {
            return false;
        }
        std::shared_ptr<Reshape> reshape5 = nullptr;
        std::shared_ptr<ov::op::v1::Multiply> multiply3 = nullptr;
        std::shared_ptr<AvgPool> avgpool = nullptr;
        for (auto child : children) {
            reshape5 = std::dynamic_pointer_cast<Reshape>(child.get_node()->shared_from_this());
            if (reshape5 != nullptr) {
                break;
            }
        }
        if (reshape5 != nullptr) {  // optional attention weights needed
            children = reshape5->output(0).get_target_inputs();
            if (children.size() != 1) {
                return false;
            }
            avgpool = std::dynamic_pointer_cast<AvgPool>(children.begin()->get_node()->shared_from_this());
            if (avgpool == nullptr) {
                return false;
            }
            children = avgpool->output(0).get_target_inputs();
            if (children.size() != 1) {
                return false;
            }
            auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
            if (multiply2 != nullptr) {
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
                multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
                if (multiply3 == nullptr) {
                    return false;
                }
            }
        }

        const Output<Node>& input1a = matmul1a->input_value(0);
        const Output<Node>& input1b = matmul1b->input_value(0);
        const Output<Node>& input1c = matmul1c->input_value(0);
        auto transpose3c_output_shape = transpose3c->output(0).get_shape();
        auto C = transpose3c_output_shape[0];
        auto H = transpose3c_output_shape[1];
        auto W = transpose3c_output_shape[2];
        OutputVector part_a = GnaTransposeSplit(reshape1a->output(0), H, W, C, true, true, false, false);
        OutputVector part_b = GnaTransposeSplit(reshape1b->output(0), H, W, C, true, false, true, false);
        OutputVector part_c = GnaTransposeSplit(reshape1c->output(0), H, W, C, true, true, true, false);
        OutputVector out_a, out_b;
        OutputVector tmp;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiply1->input_value(1).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_mul_weight->output(0));
            auto new_matmul2bc = InsertGnaMatMulAdd2D(new_multiply1->output(0), part_b[i], false, false, true);
            auto new_reshape = std::make_shared<Reshape>(new_matmul2bc->output(0),
                Constant::create(element::i64, Shape{3}, {1ull, H, H})->output(0),false);
            tmp.push_back(new_reshape->output(0));
        }

        auto combined_new_matmul2bc = std::make_shared<Concat>(tmp, 0);
        auto new_softmax = std::make_shared<Softmax>(combined_new_matmul2bc->output(0), 2);
        std::vector<size_t> split_lengths(C, 1);
        const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
        auto split_softmax = std::make_shared<VariadicSplit>(new_softmax->output(0), Constant::create(ov::element::i64, ov::Shape{}, {0}), split_lengths_const);

        for (auto i = 0; i < part_c.size(); i++) {
            auto new_reshape = std::make_shared<Reshape>(split_softmax->output(i),
                Constant::create(element::i64, Shape{2}, {H, H})->output(0),false);
            out_b.push_back(new_reshape->output(0));
            auto new_matmul2abc = InsertGnaMatMulAdd2D(new_reshape->output(0), part_a[i], false, false, true);
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

        if (reshape5 != nullptr) {  // optional attention weights are needed
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
            new_reshape5 = std::make_shared<Reshape>(new_transpose4->output(0),
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
            if (multiply3) {
                replace_output_update_name(multiply3, new_reshape6);
            } else {
                replace_output_update_name(avgpool, new_reshape6);
            }
        }

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
    auto transposeor_pattern_1a = std::make_shared<pattern::op::Or>(OutputVector{transpose_pattern_1a, transpose_pattern_2a});
    auto transposeor_pattern_1b = std::make_shared<pattern::op::Or>(OutputVector{transpose_pattern_1b, transpose_pattern_2b});
    auto transposeor_pattern_1c = std::make_shared<pattern::op::Or>(OutputVector{transpose_pattern_1c, transpose_pattern_2c});
    auto matmul_pattern_1a = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transposeor_pattern_1a, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transposeor_pattern_1b, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({transposeor_pattern_1c, pattern::any_input()});
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
        auto matmulfq2abc = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        if (children.size() != 2) {
            if (matmulfq2abc) {
                children = matmulfq2abc->output(0).get_target_inputs();
                if (children.size() != 2) {
                    return false;
                }
            } else {
                return false;
            }
        }

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
        std::shared_ptr<ov::op::v1::Multiply> multiply3 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> multiplyconstfq3 = nullptr;
        if (multiply2 != nullptr) {
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
            multiply3 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
            if (multiply3 == nullptr) {
                return false;
            }
            multiplyconstfq3 = std::dynamic_pointer_cast<FakeQuantize>(multiply3->get_input_node_shared_ptr(1));
            if (multiplyconstfq3 == nullptr) {
                return false;
            }
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
        auto weightfq = InsertWeights(Shape{num_K, 1, C * num_K, 1}, weight, true);
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
        if (multiply3) {
            replace_output_update_name(multiply3, new_reshape6);
        } else {
            replace_output_update_name(avgpool, new_reshape6);
        }

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

// GNA Multihead Self Attention factorization

GnaMhaSelfTransformation::GnaMhaSelfTransformation() {
    MATCHER_SCOPE(GnaMhaSelfTransformation);

    auto reshape_pattern_1 = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<MatMul>({reshape_pattern_1, pattern::any_input()});
    auto matmul_pattern_1b = pattern::wrap_type<MatMul>({reshape_pattern_1, pattern::any_input()});
    auto matmul_pattern_1c = pattern::wrap_type<MatMul>({reshape_pattern_1, pattern::any_input()});
    auto add_pattern_1a = pattern::wrap_type<Add>({matmul_pattern_1a, pattern::any_input()});
    auto add_pattern_1b = pattern::wrap_type<Add>({matmul_pattern_1b, pattern::any_input()});
    auto add_pattern_1c = pattern::wrap_type<Add>({matmul_pattern_1c, pattern::any_input()});
    auto reshape_pattern_2a = pattern::wrap_type<Reshape>({add_pattern_1a, pattern::any_input()});
    auto reshape_pattern_2b = pattern::wrap_type<Reshape>({add_pattern_1b, pattern::any_input()});
    auto reshape_pattern_2c = pattern::wrap_type<Reshape>({add_pattern_1c, pattern::any_input()});
    auto transpose_pattern_2a = pattern::wrap_type<Transpose>({reshape_pattern_2a, pattern::any_input()});
    auto transpose_pattern_2b = pattern::wrap_type<Transpose>({reshape_pattern_2b, pattern::any_input()});
    auto transpose_pattern_2c = pattern::wrap_type<Transpose>({reshape_pattern_2c, pattern::any_input()});
    auto trans_mult_pattern_2c = pattern::wrap_type<Multiply>({transpose_pattern_2c, pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({reshape_pattern_2c, pattern::any_input()});
    auto mult_trans_pattern_2c = pattern::wrap_type<Transpose>({multiply_pattern_1, pattern::any_input()});
    auto or_pattern2c = std::make_shared<pattern::op::Or>(OutputVector{trans_mult_pattern_2c, mult_trans_pattern_2c});
    auto matmul_pattern_2bc = pattern::wrap_type<MatMul>({or_pattern2c, transpose_pattern_2b});
    auto softmax_pattern = pattern::wrap_type<ov::op::v1::Softmax>({matmul_pattern_2bc});
    auto matmul_pattern_2abc = pattern::wrap_type<MatMul>({softmax_pattern, transpose_pattern_2a});
    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({matmul_pattern_2abc, pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
    auto matmul_pattern_5 = pattern::wrap_type<MatMul>({reshape_pattern_4, pattern::any_input()});
    auto transpose_pattern_5 = pattern::wrap_type<Transpose>({matmul_pattern_5, pattern::any_input()});
    auto add_pattern_5a = pattern::wrap_type<Add>({transpose_pattern_5, pattern::any_input()});
    auto add_pattern_5b = pattern::wrap_type<Add>({matmul_pattern_5, pattern::any_input()});
    auto add_pattern_5ab = std::make_shared<pattern::op::Or>(OutputVector{add_pattern_5a, add_pattern_5b});
    auto reshape_pattern_6 = pattern::wrap_type<Reshape>({add_pattern_5ab, pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape1 = pattern_map.at(reshape_pattern_1).get_node_shared_ptr();
        auto matmul1a_node = pattern_map.at(matmul_pattern_1a).get_node_shared_ptr();
        auto matmul1b_node = pattern_map.at(matmul_pattern_1b).get_node_shared_ptr();
        auto matmul1c_node = pattern_map.at(matmul_pattern_1c).get_node_shared_ptr();
        auto add1a_node = pattern_map.at(add_pattern_1a).get_node_shared_ptr();
        auto add1b_node = pattern_map.at(add_pattern_1b).get_node_shared_ptr();
        auto add1c_node = pattern_map.at(add_pattern_1c).get_node_shared_ptr();
        auto reshape2a = pattern_map.at(reshape_pattern_2a).get_node_shared_ptr();
        auto reshape2b = pattern_map.at(reshape_pattern_2b).get_node_shared_ptr();
        auto reshape2c = pattern_map.at(reshape_pattern_2c).get_node_shared_ptr();
        auto transpose2a = pattern_map.at(transpose_pattern_2a).get_node_shared_ptr();
        auto transpose2b = pattern_map.at(transpose_pattern_2b).get_node_shared_ptr();
        auto matmul2bc = pattern_map.at(matmul_pattern_2bc).get_node_shared_ptr();
        auto softmax = pattern_map.at(softmax_pattern).get_node_shared_ptr();
        auto matmul2abc = pattern_map.at(matmul_pattern_2abc).get_node_shared_ptr();
        auto transpose4 = pattern_map.at(transpose_pattern_4).get_node_shared_ptr();
        auto reshape4 = pattern_map.at(reshape_pattern_4).get_node_shared_ptr();
        auto matmul5_node = pattern_map.at(matmul_pattern_5).get_node_shared_ptr();
        auto reshape6 = pattern_map.at(reshape_pattern_6).get_node_shared_ptr();

        auto matmul1a = std::dynamic_pointer_cast<MatMul>(matmul1a_node);
        auto matmul1b = std::dynamic_pointer_cast<MatMul>(matmul1b_node);
        auto matmul1c = std::dynamic_pointer_cast<MatMul>(matmul1c_node);
        auto transpose_a = matmul1a->get_transpose_a();
        auto transpose_b = matmul1a->get_transpose_b();
        auto weights1a = std::dynamic_pointer_cast<Constant>(matmul1a->input_value(1).get_node_shared_ptr())->output(0);
        auto weights1b = std::dynamic_pointer_cast<Constant>(matmul1b->input_value(1).get_node_shared_ptr())->output(0);
        auto weights1c = std::dynamic_pointer_cast<Constant>(matmul1c->input_value(1).get_node_shared_ptr())->output(0);
        auto add1a = std::dynamic_pointer_cast<Add>(add1a_node);
        auto add1b = std::dynamic_pointer_cast<Add>(add1b_node);
        auto add1c = std::dynamic_pointer_cast<Add>(add1c_node);
        auto bias1a_const = std::dynamic_pointer_cast<Constant>(add1a->input_value(0).get_node_shared_ptr());
        bias1a_const = (bias1a_const) ? bias1a_const : std::dynamic_pointer_cast<Constant>(add1a->input_value(1).get_node_shared_ptr());
        auto bias1b_const = std::dynamic_pointer_cast<Constant>(add1a->input_value(0).get_node_shared_ptr());
        bias1b_const = (bias1b_const) ? bias1b_const : std::dynamic_pointer_cast<Constant>(add1b->input_value(1).get_node_shared_ptr());
        auto bias1c_const = std::dynamic_pointer_cast<Constant>(add1c->input_value(0).get_node_shared_ptr());
        bias1c_const = (bias1c_const) ? bias1c_const : std::dynamic_pointer_cast<Constant>(add1c->input_value(1).get_node_shared_ptr());
        auto bias1a = bias1a_const->output(0);
        auto bias1b = bias1b_const->output(0);
        auto bias1c = bias1c_const->output(0);
        auto new_matmul1a = InsertGnaMatMulAdd2D(reshape1->output(0), weights1a, bias1a, transpose_a, transpose_b, true);
        auto new_matmul1b = InsertGnaMatMulAdd2D(reshape1->output(0), weights1b, bias1b, transpose_a, transpose_b, true);
        auto new_matmul1c = InsertGnaMatMulAdd2D(reshape1->output(0), weights1c, bias1c, transpose_a, transpose_b, true);

        auto transpose2c = std::dynamic_pointer_cast<Transpose>(matmul2bc->input_value(0).get_node_shared_ptr());
        auto multiply1 = std::dynamic_pointer_cast<Multiply>(matmul2bc->input_value(0).get_node_shared_ptr());
        if (transpose2c) {
            multiply1 = std::dynamic_pointer_cast<Multiply>(transpose2c->input_value(0).get_node_shared_ptr());
        } else {
            transpose2c = std::dynamic_pointer_cast<Transpose>(multiply1->input_value(0).get_node_shared_ptr());
        }
        auto transpose2c_output_shape = transpose2c->output(0).get_shape();
        auto C = transpose2c_output_shape[0];
        auto H = transpose2c_output_shape[1];
        auto W = transpose2c_output_shape[2];
        OutputVector part_a = GnaTransposeSplit(new_matmul1a->output(0), H, W, C, true, true, false, false);
        OutputVector part_b = GnaTransposeSplit(new_matmul1b->output(0), H, W, C, true, false, true, false);
        OutputVector part_c = GnaTransposeSplit(new_matmul1c->output(0), H, W, C, true, true, true, false);
        OutputVector out_a, out_b;
        OutputVector tmp;
        for (auto i = 0; i < part_c.size(); i++) {
            auto mul_weight = std::dynamic_pointer_cast<Constant>(multiply1->input_value(1).get_node()->shared_from_this());
            const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
            auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);
            auto new_multiply1 = std::make_shared<Multiply>(part_c[i], new_mul_weight->output(0));
            auto new_matmul2bc = InsertGnaMatMulAdd2D(new_multiply1->output(0), part_b[i], false, false, true);
            auto new_reshape = std::make_shared<Reshape>(new_matmul2bc->output(0),
                Constant::create(element::i64, Shape{3}, {1ull, H, H})->output(0),false);
            tmp.push_back(new_reshape->output(0));
        }

        auto combined_new_matmul2bc = std::make_shared<Concat>(tmp, 0);
        auto new_softmax = std::make_shared<Softmax>(combined_new_matmul2bc->output(0), 2);
        std::vector<size_t> split_lengths(C, 1);
        const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
        auto split_softmax = std::make_shared<VariadicSplit>(new_softmax->output(0), Constant::create(ov::element::i64, ov::Shape{}, {0}), split_lengths_const);

        for (auto i = 0; i < part_c.size(); i++) {
            auto new_reshape = std::make_shared<Reshape>(split_softmax->output(i),
                Constant::create(element::i64, Shape{2}, {H, H})->output(0),false);
            out_b.push_back(new_reshape->output(0));
            auto new_matmul2abc = InsertGnaMatMulAdd2D(new_reshape->output(0), part_a[i], false, false, true);
            auto new_transpose3 = std::make_shared<Transpose>(new_matmul2abc->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            out_a.push_back(new_transpose3->output(0));
        }
        auto new_concat3 = std::make_shared<Concat>(out_a, 0);
        auto matmul5 = std::dynamic_pointer_cast<MatMul>(matmul5_node);
        if (matmul5 == nullptr) {
            return false;
        }
        transpose_a = matmul5->get_transpose_a();
        transpose_b = matmul5->get_transpose_b();
        auto weights5 = std::dynamic_pointer_cast<Constant>(matmul5->input_value(1).get_node_shared_ptr())->output(0);
        auto add5 = std::dynamic_pointer_cast<Add>(reshape6->input_value(0).get_node_shared_ptr());
        if (add5 == nullptr) {
            return false;
        }
        auto bias5_const = std::dynamic_pointer_cast<Constant>(add5->input_value(0).get_node_shared_ptr());
        bias5_const = (bias5_const) ? bias5_const : std::dynamic_pointer_cast<Constant>(add5->input_value(1).get_node_shared_ptr());
        auto bias5 = bias5_const->output(0);
        auto new_matmul5 = InsertGnaMatMulAdd2D(new_concat3->output(0), weights5, bias5, !transpose_a, transpose_b, true);
        auto transpose5 = std::dynamic_pointer_cast<Transpose>(add5->input_value(0).get_node_shared_ptr());
        if (transpose5) {
            auto new_reshape6 = std::make_shared<Reshape>(new_concat3->output(0),
                Constant::create(element::i64, Shape{3}, {1ull, C * W, H})->output(0),false);
            replace_output_update_name(reshape6, new_reshape6);
        } else {
            auto new_transpose3 = std::make_shared<Transpose>(new_concat3->output(0), 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            auto new_reshape6 = std::make_shared<Reshape>(new_transpose3->output(0),
                Constant::create(element::i64, Shape{3}, {1ull, H, C * W})->output(0),false);
            replace_output_update_name(reshape6, new_reshape6);
        }

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(reshape_pattern_6, matcher_name);
    this->register_matcher(m, callback);
}

// Split first MatMul for better GNA Multihead Self Attention performance

GnaMhaSplitSelfTransformation::GnaMhaSplitSelfTransformation() {
    MATCHER_SCOPE(GnaMhaSplitSelfTransformation);

    auto reshape_pattern_1 = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1a = pattern::wrap_type<MatMul>({reshape_pattern_1, pattern::any_input()});
    auto add_pattern = pattern::wrap_type<Add>({pattern::any_input(), matmul_pattern_1a});
    auto matmul_pattern_1b = pattern::wrap_type<ov::intel_gna::op::GNAConvolution>({reshape_pattern_1, pattern::any_input(), pattern::any_input()});
    auto or_pattern_1 = std::make_shared<pattern::op::Or>(OutputVector{add_pattern, matmul_pattern_1b});
    auto reshape_pattern_2 = pattern::wrap_type<Reshape>({or_pattern_1, pattern::any_input()});
    auto variadicsplit_pattern = pattern::wrap_type<VariadicSplit>({reshape_pattern_2, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape1 = pattern_map.at(reshape_pattern_1).get_node_shared_ptr();
        auto children = reshape1->output(0).get_target_inputs();
        auto matmul = std::dynamic_pointer_cast<MatMul>(children.begin()->get_node()->shared_from_this());
        auto gnaconv = std::dynamic_pointer_cast<ov::intel_gna::op::GNAConvolution>(children.begin()->get_node()->shared_from_this());
        auto reshape2 = pattern_map.at(reshape_pattern_2).get_node_shared_ptr();
        auto split_node = pattern_map.at(variadicsplit_pattern).get_node_shared_ptr();

        Output<Node>& input = reshape1->output(0);
        auto split = std::dynamic_pointer_cast<VariadicSplit>(split_node->shared_from_this());
        if (split == nullptr) {
            return false;
        }
        auto split_axis_const = std::dynamic_pointer_cast<Constant>(split->input_value(1).get_node_shared_ptr());
        auto axis_type = split_axis_const->get_element_type();
        auto axis_data = split_axis_const->get_data_ptr();
        int64_t split_axis = (axis_type == ov::element::i64) ? *((int64_t*)axis_data) : (int64_t)(*((int32_t*)axis_data));
        if (split_axis != 2) {
            return false;
        }
        std::vector<uint64_t> split_lengths;
        auto split_lengths_const = std::dynamic_pointer_cast<Constant>(split->input_value(2).get_node_shared_ptr());
        auto split_shape = split_lengths_const->get_shape();
        if (split_shape[0] != 3) {  // not MHA
            return false;
        }
        auto lengths_type = split_lengths_const->get_element_type();
        auto lengths_data = split_lengths_const->get_data_ptr();
        for (auto i = 0; i < split_shape[0]; i++) {
            uint64_t len = (lengths_type == ov::element::i64) ? *((int64_t*)lengths_data + i) : (int64_t)(*((int32_t*)lengths_data + i));
            split_lengths.push_back(len);
        }
        auto children0 = split->output(0).get_target_inputs();
        auto children1 = split->output(1).get_target_inputs();
        auto children2 = split->output(2).get_target_inputs();
        auto split_reshape0 = std::dynamic_pointer_cast<Reshape>(children0.begin()->get_node()->shared_from_this());
        auto split_reshape1 = std::dynamic_pointer_cast<Reshape>(children1.begin()->get_node()->shared_from_this());
        auto split_reshape2 = std::dynamic_pointer_cast<Reshape>(children2.begin()->get_node()->shared_from_this());
        if ((split_reshape0 == nullptr) || (split_reshape1 == nullptr) || (split_reshape2 == nullptr)) {
            return false;
        }
        if (matmul != nullptr) {
            auto matmul_shape = matmul->get_output_shape(0);
            auto weights = std::dynamic_pointer_cast<Constant>(matmul->input_value(1).get_node()->shared_from_this());
            auto weights_shape = weights->get_output_shape(0);
            auto transpose_a = matmul->get_transpose_a();
            auto transpose_b = matmul->get_transpose_b();
            const float* weights_ptr = weights->get_data_ptr<float>();
            auto num_weights = split_lengths[0] * ((transpose_b) ? weights_shape[1] : weights_shape[0]);
            std::vector<float> weights0(num_weights);
            num_weights = split_lengths[1] * ((transpose_b) ? weights_shape[1] : weights_shape[0]);
            std::vector<float> weights1(num_weights);
            num_weights = split_lengths[2] * ((transpose_b) ? weights_shape[1] : weights_shape[0]);
            std::vector<float> weights2(num_weights);
            if (transpose_b) {
                for (auto i = 0; i < split_lengths[0]; i++) {
                    for (auto j = 0; j < weights_shape[1]; j++) {
                        weights0[i * weights_shape[1] + j] = *(weights_ptr + i * weights_shape[1] + j);
                    }
                }
                for (auto i = 0; i < split_lengths[1]; i++) {
                    for (auto j = 0; j < weights_shape[1]; j++) {
                        weights1[i * weights_shape[1] + j] = *(weights_ptr + (i + split_lengths[0]) * weights_shape[1] + j);
                    }
                }
                for (auto i = 0; i < split_lengths[2]; i++) {
                    for (auto j = 0; j < weights_shape[1]; j++) {
                        weights2[i * weights_shape[1] + j] = *(weights_ptr + (i + split_lengths[0] + split_lengths[1]) * weights_shape[1] + j);
                    }
                }
            } else {
                for (auto i = 0; i < weights_shape[0]; i++) {
                    for (auto j = 0; j < split_lengths[0]; j++) {
                        weights0[i * weights_shape[1] + j] = *(weights_ptr + i * weights_shape[1] + j);
                    }
                    for (auto j = 0; j < split_lengths[1]; j++) {
                        weights1[i * weights_shape[1] + j] = *(weights_ptr + i * weights_shape[1] + split_lengths[0] + j);
                    }
                    for (auto j = 0; j < split_lengths[2]; j++) {
                        weights2[i * weights_shape[1] + j] = *(weights_ptr + i * weights_shape[1] + split_lengths[0] + split_lengths[1] + j);
                    }
                }
            }
            auto children = matmul->output(0).get_target_inputs();
            auto add = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
            auto bias = std::dynamic_pointer_cast<Constant>(add->input_value(0).get_node()->shared_from_this());
            if (bias == nullptr) {
                return false;
            }
            auto bias_shape = bias->get_output_shape(0);
            const float* bias_ptr = bias->get_data_ptr<float>();
            std::vector<float> bias0(split_lengths[0]);
            std::vector<float> bias1(split_lengths[1]);
            std::vector<float> bias2(split_lengths[2]);
            for (auto i = 0; i < split_lengths[0]; i++) {
                bias0[i] = bias_ptr[i];
            }
            for (auto i = 0; i < split_lengths[1]; i++) {
                bias1[i] = bias_ptr[i + split_lengths[0]];
            }
            for (auto i = 0; i < split_lengths[2]; i++) {
                bias2[i] = bias_ptr[i + split_lengths[0] + split_lengths[1]];
            }
            auto bias0_const = Constant::create(ngraph::element::f32, Shape{1, split_lengths[0]}, bias0);
            auto bias1_const = Constant::create(ngraph::element::f32, Shape{1, split_lengths[1]}, bias1);
            auto bias2_const = Constant::create(ngraph::element::f32, Shape{1, split_lengths[2]}, bias2);

            auto dim0 = (transpose_b) ? split_lengths[0] : weights_shape[0];
            auto dim1 = (transpose_b) ? weights_shape[1] : split_lengths[0];
            auto weights0_const = Constant::create(ngraph::element::f32, Shape{dim0, dim1}, weights0);
            dim0 = (transpose_b) ? split_lengths[1] : weights_shape[0];
            dim1 = (transpose_b) ? weights_shape[1] : split_lengths[1];
            auto weights1_const = Constant::create(ngraph::element::f32, Shape{dim0, dim1}, weights1);
            dim0 = (transpose_b) ? split_lengths[2] : weights_shape[0];
            dim1 = (transpose_b) ? weights_shape[1] : split_lengths[2];
            auto weights2_const = Constant::create(ngraph::element::f32, Shape{dim0, dim1}, weights2);
            //auto matmul0 = std::make_shared<MatMul>(input, weights0_const->output(0), transpose_a, transpose_b);
            //auto matmul1 = std::make_shared<MatMul>(input, weights1_const->output(0), transpose_a, transpose_b);
            //auto matmul2 = std::make_shared<MatMul>(input, weights2_const->output(0), transpose_a, transpose_b);
            //auto add0 = std::make_shared<Add>(bias0_const->output(0), matmul0->output(0));
            //auto add1 = std::make_shared<Add>(bias1_const->output(0), matmul1->output(0));
            //auto add2 = std::make_shared<Add>(bias2_const->output(0), matmul2->output(0));
            auto add0 = InsertGnaMatMulAdd2D(input, weights0_const->output(0), bias0_const->output(0), transpose_a, transpose_b, false);
            auto add1 = InsertGnaMatMulAdd2D(input, weights1_const->output(0), bias1_const->output(0), transpose_a, transpose_b, false);
            auto add2 = InsertGnaMatMulAdd2D(input, weights2_const->output(0), bias2_const->output(0), transpose_a, transpose_b, false);
            auto shape0 = split_reshape0->get_output_shape(0);
            auto shape1 = split_reshape1->get_output_shape(0);
            auto shape2 = split_reshape2->get_output_shape(0);
            auto new_reshape0 = std::make_shared<Reshape>(add0->output(0),
                Constant::create(element::i64, Shape{3}, {shape0[0], shape0[1], shape0[2]})->output(0), false);
            auto new_reshape1 = std::make_shared<Reshape>(add1->output(0),
                Constant::create(element::i64, Shape{3}, {shape1[0], shape1[1], shape1[2]})->output(0), false);
            auto new_reshape2 = std::make_shared<Reshape>(add2->output(0),
                Constant::create(element::i64, Shape{3}, {shape2[0],shape2[1], shape2[2]})->output(0), false);
        
            replace_output_update_name(split_reshape0, new_reshape0);
            replace_output_update_name(split_reshape1, new_reshape1);
            replace_output_update_name(split_reshape2, new_reshape2);

        } else {  // matmul already converted to GNAConvolution

            auto gnaconv_shape = gnaconv->get_output_shape(0);
            auto weights = std::dynamic_pointer_cast<Constant>(gnaconv->input_value(1).get_node()->shared_from_this());
            auto weights_shape = weights->get_output_shape(0);
            const float* weights_ptr = weights->get_data_ptr<float>();
            auto num_weights = split_lengths[0] * weights_shape[3];
            std::vector<float> weights0(num_weights);
            num_weights = split_lengths[1] * weights_shape[3];
            std::vector<float> weights1(num_weights);
            num_weights = split_lengths[1] * weights_shape[3];
            std::vector<float> weights2(num_weights);
            for (auto i = 0; i < split_lengths[0]; i++) {
                for (auto j = 0; j < weights_shape[3]; j++) {
                    weights0[i * weights_shape[3] + j] = *(weights_ptr + i * weights_shape[3] + j);
                }
            }
            for (auto i = 0; i < split_lengths[1]; i++) {
                for (auto j = 0; j < weights_shape[3]; j++) {
                    weights1[i * weights_shape[3] + j] = *(weights_ptr + (i + split_lengths[0]) * weights_shape[3] + j);
                }
            }
            for (auto i = 0; i < split_lengths[2]; i++) {
                for (auto j = 0; j < weights_shape[3]; j++) {
                    weights2[i * weights_shape[3] + j] = *(weights_ptr + (i + split_lengths[0] + split_lengths[1]) * weights_shape[3] + j);
                }
            }
            auto bias = std::dynamic_pointer_cast<Constant>(gnaconv->input_value(2).get_node()->shared_from_this());
            if (bias == nullptr) {
                return false;
            }
            auto bias_shape = bias->get_output_shape(0);
            const float* bias_ptr = bias->get_data_ptr<float>();
            std::vector<float> bias0(split_lengths[0]);
            std::vector<float> bias1(split_lengths[1]);
            std::vector<float> bias2(split_lengths[2]);
            for (auto i = 0; i < split_lengths[0]; i++) {
                bias0[i] = bias_ptr[i];
            }
            for (auto i = 0; i < split_lengths[1]; i++) {
                bias1[i] = bias_ptr[i + split_lengths[0]];
            }
            for (auto i = 0; i < split_lengths[2]; i++) {
                bias2[i] = bias_ptr[i + split_lengths[0] + split_lengths[1]];
            }
            auto bias0_const = Constant::create(ngraph::element::f32, Shape{1, 1, 1, split_lengths[0]}, bias0);
            auto bias1_const = Constant::create(ngraph::element::f32, Shape{1, 1, 1, split_lengths[1]}, bias1);
            auto bias2_const = Constant::create(ngraph::element::f32, Shape{1, 1, 1, split_lengths[2]}, bias2);

            auto weights0_const = Constant::create(ngraph::element::f32, Shape{split_lengths[0], 1ull, 1ull, weights_shape[3]}, weights0);
            auto weights1_const = Constant::create(ngraph::element::f32, Shape{split_lengths[1], 1ull, 1ull, weights_shape[3]}, weights1);
            auto weights2_const = Constant::create(ngraph::element::f32, Shape{split_lengths[2], 1ull, 1ull, weights_shape[3]}, weights2);
            auto matmul0 = std::make_shared<ov::intel_gna::op::GNAConvolution>(input, weights0_const->output(0), bias0_const->output(0),
                ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID);
            auto matmul1 = std::make_shared<ov::intel_gna::op::GNAConvolution>(input, weights1_const->output(0), bias1_const->output(0),
                ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID);
            auto matmul2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(input, weights2_const->output(0), bias2_const->output(0),
                ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID);
            auto shape0 = split_reshape0->get_output_shape(0);
            auto shape1 = split_reshape1->get_output_shape(0);
            auto shape2 = split_reshape2->get_output_shape(0);
            auto new_reshape0 = std::make_shared<Reshape>(matmul0->output(0),
                Constant::create(element::i64, Shape{3}, {shape0[0], shape0[1], shape0[2]})->output(0), false);
            auto new_reshape1 = std::make_shared<Reshape>(matmul1->output(0),
                Constant::create(element::i64, Shape{3}, {shape1[0], shape1[1], shape1[2]})->output(0), false);
            auto new_reshape2 = std::make_shared<Reshape>(matmul2->output(0),
                Constant::create(element::i64, Shape{3}, {shape2[0],shape2[1], shape2[2]})->output(0), false);
        
            replace_output_update_name(split_reshape0, new_reshape0);
            replace_output_update_name(split_reshape1, new_reshape1);
            replace_output_update_name(split_reshape2, new_reshape2);
        }

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(variadicsplit_pattern, matcher_name);
    this->register_matcher(m, callback);
}

