// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_dparn.hpp"

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


using namespace ov::opset12;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

GnaDparnTransformation::GnaDparnTransformation() {
    MATCHER_SCOPE(DparnTransformation);

    auto add_pattern_1 = pattern::wrap_type<Add>({pattern::any_input(), pattern::any_input()});
    auto matmul_pattern_1 = pattern::wrap_type<MatMul>({add_pattern_1, pattern::any_input()});

    auto strided_slice_pattern_1 = pattern::wrap_type<StridedSlice>({matmul_pattern_1, pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_1 = pattern::wrap_type<Reshape>({strided_slice_pattern_1, pattern::any_input()});
    auto transpose_pattern_1 = pattern::wrap_type<Transpose>({reshape_pattern_1, pattern::any_input()});
    auto multiply_pattern_1 = pattern::wrap_type<Multiply>({transpose_pattern_1, pattern::any_input()});

    auto strided_slice_pattern_2 = pattern::wrap_type<StridedSlice>({matmul_pattern_1, pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_2 = pattern::wrap_type<Reshape>({strided_slice_pattern_2, pattern::any_input()});
    auto transpose_pattern_2 = pattern::wrap_type<Transpose>({reshape_pattern_2, pattern::any_input()});

    auto matmul_pattern_2 = pattern::wrap_type<MatMul>({multiply_pattern_1, transpose_pattern_2});
    auto softmax_pattern = pattern::wrap_type<Softmax>({matmul_pattern_2});

    auto strided_slice_pattern_3 = pattern::wrap_type<StridedSlice>({matmul_pattern_1, pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape_pattern_3 = pattern::wrap_type<Reshape>({strided_slice_pattern_3, pattern::any_input()});
    auto transpose_pattern_3 = pattern::wrap_type<Transpose>({reshape_pattern_3, pattern::any_input()});
    auto matmul_pattern_3 = pattern::wrap_type<MatMul>({softmax_pattern, transpose_pattern_3});

    auto transpose_pattern_4 = pattern::wrap_type<Transpose>({matmul_pattern_3, pattern::any_input()});
    auto reshape_pattern_4 = pattern::wrap_type<Reshape>({transpose_pattern_4, pattern::any_input()});
    auto matmul_pattern_4 = pattern::wrap_type<MatMul>({reshape_pattern_4, pattern::any_input()});

    auto reshape_pattern_5 = pattern::wrap_type<Reshape>({matmul_pattern_4, pattern::any_input()});
    auto reshape_pattern_6 = pattern::wrap_type<Reshape>({reshape_pattern_5, pattern::any_input()});
    auto add_pattern_2 = pattern::wrap_type<Add>({pattern::any_input(), reshape_pattern_6});


    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto add1 = pattern_map.at(add_pattern_1).get_node_shared_ptr();
        auto matmul1 = pattern_map.at(matmul_pattern_1).get_node_shared_ptr();
        auto multiply1 = pattern_map.at(multiply_pattern_1).get_node_shared_ptr();
        auto matmul2 = pattern_map.at(matmul_pattern_2).get_node_shared_ptr();
        auto matmul3 = pattern_map.at(matmul_pattern_3).get_node_shared_ptr();
        auto matmul4 = pattern_map.at(matmul_pattern_4).get_node_shared_ptr();
        auto add2 = pattern_map.at(add_pattern_2).get_node_shared_ptr();

        const Output<Node>& input1 = add1->input_value(0);
        const Output<Node>& input2 = add1->input_value(1);
        auto add1_shape = add1->get_output_shape(0);
        auto H = add1_shape[0];
        auto W = add1_shape[2];
        
        auto new_add1 = std::make_shared<Add>(input1, input2);
        auto new_reshape1 = std::make_shared<Reshape>(new_add1->output(0), Constant::create(element::i64, Shape{2}, {H, W})->output(0), false);

        auto weights1 = std::dynamic_pointer_cast<Constant>(matmul1->input_value(1).get_node()->shared_from_this());
        auto weights1_shape = weights1->get_shape();
        const float* weights1_ptr = weights1->get_data_ptr<float>();
        auto new_weights1_0 = Constant::create(element::f32, Shape{W, W}, weights1_ptr);
        auto new_weights1_1 = Constant::create(element::f32, Shape{W, W}, weights1_ptr+W*W);
        auto new_weights1_2 = Constant::create(element::f32, Shape{W, W}, weights1_ptr+2*W*W);
        auto new_matmul1_0 = std::make_shared<MatMul>(new_reshape1->output(0), new_weights1_0->output(0), false, true);
        auto new_matmul1_1 = std::make_shared<MatMul>(new_reshape1->output(0), new_weights1_1->output(0), false, true);
        auto new_matmul1_2 = std::make_shared<MatMul>(new_reshape1->output(0), new_weights1_2->output(0), false, true);

        auto new_transpose1_0 = std::make_shared<Transpose>(new_matmul1_0->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose1_1 = std::make_shared<Transpose>(new_matmul1_1->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose1_2 = std::make_shared<Transpose>(new_matmul1_2->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));

        auto new_reshape1_0 = std::make_shared<Reshape>(new_transpose1_0->output(0), Constant::create(element::i64, Shape{3}, {4ull, W/4, H})->output(0), false);
        auto new_split1_0 = std::make_shared<Split>(new_reshape1_0->output(0), Constant::create(element::i64, Shape{}, {0}), 4);
        auto new_reshape1_1 = std::make_shared<Reshape>(new_transpose1_1->output(0), Constant::create(element::i64, Shape{3}, {4ull, W/4, H})->output(0), false);
        auto new_split1_1 = std::make_shared<Split>(new_reshape1_1->output(0), Constant::create(element::i64, Shape{}, {0}), 4);
        auto new_reshape1_2 = std::make_shared<Reshape>(new_transpose1_2->output(0), Constant::create(element::i64, Shape{3}, {4ull, W/4, H})->output(0), false);
        auto new_split1_2 = std::make_shared<Split>(new_reshape1_2->output(0), Constant::create(element::i64, Shape{}, {0}), 4);

        auto mul_weight = std::dynamic_pointer_cast<Constant>(multiply1->input_value(1).get_node()->shared_from_this());
        const float* mul_weight_ptr = mul_weight->get_data_ptr<float>();
        auto new_mul_weight = Constant::create(element::f32, {1ull,1ull}, mul_weight_ptr);

        auto new_reshape2_0 = std::make_shared<Squeeze>(new_split1_0->output(0));
        auto new_reshape2_1 = std::make_shared<Squeeze>(new_split1_0->output(1));
        auto new_reshape2_2 = std::make_shared<Squeeze>(new_split1_0->output(2));
        auto new_reshape2_3 = std::make_shared<Squeeze>(new_split1_0->output(3));

        auto new_multiply1_0 = std::make_shared<Multiply>(new_reshape2_0->output(0), new_mul_weight->output(0));
        auto new_multiply1_1 = std::make_shared<Multiply>(new_reshape2_1->output(0), new_mul_weight->output(0));
        auto new_multiply1_2 = std::make_shared<Multiply>(new_reshape2_2->output(0), new_mul_weight->output(0));
        auto new_multiply1_3 = std::make_shared<Multiply>(new_reshape2_3->output(0), new_mul_weight->output(0));

        auto new_transpose2_0 = std::make_shared<Transpose>(new_multiply1_0->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose2_1 = std::make_shared<Transpose>(new_multiply1_1->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose2_2 = std::make_shared<Transpose>(new_multiply1_2->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose2_3 = std::make_shared<Transpose>(new_multiply1_3->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));

        auto new_reshape3_0 = std::make_shared<Squeeze>(new_split1_1->output(0));
        auto new_reshape3_1 = std::make_shared<Squeeze>(new_split1_1->output(1));
        auto new_reshape3_2 = std::make_shared<Squeeze>(new_split1_1->output(2));
        auto new_reshape3_3 = std::make_shared<Squeeze>(new_split1_1->output(3));

        auto new_matmul2_0 = std::make_shared<MatMul>(new_transpose2_0->output(0), new_reshape3_0->output(0), false, false);
        auto new_matmul2_1 = std::make_shared<MatMul>(new_transpose2_1->output(0), new_reshape3_1->output(0), false, false);
        auto new_matmul2_2 = std::make_shared<MatMul>(new_transpose2_2->output(0), new_reshape3_2->output(0), false, false);
        auto new_matmul2_3 = std::make_shared<MatMul>(new_transpose2_3->output(0), new_reshape3_3->output(0), false, false);

        auto new_softmax_0 = std::make_shared<Softmax>(new_matmul2_0->output(0), 1);
        auto new_softmax_1 = std::make_shared<Softmax>(new_matmul2_1->output(0), 1);
        auto new_softmax_2 = std::make_shared<Softmax>(new_matmul2_2->output(0), 1);
        auto new_softmax_3 = std::make_shared<Softmax>(new_matmul2_3->output(0), 1);

        auto new_reshape4_0 = std::make_shared<Reshape>(new_split1_2->output(0), Constant::create(element::i64, Shape{2}, {W/4, H})->output(0), false);
        auto new_reshape4_1 = std::make_shared<Reshape>(new_split1_2->output(1), Constant::create(element::i64, Shape{2}, {W/4, H})->output(0), false);
        auto new_reshape4_2 = std::make_shared<Reshape>(new_split1_2->output(2), Constant::create(element::i64, Shape{2}, {W/4, H})->output(0), false);
        auto new_reshape4_3 = std::make_shared<Reshape>(new_split1_2->output(3), Constant::create(element::i64, Shape{2}, {W/4, H})->output(0), false);

        auto new_transpose4_0 = std::make_shared<Transpose>(new_reshape4_0->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose4_1 = std::make_shared<Transpose>(new_reshape4_1->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose4_2 = std::make_shared<Transpose>(new_reshape4_2->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto new_transpose4_3 = std::make_shared<Transpose>(new_reshape4_3->output(0), Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));

        auto new_matmul3_0 = std::make_shared<MatMul>(new_softmax_0->output(0), new_transpose4_0->output(0), false, false);
        auto new_matmul3_1 = std::make_shared<MatMul>(new_softmax_1->output(0), new_transpose4_1->output(0), false, false);
        auto new_matmul3_2 = std::make_shared<MatMul>(new_softmax_2->output(0), new_transpose4_2->output(0), false, false);
        auto new_matmul3_3 = std::make_shared<MatMul>(new_softmax_3->output(0), new_transpose4_3->output(0), false, false);

        auto weights4 = std::dynamic_pointer_cast<Constant>(matmul4->input_value(1).get_node()->shared_from_this());
        auto weights4_shape = weights4->get_shape();
        const float* weights4_ptr = weights4->get_data_ptr<float>();
        std::vector<float> weights4_0(W * W / 4, 0.0f);
        std::vector<float> weights4_1(W * W / 4, 0.0f);
        std::vector<float> weights4_2(W * W / 4, 0.0f);
        std::vector<float> weights4_3(W * W / 4, 0.0f);
        float* weights4_0_ptr = weights4_0.data();
        float* weights4_1_ptr = weights4_1.data();
        float* weights4_2_ptr = weights4_2.data();
        float* weights4_3_ptr = weights4_3.data();
        for (size_t i = 0; i < weights4_shape[0]; i++) {
            for (size_t j = 0; j < weights4_shape[1]; j++) {
                if (i >= 3 * weights4_shape[0] / 4) {
                    weights4_3_ptr[(i - 3 * weights4_shape[0] / 4) * weights4_shape[1] + j] = weights4_ptr[j * weights4_shape[0] + i];
                } else if (i >= 2 * weights4_shape[0] / 4) {
                    weights4_2_ptr[(i - 2 * weights4_shape[0] / 4) * weights4_shape[1] + j] = weights4_ptr[j * weights4_shape[0] + i];
                } else if (i >= 1 * weights4_shape[0] / 4) {
                    weights4_1_ptr[(i - 1 * weights4_shape[0] / 4) * weights4_shape[1] + j] = weights4_ptr[j * weights4_shape[0] + i];
                } else {
                    weights4_0_ptr[(i - 0 * weights4_shape[0] / 4) * weights4_shape[1] + j] = weights4_ptr[j * weights4_shape[0] + i];
                }
            }
        }
        auto new_weights4_0 = Constant::create(element::f32, Shape{W/4,W}, weights4_0_ptr);
        auto new_weights4_1 = Constant::create(element::f32, Shape{W/4,W}, weights4_1_ptr);
        auto new_weights4_2 = Constant::create(element::f32, Shape{W/4,W}, weights4_2_ptr);
        auto new_weights4_3 = Constant::create(element::f32, Shape{W/4,W}, weights4_3_ptr);
        auto new_matmul4_0 = std::make_shared<MatMul>(new_matmul3_0->output(0), new_weights4_0->output(0), false, false);
        auto new_matmul4_1 = std::make_shared<MatMul>(new_matmul3_1->output(0), new_weights4_1->output(0), false, false);
        auto new_matmul4_2 = std::make_shared<MatMul>(new_matmul3_2->output(0), new_weights4_2->output(0), false, false);
        auto new_matmul4_3 = std::make_shared<MatMul>(new_matmul3_3->output(0), new_weights4_3->output(0), false, false);

        auto new_add4_0 = std::make_shared<Add>(new_matmul4_0->output(0), new_matmul4_1->output(0));
        auto new_add4_1 = std::make_shared<Add>(new_matmul4_2->output(0), new_matmul4_3->output(0));
        auto new_add4_3 = std::make_shared<Add>(new_add4_0->output(0), new_add4_1->output(0));

        auto new_reshape5_0 = std::make_shared<Reshape>(input1, Constant::create(element::i64, Shape{3}, {1ull, H, W})->output(0), false);
        auto new_reshape5_1 = std::make_shared<Reshape>(new_add4_3->output(0), Constant::create(element::i64, Shape{3}, {1ull, H, W})->output(0), false);

        auto new_add2 = std::make_shared<Add>(new_reshape5_0->output(0), new_reshape5_1->output(0));

        replace_node(add2, new_add2);

        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(add_pattern_2, matcher_name);
    this->register_matcher(m, callback);
}
