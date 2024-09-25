/* ============================================================================
 * INTEL CONFIDENTIAL
 *
 * Copyright 2021 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material contains trade secrets and proprietary
 * and confidential information of Intel or its suppliers and licensors. The
 * Material is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or
 * disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 * ============================================================================
 */


#include "gna_transpose.hpp"

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

std::vector<size_t> FindPrimes(size_t n) {
    std::vector<size_t> factors;
    size_t n_tmp = n;
    size_t p = 2;
    while (n_tmp * n_tmp >= p * p) {
        if ((n_tmp % p) == 0) {
            factors.push_back(p);
            n_tmp = n_tmp / p;
        } else {
            p++;
        }
    }
    if (n_tmp > 1) {
        factors.push_back(n_tmp);
    }
    return (factors);
}

bool IsFactoredTransposeFeasible(std::vector<size_t> factors) {
    // check if there are any factors too large for GNA transpose
    bool feasible = true;
    for (size_t i = 0; i < factors.size(); i++) {
        if (factors[i] > 8) {
            feasible = false;
        }
    }
    return (feasible);
}

std::vector<size_t> CombineFactors(std::vector<size_t> factors) {
    // combine prime factors if possible
    std::vector<size_t> combined_factors;
    size_t new_factor = 1;
    for (size_t i = 0; i < factors.size(); i++) {
        size_t product = new_factor * factors[i];
        if (product > 8) {
            combined_factors.push_back(new_factor);
            new_factor = factors[i];
        } else {
            new_factor = product;
        }
    }
    combined_factors.push_back(new_factor);

    return (combined_factors);
}

static bool decompose(std::shared_ptr<ov::opset11::Transpose> transpose) {

    const Output<Node>& parent = transpose->input_value(0);
    auto input_shape = parent.get_shape();
    auto output_shape = transpose->output(0).get_shape();
    const Output<Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();
    if (transpose_order_dim != 1)
        return false;
    auto const_with_order_values = std::dynamic_pointer_cast<ov::opset11::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;
    std::vector<int64_t> order;
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
    if (input_shape.size() < 2) {
        return false;
    }
    size_t N = 1;
    size_t C = 1;
    size_t H = input_shape[input_shape.size() - 2];
    size_t W = input_shape[input_shape.size() - 1];
    if (input_shape.size() == 4) {
        N = input_shape[0];
        C = input_shape[1];
    } else if (input_shape.size() == 3) {
        C = input_shape[0];
    }

    if (N != 1) {
        return false;   // Batch case not yet implemented
    } else if (((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 3) && (order[2] == 1) && (order[3] == 2))) {
        return false;  // potential leading transpose needed for NHWC convolution
    } else if (((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 3) && (order[3] == 1))) {
        return false;  // potential trailing transpose needed for NHWC convolution
    } else {

        // test for simple 2D transpose
        if (((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 3) && (order[2] == 1) && (order[3] == 2))
            || ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 3) && (order[3] == 1))
            || ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 1) && (order[2] == 3) && (order[3] == 2) && (input_shape[1] == 1))
            || ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 3) && (order[2] == 2) && (order[3] == 1) && (input_shape[1] == 1))
            || ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 1) && (order[3] == 3) && (input_shape[0] == 1) && (input_shape[3] == 1))
            || ((input_shape.size() == 3) && (order[0] == 2) && (order[1] == 0) && (order[2] == 1))
            || ((input_shape.size() == 3) && (order[0] == 1) && (order[1] == 2) && (order[2] == 0))
            || ((input_shape.size() == 3) && (order[0] == 0) && (order[1] == 2) && (order[2] == 1) && (input_shape[0] == 1))
            || ((input_shape.size() == 2) && (order[0] == 1) && (order[1] == 0))) {

            size_t H_new = H;
            size_t W_new = W;
            if ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 3) && (order[2] == 1) && (order[3] == 2)) {
                H_new = C * H;
            } else if ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 3) && (order[3] == 1)) {
                H_new = C;
                W_new = H * W;
            } else if ((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 1) && (order[3] == 3)) {
                H_new = N * C;
                W_new = H * W;
            } else if ((input_shape.size() == 3) && (order[0] == 2) && (order[1] == 0) && (order[2] == 1)) {
                H_new = C * H;
            } else if ((input_shape.size() == 3) && (order[0] == 1) && (order[1] == 2) && (order[2] == 0)) {
                H_new = C;
                W_new = H * W;
            }

            // GNA-compatible transpose
            if ((H_new <= 8) || (W_new <= 8)) {
                auto new_reshape = std::make_shared<ov::opset11::Reshape>(parent,
                    ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {H_new, W_new})->output(0),false);
                auto new_transpose = std::make_shared<ov::opset11::Transpose>(new_reshape->output(0),
                    ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                new_reshape = std::make_shared<ov::opset11::Reshape>(new_transpose->output(0),
                    ov::opset11::Constant::create(ngraph::element::i64, Shape{output_shape.size()}, output_shape)->output(0),false);
                ngraph::replace_node_update_name(transpose, new_reshape);
                return true;                    
            }

            // GNA-incompatible transpose
            if (((H_new % 8) == 0) || ((W_new % 8) == 0)) {
                bool factor_W = ((W_new % 8) == 0);
                std::vector<size_t> factors = FindPrimes(factor_W ? W_new : H_new);
                bool feasible = IsFactoredTransposeFeasible(factors);

                // perform feasible transformations
                if (feasible) {
                    std::vector<size_t> combined_factors = CombineFactors(factors);

                    // generate transpose transformation
                    OutputVector upstream;
                    upstream.push_back(parent);
                    for (size_t i = 0; i < combined_factors.size(); i++) {
                        if (factor_W) {
                            auto new_reshape = std::make_shared<ov::opset11::Reshape>(upstream[0],
                                ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {(H_new)*W_new/combined_factors[i], combined_factors[i]})->output(0),false);
                            auto new_transpose = std::make_shared<ov::opset11::Transpose>(new_reshape->output(0),
                                ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            upstream[0] = new_transpose->output(0);
                        } else {
                            auto new_reshape = std::make_shared<ov::opset11::Reshape>(upstream[0],
                                ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {combined_factors[i], (H_new)*W_new/combined_factors[i]})->output(0),false);
                            auto new_transpose = std::make_shared<ov::opset11::Transpose>(new_reshape->output(0),
                                ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            upstream[0] = new_transpose->output(0);
                        }
                    }
                    auto new_reshape = std::make_shared<ov::opset11::Reshape>(upstream[0],
                        ov::opset11::Constant::create(ngraph::element::i64, Shape{output_shape.size()}, output_shape)->output(0),false);
                    ngraph::replace_node_update_name(transpose, new_reshape);
                    return true;                   
                }
            }
        }

        if (((input_shape.size() == 4) && (order[0] == 0) && (order[1] == 2) && (order[2] == 1) && (order[3] == 3))
            || ((input_shape.size() == 3) && (order[0] == 1) && (order[1] == 0) && (order[2] == 2))) {

            if (H > 8) {  // decomposition for GNA not possible unless H <= 8
                return false;
            }
            // split into separate HxW matrices
            const auto axis_node = (input_shape.size() == 3)
                ? ov::opset11::Constant::create(element::i64, Shape{}, {0}) :
                ov::opset11::Constant::create(element::i64, Shape{}, {1});
            const auto split = std::make_shared<ov::opset11::Split>(parent, axis_node, C);
            OutputVector chunks;
            for (size_t c = 0; c < C; c++) {
                auto reshape = std::make_shared<ov::opset11::Reshape>(split->output(c),
                    ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),false);
                auto transpose_part = std::make_shared<ov::opset11::Transpose>(reshape->output(0),
                    ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                chunks.push_back(transpose_part);
            }
            auto concat = std::make_shared<ov::opset11::Concat>(chunks, 0);
            auto transpose_final = std::make_shared<ov::opset11::Transpose>(concat->output(0),
                ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            auto reshape = (input_shape.size() == 3)
                ? std::make_shared<ov::opset11::Reshape>(transpose_final->output(0),
                    ov::opset11::Constant::create(ngraph::element::i64, Shape{3}, {H, C, W})->output(0),false)
                : std::make_shared<ov::opset11::Reshape>(transpose_final->output(0),
                    ov::opset11::Constant::create(ngraph::element::i64, Shape{4}, {N, H, C, W})->output(0),false);
            ngraph::replace_node_update_name(transpose, reshape);

            return true;

        } else if ((order[input_shape.size() - 2] == input_shape.size() - 1) &&
                    (order[input_shape.size() - 1] == input_shape.size() - 2)) {

            if ((input_shape.size() == 4) && (order[1] != 1)) {
                return false;  // not supported
            } else if ((H % 8 != 0) || (W % 8 != 0)) {
                return false;  // non-multiple-of-8 parts not yet handled properly
            }
            // 2D transpose case
            OutputVector chunks;
            if (C > 1) {
                size_t axis = input_shape.size() - 3;
                const auto axis_node = ov::opset11::Constant::create(element::i64, Shape{}, {axis});
                auto split = std::make_shared<ov::opset11::Split>(parent, axis_node, C);
                for (size_t c = 0; c < C; c++) {
                    chunks.push_back(split->output(c));
                }                    
            } else {
                chunks.push_back(parent);
            }

            OutputVector transpose_parts;
            for (size_t c = 0; c < C; c++) {
                if (H <= 8) {
                    std::shared_ptr<ngraph::Node> transpose_const;
                    if (input_shape.size() == 2) {
                        transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                    } else if (input_shape.size() == 3) {
                        transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                    } else {
                        transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                    }
                    auto new_transpose = std::make_shared<ov::opset11::Transpose>(chunks[c], transpose_const);
                    transpose_parts.push_back(new_transpose->output(0));
                } else {
                    // split matrix into row blocks of height 8
                    size_t H_div_8 = H / 8;
                    size_t W_div_8 = W / 8;
                    size_t axis = input_shape.size() - 2;
                    const auto axis_node = ov::opset11::Constant::create(element::i64, Shape{}, {axis});
                    auto rowblock = std::make_shared<ov::opset11::Split>(chunks[c], axis_node, H_div_8);
                    // transpose row blocks of matrix
                    std::vector<OutputVector> subblock;
                    for (size_t i = 0; i < H_div_8; i++) {
                        std::shared_ptr<ngraph::Node> transpose_const;
                        if (input_shape.size() == 2) {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                        } else if (input_shape.size() == 3) {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                        } else {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                        }
                        auto new_transpose = std::make_shared<ov::opset11::Transpose>(rowblock->output(i), transpose_const);
                        // split transposed row blocks into 8x8 blocks
                        OutputVector block;
                        auto split_rowblock = std::make_shared<ov::opset11::Split>(new_transpose->output(0), axis_node, W_div_8);
                        for (size_t j = 0; j < W_div_8; j++) {
                            block.push_back(split_rowblock->output(j));
                        }
                        subblock.push_back(block);
                    }
                    // un-transpose all 8x8 blocks of matrix
                    for (size_t i = 0; i < H_div_8; i++) {
                        for (size_t j = 0; j < W_div_8; j++) {
                            std::shared_ptr<ngraph::Node> transpose_const;
                            if (input_shape.size() == 2) {
                                transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                            } else if (input_shape.size() == 3) {
                                transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                            } else {
                                transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                            }
                            auto new_transpose = std::make_shared<ov::opset11::Transpose>(subblock[i][j], transpose_const);
                            subblock[i][j] = new_transpose->output(0);
                        }
                    }
                    // concatenate into block columns of matrix
                    OutputVector colblock;
                    for (size_t j = 0; j < W_div_8; j++) {
                        OutputVector blocks;
                        for (size_t i = 0; i < H_div_8; i++) {
                            blocks.push_back(subblock[i][j]);
                        }
                        auto concat = std::make_shared<ov::opset11::Concat>(blocks, input_shape.size() - 2);
                        colblock.push_back(concat->output(0));
                    }
                    // transpose each block column
                    OutputVector transposed_rowblock;
                    for (size_t j = 0; j < W_div_8; j++) {
                        std::shared_ptr<ngraph::Node> transpose_const;
                        if (input_shape.size() == 2) {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                        } else if (input_shape.size() == 3) {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                        } else {
                            transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                        }
                        auto new_transpose = std::make_shared<ov::opset11::Transpose>(colblock[j], transpose_const);
                        transposed_rowblock.push_back(new_transpose->output(0));
                    }
                    // contenate to form final transposed matrix
                    if (transposed_rowblock.size() > 1) {
                        auto concat = std::make_shared<ov::opset11::Concat>(transposed_rowblock, input_shape.size() - 2);
                        transpose_parts.push_back(concat->output(0));
                    } else {
                        transpose_parts.push_back(transposed_rowblock[0]);
                    }
                }
            }
            // contenate parts to form final product tensor
            if (C > 1) {
                auto concat = std::make_shared<ov::opset11::Concat>(transpose_parts, input_shape.size() - 3);
                ngraph::replace_node_update_name(transpose, concat);
            } else {
                ngraph::replace_node_update_name(transpose, transpose_parts[0].get_node_shared_ptr());
            }
            return true;



        }
    }
    
    return false;
}

GnaTransposeDecomposition::GnaTransposeDecomposition() {
    MATCHER_SCOPE(GnaTransposeDecomposition);
    auto conv = ov::pass::pattern::wrap_type<ov::opset11::Transpose>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto trsp = std::dynamic_pointer_cast<ov::opset11::Transpose>(m.get_match_root());
        return decompose(trsp);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}


// This is a quick and dirty transformation to address a specific problem in a particular model.
// It should eventually be replaced by a more general implementation.
static bool predecompose(std::shared_ptr<ov::opset11::Transpose> transpose) {

    const Output<Node>& parent = transpose->input_value(0);
    auto input_shape = parent.get_shape();
    auto output_shape = transpose->output(0).get_shape();
    const Output<Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();
    if (transpose_order_dim != 1)
        return false;
    auto const_with_order_values = std::dynamic_pointer_cast<ov::opset11::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;
    std::vector<int64_t> order;
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
    if (input_shape.size() < 2) {
        return false;
    }
    size_t N = 1;
    size_t C = 1;
    size_t H = input_shape[input_shape.size() - 2];
    size_t W = input_shape[input_shape.size() - 1];
    if (input_shape.size() == 4) {
        N = input_shape[0];
        C = input_shape[1];
    } else if (input_shape.size() == 3) {
        C = input_shape[0];
    }

    if (N != 1) {
        return false;   // Batch case not yet implemented
    } else if (((input_shape.size() == 4) && (order[0] == 2) && (order[1] == 0) && (order[2] == 1) && (order[3] == 3))) {

        auto new_reshape = std::make_shared<ov::opset11::Reshape>(parent,
            ov::opset11::Constant::create(ngraph::element::i64, Shape{3}, {input_shape[1], input_shape[2], input_shape[3]})->output(0), false);
        new_reshape->set_friendly_name("Squeeze3D");
        auto axis_node = ov::opset11::Constant::create(element::i64, Shape{}, {0});
        auto new_split = std::make_shared<ov::opset11::Split>(new_reshape->output(0), axis_node, input_shape[1]);
        OutputVector parts;
        for (auto i = 0; i < new_split->get_output_size(); i++) {
            auto new_reshape = std::make_shared<ov::opset11::Reshape>(new_split->output(i),
                ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, {input_shape[2], input_shape[3]})->output(0), false);
            auto new_transpose = std::make_shared<ov::opset11::Transpose>(new_reshape->output(0),
                ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            parts.push_back(new_transpose->output(0));
        }
        auto new_concat = std::make_shared<ov::opset11::Concat>(parts, 0);
        auto new_transpose = std::make_shared<ov::opset11::Transpose>(new_concat->output(0),
            ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        new_reshape = std::make_shared<ov::opset11::Reshape>(new_transpose->output(0),
            ov::opset11::Constant::create(ngraph::element::i64, Shape{output_shape.size()}, output_shape)->output(0), false);
        new_reshape->set_friendly_name("Unsqueeze4D");

        ngraph::replace_node_update_name(transpose, new_reshape);
        return true;

    } else {
        return false;
    }
    
    return false;
}

GnaTransposePreDecomposition::GnaTransposePreDecomposition() {
    MATCHER_SCOPE(GnaTransposeDecomposition);
    auto transpose = ov::pass::pattern::wrap_type<ov::opset11::Transpose>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto trsp = std::dynamic_pointer_cast<ov::opset11::Transpose>(m.get_match_root());
        return predecompose(trsp);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose, matcher_name);
    this->register_matcher(m, callback);
}

static bool decompose_split(std::shared_ptr<ov::opset11::Split> split) {
    auto axis = *(std::dynamic_pointer_cast<ov::opset11::Constant>(split->input_value(1).get_node_shared_ptr())
                    ->get_data_ptr<int64_t>());
    auto parent = split->input_value(0).get_node_shared_ptr();
    auto input_shape = parent->get_shape();

    if (axis != 1 || input_shape[0] == 1) {
        return false;
    }

    auto output_shape = split->output(0).get_shape();
    auto parent_copy = parent->clone_with_new_inputs(parent->input_values());
    auto transpose_const = ov::opset11::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::opset11::Transpose>(parent_copy, transpose_const);
    auto reshape_const = ov::opset11::Constant::create(ngraph::element::i64, Shape{2}, input_shape);
    auto reshape = std::make_shared<ov::opset11::Reshape>(transpose, reshape_const->output(0), false);

    ngraph::replace_node_update_name(parent, reshape);

    for (auto output : split->outputs()) {
        auto consumers = output.get_target_inputs();
        auto transpose_out = std::make_shared<ov::opset11::Transpose>(output, transpose_const);
        consumers.begin()->replace_source_output(transpose_out);
    }

    return true;
}


GnaSplitDecomposition::GnaSplitDecomposition() {
    MATCHER_SCOPE(GnaSplitDecomposition);
    auto conv = ov::pass::pattern::wrap_type<ov::opset11::Split>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto split = std::dynamic_pointer_cast<ov::opset11::Split>(m.get_match_root());
        return decompose_split(split);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
