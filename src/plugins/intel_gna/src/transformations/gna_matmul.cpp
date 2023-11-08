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

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "gna_matmul.hpp"
#include "gna_helper.hpp"
#include <layers/gna_layer_info.hpp>
#include <layers/gna_layer_type.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::GnaMatMulDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto matmul = std::dynamic_pointer_cast<MatMul>(node);
        if (nullptr == matmul) {
            continue;
        }

        const Output<Node>& parent_a = matmul->input_value(0);
        const Output<Node>& parent_b = matmul->input_value(1);
        auto const_a = std::dynamic_pointer_cast<ngraph::op::Constant>(parent_a.get_node()->shared_from_this());
        auto const_b = std::dynamic_pointer_cast<ngraph::op::Constant>(parent_b.get_node()->shared_from_this());
        auto transpose_a = matmul->get_transpose_a();
        auto transpose_b = matmul->get_transpose_b();
        auto parent_a_shape = parent_a.get_shape();
        auto parent_b_shape = parent_b.get_shape();
        auto matmul_shape = matmul->get_output_shape(0);
        size_t N = (matmul_shape.size() > 3) ? matmul_shape[matmul_shape.size() - 4] : 1;
        size_t C = (matmul_shape.size() > 2) ? matmul_shape[matmul_shape.size() - 3] : 1;
        size_t H = (matmul_shape.size() > 1) ? matmul_shape[matmul_shape.size() - 2] : 1;
        size_t W = matmul_shape[matmul_shape.size() - 1];
        size_t N_a = (parent_a_shape.size() > 3) ? parent_a_shape[parent_a_shape.size() - 4] : 1;
        size_t C_a = (parent_a_shape.size() > 2) ? parent_a_shape[parent_a_shape.size() - 3] : 1;
        size_t H_a = (parent_a_shape.size() > 1) ? parent_a_shape[parent_a_shape.size() - 2] : 1;
        size_t W_a = parent_a_shape[parent_a_shape.size() - 1];
        size_t N_b = (parent_b_shape.size() > 3) ? parent_b_shape[parent_b_shape.size() - 4] : 1;
        size_t C_b = (parent_b_shape.size() > 2) ? parent_b_shape[parent_b_shape.size() - 3] : 1;
        size_t H_b = (parent_b_shape.size() > 1) ? parent_b_shape[parent_b_shape.size() - 2] : 1;
        size_t W_b = parent_b_shape[parent_b_shape.size() - 1];
        auto input_reshape = std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(parent_a.get_node()->shared_from_this());
        if (input_reshape) {  // detect if 2D MatMul has been converted to GNA already
            auto prev_shape = input_reshape->input_value(0).get_shape();
            if ((prev_shape.size() == 2) && (H_a == prev_shape[1]) && (W_a == prev_shape[0])) {
                continue;
            }
        }

        std::shared_ptr<FakeQuantize> fq_after = nullptr;
        auto children = matmul->output(0).get_target_inputs();
        for (auto child : children) {
            fq_after = std::dynamic_pointer_cast<FakeQuantize>(child.get_node()->shared_from_this());
            if (fq_after != nullptr) {
                break;
            }
        }

        if (matmul_shape.size() == 1) {
            continue;
        } else if (N != 1) {
            continue;  // Batch case not yet implemented
        } else if (const_b != nullptr) {
            continue;  // GNA plugin now implements big matmul as convolution when 2nd arg is constant
        //} else if ((!transpose_a) && (parent_a_shape.size() == 2) && (parent_b_shape.size() == 2)) {  // use convolution for non-const 2D MatMul

            // GNA PLUGIN SCALE FACTOR CALCULATION FAILS WHEN WEIGHTS ARE NOT CONST -- NEED TO FIX THAT FIRST

            //auto new_reshape_a = std::make_shared<op::v1::Reshape>(parent_a,
            //    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, parent_a_shape[0],parent_a_shape[1], 1ull})->output(0),false);
            //auto new_transpose = std::make_shared<op::Transpose>(parent_b, 
            //    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            //auto new_reshape_b = std::make_shared<op::v1::Reshape>(new_transpose->output(0),
            //    op::Constant::create(ngraph::element::i64, Shape{4}, {parent_b_shape[1],1ull,1ull,parent_b_shape[0]})->output(0),false);
            //new_transpose = std::make_shared<op::Transpose>(new_reshape_a->output(0),
            //    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            //auto new_conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_transpose->output(0),
            //    new_reshape_b->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            //new_transpose= std::make_shared<op::Transpose>(new_conv->output(0),
            //    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            //auto new_reshape = std::make_shared<op::v1::Reshape>(new_transpose->output(0),
            //    op::Constant::create(ngraph::element::i64, Shape{2}, {parent_a_shape[0],parent_b_shape[1]})->output(0),false);
            //ngraph::replace_node(matmul, new_reshape);
            //is_graph_modfied = true;

        } else {

            OutputVector chunks_a, chunks_b;
            if (C > 1) {
                auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(parent_a,
                    op::Constant::create(ngraph::element::i64, Shape{2}, {C_a, H_a * W_a})->output(0),false);
                auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(parent_b,
                    op::Constant::create(ngraph::element::i64, Shape{2}, {C_b, H_b * W_b})->output(0),false);
                auto split_a = std::make_shared<ngraph::opset1::Split>(reshape_a,
                    ngraph::opset1::Constant::create(element::i64, Shape{}, {0}), C_a);
                auto split_b = std::make_shared<ngraph::opset1::Split>(reshape_b,
                    ngraph::opset1::Constant::create(element::i64, Shape{}, {0}), C_b);
                for (size_t c = 0; c < C; c++) {
                    chunks_a.push_back(split_a->output(c));
                    chunks_b.push_back(split_b->output(c));
                }                    
            } else {
                chunks_a.push_back(parent_a);
                chunks_b.push_back(parent_b);
            }

            bool abort = false;
            OutputVector product_parts;
            for (size_t c = 0; c < C; c++) {
                if (transpose_b) {  // B must be split before transpose is applied
                    std::shared_ptr<ov::op::v1::VariadicSplit> rowblock_b;
                    std::shared_ptr<ov::op::v0::Concat> new_concat;
                    std::vector<size_t> split_lengths;
                    size_t H_b_left = H_b;
                    while (H_b_left > 0) {
                        size_t H_b_part = (H_b_left > 8) ? 8 : H_b_left;
                        H_b_left -= H_b_part;
                        split_lengths.push_back(H_b_part * W_b);
                    }
                    if (split_lengths.size() == 1) {
                        auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                           op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_a})->output(0),false);
                        auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(chunks_b[c],
                           op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),false);
                        // want to encode A * rowblock_b^T for GNA as (rowblock_b^T * A^T)^T
                        // The GNA plugin expects that transposes and reshapes have been added at 
                        // input(0) and the output of ngraph MatMul operators since dimensions are 
                        // swapped during encoding of the GNA Affine operator.
                        if (transpose_a) {
                            // transpose/reshape trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                            auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_b->output(0),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape_btrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),false);
                            new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_a->output(0), op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape_atrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose,
                               op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),false);
                            auto new_matmul = std::make_shared<op::MatMul>(reshape_btrans->output(0), reshape_atrans->output(0), false, false);
                            auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                            // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                            auto result = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_b})->output(0), false);
                            if (matmul_shape.size() > 2) {
                                auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0),
                                    op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                                product_parts.push_back(unsqueeze->output(0));
                            } else {
                                product_parts.push_back(result->output(0));
                            }
                        } else {
                            // transpose/reshape trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                            auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_b->output(0),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),false);
                            auto new_matmul = std::make_shared<op::MatMul>(reshape->output(0), reshape_a->output(0), false, false);
                            auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                            // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding they
                            // will be swapped back
                            auto result = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, H_b})->output(0),false);
                            if (matmul_shape.size() > 2) {
                                auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0),
                                    op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                                product_parts.push_back(unsqueeze->output(0));
                            } else {
                                product_parts.push_back(result->output(0));
                            }
                        }
                    } else {
                        auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(chunks_b[c],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull,  H_b * W_b})->output(0),false);
                        const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, 
                            Shape{split_lengths.size()}, split_lengths.data());
                        rowblock_b = std::make_shared<opset1::VariadicSplit>(reshape_b->output(0), 
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), split_lengths_const);
                        OutputVector rowblock_c(rowblock_b->get_output_size());
                        for (size_t i = 0; i < rowblock_b->get_output_size(); i++) {
                            auto reshape = std::make_shared<ngraph::opset1::Reshape>(rowblock_b->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {split_lengths[i] / W_b, W_b})->output(0),false);
                            // transpose/reshape trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                            auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape->output(0),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape_btrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {split_lengths[i] / W_b, W_b})->output(0),false);
                            // want to encode A * rowblock_b^T for GNA as (rowblock_b * A^T)^T
                            if (transpose_a) {
                                auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),false);
                                auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_a->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                auto reshape_atrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                   op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),false);
                                auto new_matmul = std::make_shared<op::MatMul>(reshape_btrans->output(0), reshape_atrans->output(0), false, false);
                                auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                                // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding
                                // they will be swapped back
                                auto reshape = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, split_lengths[i] / W_b})->output(0),false);
                                new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                rowblock_c[i] = new_transpose->output(0);
                            } else {
                                auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_a})->output(0),false);
                                auto new_matmul = std::make_shared<op::MatMul>(reshape_btrans->output(0), reshape_a->output(0), false, false);
                                auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                                // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding
                                // they will be swapped back
                                auto reshape = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, split_lengths[i] / W_b})->output(0),false);
                                auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                rowblock_c[i] = new_transpose->output(0);
                            }
                        }
                        new_concat = std::make_shared<ngraph::opset1::Concat>(rowblock_c, 0);
                        auto result = std::make_shared<ngraph::opset1::Transpose>(new_concat->output(0), op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                        if (nullptr == result) {
                            abort = true;
                        } else {
                            if (matmul_shape.size() > 2) {
                                auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0), 
                                    op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                                product_parts.push_back(unsqueeze->output(0));
                            } else {
                                product_parts.push_back(result->output(0));
                            }
                        }
                    }
                } else {  // B must be split after transpose is applied
                    std::shared_ptr<ov::op::v1::VariadicSplit> rowblock_b;
                    std::shared_ptr<ov::op::v0::Concat> new_concat;
                    std::vector<size_t> split_lengths;
                    size_t W_b_left = W_b;
                    while (W_b_left > 0) {
                        size_t W_b_part = (W_b_left > 8) ? 8 : W_b_left;
                        W_b_left -= W_b_part;
                        split_lengths.push_back(H_b * W_b_part);
                    }
                    if (split_lengths.size() == 1) {
                        auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                           op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_a})->output(0),false);
                        auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(chunks_b[c],
                           op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),false);
                        // transpose/reshape trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                        auto trans_chunk_b = std::make_shared<ngraph::opset1::Reshape>(reshape_b->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{2}, {W_b, H_b})->output(0),false);
                        // want to encode A * rowblock_b for GNA as (rowblock_b^T * A^T)^T
                        if (transpose_a) {
                            auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_a->output(0), 
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape_atrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                               op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_a})->output(0),false);
                            auto new_matmul = std::make_shared<op::MatMul>(trans_chunk_b->output(0), reshape_atrans->output(0), false, false);
                            auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                            // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding they
                            // will be swapped back
                            auto result = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_b})->output(0),false);
                            if (matmul_shape.size() > 2) {
                                auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0), 
                                    op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                                product_parts.push_back(unsqueeze->output(0));
                            } else {
                                product_parts.push_back(result->output(0));
                            }
                        } else {
                            auto new_matmul = std::make_shared<op::MatMul>(trans_chunk_b->output(0), reshape_a->output(0), false, false);
                            auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                            // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding they
                            // will be swapped back
                            auto result = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, W_b})->output(0),false);
                            if (matmul_shape.size() > 2) {
                                auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0), 
                                    op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                                product_parts.push_back(unsqueeze->output(0));
                            } else {
                                product_parts.push_back(result->output(0));
                            }
                        }
                    } else {
                        auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(chunks_b[c],
                           op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),false);
                        auto trans_chunk_b = std::make_shared<ngraph::opset1::Transpose>(reshape_b->output(0),
                            op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                        auto reshape = std::make_shared<ngraph::opset1::Reshape>(trans_chunk_b->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull,  H_b * W_b})->output(0),false);
                        const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, 
                            Shape{split_lengths.size()}, split_lengths.data());
                        rowblock_b = std::make_shared<opset1::VariadicSplit>(reshape->output(0), 
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), split_lengths_const);
                        OutputVector rowblock_c(rowblock_b->get_output_size());
                        for (size_t i = 0; i < rowblock_b->get_output_size(); i++) {
                            auto reshape2_b = std::make_shared<ngraph::opset1::Reshape>(rowblock_b->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {split_lengths[i] / H_b, H_b})->output(0),false);
                            // transpose/reshape trick: pre-swap dimensions now and at time of GnaAffine encoding they will be swapped back
                            auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape2_b->output(0),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto reshape2_btrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {split_lengths[i] / H_b, H_b})->output(0),false);
                            // want to encode A * rowblock_b^T for GNA as (rowblock_b^T * A^T)^T
                            if (transpose_a) {
                                auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),false);
                                auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape_a->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                auto reshape_atrans = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                                   op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),false);
                                auto new_matmul = std::make_shared<op::MatMul>(reshape2_btrans->output(0), reshape_atrans->output(0), false, false);
                                auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                                // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding
                                // they will be swapped back
                                auto reshape = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, split_lengths[i] / H_b})->output(0),false);
                                new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                rowblock_c[i] = new_transpose->output(0);
                            } else {
                                auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(chunks_a[c],
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {W_a, H_a})->output(0),false);
                                auto new_matmul = std::make_shared<op::MatMul>(reshape2_btrans->output(0), reshape_a->output(0), false, false);
                                auto new_matmul_fq_out = InsertOutputFQ(new_matmul->output(0), fq_after);
                                // reshape/transpose trick: pre-swap dimensions now and at time of GnaAffine encoding
                                // they will be swapped back
                                auto reshape = std::make_shared<ngraph::opset1::Reshape>(new_matmul_fq_out,
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, split_lengths[i] / H_b})->output(0),false);
                                auto new_transpose = std::make_shared<ngraph::opset1::Transpose>(reshape->output(0), 
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                rowblock_c[i] = new_transpose->output(0);
                            }
                        }
                        new_concat = std::make_shared<ngraph::opset1::Concat>(rowblock_c, 0);
                        auto result = std::make_shared<ngraph::opset1::Transpose>(new_concat->output(0), op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                        if (matmul_shape.size() > 2) {
                            auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(result->output(0), 
                                op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                            product_parts.push_back(unsqueeze->output(0));
                        } else {
                            product_parts.push_back(result->output(0));
                        }
                    }
                }
            }

            if (abort) {
                continue;  // transformation failed so abort
            }

            // contenate parts to form final product tensor
            if (C > 1) {
                auto concat = std::make_shared<ngraph::opset1::Concat>(product_parts, 0);
                if (fq_after) {
                    ngraph::replace_node(fq_after, concat);
                } else {
                    ngraph::replace_node(matmul, concat);
                }
            } else {
                if (fq_after) {
                    ngraph::replace_node(fq_after, product_parts[0].get_node_shared_ptr());
                } else {
                    ngraph::replace_node(matmul, product_parts[0].get_node_shared_ptr());
                }
            }
            is_graph_modfied = true;
            continue;
        }
    }
    return is_graph_modfied;
}
