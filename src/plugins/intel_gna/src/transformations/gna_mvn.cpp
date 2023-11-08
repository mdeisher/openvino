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

#include "gna_mvn.hpp"

using namespace ngraph;
using namespace op;

bool ngraph::pass::GnaMvnDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto mvn = std::dynamic_pointer_cast<MVN>(node);
        auto mvn_v6 = std::dynamic_pointer_cast<op::v6::MVN>(node);
        if ((nullptr == mvn) && (nullptr == mvn_v6)) {
            continue;
        }

        const Output<Node>& parent = (nullptr != mvn) ? mvn->input_value(0) : mvn_v6->input_value(0);
        ov::Shape mvn_shape = (nullptr != mvn) ? mvn->get_output_shape(0) : mvn_v6->get_output_shape(0);
        bool across_channels = (nullptr != mvn) ? mvn->get_across_channels() : false;
        bool normalize_variance = (nullptr != mvn) ? mvn->get_normalize_variance() : mvn_v6->get_normalize_variance();
        double eps = (nullptr != mvn) ? mvn->get_eps() : mvn_v6->get_eps();
        ov::AxisSet reduction_axes_v1 = (nullptr != mvn) ? mvn->get_reduction_axes() : ov::AxisSet::AxisSet();
        ov::Output<ov::Node> reduction_axes_v6 = (nullptr != mvn_v6) ? mvn_v6->input_value(1) : ov::Output<ov::Node>();
        auto axis_const_data_v6 = (nullptr != mvn_v6) ? std::dynamic_pointer_cast<Constant>(reduction_axes_v6.get_node_shared_ptr())->get_data_ptr() : NULL;
        int64_t axis_value_v6 = *((int64_t*)axis_const_data_v6);
        auto eps_mode_v6 = (nullptr != mvn_v6) ? mvn_v6->get_eps_mode() : op::MVNEpsMode::INSIDE_SQRT;

        size_t N = 1, C, H, W;
        if (mvn_shape.size() == 4) {
            N = mvn_shape[0];
            C = mvn_shape[1];
            H = mvn_shape[2];
            W = mvn_shape[3];
        } else if (mvn_shape.size() == 3) {
            C = mvn_shape[0];
            H = mvn_shape[1];
            W = mvn_shape[2];
        } else {
            continue;
        }

        if (N != 1) {
            continue;   // Batch case not yet implemented
        } else if ((nullptr != mvn) && (C != 1)) {
            continue;  // v1 multi-channel case not yet implemented
        } else if ((nullptr != mvn) && (across_channels)) {
            continue;
        } else if (eps_mode_v6 == op::MVNEpsMode::OUTSIDE_SQRT) {  // not yet implemented
            continue;
        } else {

            OutputVector upstream;
            upstream.push_back(parent);
            size_t C_new = C;
            size_t H_new = H;
            size_t W_new = W;

            // reshape tensor into supported shape and abort if not possible
            if (nullptr != mvn_v6) {
                if ((axis_value_v6 != -1) && (axis_value_v6 != mvn_shape.size() - 1)) {
                    if ((axis_value_v6 == mvn_shape.size() - 2) && (W == 1)) { // reshapable into 1,1,W_new
                        C_new = 1;
                        H_new = C;
                        W_new = H;
                        auto reshape = std::make_shared<ngraph::opset1::Reshape>(parent,
                            op::Constant::create(ngraph::element::i64, Shape{4}, {N, C_new, H_new, W_new})->output(0),false);
                        upstream[0] = reshape->output(0);
                    } else {
                        continue;
                    }
                }
            } else {
                if (nullptr == mvn) {
                    C_new = 1;
                    H_new = C * H;
                    auto reshape = std::make_shared<ngraph::opset1::Reshape>(parent,
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C_new, H_new, W_new})->output(0),false);
                    upstream[0] = reshape->output(0);
                }
            }

            // Check if average must be split
            size_t num_parts = 1;
            while (W_new / num_parts > 768) {  // 768 is maximum GNA1/2 kernel size
                num_parts *= 2;
            }
            // Abort if W_new is not divisible by power of 2
            if ((W_new / num_parts) * num_parts != W_new) {
                continue;
            }

            // Create MVN averaging weights --------
            std::vector<float> neg_avg_weights(8 * W_new / num_parts, -1.0f / W_new);
            std::vector<float> avg_weights(8 * W_new / num_parts, 1.0f / W_new);
            std::vector<float> avg_broadcast(8 * W_new * num_parts, 0.0f);
            std::vector<float> minus_half(H_new * W_new, -0.5f);
            std::vector<float> eps_tensor(H_new * W_new, (float)eps);
            for (size_t i = 0; i < W_new * num_parts; i++) {
                avg_broadcast[i * 8] = 1.0f;
            }
            auto neg_avg_weights_const = op::Constant::create(ngraph::element::f32, Shape{8, W_new / num_parts, 1, 1}, neg_avg_weights);
            auto avg_weights_const = op::Constant::create(ngraph::element::f32, Shape{8, W_new / num_parts, 1, 1}, avg_weights);
            auto avg_broadcast_const = op::Constant::create(ngraph::element::f32, Shape{W_new, 8 * num_parts, 1, 1}, avg_broadcast);
            auto minus_half_const = op::Constant::create(ngraph::element::f32, Shape{1, H_new * W_new}, minus_half);
            auto eps_tensor_const = op::Constant::create(ngraph::element::f32, Shape{1, H_new * W_new}, eps_tensor);

            // Assumes C=1 case
            auto input_4d = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, H_new * num_parts, 1ull, W_new / num_parts})->output(0),false);
            auto input_2d = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H_new * W_new})->output(0),false);
            auto transposed_input_1 = std::make_shared<op::Transpose>(input_4d->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_1 = std::make_shared<opset1::Convolution>(transposed_input_1->output(0),
                neg_avg_weights_const->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            transposed_avg_conv_1->set_friendly_name("MvnAvg1");
            auto avg_conv_1 = std::make_shared<op::Transpose>(transposed_avg_conv_1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape_avg_conv_1 = std::make_shared<ngraph::opset1::Reshape>(avg_conv_1->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, H_new, 8*num_parts})->output(0),false);
            auto transposed_input_2 = std::make_shared<op::Transpose>(reshape_avg_conv_1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_2 = std::make_shared<opset1::Convolution>(transposed_input_2->output(0),
                avg_broadcast_const->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            transposed_avg_conv_2->set_friendly_name("MvnAvg2");
            auto avg_conv_2 = std::make_shared<op::Transpose>(transposed_avg_conv_2->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto avg_conv_2_2d = std::make_shared<ngraph::opset1::Reshape>(avg_conv_2,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H_new * W_new})->output(0),false);
            auto subtract_mean = std::make_shared<op::v1::Add>(input_2d->output(0), avg_conv_2_2d->output(0));
            subtract_mean->set_friendly_name("MvnSubMean");
            if (normalize_variance) {
                auto squared_diff = std::make_shared<op::v1::Multiply>(subtract_mean->output(0), subtract_mean->output(0));
                squared_diff->set_friendly_name("MvnSqrDiff");
                auto squared_diff_reshape = std::make_shared<ngraph::opset1::Reshape>(squared_diff->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, H_new * num_parts, 1ull, W_new / num_parts})->output(0),false);
                auto transposed_input_3 = std::make_shared<op::Transpose>(squared_diff_reshape->output(0),
                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto transposed_avg_conv_3 = std::make_shared<opset1::Convolution>(transposed_input_3->output(0),
                    avg_weights_const->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
                transposed_avg_conv_3->set_friendly_name("MvnAvg3");
                auto avg_conv_3 = std::make_shared<op::Transpose>(transposed_avg_conv_3->output(0),
                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto reshape_avg_conv_3 = std::make_shared<ngraph::opset1::Reshape>(avg_conv_3->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, H_new, 8*num_parts})->output(0),false);
                auto transposed_input_4 = std::make_shared<op::Transpose>(reshape_avg_conv_3->output(0),
                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto transposed_avg_conv_4 = std::make_shared<opset1::Convolution>(transposed_input_4->output(0),
                    avg_broadcast_const->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
                transposed_avg_conv_4->set_friendly_name("MvnAvg4");
                auto avg_conv_4 = std::make_shared<op::Transpose>(transposed_avg_conv_4->output(0),
                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto reshape_avg_conv_4 = std::make_shared<ngraph::opset1::Reshape>(avg_conv_4->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H_new * W_new})->output(0),false);
                OutputVector avg;
                avg.push_back(reshape_avg_conv_4->output(0));
                if ((nullptr != mvn) || (eps_mode_v6 == op::MVNEpsMode::INSIDE_SQRT)) {
                    auto add_epsilon = std::make_shared<op::v1::Add>(eps_tensor_const->output(0), reshape_avg_conv_4->output(0));
                    avg[0] = add_epsilon->output(0);
                }
                auto log_var = std::make_shared<ngraph::opset1::Log>(avg[0]);
                log_var->set_friendly_name("MvnLogVar");
                auto log_inv_stdev = std::make_shared<op::v1::Multiply>(log_var->output(0), minus_half_const->output(0));
                log_inv_stdev->set_friendly_name("MvnLogInvStdev");
                auto inv_stdev = std::make_shared<ngraph::opset1::Exp>(log_inv_stdev->output(0));
                inv_stdev->set_friendly_name("MvnInvStdev");
                auto normalized_output = std::make_shared<op::v1::Multiply>(subtract_mean->output(0), inv_stdev->output(0));
                normalized_output->set_friendly_name("MvnOutput");
                auto mvn_output = (mvn_shape.size() == 3) ? std::make_shared<ngraph::opset1::Reshape>(normalized_output->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),false)
                    : std::make_shared<ngraph::opset1::Reshape>(normalized_output->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),false);
                if (nullptr != mvn) {
                    ngraph::replace_node_update_name(mvn, mvn_output);
                } else {
                    ngraph::replace_node_update_name(mvn_v6, mvn_output);
                }

            } else {
                auto mvn_output = (mvn_shape.size() == 3) ? std::make_shared<ngraph::opset1::Reshape>(subtract_mean->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),false)
                    : std::make_shared<ngraph::opset1::Reshape>(subtract_mean->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),false);
                if (nullptr != mvn) {
                    ngraph::replace_node_update_name(mvn, mvn_output);
                } else {
                    ngraph::replace_node_update_name(mvn_v6, mvn_output);
                }
            }
            is_graph_modfied = true;
        }
    }
    return is_graph_modfied;
}
