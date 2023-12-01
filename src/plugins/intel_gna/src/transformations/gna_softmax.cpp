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

#include "transformations/gna_softmax.hpp"
#include "gna_helper.hpp"

using namespace ngraph;
using namespace op;

std::shared_ptr<ov::op::v0::FakeQuantize> InsertFQ(const ov::Output<ov::Node> &parent, float val) {
    size_t levels = 65535;  // value used by POT 2023.0.1 when quantizing a zero const
    auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;
    auto fq_dim = parent.get_shape().size();
    auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
    auto fq_type = parent.get_element_type();
    auto fq_low = -val;
    auto fq_high = val;
    auto input_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
    auto input_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
    auto output_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
    auto output_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
    auto fq = std::make_shared<FakeQuantize>(parent, input_low->output(0), input_high->output(0), output_low->output(0), output_high->output(0), levels, auto_broadcast);
    return fq;
}

size_t ShapeSize(ov::Shape shape) {
    size_t num_elements = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        num_elements *= shape[i];
    }
    return num_elements;
}

ov::Output<ov::Node> MakeWeights(ov::Shape shape, std::vector<float> weights, float val, bool use_fq) {
    auto weights_const = op::Constant::create(ngraph::element::f32, shape, weights);
    if (use_fq) {
        auto fq = InsertFQ(weights_const->output(0), val);
        return fq->output(0);
    } else {
        return weights_const->output(0);
    }
}

ov::Output<ov::Node> MakeWeights1x1(ov::Shape shape, float val, bool use_fq) {
    std::vector<float> weights(ShapeSize(shape), val);

    return MakeWeights(shape, weights, val, use_fq);
}

bool ngraph::pass::GnaSoftmaxDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto softmax_v1 = std::dynamic_pointer_cast<v1::Softmax>(node);
        auto softmax_v8 = std::dynamic_pointer_cast<v8::Softmax>(node);
        if ((nullptr == softmax_v1) && (nullptr == softmax_v8)) {
            continue;
        }

        const Output<Node>& parent = (nullptr != softmax_v1) ? softmax_v1->input_value(0) : softmax_v8->input_value(0);
        auto softmax_shape = (nullptr != softmax_v1) ? softmax_v1->get_output_shape(0) : softmax_v8->get_output_shape(0);
        auto axis = (nullptr != softmax_v1) ? softmax_v1->get_axis() : softmax_v8->get_axis();
        auto auto_broadcast = (nullptr != softmax_v1) ? softmax_v1->get_autob() : softmax_v8->get_autob();
        auto fq_input = FindFqUpstream(parent);  // note:  POT bug requires that we invent FQ layers
        size_t N = 1, C = 1, H, W;

        if (softmax_shape.size() == 4) {
            N = softmax_shape[0];
            C = softmax_shape[1];
            H = softmax_shape[2];
            W = softmax_shape[3];
        } else if (softmax_shape.size() == 3) {
            C = softmax_shape[0];
            H = softmax_shape[1];
            W = softmax_shape[2];
        } else if (softmax_shape.size() == 2) {
            H = softmax_shape[0];
            W = softmax_shape[1];
        } else {
            continue;
        }

        if (N != 1) {
            continue;   // Batch case not yet implemented
        } else if ((axis != softmax_shape.size() - 1) && (axis != -1)) {
            continue; // only support row softmax at this time
        }

        std::shared_ptr<FakeQuantize> fq_before = nullptr;
        std::shared_ptr<FakeQuantize> fq_after = nullptr;
        fq_before = std::dynamic_pointer_cast<FakeQuantize>(parent.get_node_shared_ptr());
        auto children = (softmax_v1) ? softmax_v1->output(0).get_target_inputs() : softmax_v8->output(0).get_target_inputs();
        for (auto child : children) {
            fq_after = std::dynamic_pointer_cast<FakeQuantize>(child.get_node()->shared_from_this());
            if (fq_after != nullptr) {
                break;
            }
        }
        bool use_fq = (fq_before || fq_after);
        float in_scale = 1.0f;
        float out_scale = 1.0f;
        if (fq_before) {
            auto out_high_const = std::dynamic_pointer_cast<Constant>(fq_before->input_value(4).get_node_shared_ptr());
            in_scale = *((float*)out_high_const->get_data_ptr());
        }
        if (fq_after) {
            auto out_high_const = std::dynamic_pointer_cast<Constant>(fq_after->input_value(4).get_node_shared_ptr());
            out_scale = *((float*)out_high_const->get_data_ptr());
        }

        // simple binary kernels used only for copying data
        std::vector<float> copy_weights(8 * 8, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create identity kernels
            copy_weights[i * 8 + i] = 1.0f;
        }
        std::vector<float> neg_broadcast_weights(8 * 8 * W, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create broadcast kernels
            for (size_t j = 0; j < W; j++) {
                for (size_t k = 0; k < 8; k++) {
                    if (k == i) {
                        neg_broadcast_weights[i * W  * 8 + j * 8 + k] = -1.0f;
                    }
                }
            }
        }
        auto copy_weights_out = MakeWeights(Shape{8, 8, 1, 1}, copy_weights, 1.0f, (bool)fq_input);
        auto neg_broadcast_weights_out = MakeWeights(Shape{8 * W, 8, 1, 1}, neg_broadcast_weights, 1.0f, (bool)fq_input);

        // Prepare to perform softmax sum in parts
        size_t num_parts = 1;
        while (W / num_parts > 768) {  // 768 is maximum GNA1/2 kernel size
            num_parts *= 2;
        }
        // Abort if W is not divisible by power of 2
        if ((W / num_parts) * num_parts != W) {
            continue;
        }
        std::vector<float> avg_weights(8 * W / num_parts, 1.0f / W);
        std::vector<float> avg_broadcast(8 * W * num_parts, 0.0f);
        std::vector<float> minus_log_W(C * H * W, -log((float)W));
        std::vector<float> minus_log_W_partial(8 * W, -log((float)W));
        for (size_t i = 0; i < W * num_parts; i++) {
            avg_broadcast[i * 8] = 1.0f;
        }
        // weights tensor orders are Cout,Kh,Kw,Cin
        auto avg_weights_out = MakeWeights(Shape{8, 1, 1, W / num_parts}, avg_weights, 1.0f / W, (bool)fq_input);
        auto avg_broadcast_out = MakeWeights(Shape{W, 1, 1, 8 * num_parts}, avg_broadcast, 1.0f, (bool)fq_input);
        auto minus_log_W_out = MakeWeights(Shape{1, C * H * W}, minus_log_W, -log((float)W), (bool)fq_input);
        auto minus_log_W_partial_out = MakeWeights(Shape{1, 8 * W}, minus_log_W_partial, -log((float)W), (bool)fq_input);

        auto parent_2d = std::make_shared<ngraph::opset1::Reshape>(parent,
            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),false);
        OutputVector upstream;
        upstream.push_back(parent_2d);

        auto reshape_2 = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
            op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, C*H, W, 1ull})->output(0),false);
        auto identity_out = MakeWeights1x1(Shape{1, 1, 1, 1}, 1.0f, (bool)fq_input);
        auto conv_1 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_2->output(0),
            identity_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
        auto pool_1 = std::make_shared<ov::intel_gna::op::GNAMaxPool>(conv_1->output(0), Strides{1,W}, Shape{0,0}, Shape{0,0},
            Shape{1,W}, ov::op::RoundingType::FLOOR, op::PadType::VALID);
        auto broadcast_out = MakeWeights1x1(Shape{W, 1, 1, 1}, 1.0f, (bool)fq_input);  // identity
        Output<Node> prev;
        if (use_fq) {
            auto pool_fq = InsertFQ(pool_1->output(0), in_scale);
            auto conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(pool_fq->output(0),
                broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            auto conv2_fq = InsertFQ(conv_2->output(0), in_scale);
            prev = conv2_fq->output(0);
        } else {
            auto conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(pool_1->output(0),
                broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
           prev = conv_2->output(0);
        }
        auto reshape_3 = std::make_shared<ngraph::opset1::Reshape>(prev,
            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),false);
        auto x_minus_max = std::make_shared<op::v1::Subtract>(upstream[0], reshape_3->output(0));
        // perform softmax in log domain
        if (use_fq) {
            auto x_minus_max_fq = InsertFQ(x_minus_max->output(0), 2 * in_scale); // after subtracting max range is [-2*M,0]
            auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max_fq->output(0));
            auto exp_x_minus_max_fq = InsertFQ(exp_x_minus_max->output(0), 1.0f); // max is exp(0) == 1.0
            prev = exp_x_minus_max_fq->output(0);
        } else {
            auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max->output(0));
            prev = exp_x_minus_max->output(0);
        }
        auto reshape_7 = std::make_shared<op::v1::Reshape>(prev,
            op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, C*H*num_parts,1ull,W/num_parts})->output(0),false);
        auto avg_conv_1 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_7->output(0),
            avg_weights_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
        auto reshape_avg_conv_1 = std::make_shared<ngraph::opset1::Reshape>(avg_conv_1->output(0),
            op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, C*H, 8*num_parts})->output(0),false);
        auto avg_conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_avg_conv_1->output(0),
            avg_broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
        auto avg_conv_2_1d = std::make_shared<ngraph::opset1::Reshape>(avg_conv_2,
            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),false);
        auto log_avg_1d = std::make_shared<op::Log>(avg_conv_2_1d->output(0));
        auto diff_1 = std::make_shared<op::v1::Add>(x_minus_max->output(0), minus_log_W_out);
        auto diff_2 = std::make_shared<op::v1::Subtract>(diff_1->output(0), log_avg_1d->output(0));
        auto softmax_output_1d = std::make_shared<op::Exp>(diff_2->output(0));
        if (softmax_shape.size() == 4) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        } else if (softmax_shape.size() == 3) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        } else if (softmax_shape.size() == 2) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        }

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}

// Lumping C and H together can degrade accuracy due to high dynamic range for large problem sizes.
// In that case it may be better to split by channel.
// But elementwise operations are very expensive on GNA so those need to be merged.

bool ngraph::pass::GnaSoftmaxParallelDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto softmax_v1 = std::dynamic_pointer_cast<v1::Softmax>(node);
        auto softmax_v8 = std::dynamic_pointer_cast<v8::Softmax>(node);
        if ((nullptr == softmax_v1) && (nullptr == softmax_v8)) {
            continue;
        }

        const Output<Node>& parent = (nullptr != softmax_v1) ? softmax_v1->input_value(0) : softmax_v8->input_value(0);
        auto softmax_shape = (nullptr != softmax_v1) ? softmax_v1->get_output_shape(0) : softmax_v8->get_output_shape(0);
        auto axis = (nullptr != softmax_v1) ? softmax_v1->get_axis() : softmax_v8->get_axis();
        auto auto_broadcast = (nullptr != softmax_v1) ? softmax_v1->get_autob() : softmax_v8->get_autob();
        auto fq_input = FindFqUpstream(parent);  // note:  POT bug requires that we invent FQ layers
        size_t N = 1, C = 1, H, W;

        if (softmax_shape.size() == 4) {
            N = softmax_shape[0];
            C = softmax_shape[1];
            H = softmax_shape[2];
            W = softmax_shape[3];
        } else if (softmax_shape.size() == 3) {
            C = softmax_shape[0];
            H = softmax_shape[1];
            W = softmax_shape[2];
        } else if (softmax_shape.size() == 2) {
            H = softmax_shape[0];
            W = softmax_shape[1];
        } else {
            continue;
        }

        if (N != 1) {
            continue;   // Batch case not yet implemented
        } else if ((axis != softmax_shape.size() - 1) && (axis != -1)) {
            continue; // only support row softmax at this time
        }

        std::shared_ptr<FakeQuantize> fq_before = nullptr;
        std::shared_ptr<FakeQuantize> fq_after = nullptr;
        fq_before = std::dynamic_pointer_cast<FakeQuantize>(parent.get_node_shared_ptr());
        auto children = (softmax_v1) ? softmax_v1->output(0).get_target_inputs() : softmax_v8->output(0).get_target_inputs();
        for (auto child : children) {
            fq_after = std::dynamic_pointer_cast<FakeQuantize>(child.get_node()->shared_from_this());
            if (fq_after != nullptr) {
                break;
            }
        }
        bool use_fq = (fq_before || fq_after);
        float in_scale = 1.0f;
        float out_scale = 1.0f;
        if (fq_before) {
            auto out_high_const = std::dynamic_pointer_cast<Constant>(fq_before->input_value(4).get_node_shared_ptr());
            in_scale = *((float*)out_high_const->get_data_ptr());
        }
        if (fq_after) {
            auto out_high_const = std::dynamic_pointer_cast<Constant>(fq_after->input_value(4).get_node_shared_ptr());
            out_scale = *((float*)out_high_const->get_data_ptr());
        }

        // simple binary kernels used only for copying data
        std::vector<float> copy_weights(8 * 8, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create identity kernels
            copy_weights[i * 8 + i] = 1.0f;
        }
        std::vector<float> neg_broadcast_weights(8 * 8 * W, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create broadcast kernels
            for (size_t j = 0; j < W; j++) {
                for (size_t k = 0; k < 8; k++) {
                    if (k == i) {
                        neg_broadcast_weights[i * W  * 8 + j * 8 + k] = -1.0f;
                    }
                }
            }
        }
        auto copy_weights_out = MakeWeights(Shape{8, 8, 1, 1}, copy_weights, 1.0f, (bool)fq_input);
        auto neg_broadcast_weights_out = MakeWeights(Shape{8 * W, 8, 1, 1}, neg_broadcast_weights, 1.0f, (bool)fq_input);

        // Prepare to perform softmax sum in parts
        size_t num_parts = 1;
        while (W / num_parts > 768) {  // 768 is maximum GNA1/2 kernel size
            num_parts *= 2;
        }
        // Abort if W is not divisible by power of 2
        if ((W / num_parts) * num_parts != W) {
            continue;
        }
        std::vector<float> avg_weights(8 * W / num_parts, 1.0f / W);
        std::vector<float> avg_broadcast(8 * W * num_parts, 0.0f);
        std::vector<float> minus_log_W(C * H * W, -log((float)W));
        std::vector<float> minus_log_W_partial(8 * W, -log((float)W));
        for (size_t i = 0; i < W * num_parts; i++) {
            avg_broadcast[i * 8] = 1.0f;
        }
        // weights tensor orders are Cout,Kh,Kw,Cin
        auto avg_weights_out = MakeWeights(Shape{8, 1, 1, W / num_parts}, avg_weights, 1.0f / W, (bool)fq_input);
        auto avg_broadcast_out = MakeWeights(Shape{W, 1, 1, 8 * num_parts}, avg_broadcast, 1.0f, (bool)fq_input);
        auto minus_log_W_out = MakeWeights(Shape{1, C * H * W}, minus_log_W, -log((float)W), (bool)fq_input);
        auto minus_log_W_partial_out = MakeWeights(Shape{1, 8 * W}, minus_log_W_partial, -log((float)W), (bool)fq_input);

        auto parent_2d = std::make_shared<ngraph::opset1::Reshape>(parent,
            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),false);
        OutputVector upstream;
        upstream.push_back(parent_2d);

        if (C > 1) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{2}, {C, H * W})->output(0),false);
            std::vector<size_t> split_lengths(C, 1);
            const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
            auto split = std::make_shared<VariadicSplit>(reshape->output(0), Constant::create(ov::element::i64, ov::Shape{}, {0}), split_lengths_const);
            upstream.clear();
            for (size_t c = 0; c < C; c++) {
                upstream.push_back(split->output(c));
            }
        }

        for (size_t c = 0; c < C; c++) {
            auto reshape_2 = std::make_shared<ngraph::opset1::Reshape>(upstream[c],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, H, W, 1ull})->output(0),false);
            auto identity_out = MakeWeights1x1(Shape{1, 1, 1, 1}, 1.0f, (bool)fq_input);
            auto conv_1 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_2->output(0),
                identity_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            auto pool_1 = std::make_shared<ov::intel_gna::op::GNAMaxPool>(conv_1->output(0), Strides{1,W}, Shape{0,0}, Shape{0,0},
                Shape{1,W}, ov::op::RoundingType::FLOOR, op::PadType::VALID);
            auto broadcast_out = MakeWeights1x1(Shape{W, 1, 1, 1}, 1.0f, (bool)fq_input);  // identity
            Output<Node> prev;
            if (use_fq) {
                auto pool_fq = InsertFQ(pool_1->output(0), in_scale);
                auto conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(pool_fq->output(0),
                    broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
                auto conv2_fq = InsertFQ(conv_2->output(0), in_scale);
                prev = conv2_fq->output(0);
            } else {
                auto conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(pool_1->output(0),
                    broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
               prev = conv_2->output(0);
            }
            auto reshape_3 = std::make_shared<ngraph::opset1::Reshape>(prev,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),false);
            upstream[c] = reshape_3->output(0);
        }

        auto combined_max = std::make_shared<ov::op::v0::Concat>(upstream, 1);

        std::shared_ptr<ov::Node> x_minus_max = nullptr;
#define PAR_SUBTRACT
#define NUM_K 32
#ifdef PAR_SUBTRACT
        OutputVector subtract_operand;
        subtract_operand.push_back(parent_2d);
        subtract_operand.push_back(combined_max);
        auto combined_sub = std::make_shared<ov::op::v0::Concat>(subtract_operand, 0);
        auto new_transpose = std::make_shared<Transpose>(combined_sub->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        size_t num_K = NUM_K;
        while ((2 * C * H * W) % num_K != 0) {  // reduce parallelism if sizes don't match
            num_K = num_K / 2;
        }
        if (num_K != NUM_K) {
            printf("Warning:  failed to optimize parallel subtract\n");
        }
        auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_transpose->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, C * H * W / num_K, 2 * num_K, 1ull})->output(0),false);
        std::vector<float> weight(num_K * 2 * num_K, 0.0);
        for (size_t i = 0; i < num_K; i++) {
            weight[i * num_K * 2 + i * 2 + 0] = 1.0f;
            weight[i * num_K * 2 + i * 2 + 1] = -1.0f;
        }
        auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, 2 * num_K, 1}, weight);
        auto new_conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape->output(0),weight_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        x_minus_max = std::make_shared<ov::op::v1::Reshape>(new_conv->output(0),
            Constant::create(element::i64, Shape{2}, {1ull, C * H * W})->output(0),false);
#else
        x_minus_max = std::make_shared<op::v1::Subtract>(parent_2d, combined_max->output(0));
#endif
        // perform softmax in log domain
        Output<Node> prev;
#define EXP_CONV
#define NUM_K_EXP 32
#ifdef EXP_CONV
        if (use_fq) {
            auto x_minus_max_fq = InsertFQ(x_minus_max->output(0), 2 * in_scale);  // after subtracting max range is [-2*M,0]
            prev = x_minus_max_fq->output(0);
        } else {
            prev = x_minus_max->output(0);
        }
        size_t num_K_exp = NUM_K_EXP;
        while ((C * H * W) % num_K_exp != 0) {  // reduce parallelism if sizes don't match
            num_K_exp = num_K_exp / 2;
        }
        if (num_K_exp != NUM_K_EXP) {
            printf("Warning:  failed to optimize parallel exp\n");
        }
        auto new_reshape_exp = std::make_shared<ov::op::v1::Reshape>(prev,
            Constant::create(element::i64, Shape{4}, {1ull, C * H * W / num_K_exp, num_K_exp, 1ull})->output(0),false);
        std::vector<float> weight_exp(num_K_exp * num_K_exp, 0.0);
        for (size_t i = 0; i < num_K_exp; i++) {
            for (size_t j = 0; j < num_K_exp; j++) {
                if (i == j) {
                    weight_exp[i * num_K_exp + j] = 1.0f;
                }
            }
        }
        auto weight_exp_const = Constant::create(ngraph::element::f32, Shape{num_K_exp, 1, num_K_exp, 1}, weight_exp);
        auto new_conv_exp = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape_exp->output(0),weight_exp_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        if (use_fq) {
            auto exp_x_minus_max = std::make_shared<op::Exp>(new_conv_exp->output(0));
            auto exp_x_minus_max_fq = InsertFQ(exp_x_minus_max->output(0), 1.0f);  // max is exp(0) == 1.0
            prev = exp_x_minus_max_fq->output(0);
        } else {
            auto exp_x_minus_max = std::make_shared<op::Exp>(new_conv_exp->output(0));
            prev = exp_x_minus_max->output(0);
        }
#else
        if (use_fq) {
            auto x_minus_max_fq = InsertFQ(x_minus_max->output(0), 2 * in_scale); // after subtracting max range is [-2*M,0]
            auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max_fq->output(0));
            auto exp_x_minus_max_fq = InsertFQ(exp_x_minus_max->output(0), 1.0f); // max is exp(0) == 1.0
            prev = exp_x_minus_max_fq->output(0);
        } else {
            auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max->output(0));
            prev = exp_x_minus_max->output(0);
        }
#endif

        OutputVector upstream_exp;
        if (C > 1) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(prev,
                op::Constant::create(ngraph::element::i64, Shape{2}, {C, H * W})->output(0),false);
            std::vector<size_t> split_lengths(C, 1);
            const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
            auto split = std::make_shared<VariadicSplit>(reshape->output(0), Constant::create(ov::element::i64, ov::Shape{}, {0}), split_lengths_const);
            upstream.clear();
            for (size_t c = 0; c < C; c++) {
                upstream.push_back(split->output(c));
            }
        }

        for (size_t c = 0; c < C; c++) {
            auto reshape_7 = std::make_shared<op::v1::Reshape>(upstream[c],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, H*num_parts,1ull,W/num_parts})->output(0),false);
            auto avg_conv_1 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_7->output(0),
                avg_weights_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            auto reshape_avg_conv_1 = std::make_shared<ngraph::opset1::Reshape>(avg_conv_1->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, H, 8*num_parts})->output(0),false);
            auto avg_conv_2 = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_avg_conv_1->output(0),
                avg_broadcast_out,Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            auto avg_conv_2_1d = std::make_shared<ngraph::opset1::Reshape>(avg_conv_2,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),false);
            auto log_avg_1d = std::make_shared<op::Log>(avg_conv_2_1d->output(0));
            upstream[c] = log_avg_1d->output(0);
        }

        auto combined_log_avg = std::make_shared<ov::op::v0::Concat>(upstream, 1);

        std::shared_ptr<ov::Node> softmax_output_1d = nullptr;
#define PAR_ADD
#define NUM_K_ADD 32
#ifdef PAR_ADD
        OutputVector add_operand;
        add_operand.push_back(x_minus_max->output(0));
        add_operand.push_back(minus_log_W_out);
        add_operand.push_back(combined_log_avg);
        auto combined_add = std::make_shared<ov::op::v0::Concat>(add_operand, 0);
        auto new_transpose_add = std::make_shared<Transpose>(combined_add->output(0), 
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        size_t num_K_add = NUM_K_ADD;
        while ((3 * C * H * W) % num_K_add != 0) {  // reduce parallelism if sizes don't match
            num_K_add = num_K_add / 2;
        }
        if (num_K_add != NUM_K_ADD) {
            printf("Warning:  failed to optimize parallel add\n");
        }
        auto new_reshape_add = std::make_shared<ov::op::v1::Reshape>(new_transpose_add->output(0),
            Constant::create(element::i64, Shape{4}, {1ull, C * H * W / num_K_add, 3 * num_K_add, 1ull})->output(0),false);
        std::vector<float> weight_add(num_K_add * 3 * num_K_add, 0.0);
        for (size_t i = 0; i < num_K_add; i++) {
            weight_add[i * num_K_add * 3 + i * 3 + 0] = 1.0f;
            weight_add[i * num_K_add * 3 + i * 3 + 1] = 1.0f;
            weight_add[i * num_K_add * 3 + i * 3 + 2] = -1.0f;
        }
        auto weight_add_const = Constant::create(ngraph::element::f32, Shape{num_K_add, 1, 3 * num_K_add, 1}, weight_add);
        auto new_conv_add = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape_add->output(0),weight_add_const->output(0), 
            Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
        softmax_output_1d = std::make_shared<op::Exp>(new_conv_add->output(0));
#else
        auto diff_1 = std::make_shared<op::v1::Add>(x_minus_max->output(0), minus_log_W_out);
        auto diff_2 = std::make_shared<op::v1::Subtract>(diff_1->output(0), combined_log_avg->output(0));
        softmax_output_1d = std::make_shared<op::Exp>(diff_2->output(0));
#endif
        if (softmax_shape.size() == 4) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        } else if (softmax_shape.size() == 3) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        } else if (softmax_shape.size() == 2) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(softmax_output_1d->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),false);
            if (nullptr != softmax_v1) {
                ngraph::replace_node_update_name(softmax_v1, new_reshape);
            } else {
                ngraph::replace_node_update_name(softmax_v8, new_reshape);
            }
        }
        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
