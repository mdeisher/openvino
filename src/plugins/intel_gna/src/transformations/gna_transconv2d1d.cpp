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

#include "gna_transconv2d1d.hpp"
#include "gna_helper.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;


bool ngraph::pass::GnaTransposeConvolution2d1dDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node);
        if (nullptr == conv) {
            continue;
        }

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& weights = conv->input_value(1);
        auto conv_name = conv->get_friendly_name();
        auto input_shape = input.get_shape();
        auto weights_shape = weights.get_shape();
        auto output_shape = conv->get_output_shape();
        auto auto_pad= conv->get_auto_pad();
        auto dilations = conv->get_dilations();
        auto output_padding = conv->get_output_padding();
        auto pads_begin = conv->get_pads_begin();
        auto pads_end = conv->get_pads_end();
        auto strides = conv->get_strides();
        auto weights_fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
        bool nchw_only = false;  // keep graph NCHW and rely on layout transformation to convert to NHWC for GNA
        bool layout_nchw = true;
        bool do_folding = true;

        if (weights_fq) {
            weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(weights_fq->input_value(0).get_node_shared_ptr());
            if (weights_const == nullptr) {
                continue;
            }
        }

        if (weights_shape.size() == 4) {  // ConvTranspose2d: 4D input with N=1, 4D filters, 2D stride, 2D dilation, 2D padding
            if (input_shape.size() != 4 ||
                pads_begin.size() != 2 ||
                pads_end.size() != 2 ||
                dilations.size() != 2 ||
                strides.size() != 2 ||
                input_shape[0] != 1 ||
                pads_begin[0] != pads_end[0] ||  // only symmetric padding is supported
                pads_begin[1] != pads_end[1] ||
                strides[0] > weights_shape[2] ||  // only support cases where kernel overlaps input
                strides[1] > weights_shape[3]) {
                continue;
            }
        } else {
            continue;
        }

        // find Convolution--><Add>--><Activation> pattern OR
        // find Convolution--><Add>--><FQ>--><Activation>--><FQ> pattern else skip
        const Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto add_after = std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        auto addfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(children.begin()->get_node()->shared_from_this());
        auto relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(children.begin()->get_node()->shared_from_this());
        auto sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(children.begin()->get_node()->shared_from_this());
        auto tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(children.begin()->get_node()->shared_from_this());
        auto actfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto slice_after = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            addfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(add_children.begin()->get_node()->shared_from_this());
            prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(add_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(add_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(add_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(add_children.begin()->get_node()->shared_from_this());
            slice_after = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(add_children.begin()->get_node()->shared_from_this());
        }
        if (addfq_after) {
            auto addfq_children = addfq_after->output(0).get_target_inputs();
            if (addfq_children.size() != 1) {
                continue;
            }
            prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(addfq_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(addfq_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(addfq_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(addfq_children.begin()->get_node()->shared_from_this());
            slice_after = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(addfq_children.begin()->get_node()->shared_from_this());
        }
        if (slice_after == nullptr) {
            OutputVector upstream;
            if (prelu_after) {
                upstream.push_back(prelu_after->output(0));
            } else if (relu_after) {
                upstream.push_back(relu_after->output(0));
            } else if (sigmoid_after) {
                upstream.push_back(sigmoid_after->output(0));
            } else if (tanh_after) {
                upstream.push_back(tanh_after->output(0));
            }
            if (upstream.size() > 0) {
                auto act_children = upstream[0].get_target_inputs();
                if (act_children.size() != 1) {
                    continue;
                }
                actfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(act_children.begin()->get_node()->shared_from_this());
                slice_after = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(act_children.begin()->get_node()->shared_from_this());
                if (actfq_after) {
                    auto actfq_children = actfq_after->output(0).get_target_inputs();
                    if (actfq_children.size() != 1) {
                        continue;
                    }
                    slice_after = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(actfq_children.begin()->get_node()->shared_from_this());
                }
            } else {
                continue;
            }
        }

        // find padded dimensions of virtual equivalent convolution
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t H = input_shape[2];
        size_t W = input_shape[3];
        size_t H_pad_inner = strides[0] - 1;  // number of zero rows to insert between rows of input
        size_t W_pad_inner = strides[1] - 1;  // number of zero columns to insert between columns of input
        size_t H_pad_outer = weights_shape[2] - pads_begin[0] - 1;             // new amount of outer padding in H
        size_t W_pad_outer = weights_shape[3] - pads_begin[1] - 1;  // new amount of outer padding in W
        size_t H_pad_additional = output_padding[0];  // additional padding in H to get the proper number of outputs
        size_t W_pad_additional = output_padding[1];  // additional padding in W to get the proper number of outputs
        size_t H_new = H + (H - 1) * H_pad_inner + 2 * H_pad_outer + H_pad_additional;
        size_t W_new = W + (W - 1) * W_pad_inner + 2 * W_pad_outer + W_pad_additional;

        if (N != 1) {
            continue;
        }

        if ((input_shape[3] == 1) && (weights_shape[3] > 1)) {  // 1D in W
            std::vector<size_t> W_indices;
            if (slice_after) {  // check to see if we can avoid part of calculation
                auto slice_shape = slice_after->get_input_shape(0);
                auto begin_mask = slice_after->get_begin_mask();
                auto end_mask = slice_after->get_end_mask();
                auto begin_const = std::dynamic_pointer_cast<ngraph::op::Constant>(slice_after->input_value(1).get_node_shared_ptr());
                auto end_const = std::dynamic_pointer_cast<ngraph::op::Constant>(slice_after->input_value(2).get_node_shared_ptr());
                auto begin_data = begin_const->get_data_ptr<int64_t>();
                auto end_data = end_const->get_data_ptr<int64_t>();
                int32_t b32 = (int32_t)begin_data[3];
                int32_t e32 = (int32_t)end_data[3];
                size_t begin = (begin_mask[3] == 1) ? 0 : ((b32 < 0) ? slice_shape[3] + b32 + 1 : b32);
                size_t end = (end_mask[3] == 1) ? slice_shape[3] : ((e32 < 0) ? slice_shape[3] + e32 + 1 : e32);
                for (auto i = begin; i < end; i++) {
                    W_indices.push_back(i);
                }
            } else {
                for (auto i = 0; i < weights_shape[3]; i++) {
                    W_indices.push_back(i);
                }
            }
            OutputVector upstream;
            std::shared_ptr<ov::op::v1::ConvolutionBackpropData> new_conv = nullptr;
            for (auto i = 0; i < W_indices.size(); i++) {
                auto new_auto_pad = conv->get_auto_pad();
                auto new_dilations = conv->get_dilations();
                auto new_output_padding = conv->get_output_padding();
                auto new_pads_begin = conv->get_pads_begin();
                auto new_pads_end = conv->get_pads_end();
                auto new_strides = conv->get_strides();
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(weights_shape[0] * weights_shape[1] * weights_shape[2],0.0f);
                float* new_weight_ptr = new_weights.data();
                ov::Shape new_weights_shape;
                new_weights_shape.push_back(weights_shape[0]);
                new_weights_shape.push_back(weights_shape[1]);
                new_weights_shape.push_back(weights_shape[2]);
                new_weights_shape.push_back(1);
                for (size_t j = 0; j < new_weights_shape[0]; j++) {  // Co
                    for (size_t k = 0; k < new_weights_shape[1]; k++) {  // Ci
                        auto kernel = weight_ptr + j * weights_shape[1] * weights_shape[2] * weights_shape[3] +
                                      k * weights_shape[2] * weights_shape[3];
                        auto new_kernel = new_weight_ptr + j * new_weights_shape[1] * new_weights_shape[2] * new_weights_shape[3] +
                                          k * new_weights_shape[2] * new_weights_shape[3];
                        for (size_t m = 0; m < new_weights_shape[2]; m++) {  // Kh
                            new_kernel[m] = kernel[m * weights_shape[3] + W_indices[i]];
                        }
                    }
                }

                auto new_weights_const = op::Constant::create(ngraph::element::f32,new_weights_shape, new_weights);
                if (weights_fq) {
                    auto new_weights_fq = CopyFQ(new_weights_const->output(0), weights_fq);
                    new_conv = std::make_shared<opset1::ConvolutionBackpropData>(parent,
                        new_weights_fq->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,new_auto_pad, new_output_padding);
                    upstream.push_back(new_conv->output(0));
                } else {
                    new_conv = std::make_shared<opset1::ConvolutionBackpropData>(parent,
                        new_weights_const->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,new_auto_pad, new_output_padding);
                    upstream.push_back(new_conv->output(0));
                }
            }
            if (upstream.size() == 1) {
                if (!(add_after || prelu_after || relu_after || sigmoid_after || tanh_after || slice_after)) {
                    ngraph::replace_node(conv, new_conv);
                }
            } else {
                auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 3);
                upstream.resize(0);
                upstream.push_back(new_concat->output(0));
                if (!(add_after || prelu_after || relu_after || sigmoid_after || tanh_after || slice_after)) {
                    ngraph::replace_node(conv, new_concat);
                }
            }
            if (add_after) {
                auto new_add = std::make_shared<opset1::Add>(upstream[0], add_after->input_value(1));
                upstream[0] = new_add->output(0);
                if (!(prelu_after || relu_after || sigmoid_after || tanh_after || slice_after)) {
                    ngraph::replace_node(add_after, new_add);
                }
            }
            if (addfq_after) {
                auto new_addfq_after = CopyFQ(upstream[0], addfq_after);
                upstream[0] = new_addfq_after->output(0);
            }
            InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
            if (slice_after && actfq_after) {
                auto new_actfq_after = CopyFQ(upstream[0], actfq_after);
                upstream[0] = new_actfq_after->output(0);
            }
            if (prelu_after) {
                if (!slice_after) {
                    ngraph::replace_node_update_name(prelu_after, upstream[0].get_node_shared_ptr());
                } else {
                    ngraph::replace_node_update_name(slice_after, upstream[0].get_node_shared_ptr());
                }
            } else if (relu_after) {
                if (!slice_after) {
                    ngraph::replace_node_update_name(relu_after, upstream[0].get_node_shared_ptr());
                } else {
                    ngraph::replace_node_update_name(slice_after, upstream[0].get_node_shared_ptr());
                }
            } else if (sigmoid_after) {
                if (!slice_after) {
                    ngraph::replace_node_update_name(sigmoid_after, upstream[0].get_node_shared_ptr());
                } else {
                    ngraph::replace_node_update_name(slice_after, upstream[0].get_node_shared_ptr());
                }
            } else if (tanh_after) {
                if (!slice_after) {
                    ngraph::replace_node_update_name(tanh_after, upstream[0].get_node_shared_ptr());
                } else {
                    ngraph::replace_node_update_name(slice_after, upstream[0].get_node_shared_ptr());
                }
            } else if (add_after && slice_after) {
                ngraph::replace_node_update_name(slice_after, upstream[0].get_node_shared_ptr());
            }

            is_graph_modfied = true;

        }
    }

    return is_graph_modfied;
}
