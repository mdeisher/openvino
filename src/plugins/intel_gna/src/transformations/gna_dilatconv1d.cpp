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

#include "gna_dilatconv1d.hpp"
#include "gna_helper.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::GnaDilatConvolution1dDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution> (node);
        if (nullptr == conv) {
            continue;
        }

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& weights = conv->input_value(1);
        auto input_shape = input.get_shape();
        auto weights_shape = weights.get_shape();
        auto output_shape = conv->get_output_shape(0);
        auto auto_pad= conv->get_auto_pad();
        auto dilations = conv->get_dilations();
        auto pads_begin = conv->get_pads_begin();
        auto pads_end = conv->get_pads_end();
        auto strides = conv->get_strides();
        const float* weight_ptr = NULL; 
        auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
        auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        if (fq) {
            weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(fq->input_value(0).get_node_shared_ptr());
            if (weights_const == nullptr) {
                continue;
            }
        }
        weight_ptr = weights_const->get_data_ptr<float>();

        // only support 3D input with N=1, 3D filters, 1D stride, 1D dilation, 1D padding
        if (input_shape.size() < 3 ||
            weights_shape.size() < 3 ||
            output_shape.size() < 3 ||
            pads_begin.size() != 1 ||
            pads_end.size() != 1 ||
            dilations.size() != 1 ||
            strides.size() != 1 ||
            input_shape[0] != 1) {
            continue;
        }

        if (dilations[0] == 1) {  // not dilated
            continue;
        }

        if (strides[0] != 1) {  // only stride 1 supported at this time
            continue;
        }

        // find Convolution--><Add>--><Activation> pattern, skip if multiple consumers
        const Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto add_after = std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        auto prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(children.begin()->get_node()->shared_from_this());
        auto relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(children.begin()->get_node()->shared_from_this());
        auto sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(children.begin()->get_node()->shared_from_this());
        auto tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(add_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(add_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(add_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(add_children.begin()->get_node()->shared_from_this());
        }

        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t H = input_shape[2];
        size_t Co = weights_shape[0];
        size_t Ci = weights_shape[1];
        size_t Kh = weights_shape[2];
        size_t Kh_dilated = Kh + (Kh - 1) * (dilations[0] - 1);

        if (Kh > H) {  // oversized 
            continue;  // don't yet support this case
        }

        OutputVector parts;

        // beginning part
        size_t H_start = 0;
        size_t H_stop = Kh_dilated - 1;
        auto slice_start = op::Constant::create(ngraph::element::i64, Shape{3}, {0ull, 0ull, H_start});
        auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{3}, {N, C, H_stop});
        auto slice_step = op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 1ull, 1ull});
        auto new_slice = std::make_shared<v8::Slice>(input, slice_start, slice_stop, slice_step);
        parts.push_back(new_slice->output(0));

        // middle part
        auto H_nonfoldable = H % dilations[0];
        if (H_nonfoldable > 0) {  
            H_start = 0;
            H_stop = H - H_nonfoldable;
            slice_start = op::Constant::create(ngraph::element::i64, Shape{3}, {0ull, 0ull, H_start});
            slice_stop = op::Constant::create(ngraph::element::i64, Shape{3}, {N, C, H_stop});
            slice_step = op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 1ull, 1ull});
            new_slice = std::make_shared<v8::Slice>(input, slice_start, slice_stop, slice_step);
            parts.push_back(new_slice->output(0));
        } else {
            parts.push_back(input);
        }

        // end part
        H_start = H - H_nonfoldable - Kh_dilated + 1;
        H_stop = H;
        slice_start = op::Constant::create(ngraph::element::i64, Shape{3}, {0ull, 0ull, H_start});
        slice_stop = op::Constant::create(ngraph::element::i64, Shape{3}, {N, C, H_stop});
        slice_step = op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 1ull, 1ull});
        new_slice = std::make_shared<v8::Slice>(input, slice_start, slice_stop, slice_step);
        parts.push_back(new_slice->output(0));

        // create dilated kernel
        std::vector<float> dilated_kernel(Co * Ci * Kh_dilated, 0.0f);
        float* dilated_kernel_ptr = dilated_kernel.data();
        for (auto i = 0; i < weights_shape[0]; i++) {  // Co
            for (auto j = 0; j < weights_shape[1]; j++) {  // Ci
                size_t m = 0;
                for (auto k = 0; k < weights_shape[2] - 1; k++) {
                    dilated_kernel_ptr[i * Ci * Kh_dilated + j * Kh_dilated + m] =
                        weight_ptr[i * Ci * Kh + j * Kh + k];
                    m++;
                    for (auto n = 0; n < dilations[0] - 1; n++) {
                        dilated_kernel_ptr[i * Ci * Kh_dilated + j * Kh_dilated + m] = 0.0f;
                        m++;
                    }
                }
                dilated_kernel_ptr[i * Ci * Kh_dilated + j * Kh_dilated + m] =
                    weight_ptr[i * Ci * Kh + j * Kh + Kh - 1];
            }
        }

        // beginning convolution
        ov::Strides new_strides = {1};
        ov::CoordinateDiff new_pads_begin = {pads_begin[0]};
        ov::CoordinateDiff new_pads_end = {0};
        ov::Strides new_dilations = {1};
        auto dilated_kernel_const = op::Constant::create(ngraph::element::f32, Shape{Co, Ci, Kh_dilated}, dilated_kernel);
        auto new_conv = std::make_shared<opset1::Convolution>(parts[0],
            dilated_kernel_const->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::EXPLICIT);
        parts[0] = new_conv->output(0);

        // middle convolution
        new_pads_begin = {0};
        new_pads_end = {0};
        new_conv = std::make_shared<opset1::Convolution>(parts[1],
            dilated_kernel_const->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::VALID);
        parts[1] = new_conv->output(0);

        // end convolution
        new_pads_begin = {0};
        new_pads_end = {pads_end[0]};
        new_conv = std::make_shared<opset1::Convolution>(parts[2],
            dilated_kernel_const->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::EXPLICIT);
        parts[2] = new_conv->output(0);

        // concatenate the beginning, middle, and end
        auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 2);

        ngraph::replace_node_update_name(conv, new_concat);

        //if (prelu_after) {
        //    ngraph::replace_node_update_name(prelu_after, new_transpose);
        //} else if (relu_after) {
        //    ngraph::replace_node_update_name(relu_after, new_transpose);
        //} else if (sigmoid_after) {
        //    ngraph::replace_node_update_name(sigmoid_after, new_transpose);
        //} else if (tanh_after) {
        //    ngraph::replace_node_update_name(tanh_after, new_transpose);
        //} else if (add_after) {
        //    ngraph::replace_node_update_name(add_after, new_transpose);
        //} else {
        //    ngraph::replace_node_update_name(conv, new_transpose);
        //}
        is_graph_modfied = true;
    }

    return is_graph_modfied;
}
