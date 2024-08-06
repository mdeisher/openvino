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

#include "gna_transconv_before.hpp"
#include "gna_helper.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::GnaTransposeConvolutionPreDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
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
        auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
        auto fq_input = FindFqUpstream(input);

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
        } else if (weights_shape.size() == 3) {  // ConvTranspose1d: 3D input with N=1, 3D filters, 1D stride, 1D dilation, 1D padding
            if (input_shape.size() != 3 ||
                pads_begin.size() != 1 ||
                pads_end.size() != 1 ||
                dilations.size() != 1 ||
                strides.size() != 1 ||
                input_shape[0] != 1 ||
                pads_begin[0] != pads_end[0] || // only symmetric padding is supported
                strides[0] > weights_shape[2]) {  // only support cases where kernel overlaps input 
                continue;
            }
        } else {
            continue;
        }

        const Output<Node>& parent = conv->input_value(0);

        // find padded dimensions of virtual equivalent convolution
        bool is_1D = (weights_shape.size() == 3);
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t H = input_shape[2];
        size_t W = is_1D ? 1 : input_shape[3];
        size_t H_pad_inner = strides[0] - 1;  // number of zero rows to insert between rows of input
        size_t W_pad_inner = is_1D ? 0 : strides[1] - 1;  // number of zero columns to insert between columns of input
        size_t H_pad_outer = weights_shape[2] - pads_begin[0] - 1;             // new amount of outer padding in H
        size_t W_pad_outer = is_1D ? 0 : weights_shape[3] - pads_begin[1] - 1;  // new amount of outer padding in W
        size_t H_pad_additional = output_padding[0];  // additional padding in H to get the proper number of outputs
        size_t W_pad_additional = is_1D ? 0 : output_padding[1];  // additional padding in W to get the proper number of outputs
        size_t H_new = H + (H - 1) * H_pad_inner + 2 * H_pad_outer + H_pad_additional;
        size_t W_new = is_1D ? 1 : W + (W - 1) * W_pad_inner + 2 * W_pad_outer + W_pad_additional;

        if (N != 1) {
            continue;
        }

        // only stride 1 cases can be handled before layout conversion

        if (((strides.size() == 2) && (strides[0] == 1) && (strides[1] == 1))
            || ((strides.size() == 1) && (strides[0] == 1))) {

            const float* weight_ptr = weights_const->get_data_ptr<float>();
            std::vector<float> new_weights(weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3],0.0f);
            float* new_weight_ptr = new_weights.data();
            ov::Shape new_weights_shape;
            ov::CoordinateDiff new_pads_begin;
            ov::CoordinateDiff new_pads_end;
            new_weights_shape.push_back(weights_shape[1]);
            new_weights_shape.push_back(weights_shape[0]);
            new_weights_shape.push_back(weights_shape[2]);
            new_weights_shape.push_back(weights_shape[3]);
            for (size_t i = 0; i < new_weights_shape[0]; i++) {  // Co
                for (size_t j = 0; j < new_weights_shape[1]; j++) {  // Ci
                    auto Kh = weights_shape[2];
                    auto Kw = weights_shape[3];
                    auto kernel_size = (weights_shape.size() == 4) ? new_weights_shape[2] * new_weights_shape[3] : new_weights_shape[2];
                    auto kernel = weight_ptr + j * weights_shape[1] * kernel_size + i * kernel_size;
                    auto new_kernel = new_weight_ptr + i * new_weights_shape[1] * kernel_size + j * kernel_size;
                    for (size_t k = 0; k < Kh; k++) {  // store kernels in weird OpenVINO backwards order
                        for (size_t m = 0; m < Kw; m++) {
                            new_kernel[k * Kw + m] = kernel[(Kh - k - 1) * Kw + (Kw - m - 1)];
                        }
                    }
                }
            }
            auto new_weights_node = InsertWeights(new_weights_shape, new_weights, (bool)fq_input);
            new_pads_begin.push_back(H_pad_outer);
            new_pads_end.push_back(H_pad_outer);
            if (strides.size() == 2) {
                new_pads_begin.push_back(W_pad_outer);
                new_pads_end.push_back(W_pad_outer);
            }
            auto new_conv = std::make_shared<opset1::Convolution>(parent,
                new_weights_node->output(0),strides,new_pads_begin,new_pads_end,dilations,PadType::EXPLICIT);
            ngraph::replace_node_update_name(conv, new_conv);
            is_graph_modfied = true;
        }
    }

    return is_graph_modfied;
}
