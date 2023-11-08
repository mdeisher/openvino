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

#include "gna_transconv_after.hpp"
#include "gna_helper.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

void BuildKernelMap1D2(size_t dim_new, size_t pad_outer, size_t pad_additional, size_t stride, size_t n_weights, bool do_folding,
    std::vector<std::vector<std::vector<size_t>>> &kernel_list,  // list of kernel layouts
    std::vector<std::vector<int32_t>> &input_list)               // list of input endpoints
{
    bool print_stuff = false;

    // Rather than calculating the indices of the kernels and inputs operations on-the-fly
    // we simply construct vectors indicating the position of each kernel row/column and non-zero input.
    std::vector<int32_t> input_index(dim_new, -1);  // holds index of input row/column or -1 if padding
    // Find input layout
    int32_t j = 0;
    for (size_t i = pad_outer; i < dim_new - pad_outer - pad_additional; i += stride) {
        input_index[i] = j++;
    }
    std::vector<std::vector<size_t>> kernel_pos;  // offsets for convolution kernel
    // Find kernel layout required for each virtual convolution output
    int32_t input_start_prev = -1;
    int32_t input_end_prev = -1;
    for (size_t i = 0; i < dim_new - n_weights + 1; i++) {
        int32_t input_start = -1;
        int32_t input_end = -1;
        kernel_pos.resize(kernel_pos.size() + 1);
        for (size_t j = 0; j < n_weights; j++) {
            if (input_index[i + j] >= 0) {  // do not include multiply by zero (indicated by index == -1)
                if (input_start < 0) {
                    input_start = input_index[i + j];
                    input_end = input_start;
                } else {
                    input_end = input_index[i + j];
                }
                kernel_pos[kernel_pos.size()-1].push_back(j);
            }
        }
        if (input_start_prev < 0) {
            input_start_prev = input_start;
            input_end_prev = input_end;
        } else {
            if ((input_start != input_start_prev) || (input_end != input_end_prev)) { // need a new layer
                std::vector<size_t> new_row = kernel_pos[kernel_pos.size() - 1];
                kernel_list.push_back(kernel_pos);  // save kernel layout
                kernel_list[kernel_list.size() - 1].pop_back();  // remove last row since it is really part of the next kernel
                std::vector<int32_t> input_range;
                input_range.push_back(input_start_prev);
                input_range.push_back(input_end_prev);
                input_list.push_back(input_range);
                kernel_pos.clear();
                kernel_pos.push_back(new_row);
                input_start_prev = input_start;
                input_end_prev = input_end;
            }
        }
    }

    // save final kernel and input layout
    std::vector<int32_t> input_range;
    input_range.push_back(input_start_prev);
    input_range.push_back(input_end_prev);
    input_list.push_back(input_range);
    std::vector<std::vector<size_t>> new_kernel;
    for (uint32_t m = 0; m < kernel_pos.size(); m++) {
        std::vector<size_t> new_kernel_element;
        for (uint32_t n = 0; n < kernel_pos[m].size(); n++) {
            new_kernel_element.push_back(kernel_pos[m][n]);
        }
        new_kernel.push_back(new_kernel_element);
    }
    kernel_list.push_back(new_kernel);  // save kernel layout

    // extend kernels by folding into output channels if possible
    if (do_folding) {
        if (input_list.size() > 2) {
            size_t i = 2;
            while (i < input_list.size()) {
                auto curr_low = input_list[i][0];
                auto prev_low = input_list[i - 1][0];
                auto curr_high = input_list[i][1];
                auto prev_high = input_list[i - 1][1];
                if ((curr_low >= prev_low) && (curr_high <= prev_high)) {  // time periods overlap
                    std::vector<std::vector<size_t>> append_kernel;
                    for (size_t j = 0; j < kernel_list[i].size(); j++) {
                        std::vector<size_t> new_kernel(kernel_list[i - 1][0].size(), PAD_VALUE);
                        for (size_t k = 0; k < kernel_list[i - 1][0].size(); k++) {
                            for (size_t m = 0; m < kernel_list[i][j].size(); m++) {
                                if (input_list[i][0] + m == input_list[i - 1][0] + k) {
                                    new_kernel[k] = kernel_list[i][j][m];
                                }
                            }
                        }
                        append_kernel.push_back(new_kernel);
                    }
                    for (size_t j = 0; j < append_kernel.size(); j++) {
                        kernel_list[i - 1].push_back(append_kernel[j]);
                    }
                    kernel_list.erase(kernel_list.begin() + i);  // remove redundant kernel
                    input_list.erase(input_list.begin() + i);    // remove redundant input
                } else {
                    i++;
                }
            }
        }
    }

    // consolidate convolutions when possible
    size_t i = 1;
    while (i < input_list.size()) {
        if (kernel_list[i].size() == kernel_list[i - 1].size()) {
            bool kernel_match = true;
            for (size_t j = 0; j < kernel_list[i].size(); j++) {
                if (kernel_list[i][j].size() == kernel_list[i - 1][j].size()) {
                    for (size_t j = 0; j < kernel_list[i].size(); j++) {
                        if (kernel_list[i][j] != kernel_list[i - 1][j]) {
                            kernel_match = false;
                        }
                    }
                } else {
                    kernel_match = false;
                }
            }

            if (kernel_match) {
                input_list[i - 1][1] = input_list[i][1];  // increase input length
                kernel_list.erase(kernel_list.begin() + i);  // remove redundant kernel
                input_list.erase(input_list.begin() + i);  // remove redundant input
            } else {
                i++;
            }
        } else {
            i++;
        }
    }

    if (print_stuff)
    {
        size_t output_index = 0;
        for (size_t i = 0; i < input_list.size(); i++) {
            printf("position %llu:\nx%d .. x%d\n", output_index, input_list[i][0], input_list[i][1]);
            for (uint32_t m = 0; m < kernel_list[i].size(); m++) {
                for (uint32_t n = 0; n < kernel_list[i][m].size(); n++) {
                    if (kernel_list[i][m][n] == PAD_VALUE) {
                        printf("00 ");
                    } else {
                        printf("k%llu ", kernel_list[i][m][n]);
                    }
                }
                printf("\n");
            }
            output_index += kernel_list[i].size() * ((input_list[i][1] - input_list[i][0] + 1) - kernel_list[i][0].size() + 1);
            printf("\n");
        }
    }

    // reverse kernel -- OpenVINO stores them backwards
    for (size_t i = 0; i < input_list.size(); i++) {
        for (uint32_t m = 0; m < kernel_list[i].size(); m++) {
            for (uint32_t n = 0; n < kernel_list[i][m].size(); n++) {
                if (kernel_list[i][m][n] != PAD_VALUE) {
                    kernel_list[i][m][n] = n_weights - kernel_list[i][m][n] - 1;
                }
            }
        }
    }
}

bool ngraph::pass::GnaTransposeConvolutionPostDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node);
        if (nullptr == conv) {
            continue;
        }

        Output<Node>& input = conv->input_value(0);
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
        float weights_min = 0.0f, weights_max = 0.0f;
        auto fq_input = FindFqUpstream(input);  // note:  POT bug requires that we invent FQ layers for transconv ops
        auto fq_before = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(input.get_node_shared_ptr());
        auto transpose_before = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(input.get_node_shared_ptr());

        if (transpose_before) {
            fq_before = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(transpose_before->input_value(0).get_node_shared_ptr());
            input = transpose_before->input_value(0);
        }
        if (fq_before) {
            input = fq_before->input_value(0);
        } 

        if (weights_fq) {
            weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(weights_fq->input_value(0).get_node_shared_ptr());
            if (weights_const == nullptr) {
                continue;
            }
        }

        if (fq_input) {  // weights stats needed to insert FQ layers later
            size_t n = 1;
            for (size_t i = 0; i < weights_shape.size(); i++) {
                n *= weights_shape[i];
            }
            const float* weight_ptr = weights_const->get_data_ptr<float>();
            weights_min = weights_max = weight_ptr[0];
            for (size_t i = 0; i < n; i++) {
                if (weights_min > weight_ptr[i]) {
                    weights_min = weight_ptr[i];
                }
                if (weights_max < weight_ptr[i]) {
                    weights_max = weight_ptr[i];
                }
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

        // find Convolution--><Add>--><FQ>--><Activation>--><FQ> pattern else skip
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto fq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        if (fq_after) {
            children = fq_after->output(0).get_target_inputs();
            if (children.size() != 1) {
                continue;
            }
        }
        auto add_after = std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        auto addfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(children.begin()->get_node()->shared_from_this());
        auto relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(children.begin()->get_node()->shared_from_this());
        auto sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(children.begin()->get_node()->shared_from_this());
        auto tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(children.begin()->get_node()->shared_from_this());
        auto actfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto reshape_after = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            addfq_after = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(add_children.begin()->get_node()->shared_from_this());
            if (addfq_after) {
                add_children = addfq_after->output(0).get_target_inputs();
                if (add_children.size() != 1) {
                    continue;
                }
            }
            prelu_after = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(add_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<ngraph::opset1::Relu>(add_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(add_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<ngraph::opset1::Tanh>(add_children.begin()->get_node()->shared_from_this());
            reshape_after = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(add_children.begin()->get_node()->shared_from_this());
        }
        if (reshape_after == nullptr) {
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
                if (actfq_after) {
                    act_children = actfq_after->output(0).get_target_inputs();
                    if (act_children.size() != 1) {
                        continue;
                    }
                }
                reshape_after = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(act_children.begin()->get_node()->shared_from_this());
            }
        }

        // look for and absorb transpose before if it is converting to NCHW
        Output<Node> transpose_parent; 
        if (transpose_before) {
            transpose_parent = transpose_before->input_value(0);
            auto transpose_input_shape = transpose_before->input(0).get_shape();
            auto transpose_output_shape = transpose_before->output(0).get_shape();
            const Output<Node>& transpose_order = transpose_before->input_value(1);
            auto transpose_order_dim = transpose_order.get_shape().size();
            if (transpose_order_dim != 1)
                continue;
            auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
            if (!const_with_order_values)
                continue;
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
            if (!((order[0] == 0) && (order[1] == 3) && (order[2] == 1) && (order[3] == 2))) {
                transpose_before = nullptr;
            }
        }

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

        // ========== Stride 1 cases ===========
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
                            //printf("%e ", new_kernel[k * Kw + m]);
                        }
                        //printf("\n");
                    }
                }
            }
            auto new_weights_node = InsertWeights(new_weights_shape, new_weights, (bool)fq_input, weights_min, weights_max);
            new_pads_begin.push_back(H_pad_outer);
            new_pads_end.push_back(H_pad_outer);
            if (strides.size() == 2) {
                new_pads_begin.push_back(W_pad_outer);
                new_pads_end.push_back(W_pad_outer);
            }
            OutputVector upstream;
            upstream.push_back(input);
            if (fq_input) {
                auto new_fq = CopyFQ(upstream[0], fq_input);
                upstream[0] = new_fq->output(0);
            }
            auto new_conv = std::make_shared<opset1::Convolution>(upstream[0],
                new_weights_node->output(0),strides,new_pads_begin,new_pads_end,dilations,PadType::EXPLICIT);
            ngraph::replace_node_update_name(conv, new_conv);
            is_graph_modfied = true;
            continue;

        }
        // ========== 1D cases ===========
        else if (is_1D || (W_new ==  weights_shape[3])) { // input_shape[3])) {  // 1D in H direction case

            Output<Node> source = input;
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D2(H_new, H_pad_outer, H_pad_additional, strides[0], weights_shape[2], true, kernel_list, input_list);

            if (is_1D) {  // if 3D input tensor then reshape to 4D
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(source,
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),false);
                source = new_reshape->output(0);
            }

            if (!transpose_before) {  // convert to NHWC for GNA
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(source,
                    op::Constant::create(ngraph::element::i64, Shape{2}, {N * C, H * W})->output(0),false);
                auto new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0), 
                    op::Constant::create(element::Type_t::i64, Shape{2}, {1,0}));
                new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),false);
                source = new_reshape->output(0);
            }

            // Insert convolutions
            OutputVector upstream;
            upstream.push_back(source);
            OutputVector parts;
            for (size_t n = 0; n < input_list.size(); n++) {
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(source,
                    op::Constant::create(ngraph::element::i64, Shape{3}, {H, W, C})->output(0),false);
                size_t H_start = input_list[n][0];
                size_t H_stop = input_list[n][1] + 1;
                auto slice_start = op::Constant::create(ngraph::element::i64, Shape{3}, {H_start, 0ull, 0ull});
                auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{3}, {H_stop, W, C});
                auto slice_step = op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 1ull, 1ull});
                auto new_slice = std::make_shared<v8::Slice>(new_reshape->output(0), slice_start, slice_stop, slice_step);
                new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_slice->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, H_stop - H_start, W, C})->output(0),false);
                upstream[0] = new_reshape->output(0);
                if (fq_input) {
                    auto new_fq = CopyFQ(upstream[0], fq_input);
                    upstream[0] = new_fq->output(0);
                }
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //   
                size_t C_out = kernel_list[n].size() * weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;  // preserve input channels
                size_t K_h = kernel_list[n][0].size(); // include only kernel elements that would not be multiplied by zero padding
                size_t K_w = 1;  // 1D case
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = (weights_shape.size() == 3) ? weights_shape[1] * weights_shape[2] : weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = (weights_shape.size() == 3) ? weights_shape[2] : weights_shape[2] * weights_shape[3];
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + j) = 0;
                            } else {
                                *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + j) =
                                    *(weight_ptr + i_prev * i_step + j_prev * j_step + k_prev);
                            }
                        }
                    }
                }
                ov::Strides new_strides = {1, 1};
                ov::CoordinateDiff new_pads_begin = {0, (int64_t)W_pad_outer};
                ov::CoordinateDiff new_pads_end = {0, (int64_t)W_pad_outer};
                ov::Strides new_dilations = {1, 1};
                auto new_weights_node = InsertWeights(Shape{C_out, K_h, K_w, C_in}, new_weights, (bool)fq_input, weights_min, weights_max);
                auto new_conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(upstream[0],
                    new_weights_node->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::EXPLICIT);
                auto conv_shape = new_conv->get_output_shape(0);
                auto conv_N = conv_shape[0];
                auto conv_H = conv_shape[1];
                auto conv_W = conv_shape[2];
                auto conv_C = conv_shape[3];
                auto H_out = conv_C * conv_H / weights_shape[1];
                upstream[0] = new_conv->output(0);
                if (fq_after) {
                    auto new_fq = CopyFQ(upstream[0], fq_after);
                    upstream[0] = new_fq->output(0);
                }
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const = op::Constant::create(ngraph::element::f32,Shape{1ull, 1ull, 1ull, C_out}, new_bias);
                        auto new_add = std::make_shared<opset1::Add>(upstream[0], new_bias_const->output(0));
                        if (addfq_after) {
                            auto new_addfq = CopyFQ(new_add->output(0), addfq_after);
                            upstream[0] = new_addfq->output(0);
                        } else {
                            upstream[0] = new_add->output(0);
                        }
                    }
                }
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                if (actfq_after) {
                    auto new_actfq = CopyFQ(upstream[0], actfq_after);
                    upstream[0] = new_actfq->output(0);
                }
                ov::Shape old_shape = upstream[0].get_shape();
                ov::Shape new_shape = {N, weights_shape[1], H_out, 1ull};
                if (new_shape == old_shape) {
                    parts.push_back(upstream[0]);
                } else {
                    auto num_splits = conv_H;
                    if (num_splits > 1) {
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {conv_C * conv_H / weights_shape[1], weights_shape[1]})->output(0),false);
                        upstream[0] = new_reshape->output(0);
                    } else {
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {conv_C/weights_shape[1], weights_shape[1]})->output(0),false);
                        upstream[0] = new_reshape->output(0);
                    }
                    parts.push_back(upstream[0]);
                }
            }
            std::shared_ptr<ov::op::v1::Reshape> final_reshape = nullptr;
            auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 0);
            auto new_transpose = std::make_shared<op::Transpose>(new_concat->output(0), 
                op::Constant::create(element::Type_t::i64, Shape{2}, {1,0}));
            final_reshape = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, weights_shape[1], H_new - weights_shape[2] + 1, 1ull})->output(0),false);
            upstream[0] = final_reshape->output(0);
            if (is_1D) {  // if 3D original input tensor then reshape back to 3D
                final_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{3}, {N, weights_shape[1], H_new - weights_shape[2] + 1})->output(0),false);
            }
            if (reshape_after != nullptr) {
                ngraph::replace_node_update_name(reshape_after, final_reshape);
            } else if (actfq_after) {
                ngraph::replace_node_update_name(actfq_after, final_reshape);
            } else if (prelu_after) {
                ngraph::replace_node_update_name(prelu_after, final_reshape);
            } else if (relu_after) {
                ngraph::replace_node_update_name(relu_after, final_reshape);
            } else if (sigmoid_after) {
                ngraph::replace_node_update_name(sigmoid_after, final_reshape);
            } else if (tanh_after) {
                ngraph::replace_node_update_name(tanh_after, final_reshape);
            } else if (addfq_after) {
                ngraph::replace_node_update_name(addfq_after, final_reshape);
            } else if (add_after) {
                ngraph::replace_node_update_name(add_after, final_reshape);
            } else if (fq_after) {
                ngraph::replace_node_update_name(fq_after, final_reshape);
            } else {
                ngraph::replace_node_update_name(conv, final_reshape);
            }

            is_graph_modfied = true;

        } else if (H_new == input_shape[2]) {
            Output<Node> source = input;
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D2(W_new, W_pad_outer, W_pad_additional, strides[1], weights_shape[3], true, kernel_list, input_list);

            // Insert convolutions
            OutputVector parts;
            OutputVector upstream;
            upstream.push_back(source);
            for (size_t n = 0; n < input_list.size(); n++) {
                size_t W_start = input_list[n][0];
                size_t W_stop = input_list[n][1] + 1;
                auto slice_start = op::Constant::create(ngraph::element::i64, Shape{4}, {0ull, 0ull, 0ull, W_start});
                auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W_stop});
                auto slice_step = op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, 1ull});
                auto new_slice = std::make_shared<v8::Slice>(source, slice_start, slice_stop, slice_step);
                upstream[0] = new_slice->output(0);
                if (fq_input) {
                    auto new_fq = CopyFQ(upstream[0], fq_input);
                    upstream[0] = new_fq->output(0);
                }
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //   
                size_t C_out = kernel_list[n].size() * weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;  // preserve input channels
                size_t K_h = 1;   // 1D case
                size_t K_w = kernel_list[n][0].size(); // include only kernel elements that would not be multiplied by zero padding
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[3]; k++) {  // 1D in W
                                if ((j == 2) && (k == 1)) {
                                    printf("%e\n", *(weight_ptr + i * i_step + j * j_step + k));
                                }
                                //printf("%d %d %d: %f\n", (int)i, (int)j, (int)k, *(weight_ptr + i * i_step + j * j_step + k));
                            }
                        }
                    }
                    printf("bias\n");
                    auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    for (size_t j = 0; j < weights_shape[1]; j++) {
                        printf("%e\n", *(bias_ptr + j));
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                *(new_weight_ptr + i * C_in * K_w + j * K_w + m) = 0;
                                if (print_stuff) {
                                    //printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in * K_h + j * K_h + m));
                                }
                            } else {
                                *(new_weight_ptr + i * C_in * K_w + j * K_w + m) =
                                    *(weight_ptr + i_prev * i_step + j_prev * j_step + k_prev);
                                if (print_stuff) {
                                    //printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in * K_h + j * K_h + m));
                                }
                            }
                        }
                    }
                }
                auto new_weights_node = InsertWeights(Shape{C_out, C_in, K_h, K_w}, new_weights, (bool)fq_input, weights_min, weights_max);
                auto new_conv = std::make_shared<opset1::Convolution>(upstream[0],
                    new_weights_node->output(0),Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
                upstream[0] = new_conv->output(0);
                if (fq_after) {
                    auto new_fq = CopyFQ(upstream[0], fq_after);
                    upstream[0] = new_fq->output(0);
                }
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const = op::Constant::create(ngraph::element::f32,Shape{1ull, C_out, 1ull, 1ull}, new_bias);
                        auto new_add = std::make_shared<opset1::Add>(upstream[0], new_bias_const->output(0));
                        auto add_shape = new_add->get_output_shape(0);
                        if (addfq_after) {
                            auto new_addfq = CopyFQ(new_add->output(0), addfq_after);
                            upstream[0] = new_addfq->output(0);
                        } else {
                            upstream[0] = new_add->output(0);
                        }
                        auto W_out = add_shape[1] * add_shape[3] / weights_shape[1];
                        InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                        if (actfq_after) {
                            auto new_actfq = CopyFQ(upstream[0], actfq_after);
                            upstream[0] = new_actfq->output(0);
                        }
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{4}, {N, weights_shape[1], 1ull, W_out})->output(0),false);
                        parts.push_back(new_reshape->output(0));
                    }
                } else {
                    auto conv_shape = new_conv->get_output_shape(0);
                    auto W_out = conv_shape[1] * conv_shape[3] / weights_shape[1];
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                    if (actfq_after) {
                        auto new_actfq = CopyFQ(upstream[0], actfq_after);
                        upstream[0] = new_actfq->output(0);
                    }
                    auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, weights_shape[1], 1ull, W_out})->output(0),false);
                    parts.push_back(new_reshape->output(0));
                }
            }
            auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 2);
            if (reshape_after != nullptr) {
                ngraph::replace_node_update_name(reshape_after, new_concat);
            } else if (actfq_after) {
                ngraph::replace_node_update_name(actfq_after, new_concat);
            } else if (prelu_after) {
                ngraph::replace_node_update_name(prelu_after, new_concat);
            } else if (relu_after) {
                ngraph::replace_node_update_name(relu_after, new_concat);
            } else if (sigmoid_after) {
                ngraph::replace_node_update_name(sigmoid_after, new_concat);
            } else if (tanh_after) {
                ngraph::replace_node_update_name(tanh_after, new_concat);
            } else if (addfq_after) {
                ngraph::replace_node_update_name(addfq_after, new_concat);
            } else if (add_after) {
                ngraph::replace_node_update_name(add_after, new_concat);
            } else if (fq_after) {
                ngraph::replace_node_update_name(fq_after, new_concat);
            } else {
                ngraph::replace_node_update_name(conv, new_concat);
            }

            is_graph_modfied = true;

        }
        // ========== 2D stride (S,1) cases ===========
        else if (strides[1] == 1) {  // zero insertion in H direction only

            Output<Node> source = input;
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D2(H_new, H_pad_outer, H_pad_additional, strides[0], weights_shape[2], true, kernel_list, input_list);

            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(source,
                op::Constant::create(ngraph::element::i64, Shape{2}, {input_shape[1], input_shape[2] * input_shape[3]})->output(0),false);
            auto new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0), op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            auto new_source = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, input_shape[2], input_shape[3], input_shape[1]})->output(0),false);
            // Insert convolutions
            std::vector<OutputVector> stream;  // split transpose convolution kernel and compute in streams by partial kernel (possibly broken down by channel group)
            for (size_t n = 0; n < input_list.size(); n++) {
                OutputVector upstream;
                size_t H_start = input_list[n][0];
                size_t H_stop = input_list[n][1] + 1;
                std::vector<size_t> split_lengths;
                size_t split_index = 0;
                if (H_start > 0) {
                    split_lengths.push_back(H_start);
                    split_index++;
                }
                split_lengths.push_back(H_stop - H_start);
                if (H_stop < H) {
                    split_lengths.push_back(H - H_stop);
                }
                const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, Shape{split_lengths.size()}, split_lengths.data());
                auto new_split = std::make_shared<opset1::VariadicSplit>(new_source->output(0), ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), split_lengths_const);
                upstream.push_back(new_split->output(split_index));
                if (fq_input) {
                    auto new_fq = CopyFQ(upstream[0], fq_input);
                    upstream[0] = new_fq->output(0);
                }
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //   
                size_t C_out = kernel_list[n].size() * weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;  // preserve input channels
                size_t K_h = kernel_list[n][0].size(); // include only kernel elements that would not be multiplied by zero padding
                size_t K_w = weights_shape[3];
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                size_t k_step = weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[2]; k++) {
                                for (size_t n = 0; n < weights_shape[3]; n++) {
                                    printf("%d %d %d %d: %f\n", (int)i, (int)j, (int)k, (int)n, *(weight_ptr + i * i_step + j * j_step + k * k_step + n));
                                }
                            }
                        }
                    }
                    if (add_after) {
                        printf("bias\n");
                        auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            printf("%e\n", *(bias_ptr + j));
                        }
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                for (size_t n = 0; n < K_w; n++) {
                                    *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + n * C_in + j) = 0;
                                    if (print_stuff) {
                                        printf("%d %d %d %d: %f\n", (int)j, (int)m, (int)n, (int)i, *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + n * C_in + j));
                                    }
                                }
                            } else {
                                for (size_t n = 0; n < K_w; n++) {
                                    *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + n * C_in + j) =
                                        *(weight_ptr + i_prev * i_step + j_prev * j_step + k_prev * k_step + (K_w - n - 1));
                                    if (print_stuff) {
                                        printf("%d %d %d %d: %f\n", (int)j, (int)m, (int)n, (int)i, *(new_weight_ptr + i * K_h * K_w * C_in + m * K_w * C_in + n * C_in + j));
                                    }
                                }
                            }
                        }
                    }
                }
                ov::Strides new_strides = {1, 1};
                ov::CoordinateDiff new_pads_begin = {0, (int64_t)W_pad_outer};
                ov::CoordinateDiff new_pads_end = {0, (int64_t)W_pad_outer};
                ov::Strides new_dilations = {1, 1};
                auto new_weights_node = InsertWeights(Shape{C_out, K_h, K_w, C_in}, new_weights, (bool)fq_input, weights_min, weights_max);
                auto new_conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(upstream[0],
                    new_weights_node->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::EXPLICIT);
                auto conv_shape = new_conv->get_output_shape(0);
                auto conv_N = conv_shape[0];
                auto conv_H = conv_shape[1];
                auto conv_W = conv_shape[2];
                auto conv_C = conv_shape[3];
                auto H_out = conv_C * conv_H / weights_shape[1];
                auto W_out = conv_W;
                upstream[0] = new_conv->output(0);
                if (fq_after) {
                    auto new_fq = CopyFQ(upstream[0], fq_after);
                    upstream[0] = new_fq->output(0);
                }
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const = op::Constant::create(ngraph::element::f32,Shape{1ull, 1ull, 1ull, C_out}, new_bias);
                        auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                        if (addfq_after) {
                            auto new_addfq = CopyFQ(new_add->output(0), addfq_after);
                            upstream[0] = new_addfq->output(0);
                        } else {
                            upstream[0] = new_add->output(0);
                        }
                    }
                }
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                if (actfq_after) {
                    auto new_actfq = CopyFQ(upstream[0], actfq_after);
                    upstream[0] = new_actfq->output(0);
                }
                OutputVector tmp;
                stream.push_back(tmp);
                if (conv_C == weights_shape[1]) {

                    // to avoid big transposes later, split channels and process in groups of 8
                    auto num_splits = conv_C / 8;
                    if ((num_splits > 1) && (num_splits * 8 == conv_C)) {  // TO DO:  extend this for non-multiples-of-8
                        // note:  reshape to 1D to work around graph compiler bug
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, conv_N * conv_H * conv_W * conv_C})->output(0),false);
                        auto new_split = std::make_shared<ngraph::opset1::Split>(new_reshape->output(0),
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), num_splits);
                        new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{4}, {conv_N, conv_H, conv_W, conv_C / num_splits})->output(0),false);
                        stream[n].push_back(new_reshape->output(0));
                        for (size_t i = 1; i < num_splits; i++) {
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{4}, {conv_N, conv_H, conv_W, conv_C / num_splits})->output(0),false);
                            stream[n].push_back(new_reshape->output(0));
                        }
                    } else {
                        stream[n].push_back(upstream[0]);
                    }

                } else {

                    size_t pad0 = 0;
                    auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {conv_N*conv_H*conv_W, conv_C})->output(0),false);
                    upstream[0] = new_reshape->output(0);
                    auto new_shape = new_reshape->get_output_shape(0);
                    pad0 = ((new_shape[0] % 8) == 0) ? 0 : 8 - (new_shape[0] % 8);
                    if ((new_shape[0] <= 8) && ((new_shape[1] % 8) == 0)) {
                        pad0 = 0;  // skip padding if transpose is legal in second dimension
                    }
                    if (pad0 > 0) {
                        std::vector<float> padding(pad0 * new_shape[1], 0.0f);
                        auto padding_const = op::Constant::create(ngraph::element::f32, {pad0, new_shape[1]}, padding);
                        upstream.push_back(padding_const->output(0));
                        auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 0);
                        upstream.resize(1);
                        upstream[0] = new_concat->output(0);
                    }
                    new_transpose = std::make_shared<op::Transpose>(upstream[0],
                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                    stream[n].push_back(new_transpose->output(0));

                    // note:  reshape to 1D to work around graph compiler bug
                    new_shape = stream[n][0].get_shape();
                    new_reshape = std::make_shared<ngraph::opset1::Reshape>(stream[n][0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0] * new_shape[1]})->output(0),false);
                    auto new_split = std::make_shared<ngraph::opset1::Split>(new_reshape->output(0),
                        ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), conv_C);
                    std::vector<OutputVector> substream(weights_shape[1]);
                    if (conv_H > 1) {
                        for (size_t i = 0; i < conv_C; i++) {
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {new_shape[0] / conv_C, new_shape[1]})->output(0),false);
                            upstream[0] = new_reshape->output(0);
                            if (pad0 > 0) {
                                std::vector<size_t> split_lengths;
                                split_lengths.push_back(conv_H * conv_W);
                                if (conv_H * conv_W < new_shape[1]) {
                                    split_lengths.push_back(new_shape[1] - conv_H * conv_W);
                                }
                                const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, Shape{split_lengths.size()}, split_lengths.data());
                                auto new_split = std::make_shared<opset1::VariadicSplit>(upstream[0], ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), split_lengths_const);
                                upstream[0] = new_split->output(0);
                            }
                            new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                                op::Constant::create(ngraph::element::i64, Shape{2}, {conv_H, conv_W})->output(0),false);
                            upstream[0] = new_reshape->output(0);
                            auto new_shape = new_reshape->get_output_shape(0);
                            pad0 = ((new_shape[0] % 8) == 0) ? 0 : 8 - (new_shape[0] % 8);
                            if (pad0 > 0) {
                                std::vector<float> padding(pad0 * new_shape[1], 0.0f);
                                auto padding_const = op::Constant::create(ngraph::element::f32, {pad0, new_shape[1]}, padding);
                                upstream.push_back(padding_const->output(0));
                                auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 0);
                                upstream.resize(1);
                                upstream[0] = new_concat->output(0);
                            }
                            auto new_transpose = std::make_shared<op::Transpose>(upstream[0],
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto k = i % weights_shape[1];
                            substream[k].push_back(new_transpose->output(0));
                        }
                        upstream.resize(0);
                        for (size_t i = 0; i < weights_shape[1]; i++) {
                            auto new_concat = std::make_shared<ngraph::opset1::Concat>(substream[i], 0);
                            auto new_transpose = std::make_shared<op::Transpose>(new_concat->output(0),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            auto new_shape = new_transpose->get_output_shape(0);
                            OutputVector prev;
                            prev.push_back(new_transpose->output(0));
                            if (pad0 > 0) {
                                std::vector<size_t> split_lengths;
                                split_lengths.push_back(new_shape[0] - pad0);
                                split_lengths.push_back(pad0);
                                const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, Shape{split_lengths.size()}, split_lengths.data());
                                auto new_split = std::make_shared<opset1::VariadicSplit>(prev[0], ngraph::opset1::Constant::create(element::i64, Shape{}, {0}), split_lengths_const);
                                prev[0] = new_split->output(0);
                            }
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(prev[0],
                                op::Constant::create(ngraph::element::i64,Shape{2},{conv_C * conv_H / weights_shape[1], conv_W})->output(0),false);
                            upstream.push_back(new_reshape->output(0));
                        }
                    } else {
                        for (size_t i = 0; i < conv_C; i++) {
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{2}, {new_shape[0] / conv_C, new_shape[1]})->output(0),false);
                            upstream[0] = new_reshape->output(0);
                            auto new_shape = new_reshape->get_output_shape(0);
                            if (pad0 > 0) {
                                std::vector<size_t> split_lengths;
                                split_lengths.push_back(new_shape[1] - pad0);
                                split_lengths.push_back(pad0);
                                const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64, Shape{split_lengths.size()}, split_lengths.data());
                                auto new_split = std::make_shared<opset1::VariadicSplit>(new_reshape->output(0), ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), split_lengths_const);
                                upstream[0] = new_split->output(0);
                            }
                            auto k = i % weights_shape[1];
                            substream[k].push_back(upstream[0]);
                        }
                        upstream.resize(0);
                        for (size_t i = 0; i < weights_shape[1]; i++) {
                            for (size_t j = 0; j < substream[i].size(); j++) {
                                upstream.push_back(substream[i][j]);
                            }
                        }
                    }
                    auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 0);

                    // to avoid big transposes later, split channels and process in groups of 8
                    auto num_splits = weights_shape[1] / 8;
                    if ((num_splits > 1) && (num_splits * 8 == weights_shape[1])) {  // TO DO:  extend this for non-multiples-of-8
                        // note:  reshape to 1D to work around graph compiler bug
                        auto new_shape = stream[n][0].get_shape();
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_concat->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, conv_N * weights_shape[1] * H_out * W_out})->output(0),false);
                        auto new_split = std::make_shared<ngraph::opset1::Split>(new_reshape->output(0),
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), num_splits);
                        new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{4}, {conv_N, weights_shape[1] / num_splits, H_out, W_out})->output(0),false);
                        stream[n][0] = new_reshape->output(0);
                        for (size_t i = 1; i < num_splits; i++) {
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(i),
                                op::Constant::create(ngraph::element::i64, Shape{4}, {conv_N, weights_shape[1] / num_splits, H_out, W_out})->output(0),false);
                            stream[n].push_back(new_reshape->output(0));
                        }
                    } else {
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_concat->output(0),
                            op::Constant::create(ngraph::element::i64, Shape{4}, {conv_N, weights_shape[1], H_out, W_out})->output(0),false);
                        stream[n][0] = new_reshape->output(0);
                    }
                }
            }
            // recombine channels if necessary

            OutputVector subchannels;
            std::shared_ptr<ov::op::v0::Concat> final_concat;
            if (stream[0].size() == 1) {
                OutputVector tmp;
                for (size_t i = 0; i < stream.size(); i++) {
                    tmp.push_back(stream[i][0]);
                }
                final_concat = std::make_shared<ngraph::opset1::Concat>(tmp, 2);
            } else {
                for (size_t j = 0; j < stream[0].size(); j++) {
                    OutputVector tmp;
                    for (size_t i = 0; i < stream.size(); i++) {
                        tmp.push_back(stream[i][j]);
                    }
                    auto new_concat = std::make_shared<ngraph::opset1::Concat>(tmp, 2);
                    subchannels.push_back(new_concat->output(0));
                }
                final_concat = std::make_shared<ngraph::opset1::Concat>(subchannels, 1);
            }
            auto concat_shape = final_concat->get_output_shape(0);
            new_reshape = std::make_shared<ngraph::opset1::Reshape>(final_concat->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {concat_shape[1], concat_shape[0]*concat_shape[2]*concat_shape[3]})->output(0),false);
            new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0),
                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {concat_shape[0], concat_shape[2], concat_shape[3], concat_shape[1]})->output(0),false);
            new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            if (reshape_after != nullptr) {
                ngraph::replace_node_update_name(reshape_after, new_transpose);
            } else if (actfq_after) {
                ngraph::replace_node_update_name(actfq_after, new_transpose);
            } else if (prelu_after) {
                ngraph::replace_node_update_name(prelu_after, new_transpose);
            } else if (relu_after) {
                ngraph::replace_node_update_name(relu_after, new_transpose);
            } else if (sigmoid_after) {
                ngraph::replace_node_update_name(sigmoid_after, new_transpose);
            } else if (tanh_after) {
                ngraph::replace_node_update_name(tanh_after, new_transpose);
            } else if (addfq_after) {
                ngraph::replace_node_update_name(addfq_after, new_transpose);
            } else if (add_after) {
                ngraph::replace_node_update_name(add_after, new_transpose);
            } else if (fq_after) {
                ngraph::replace_node_update_name(fq_after, new_transpose);
            } else {
                ngraph::replace_node_update_name(conv, new_transpose);
            }

            is_graph_modfied = true;
        }
        // ========== 2D stride (1,S) cases ===========
        else if (strides[0] == 1) {  // zero insertion in W direction only

            Output<Node> source = input;
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D2(W_new, W_pad_outer, W_pad_additional, strides[1], weights_shape[3], true, kernel_list, input_list);

            // Insert convolutions
            OutputVector parts;
            for (size_t n = 0; n < input_list.size(); n++) {
                OutputVector upstream;
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(source,
                    op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),false);
                upstream.push_back(new_reshape->output(0));
                if (fq_input) {
                    auto new_fq = CopyFQ(new_reshape->output(0), fq_input);
                    upstream[0] = new_fq->output(0);
                }
                size_t W_start = input_list[n][0];
                size_t W_stop = input_list[n][1] + 1;
                auto slice_start = op::Constant::create(ngraph::element::i64, Shape{3}, {0ull, 0ull, W_start});
                auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W_stop});
                auto slice_step = op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 1ull, 1ull});
                auto new_slice = std::make_shared<v8::Slice>(new_reshape->output(0), slice_start, slice_stop, slice_step);
                new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_slice->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W_stop -  W_start})->output(0),false);
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //   
                size_t C_out = kernel_list[n].size() * weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;  // preserve input channels
                size_t K_h = weights_shape[2];
                size_t K_w = kernel_list[n][0].size(); // include only kernel elements that would not be multiplied by zero padding
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                size_t k_step = weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[2]; k++) {
                                for (size_t n = 0; n < weights_shape[3]; n++) {
                                    printf("%d %d %d %d: %f\n", (int)i, (int)j, (int)k, (int)n, *(weight_ptr + i * i_step + j * j_step + k * k_step + n));
                                }
                            }
                        }
                    }
                    if (add_after) {
                        printf("bias\n");
                        auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            printf("%e\n", *(bias_ptr + j));
                        }
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                for (size_t n = 0; n < K_h; n++) {
                                    *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + n * K_w + m) = 0;
                                    if (print_stuff) {
                                        printf("%d %d %d %d: %f\n", (int)i, (int)j, (int)n, (int)m, *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + n * K_w + m));
                                    }
                                }
                            } else {
                                for (size_t n = 0; n < K_h; n++) {
                                    *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + n * K_w + m) =
                                        *(weight_ptr + i_prev * i_step + j_prev * j_step + (K_h - n - 1) * k_step + k_prev);
                                    if (print_stuff) {
                                        printf("%d %d %d %d: %f\n", (int)i, (int)j, (int)n, (int)m, *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + n * K_w + m));
                                    }
                                }
                            }
                        }
                    }
                }
                ov::Strides new_strides = {1, 1};
                ov::CoordinateDiff new_pads_begin = {(int64_t)H_pad_outer, 0};
                ov::CoordinateDiff new_pads_end = {(int64_t)H_pad_outer, 0};
                ov::Strides new_dilations = {1, 1};
                auto new_weights_node = InsertWeights(Shape{C_out, C_in, K_h, K_w}, new_weights, (bool)fq_input, weights_min, weights_max);
                auto new_conv = std::make_shared<opset1::Convolution>(upstream[0],
                    new_weights_node->output(0),new_strides,new_pads_begin,new_pads_end,new_dilations,PadType::EXPLICIT);
                auto conv_shape = new_conv->get_output_shape(0);
                auto H_out = conv_shape[1] * conv_shape[2] / weights_shape[1];
                auto W_out = conv_shape[3];
                upstream[0] = new_conv->output(0);
                if (fq_after) {
                    auto new_fq = CopyFQ(upstream[0], fq_after);
                    upstream[0] = new_fq->output(0);
                }
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const = std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const = op::Constant::create(ngraph::element::f32,Shape{1ull, C_out, 1ull, 1ull}, new_bias);
                        auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                        if (addfq_after) {
                            auto new_addfq = CopyFQ(new_add->output(0), addfq_after);
                            upstream[0] = new_addfq->output(0);
                        } else {
                            upstream[0] = new_add->output(0);
                        }
                    }
                }
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                if (actfq_after) {
                    auto new_actfq = CopyFQ(upstream[0], actfq_after);
                    upstream[0] = new_actfq->output(0);
                }
                if (conv_shape[1] == weights_shape[1]) {
                    auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, weights_shape[1], H_out, W_out})->output(0),false);
                    parts.push_back(new_reshape->output(0));
                } else {
                    if (conv_shape[3] == 1) {
                        auto num_splits = (conv_shape[0] * conv_shape[1]) / weights_shape[1];
                        if (num_splits > 1) {
                            auto new_split = std::make_shared<ngraph::opset1::Split>(upstream[0],
                                ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), num_splits);
                            for (uint32_t i = 0; i < num_splits; i++) {
                                parts.push_back(new_split->output(i));
                            }
                        } else {
                            parts.push_back(upstream[0]);
                        }
                    } else {
                        auto num_channels = weights_shape[1];
                        auto num_outputs_per_chan = conv_shape[1] / weights_shape[1];
                        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {num_outputs_per_chan, num_channels*conv_shape[3]})->output(0),false);
                        auto new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0), 
                            op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                        upstream[0] = new_transpose->output(0);
                        if (num_channels > 1) {
                            // note:  reshape to 1D to work around graph compiler bug
                            auto new_shape = upstream[0].get_shape();
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0] * new_shape[1]})->output(0),false);
                            auto new_split = std::make_shared<ngraph::opset1::Split>(new_reshape->output(0),
                                ngraph::opset1::Constant::create(element::i64, Shape{}, {1}), num_channels);
                            OutputVector subparts;
                            for (uint32_t i = 0; i < num_channels; i++) {
                                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_split->output(i),
                                    op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, num_outputs_per_chan*conv_shape[3]})->output(0),false);
                                subparts.push_back(new_reshape->output(0));
                            }                                
                            auto new_concat = std::make_shared<ngraph::opset1::Concat>(subparts, 0);
                            new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_concat->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, num_channels, 1ull, num_outputs_per_chan*conv_shape[3]})->output(0),false);
                            parts.push_back(new_reshape->output(0));
                        } else {
                            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream[0],
                                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, num_channels, 1ull, num_outputs_per_chan*conv_shape[3]})->output(0),false);
                            parts.push_back(new_reshape->output(0));
                        }
                    }
                }
            }
            auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 3);
            if (reshape_after != nullptr) {
                ngraph::replace_node_update_name(reshape_after, new_concat);
            } else if (actfq_after) {
                ngraph::replace_node_update_name(actfq_after, new_concat);
            } else if (prelu_after) {
                ngraph::replace_node_update_name(prelu_after, new_concat);
            } else if (relu_after) {
                ngraph::replace_node_update_name(relu_after, new_concat);
            } else if (sigmoid_after) {
                ngraph::replace_node_update_name(sigmoid_after, new_concat);
            } else if (tanh_after) {
                ngraph::replace_node_update_name(tanh_after, new_concat);
            } else if (addfq_after) {
                ngraph::replace_node_update_name(addfq_after, new_concat);
            } else if (add_after) {
                ngraph::replace_node_update_name(add_after, new_concat);
            } else if (fq_after) {
                ngraph::replace_node_update_name(fq_after, new_concat);
            } else {
                ngraph::replace_node_update_name(conv, new_concat);
            }

            is_graph_modfied = true;

        } else {
            continue;
        }
    }
    return is_graph_modfied;
}
