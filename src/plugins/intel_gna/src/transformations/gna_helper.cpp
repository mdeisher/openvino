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

#include "gna_helper.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

size_t GetChannels(Output<Node>& parent) {
    size_t C = 0;
    auto input_shape = parent.get_shape();
    if (input_shape.size() > 2) {
        C = input_shape[input_shape.size() - 3];
    }
    return (C);
}

std::shared_ptr<ov::Node> AdlChannelPadTensor(Output<Node> parent) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto input_shape = parent.get_shape();
    if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
        size_t N = input_shape[input_shape.size() - 4];
        size_t C = input_shape[input_shape.size() - 3];
        size_t H = input_shape[input_shape.size() - 2];
        size_t W = input_shape[input_shape.size() - 1];
        // if not a multiple of 8 must pad
        if ((C < 8) && (((C * H * W) % 32) == 0)) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                parent,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),
                false);
            std::vector<float> padding((8 - C) * H * W, 0.0f);
            auto padding_const = op::Constant::create(ngraph::element::f32, Shape{8 - C, H * W}, padding);
            OutputVector chunks;
            chunks.push_back(reshape->output(0));
            chunks.push_back(padding_const->output(0));
            auto concat = std::make_shared<op::Concat>(chunks, 0);
            result = std::make_shared<ngraph::opset1::Reshape>(
                concat->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, 8ull, H, W})->output(0),
                false);
        } else if (C < 8) {  // unaligned copy ==> must work-around GNA plugin memory explosion
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                parent,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, C * H * W})->output(0),
                false);
            // perform smallest amount of padding needed to bring size to multiple of 32
            size_t padding_size = ((C * H * W) / 32 + 1) * 32 - (C * H * W);
            size_t padded_size = C * H * W + padding_size;
            std::vector<float> padding(padding_size, 0.0f);
            auto padding_const = op::Constant::create(ngraph::element::f32, Shape{1, padding_size}, padding);
            OutputVector chunks;
            chunks.push_back(reshape->output(0));
            chunks.push_back(padding_const->output(0));
            auto concat = std::make_shared<op::Concat>(chunks, 1);
            // perform rest of padding using aligned copy
            if (((8 * H * W - padded_size) % 32) == 0) {
                padding_size = 8 * H * W - padded_size;
            } else {
                padding_size = ((8 * H * W - padded_size) / 32 + 1) * 32 - (8 * H * W);
            }
            padding.resize(padding_size, 0.0f);
            padding_const = op::Constant::create(ngraph::element::f32, Shape{1, padding_size}, padding);
            padded_size += padding_size;
            chunks.clear();
            chunks.push_back(concat->output(0));
            chunks.push_back(padding_const->output(0));
            concat = std::make_shared<op::Concat>(chunks, 1);
            chunks[0] = concat->output(0);
            if (padded_size > 8 * H * W) {
                auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                    chunks[0],
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{2},
                                                     {0ull, 0ull}),  // begin slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{2},
                                                     {0ull, 8 * H * W}),  // end slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1ull, 1ull}),  // strides
                    std::vector<int64_t>{1, 0},  // begin mask
                    std::vector<int64_t>{1, 0},  // end mask
                    std::vector<int64_t>{0, 0},
                    std::vector<int64_t>{0, 0},
                    std::vector<int64_t>{0, 0});
                chunks[0] = slice->output(0);
            }
            result = std::make_shared<ngraph::opset1::Reshape>(
                chunks[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, 8ull, H, W})->output(0),
                false);
        }
    }

    return (result);
}

std::shared_ptr<ov::Node> AdlChannelPadKernel(Output<Node>& weights_const_output, size_t C) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto weights_const =
        std::dynamic_pointer_cast<ngraph::opset1::Constant>(weights_const_output.get_node_shared_ptr());
    auto kernel_shape = weights_const_output.get_shape();
    const float* weights = weights_const->get_data_ptr<float>();
    if (kernel_shape[1] < C) {
        std::vector<float> new_weights(kernel_shape[0] * C * kernel_shape[2] * kernel_shape[3], 0.0f);
        float* new_weight_ptr = new_weights.data();
        const float* weight_ptr = weights;
        for (size_t i = 0; i < kernel_shape[0]; i++) {
            for (size_t j = 0; j < C; j++) {
                for (size_t k = 0; k < kernel_shape[2]; k++) {
                    for (size_t p = 0; p < kernel_shape[3]; p++) {
                        if (j < kernel_shape[1]) {
                            *new_weight_ptr++ = *weight_ptr++;
                        } else {
                            *new_weight_ptr++ = 0.0f;
                        }
                    }
                }
            }
        }
        result = op::Constant::create(ngraph::element::f32,
                                      Shape{kernel_shape[0], C, kernel_shape[2], kernel_shape[3]},
                                      new_weights);
    }

    return (result);
}

std::shared_ptr<ov::Node> NchwToNhwc(Output<Node> parent) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto input_shape = parent.get_shape();
    if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
        size_t N = input_shape[input_shape.size() - 4];
        size_t C = input_shape[input_shape.size() - 3];
        size_t H = input_shape[input_shape.size() - 2];
        size_t W = input_shape[input_shape.size() - 1];
        if (C <= 8) {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                parent,
                op::Constant::create(ngraph::element::i64, Shape{2}, {C, H * W})->output(0),
                false);
            std::shared_ptr<ngraph::Node> transpose_const;
            transpose_const = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
            auto new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0), transpose_const);
            result = std::make_shared<ngraph::opset1::Reshape>(
                new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, H, W, C})->output(0),
                false);
        }
    }

    return (result);
}

// inserts given NCHW Convolution wrapped in Transposes into NHWC graph
std::shared_ptr<ov::Node> AdlInsertConvolutionAddRelu(Output<Node> parent,
                                                      std::shared_ptr<opset1::Convolution> conv,
                                                      std::shared_ptr<opset1::Add> add) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto input_shape = parent.get_shape();
    auto output_shape = conv->output(0).get_shape();
    if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
        size_t N = input_shape[input_shape.size() - 4];
        size_t H = input_shape[input_shape.size() - 3];
        size_t W = input_shape[input_shape.size() - 2];
        size_t C = input_shape[input_shape.size() - 1];
        size_t Co = output_shape[output_shape.size() - 3];
        if ((C <= 8) && (W >= 16)) {
            auto kernel_shape = conv->input_value(1).get_shape();
            auto stride = conv->get_strides();
            auto dilation = conv->get_dilations();
            auto pad_begin = conv->get_pads_begin();
            auto pad_end = conv->get_pads_end();
            auto pad_type = conv->get_auto_pad();
            auto weights_const =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
            const float* weights = weights_const->get_data_ptr<float>();

            auto new_transpose =
                std::make_shared<op::Transpose>(parent,
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto new_weights_const_output = AdlChannelPadKernel(weights_const->output(0), C)->output(0);
            auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                  new_weights_const_output,
                                                                  stride,
                                                                  pad_begin,
                                                                  pad_end,
                                                                  dilation,
                                                                  pad_type);
            auto add_const = std::dynamic_pointer_cast<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
            if (add_const != nullptr) {
                auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), add_const->output(0));
                auto new_relu = std::make_shared<opset1::Relu>(new_add->output(0));
                result =
                    std::make_shared<op::Transpose>(new_relu->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            }
        }
    }

    return (result);
}

// splits given NCHW Convolution wrapped in Transposes and inserts into NHWC graph
std::vector<std::shared_ptr<ov::Node>> AdlInsertSplitConvolutionAddRelu(Output<Node> parent,
                                                                        std::shared_ptr<opset1::Convolution> conv,
                                                                        std::shared_ptr<opset1::Add> add) {
    std::vector<std::shared_ptr<ov::Node>> result;

    auto input_shape = parent.get_shape();
    auto output_shape = conv->output(0).get_shape();
    if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
        size_t N = input_shape[input_shape.size() - 4];
        size_t H = input_shape[input_shape.size() - 3];
        size_t W = input_shape[input_shape.size() - 2];
        size_t C = input_shape[input_shape.size() - 1];
        size_t Co = output_shape[output_shape.size() - 3];
        // must split NHWC channel-wise
        if ((C <= 8) && (W >= 16)) {
            auto kernel_shape = conv->input_value(1).get_shape();
            auto stride = conv->get_strides();
            auto dilation = conv->get_dilations();
            auto pad_begin = conv->get_pads_begin();
            auto pad_end = conv->get_pads_end();
            auto pad_type = conv->get_auto_pad();
            auto weights_const =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
            const float* weights = weights_const->get_data_ptr<float>();
            std::vector<size_t> co_split_lengths;
            size_t channels_left = Co;
            while (channels_left > 0) {
                size_t Co_part = (channels_left > ADL_GNA_MAX_CHANNELS) ? ADL_GNA_MAX_CHANNELS : channels_left;
                channels_left -= Co_part;
                co_split_lengths.push_back(Co_part);
            }
            size_t start_kernel_index = 0;
            for (size_t p = 0; p < co_split_lengths.size(); p++) {
                auto new_transpose =
                    std::make_shared<op::Transpose>(parent,
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(
                    co_split_lengths[p] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3],
                    0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
                size_t j_step = kernel_shape[2] * kernel_shape[3];
                size_t k_step = kernel_shape[3];
                for (size_t i = start_kernel_index; i < start_kernel_index + co_split_lengths[p]; i++) {
                    for (size_t j = 0; j < kernel_shape[1]; j++) {
                        for (size_t k = 0; k < kernel_shape[2]; k++) {
                            for (size_t m = 0; m < kernel_shape[3]; m++) {
                                if (j < C) {
                                    *new_weight_ptr++ = *(weight_ptr + i * i_step + j * j_step + k * k_step + m);
                                }
                            }
                        }
                    }
                }
                auto new_weights_const =
                    op::Constant::create(ngraph::element::f32,
                                         Shape{co_split_lengths[p], kernel_shape[1], kernel_shape[2], kernel_shape[3]},
                                         new_weights);
                auto new_weights_const_output = AdlChannelPadKernel(new_weights_const->output(0), C)->output(0);
                auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                      new_weights_const_output,
                                                                      stride,
                                                                      pad_begin,
                                                                      pad_end,
                                                                      dilation,
                                                                      pad_type);
                auto add_const = std::dynamic_pointer_cast<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
                if (add_const != nullptr) {
                    const float* bias_ptr = add_const->get_data_ptr<float>();
                    std::vector<float> new_bias(co_split_lengths[p], 0.0f);
                    for (size_t i = start_kernel_index; i < start_kernel_index + co_split_lengths[p]; i++) {
                        new_bias[i - start_kernel_index] = bias_ptr[i];
                    }
                    auto new_bias_const =
                        op::Constant::create(ngraph::element::f32, Shape{1, co_split_lengths[p], 1, 1}, new_bias);
                    auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                    auto new_relu = std::make_shared<opset1::Relu>(new_add->output(0));
                    result.push_back(std::make_shared<op::Transpose>(
                        new_relu->output(0),
                        op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1})));
                }
                start_kernel_index += co_split_lengths[p];
            }
        }
    }

    return (result);
}

std::shared_ptr<ov::Node> AdlInsertConvolutionAddReluHpadCsplit(Output<Node> parent,
                                                                std::shared_ptr<opset1::Convolution> conv,
                                                                std::shared_ptr<opset1::Add> add) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto input_shape = parent.get_shape();
    auto output_shape = conv->output(0).get_shape();
    if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
        size_t N = input_shape[input_shape.size() - 4];
        size_t H = input_shape[input_shape.size() - 3];
        size_t W = input_shape[input_shape.size() - 2];
        size_t C = input_shape[input_shape.size() - 1];
        size_t H_new = H;
        OutputVector upstream(1);
        upstream[0] = parent;

        // pad to minimum height for GNA3 on Alder Lake if needed
        if (H < 16) {
            std::vector<float> padding(N * (16 - H) * W * C, 0.0f);
            auto padding_const = op::Constant::create(ngraph::element::f32, Shape{N, (16 - H), W, C}, padding);
            OutputVector chunks;
            chunks.push_back(upstream[0]);
            chunks.push_back(padding_const->output(0));
            auto concat = std::make_shared<op::Concat>(chunks, 1);
            upstream[0] = concat->output(0);
            H_new = 16;
        }
        auto kernel_shape = conv->input_value(1).get_shape();
        auto stride = conv->get_strides();
        auto dilation = conv->get_dilations();
        auto pad_begin = conv->get_pads_begin();
        auto pad_end = conv->get_pads_end();
        auto pad_type = conv->get_auto_pad();
        auto weights_const =
            std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
        const float* weights = weights_const->get_data_ptr<float>();
        std::shared_ptr<ov::op::v1::Transpose> new_transpose;

        // must split NHWC channel-wise
        std::vector<size_t> c_split_lengths;
        size_t channels_left = C;
        while (channels_left > 0) {
            size_t C_part = (channels_left > 120) ? 120 : channels_left;
            channels_left -= C_part;
            c_split_lengths.push_back(C_part);
        }
        if (c_split_lengths.size() > 1) {
            std::vector<size_t> hw_split_lengths;
            size_t HW_left = H_new * W;
            while (HW_left > 0) {
                size_t HW_part = (HW_left > 8) ? 8 : HW_left;
                HW_left -= HW_part;
                hw_split_lengths.push_back(HW_part);
            }
            std::shared_ptr<opset1::VariadicSplit> hw_split;
            if (hw_split_lengths.size() > 1) {
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{2}, {H_new * W, C})->output(0),
                    false);
                const auto axis_const = ngraph::opset1::Constant::create(element::i64, Shape{}, {0});
                const auto hw_split_lengths_const = ngraph::opset1::Constant::create(element::i64,
                                                                                     Shape{hw_split_lengths.size()},
                                                                                     hw_split_lengths.data());
                hw_split =
                    std::make_shared<opset1::VariadicSplit>(new_reshape->output(0), axis_const, hw_split_lengths_const);
            }
            std::vector<OutputVector> chunks(c_split_lengths.size());
            for (size_t i = 0; i < hw_split_lengths.size(); i++) {
                if (hw_split_lengths.size() == 1) {
                    new_transpose =
                        std::make_shared<op::Transpose>(upstream[0],
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                } else {
                    new_transpose =
                        std::make_shared<op::Transpose>(hw_split->output(i),
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                }
                std::shared_ptr<opset1::VariadicSplit> c_split;
                const auto axis_const = ngraph::opset1::Constant::create(element::i64, Shape{}, {0});
                const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64,
                                                                                  Shape{c_split_lengths.size()},
                                                                                  c_split_lengths.data());
                c_split =
                    std::make_shared<opset1::VariadicSplit>(new_transpose->output(0), axis_const, split_lengths_const);
                for (size_t j = 0; j < c_split_lengths.size(); j++) {
                    new_transpose =
                        std::make_shared<op::Transpose>(c_split->output(j),
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                    chunks[j].push_back(new_transpose->output(0));
                }
            }
            upstream.clear();
            for (size_t j = 0; j < c_split_lengths.size(); j++) {
                auto concat = std::make_shared<op::Concat>(chunks[j], 0);
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    concat->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, H_new, W, c_split_lengths[j]})
                        ->output(0),
                    false);
                upstream.push_back(new_reshape->output(0));
            }
        }
        // perform separate convolutions
        size_t start_channel = 0;
        for (size_t p = 0; p < upstream.size(); p++) {
            std::vector<float> new_weights(kernel_shape[0] * c_split_lengths[p] * kernel_shape[2] * kernel_shape[3],
                                           0.0f);
            float* new_weight_ptr = new_weights.data();
            const float* weight_ptr = weights;
            size_t i_step = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
            size_t j_step = kernel_shape[2] * kernel_shape[3];
            size_t k_step = kernel_shape[3];
            for (size_t i = 0; i < kernel_shape[0]; i++) {
                for (size_t j = start_channel; j < start_channel + c_split_lengths[p]; j++) {
                    for (size_t k = 0; k < kernel_shape[2]; k++) {
                        for (size_t m = 0; m < kernel_shape[3]; m++) {
                            if (j < C) {
                                *new_weight_ptr++ = *(weight_ptr + i * i_step + j * j_step + k * k_step + m);
                            }
                        }
                    }
                }
            }
            auto new_weights_const =
                op::Constant::create(ngraph::element::f32,
                                     Shape{kernel_shape[0], c_split_lengths[p], kernel_shape[2], kernel_shape[3]},
                                     new_weights);
            new_transpose =
                std::make_shared<op::Transpose>(upstream[p],
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                  new_weights_const->output(0),
                                                                  stride,
                                                                  pad_begin,
                                                                  pad_end,
                                                                  dilation,
                                                                  pad_type);
            if (p == 0) {
                auto add_const = std::dynamic_pointer_cast<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
                if (add_const == nullptr) {
                    return (result);  // bad graph
                }
                auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), add_const->output(0));
                upstream[0] = new_add->output(0);
            } else {
                upstream[p] = new_conv->output(0);
            }
            // if convolution is not split then insert bias and activation now
            if (c_split_lengths.size() == 1) {
                auto new_relu = std::make_shared<opset1::Relu>(upstream[0]);
                upstream[p] = new_relu->output(0);
            }
            new_transpose =
                std::make_shared<op::Transpose>(upstream[p],
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            upstream[p] = new_transpose->output(0);
            // remove padding if necessary
            if (H < 16) {
                auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                    upstream[p],
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {0ull, 0ull, 0ull, 0ull}),  // begin slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {0ull, H / stride[0], 0ull, 0ull}),  // end slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {1ull, 1ull, 1ull, 1ull}),  // strides
                    std::vector<int64_t>{1, 0, 1, 1},                            // begin mask
                    std::vector<int64_t>{1, 0, 1, 1},                            // end mask
                    std::vector<int64_t>{0, 0, 0, 0},
                    std::vector<int64_t>{0, 0, 0, 0},
                    std::vector<int64_t>{0, 0, 0, 0});
                upstream[p] = slice->output(0);
            }
            start_channel += c_split_lengths[p];
        }
        if (upstream.size() > 1) {
            OutputVector prev_layer;
            prev_layer.push_back(upstream[0]);
            for (size_t i = 1; i < upstream.size(); i++) {
                auto new_add = std::make_shared<opset1::Add>(prev_layer[0], upstream[i]);
                prev_layer[0] = new_add->output(0);
            }
            auto new_relu = std::make_shared<opset1::Relu>(prev_layer[0]);
            upstream.clear();
            upstream.push_back(new_relu->output(0));
        }
        result = upstream[0].get_node_shared_ptr();
    }

    return (result);
}

std::shared_ptr<ov::Node> AdlInsertSplitConvolutionAddReluHpadCsplit(std::vector<std::shared_ptr<ov::Node>> parent,
                                                                     std::shared_ptr<opset1::Convolution> conv,
                                                                     std::shared_ptr<opset1::Add> add) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto output_shape = conv->output(0).get_shape();
    size_t start_channel = 0;

    for (size_t p = 0; p < parent.size(); p++) {
        auto input_shape = parent[p]->output(0).get_shape();
        if ((input_shape.size() == 4) && (input_shape[0] == 1)) {
            size_t N = input_shape[input_shape.size() - 4];
            size_t H = input_shape[input_shape.size() - 3];
            size_t W = input_shape[input_shape.size() - 2];
            size_t C = input_shape[input_shape.size() - 1];
            size_t H_new = H;
            OutputVector upstream(1);
            upstream[0] = parent[p]->output(0);

            // pad to minimum height for GNA3 on Alder Lake if needed
            if (H < 16) {
                std::vector<float> padding(N * (16 - H) * W * C, 0.0f);
                auto padding_const = op::Constant::create(ngraph::element::f32, Shape{N, (16 - H), W, C}, padding);
                OutputVector chunks;
                chunks.push_back(upstream[0]);
                chunks.push_back(padding_const->output(0));
                auto concat = std::make_shared<op::Concat>(chunks, 1);
                upstream[0] = concat->output(0);
                H_new = 16;
            }
            auto kernel_shape = conv->input_value(1).get_shape();
            auto stride = conv->get_strides();
            auto dilation = conv->get_dilations();
            auto pad_begin = conv->get_pads_begin();
            auto pad_end = conv->get_pads_end();
            auto pad_type = conv->get_auto_pad();
            auto weights_const =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
            const float* weights = weights_const->get_data_ptr<float>();
            std::shared_ptr<ov::op::v1::Transpose> new_transpose;

            // perform convolutions
            std::vector<float> new_weights(kernel_shape[0] * C * kernel_shape[2] * kernel_shape[3], 0.0f);
            float* new_weight_ptr = new_weights.data();
            const float* weight_ptr = weights;
            size_t i_step = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
            size_t j_step = kernel_shape[2] * kernel_shape[3];
            size_t k_step = kernel_shape[3];
            for (size_t i = 0; i < kernel_shape[0]; i++) {
                for (size_t j = start_channel; j < start_channel + C; j++) {
                    for (size_t k = 0; k < kernel_shape[2]; k++) {
                        for (size_t m = 0; m < kernel_shape[3]; m++) {
                            *new_weight_ptr++ = *(weight_ptr + i * i_step + j * j_step + k * k_step + m);
                        }
                    }
                }
            }
            auto new_weights_const = op::Constant::create(ngraph::element::f32,
                                                          Shape{kernel_shape[0], C, kernel_shape[2], kernel_shape[3]},
                                                          new_weights);
            new_transpose =
                std::make_shared<op::Transpose>(upstream[0],
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                  new_weights_const->output(0),
                                                                  stride,
                                                                  pad_begin,
                                                                  pad_end,
                                                                  dilation,
                                                                  pad_type);
            if (p == 0) {  // add bias to the first convolution in the set
                auto add_const = std::dynamic_pointer_cast<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
                if (add_const == nullptr) {
                    return (result);  // bad graph
                }
                auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), add_const->output(0));
                upstream[0] = new_add->output(0);
            } else {
                upstream[0] = new_conv->output(0);
            }
            // if convolution is not split then insert activation now
            if (parent.size() == 1) {
                auto new_relu = std::make_shared<opset1::Relu>(upstream[0]);
                upstream[0] = new_relu->output(0);
            }
            new_transpose =
                std::make_shared<op::Transpose>(upstream[0],
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            upstream[0] = new_transpose->output(0);
            // remove padding if necessary
            if (H < 16) {
                auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                    upstream[0],
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {0ull, 0ull, 0ull, 0ull}),  // begin slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {0ull, H / stride[0], 0ull, 0ull}),  // end slice index
                    ngraph::opset1::Constant::create(ngraph::element::i64,
                                                     ngraph::Shape{4},
                                                     {1ull, 1ull, 1ull, 1ull}),  // strides
                    std::vector<int64_t>{1, 0, 1, 1},                            // begin mask
                    std::vector<int64_t>{1, 0, 1, 1},                            // end mask
                    std::vector<int64_t>{0, 0, 0, 0},
                    std::vector<int64_t>{0, 0, 0, 0},
                    std::vector<int64_t>{0, 0, 0, 0});
                upstream[0] = slice->output(0);
            }
            if (p > 0) {
                auto new_add = std::make_shared<opset1::Add>(result->output(0), upstream[0]);
                upstream[0] = new_add->output(0);
                if (p == parent.size() - 1) {
                    auto new_relu = std::make_shared<opset1::Relu>(upstream[0]);
                    upstream[0] = new_relu->output(0);
                }
            }
            result = upstream[0].get_node_shared_ptr();
            start_channel += C;
        }
    }

    return (result);
}

std::shared_ptr<ov::Node> AdlBigTranspose2d(Output<Node> parent) {
    std::shared_ptr<ov::Node> result = nullptr;

    auto input_shape = parent.get_shape();
    if (((input_shape.size() == 4) && (input_shape[0] == 1) && (input_shape[1] == 1)) ||
        ((input_shape.size() == 3) && (input_shape[0] == 1)) || (input_shape.size() == 2)) {
        OutputVector upstream(1);
        size_t H = input_shape[input_shape.size() - 2];
        size_t W = input_shape[input_shape.size() - 1];
        size_t H_div_8 = H / 8;
        size_t W_div_8 = W / 8;

        if (((H % 8) != 0) || ((W % 8) != 0)) {
            return (result);  // abort if not multiple of 8 rows and columns
        }

        // if not 2D then reshape to 2D
        upstream[0] = parent;
        if ((input_shape.size() == 4) && (input_shape.size() == 3)) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),
                false);
            upstream[0] = reshape->output(0);
        }

        // split into 8xW row blocks
        auto h_split =
            std::make_shared<ngraph::opset1::Split>(upstream[0],
                                                    ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                                                    H_div_8);

        // transpose row blocks to Wx8
        std::vector<OutputVector> subblock;
        for (size_t i = 0; i < H_div_8; i++) {
            auto transpose =
                std::make_shared<op::Transpose>(h_split->output(i),
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            // split transposed row blocks into 8x8 blocks
            OutputVector block;
            auto split_rowblock =
                std::make_shared<ngraph::opset1::Split>(transpose->output(0),
                                                        ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                                                        W_div_8);
            for (size_t j = 0; j < W_div_8; j++) {
                block.push_back(split_rowblock->output(j));
            }
            subblock.push_back(block);
        }

        // un-transpose all 8x8 blocks
        for (size_t i = 0; i < H_div_8; i++) {
            for (size_t j = 0; j < W_div_8; j++) {
                auto transpose =
                    std::make_shared<op::Transpose>(subblock[i][j],
                                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                subblock[i][j] = transpose->output(0);
            }
        }

        // concatenate into block columns
        OutputVector colblock;
        for (size_t j = 0; j < W_div_8; j++) {
            OutputVector blocks;
            for (size_t i = 0; i < H_div_8; i++) {
                blocks.push_back(subblock[i][j]);
            }
            auto concat = std::make_shared<ngraph::opset1::Concat>(blocks, 0);
            colblock.push_back(concat->output(0));
        }

        // un-transpose all block columns
        for (size_t i = 0; i < colblock.size(); i++) {
            auto transpose =
                std::make_shared<op::Transpose>(colblock[i],
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            colblock[i] = transpose->output(0);
        }

        // concatenate transposed block columns
        auto concat = std::make_shared<ngraph::opset1::Concat>(colblock, 0);
        upstream[0] = concat->output(0);

        if (input_shape.size() == 3) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, W, H})->output(0),
                false);
            upstream[0] = reshape->output(0);
        } else if (input_shape.size() == 4) {
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, W, H})->output(0),
                false);
            upstream[0] = reshape->output(0);
        }

        result = upstream[0].get_node_shared_ptr();
    }

    return (result);
}

void InsertActivation(OutputVector& upstream,
                      std::shared_ptr<ov::op::v0::PRelu> prelu,
                      std::shared_ptr<ov::op::v0::Relu> relu,
                      std::shared_ptr<ov::op::v0::Sigmoid> sigmoid,
                      std::shared_ptr<ov::op::v0::Tanh> tanh) {
    if (prelu) {
        auto slope_const = std::dynamic_pointer_cast<opset1::Constant>(prelu->input_value(1).get_node_shared_ptr());
        const float* slope_ptr = slope_const->get_data_ptr<float>();
        std::vector<float> new_slope(1, 0.0f);
        float* new_slope_ptr = new_slope.data();
        *new_slope_ptr = *slope_ptr;
        auto new_prelu_slope = op::Constant::create(ngraph::element::f32, Shape{1ull}, new_slope);
        auto new_prelu = std::make_shared<opset1::PRelu>(upstream[0], new_prelu_slope->output(0));
        upstream[0] = new_prelu->output(0);
    } else if (relu) {
        auto new_relu = std::make_shared<opset1::Relu>(upstream[0]);
        upstream[0] = new_relu->output(0);
    } else if (sigmoid) {
        auto new_sigmoid = std::make_shared<opset1::Sigmoid>(upstream[0]);
        upstream[0] = new_sigmoid->output(0);
    } else if (tanh) {
        auto new_tanh = std::make_shared<opset1::Tanh>(upstream[0]);
        upstream[0] = new_tanh->output(0);
    }
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> get_input_dimensions(ov::Shape input_shape) {
    uint64_t d0 = input_shape[0];
    uint64_t d1 = input_shape[1];
    uint64_t d2 = input_shape[2];
    uint64_t d3 = input_shape[3];
    return std::make_tuple(d0, d1, d2, d3);
}

std::tuple<int64_t, int64_t, int64_t> extract_height_padding(ov::CoordinateDiff pads_begin,
                                                                    ov::CoordinateDiff pads_end) {
    auto height_begin = pads_begin[0];
    auto height_end = pads_end[0];
    return std::make_tuple(height_begin, height_end, std::abs(height_begin - height_end));
}

std::tuple<int64_t, int64_t, int64_t> extract_width_padding(ov::CoordinateDiff pads_begin,
                                                                   ov::CoordinateDiff pads_end) {
    auto width_begin = pads_begin[1];
    auto width_end = pads_end[1];
    return std::make_tuple(width_begin, width_end, std::abs(width_begin - width_end));
}

std::shared_ptr<ov::opset11::Reshape> create_reshape(const ov::Output<ov::Node>& input,
                                                            uint64_t ndims,
                                                            ov::Shape shape) {
    return std::make_shared<ov::opset11::Reshape>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{ndims}, shape)->output(0),
        false);
}

std::shared_ptr<ov::opset11::Constant> create_zero_const(ov::Shape shape) {
    return ov::opset11::Constant::create(ov::element::f32, shape, std::vector<float>(shape[0] * shape[1], 0.0f));
}

std::shared_ptr<ov::op::v0::Concat> concatenate_zeros(uint64_t pad_begin,
                                                             uint64_t pad_end,
                                                             std::shared_ptr<ov::Node> padding_const,
                                                             std::shared_ptr<ov::Node> input_node) {
    ov::OutputVector concat_vector;
    if (pad_begin > pad_end) {
        concat_vector.push_back(padding_const->output(0));
        concat_vector.push_back(input_node->output(0));
    } else {
        concat_vector.push_back(input_node->output(0));
        concat_vector.push_back(padding_const->output(0));
    }
    return std::make_shared<ov::opset11::Concat>(concat_vector, 0);
}

void trimm_padding(ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end) {
    if (pads_begin[0] > pads_end[0]) {
        pads_begin[0] = pads_end[0];
    } else {
        pads_end[0] = pads_begin[0];
    }
    if (pads_begin[1] > pads_end[1]) {
        pads_begin[1] = pads_end[1];
    } else {
        pads_end[1] = pads_begin[1];
    }
}

std::shared_ptr<ov::Node> modify_padding(std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv,
                                                    const ov::Output<ov::Node>& input,
                                                    ov::CoordinateDiff pads_begin,
                                                    ov::CoordinateDiff pads_end) {
    trimm_padding(pads_begin, pads_end);

    if (nullptr != conv) {
        return std::make_shared<ov::intel_gna::op::GNAConvolution>(input,
                                                          conv->input_value(1),
                                                          conv->get_strides(),
                                                          pads_begin,
                                                          pads_end,
                                                          conv->get_dilations(),
                                                          conv->get_auto_pad());
    }

    return nullptr;
}

std::shared_ptr<ov::opset11::Transpose> create_2d_transpose(const ov::Output<ov::Node>& input) {
    return std::make_shared<ov::opset11::Transpose>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
}

std::shared_ptr<ov::op::v0::FakeQuantize> CopyFQ(const ov::Output<ov::Node> &parent, std::shared_ptr<ov::Node> old) {
    auto old_FQ = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(old);
    if (old_FQ) {
        auto levels = old_FQ->get_levels();
        auto auto_broadcast = old_FQ->get_auto_broadcast();
        auto input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(1).get_node_shared_ptr());
        auto input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(2).get_node_shared_ptr());
        auto output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(3).get_node_shared_ptr());
        auto output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(4).get_node_shared_ptr());
        auto fq_dim = parent.get_shape().size();
        auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
        auto fq_type = parent.get_element_type();
        auto input_low_data = input_low->get_data_ptr<float>();
        auto input_high_data = input_high->get_data_ptr<float>();
        auto output_low_data = output_low->get_data_ptr<float>();
        auto output_high_data = output_high->get_data_ptr<float>();
        auto new_input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
        auto new_input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
        auto new_output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
        auto new_output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
        auto new_FQ = std::make_shared<FakeQuantize>(parent, new_input_low->output(0), new_input_high->output(0), new_output_low->output(0), new_output_high->output(0), levels, auto_broadcast);
        return new_FQ;
    } else {
        return nullptr;
    }
}

ov::Output<ov::Node> InsertOutputFQ(const ov::Output<ov::Node>& matmul_out, std::shared_ptr<ov::Node> old) {
    auto old_FQ = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(old);
    if (old_FQ) {
        auto levels = old_FQ->get_levels();
        auto auto_broadcast = old_FQ->get_auto_broadcast();
        auto input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(1).get_node_shared_ptr());
        auto input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(2).get_node_shared_ptr());
        auto output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(3).get_node_shared_ptr());
        auto output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(old_FQ->input_value(4).get_node_shared_ptr());
        auto fq_dim = matmul_out.get_shape().size();
        auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
        auto fq_type = matmul_out.get_element_type();
        auto input_low_data = input_low->get_data_ptr<float>();
        auto input_high_data = input_high->get_data_ptr<float>();
        auto output_low_data = output_low->get_data_ptr<float>();
        auto output_high_data = output_high->get_data_ptr<float>();
        auto new_input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
        auto new_input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
        auto new_output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
        auto new_output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
        auto new_FQ = std::make_shared<FakeQuantize>(matmul_out, new_input_low->output(0), new_input_high->output(0), new_output_low->output(0), new_output_high->output(0), levels, auto_broadcast);
        return new_FQ->output(0);
    } else {
        return matmul_out;
    }
}

std::shared_ptr<ov::Node> InsertWeights(ov::Shape shape, std::vector<float> data, bool use_fq) {
    std::shared_ptr<ov::Node> weight_node = nullptr; 
    auto new_weights_const = op::Constant::create(ngraph::element::f32, shape, data);
    if (use_fq) {
        size_t data_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            data_size *= shape[i];
        }
        float min = data[0];
        float max = data[0];
        bool is_binary = true;
        for (size_t i = 0; i < data_size; i++) {
            if (data[i] > max) {
                max = data[i];
            }
            if (data[i] < min) {
                min = data[i];
            }
            if ((data[i] != 0.0) && (data[i] != 1.0)) {
                is_binary = false;
            }
        }
        size_t levels = (is_binary) ? 3 : 65535;
        auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;
        auto fq_dim = shape.size();
        auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
        auto fq_type = ngraph::element::f32;
        auto fq_low = min;
        auto fq_high = max;
        auto input_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
        auto input_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
        auto output_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
        auto output_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
        auto fq = std::make_shared<FakeQuantize>(new_weights_const->output(0), input_low->output(0), input_high->output(0), 
            output_low->output(0), output_high->output(0), levels, auto_broadcast);
        weight_node = fq;
    } else {
        weight_node = new_weights_const;
    }

    return weight_node;
}

std::shared_ptr<ov::Node> InsertWeights(ov::Shape shape, std::vector<float> data, bool use_fq, float min, float max) {
    std::shared_ptr<ov::Node> weight_node = nullptr; 
    auto new_weights_const = op::Constant::create(ngraph::element::f32, shape, data);
    if (use_fq) {
        size_t data_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            data_size *= shape[i];
        }
        bool is_binary = true;
        for (size_t i = 0; i < data_size; i++) {
            if ((data[i] != 0.0) && (data[i] != 1.0)) {
                is_binary = false;
            }
        }
        size_t levels = (is_binary) ? 3 : 65535;
        auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;
        auto fq_dim = shape.size();
        auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
        auto fq_type = ngraph::element::f32;
        auto fq_low = min;
        auto fq_high = max;
        auto input_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
        auto input_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
        auto output_low = std::make_shared<Constant>(fq_type, fq_shape, fq_low);
        auto output_high = std::make_shared<Constant>(fq_type, fq_shape, fq_high);
        auto fq = std::make_shared<FakeQuantize>(new_weights_const->output(0), input_low->output(0), input_high->output(0), 
            output_low->output(0), output_high->output(0), levels, auto_broadcast);
        weight_node = fq;
    } else {
        weight_node = new_weights_const;
    }

    return weight_node;
}

std::shared_ptr<ov::op::v0::FakeQuantize> FindFqUpstream(const Output<Node>& parent) {
    auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(parent.get_node_shared_ptr());
    if (fq) {
        return fq;
    } else {
        if (parent.get_node_shared_ptr()->input_values().size() > 0) {
            auto new_parent = parent.get_node_shared_ptr()->input_value(0);
            return FindFqUpstream(new_parent);
        } else {
            return nullptr;
        }
    }
}

std::shared_ptr<ov::Node> GnaNewConvWeights(ov::Output<ov::Node>& B, bool transpose_b) {
    std::shared_ptr<ov::Node> new_conv_weights = nullptr;
    std::shared_ptr<ov::op::v0::Constant> weights_const = nullptr;
    auto weights_fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(B.get_node_shared_ptr());
    if (weights_fq) {
        const ov::Output<ov::Node>& inputfq = weights_fq->input_value(0);
        weights_const = std::dynamic_pointer_cast<ngraph::op::Constant>(inputfq.get_node()->shared_from_this());
    } else {
        weights_const = std::dynamic_pointer_cast<ngraph::op::Constant>(B.get_node_shared_ptr());
    }
    if (weights_const) {
        auto weights_shape = weights_const->get_output_shape(0);
        const float* weight_ptr = weights_const->get_data_ptr<float>();
        std::vector<float> new_weights(weights_shape[0] * weights_shape[1], 0.0f);
        float* new_weight_ptr = new_weights.data();
        ov::Shape new_weights_shape;
        if (transpose_b) {
            // leave weights alone since transpose for MatMul and transpose for convolution cancel each other
            new_weights_shape.push_back(weights_shape[0]);
            new_weights_shape.push_back(1);
            new_weights_shape.push_back(1);
            new_weights_shape.push_back(weights_shape[1]);
            memcpy(new_weight_ptr, weight_ptr, new_weights.size() * sizeof(float));
        } else {
            // transpose weight matrix
            new_weights_shape.push_back(weights_shape[1]);
            new_weights_shape.push_back(1);
            new_weights_shape.push_back(1);
            new_weights_shape.push_back(weights_shape[0]);
            for (auto i = 0; i < weights_shape[0]; i++) {
                for (auto j = 0; j < weights_shape[1]; j++) {
                    new_weight_ptr[j * weights_shape[0] + i] = weight_ptr[i * weights_shape[1] + j];
                }
            }
        }
        auto new_weights_const = ov::op::v0::Constant::create(ngraph::element::f32, new_weights_shape, new_weights);
        new_conv_weights = new_weights_const;
        if (weights_fq) {
            auto new_weights_fq = CopyFQ(new_weights_const->output(0), weights_fq);
            new_conv_weights = new_weights_fq;
        }
    } else {
        auto weights_shape = B.get_shape();
        if (transpose_b) {
            // no additional transpose since transpose for MatMul and transpose for convolution cancel each other
            auto reshape = std::make_shared<ov::op::v1::Reshape>(B,
                Constant::create(ov::element::i64, ov::Shape{4}, {weights_shape[0], 1ull, 1ull, weights_shape[1]})->output(0),false);
            new_conv_weights = reshape;
        } else {
            // transpose weight tensor
            auto transpose = std::make_shared<Transpose>(B, 
                Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose->output(0), 
                Constant::create(ov::element::i64, ov::Shape{4}, {weights_shape[1], 1ull, 1ull, weights_shape[0]})->output(0),false);
            new_conv_weights = reshape;
        }
    }

    return new_conv_weights;
}

std::shared_ptr<ov::Node> GnaNewConvBias(ov::Output<ov::Node>& C) {
    std::shared_ptr<ov::Node> new_conv_bias = nullptr;
    auto bias_fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(C.get_node_shared_ptr());
    auto bias_const = std::dynamic_pointer_cast<ngraph::op::Constant>(C.get_node_shared_ptr());
    if (bias_fq) {
        bias_const = std::dynamic_pointer_cast<ngraph::op::Constant>(bias_fq->input_value(0).get_node_shared_ptr());
    }
    if (bias_const) {
        auto bias_shape = bias_const->get_output_shape(0);
        const float* bias_ptr = bias_const->get_data_ptr<float>();
        size_t len = 1;
        for (size_t i = 0; i < bias_shape.size(); i++) {
            len *= bias_shape[i];
        }
        std::vector<float> new_bias(len, 0.0f);
        float* new_bias_ptr = new_bias.data();
        ov::Shape new_bias_shape;
        new_bias_shape.push_back(1);
        new_bias_shape.push_back(1);
        new_bias_shape.push_back(1);
        new_bias_shape.push_back(len);
        memcpy(new_bias_ptr, bias_ptr, new_bias.size() * sizeof(float));
        auto new_bias_const = ov::op::v0::Constant::create(ngraph::element::f32, new_bias_shape, new_bias);
        new_conv_bias = new_bias_const;
    } else {
        auto C_shape = C.get_shape();
        auto C_size = C_shape[0];
        for (size_t i = 1; i < C_shape.size(); i++) {
            C_size *= C_shape[i];
        }
        auto bias = (bias_fq) ? bias_fq->output(0) : C;
        auto reshape = std::make_shared<ov::op::v1::Reshape>(bias,
            Constant::create(ov::element::i64, ov::Shape{4}, {1ull, 1ull, 1ull, C_size})->output(0), false);
        new_conv_bias = reshape;
    }

    return new_conv_bias;
}

// insert subgraph to perform A * B
std::shared_ptr<ov::Node> InsertGnaMatMulAdd2D(ov::Output<ov::Node>& A, ov::Output<ov::Node>& B, bool transpose_a, bool transpose_b, bool out_2D) {
    auto A_shape = A.get_shape();
    auto B_shape = B.get_shape();
    auto H = A_shape[0];
    auto W = A_shape[1];
    ov::Shape out_shape = {A_shape[0], B_shape[1]};
    ov::Output<ov::Node> upstream = A;
    if (transpose_b) out_shape[1] = B_shape[0];
    if (transpose_a) {
        auto transpose = std::make_shared<Transpose>(A, Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        upstream = transpose->output(0);
        auto H = A_shape[0];
        auto W = A_shape[1];
        out_shape[0] = A_shape[1];
    }
    auto reshape = std::make_shared<ov::op::v1::Reshape>(upstream, Constant::create(ov::element::i64, ov::Shape{4}, {1ull, A_shape[0], 1ull, A_shape[1]})->output(0), false);
    auto weights = GnaNewConvWeights(B, transpose_b);
    auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0), weights->output(0),
        ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID );
    if (out_2D) {
        reshape = std::make_shared<ov::op::v1::Reshape>(conv->output(0), 
            Constant::create(ov::element::i64, ov::Shape{2}, out_shape)->output(0), false);
        return reshape;
    } else {
        return conv;
    }
}

// insert subgraph to perform A * B + C
std::shared_ptr<ov::Node> InsertGnaMatMulAdd2D(ov::Output<ov::Node>& A, ov::Output<ov::Node>& B, ov::Output<ov::Node>& C, bool transpose_a, bool transpose_b, bool out_2D) {
    auto A_shape = A.get_shape();
    auto B_shape = B.get_shape();
    auto H = A_shape[0];
    auto W = A_shape[1];
    ov::Shape out_shape = {A_shape[0], B_shape[1]};
    ov::Output<ov::Node> upstream = A;
    if (transpose_b) out_shape[1] = B_shape[0];
    if (transpose_a) {
        auto transpose = std::make_shared<Transpose>(A, Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
        upstream = transpose->output(0);
        H = A_shape[1];
        W = A_shape[0];
        out_shape[0] = A_shape[1];
    }
    auto reshape = std::make_shared<ov::op::v1::Reshape>(upstream, Constant::create(ov::element::i64, ov::Shape{4}, {1ull, H, 1ull, W})->output(0), false);
    auto weights = GnaNewConvWeights(B, transpose_b);
    auto bias = GnaNewConvBias(C);
    auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0), weights->output(0), bias->output(0),
        ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1}, ov::op::PadType::VALID );
    if (out_2D) {
        reshape = std::make_shared<ov::op::v1::Reshape>(conv->output(0), 
            Constant::create(ov::element::i64, ov::Shape{2}, out_shape)->output(0), false);
        return reshape;
    } else {
        return conv;
    }
}

bool Is2DTranspose(std::shared_ptr<ov::op::v1::Transpose> transpose) {
    bool is_2D = false;

    if (transpose) {
        auto input_shape = transpose->input_value(0).get_shape();
        auto output_shape = transpose->output(0).get_shape();
        const ov::Output<ov::Node>& transpose_order = transpose->input_value(1);
        auto transpose_order_dim = transpose_order.get_shape().size();
        if (transpose_order_dim == 1) {
            auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
            if (const_with_order_values) {
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
                std::vector<int64_t> squeezed_order;
                int64_t count = 0;
                for (size_t i = 0; i < order.size(); i++) {
                    if (order[i] != i) {
                        squeezed_order.push_back(order[i] - count);
                    } else {
                        count++;
                    }
                }
                if ((squeezed_order.size() == 2) && (squeezed_order[0] == 1) && (squeezed_order[1] == 0)) {
                    is_2D = true;
                }
            }
        }
    }

    return is_2D;
}

