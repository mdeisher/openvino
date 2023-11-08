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

#pragma once

#include <transformations_visibility.hpp>
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "openvino/opsets/opset11.hpp"
#include <layers/gna_layer_info.hpp>
#include <layers/gna_layer_type.hpp>

#define ADL_GNA_MAX_CHANNELS 96

size_t GetChannels(ov::Output<ov::Node>& parent);
std::shared_ptr<ov::Node> AdlChannelPadTensor(ov::Output<ov::Node> parent);
std::shared_ptr<ov::Node> AdlChannelPadKernel(ov::Output<ov::Node>& weights_const_output, size_t C);
std::shared_ptr<ov::Node> NchwToNhwc(ov::Output<ov::Node> parent);
std::shared_ptr<ov::Node> AdlInsertConvolutionAddRelu(ov::Output<ov::Node> parent,
    std::shared_ptr<ngraph::opset1::Convolution> conv,
    std::shared_ptr<ngraph::opset1::Add> add);
std::vector<std::shared_ptr<ov::Node>> AdlInsertSplitConvolutionAddRelu(ov::Output<ov::Node> parent,
    std::shared_ptr<ngraph::opset1::Convolution> conv,
    std::shared_ptr<ngraph::opset1::Add> add);
std::shared_ptr<ov::Node> AdlInsertConvolutionAddReluHpadCsplit(ov::Output<ov::Node> parent,
    std::shared_ptr<ngraph::opset1::Convolution> conv,
    std::shared_ptr<ngraph::opset1::Add> add);
std::shared_ptr<ov::Node> AdlInsertSplitConvolutionAddReluHpadCsplit(std::vector<std::shared_ptr<ov::Node>> parent,
    std::shared_ptr<ngraph::opset1::Convolution> conv,
    std::shared_ptr<ngraph::opset1::Add> add);
std::shared_ptr<ov::Node> AdlBigTranspose2d(ov::Output<ov::Node> parent);
void InsertActivation(ov::OutputVector& upstream,
                      std::shared_ptr<ov::op::v0::PRelu> prelu,
                      std::shared_ptr<ov::op::v0::Relu> relu,
                      std::shared_ptr<ov::op::v0::Sigmoid> sigmoid,
                      std::shared_ptr<ov::op::v0::Tanh> tanh);
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> get_input_dimensions(ov::Shape input_shape);
std::tuple<int64_t, int64_t, int64_t> extract_height_padding(ov::CoordinateDiff pads_begin, ov::CoordinateDiff pads_end);
std::tuple<int64_t, int64_t, int64_t> extract_width_padding(ov::CoordinateDiff pads_begin, ov::CoordinateDiff pads_end);
std::shared_ptr<ov::opset11::Reshape> create_reshape(const ov::Output<ov::Node>& input,
                                                     uint64_t ndims,
                                                     ov::Shape shape);
std::shared_ptr<ov::opset11::Constant> create_zero_const(ov::Shape shape);
std::shared_ptr<ov::op::v0::Concat> concatenate_zeros(uint64_t pad_begin,
                                                      uint64_t pad_end,
                                                      std::shared_ptr<ov::Node> padding_const,
                                                      std::shared_ptr<ov::Node> input_node);
void trimm_padding(ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end);
std::shared_ptr<ov::Node> modify_padding(std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv,
                                             const ov::Output<ov::Node>& input,
                                             ov::CoordinateDiff pads_begin,
                                             ov::CoordinateDiff pads_end);
std::shared_ptr<ov::opset11::Transpose> create_2d_transpose(const ov::Output<ov::Node>& input);
std::shared_ptr<ov::op::v0::FakeQuantize> CopyFQ(const ov::Output<ov::Node>& parent, std::shared_ptr<ov::Node> old);
ov::Output<ov::Node> InsertOutputFQ(const ov::Output<ov::Node>& matmul_out, std::shared_ptr<ov::Node> old);
std::shared_ptr<ov::Node> InsertWeights(ov::Shape shape, std::vector<float> data, bool use_fq);
std::shared_ptr<ov::Node> InsertWeights(ov::Shape shape, std::vector<float> data, bool use_fq, float min, float max);
std::shared_ptr<ov::op::v0::FakeQuantize> FindFqUpstream(const ov::Output<ov::Node>& parent);
