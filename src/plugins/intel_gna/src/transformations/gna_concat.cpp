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

#include "gna_concat.hpp"
#include "gna_helper.hpp"

using namespace ngraph;
using namespace op;

std::vector<size_t> ReduceDims(Shape shape, int64_t axis) {
    std::vector<size_t> result;
    size_t dim0 = 1;
    for (size_t j = 0; j < (size_t)axis; j++) {
        dim0 = dim0 * shape[j];
    }
    result.push_back(dim0);
    size_t dim1 = 1;
    for (size_t j = (size_t)axis; j < shape.size(); j++) {
        dim1 = dim1 * shape[j];
    }
    result.push_back(dim1);

    return result;
}

bool ngraph::pass::GnaConcatDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto concat = std::dynamic_pointer_cast<Concat>(node);
        if (nullptr == concat) {
            continue;
        }

        auto output_shape = concat->output(0).get_shape();
        auto axis = concat->get_axis();
        bool trivial_concat = true;
        for (int64_t i = 0; i < axis; i++) {
            for (size_t j = 0; j < concat->input_values().size(); j++) {
                auto dim = concat->input(j).get_shape()[i];
                if (dim > 1) {
                    trivial_concat = false;
                }
            }
        }
        if (trivial_concat) {    // concat is already supported natively by GNA
            // Concat with an input does not work properly.
            // It either leads to an error condition or very bad accuracy.
            // Here we attampt to implement the concat using a Multiply operation so that scale factors are properly set.
            // For now, we limit the implementation to concat with 2 inputs (it is trivial to extend it).
            auto num_inputs = concat->get_input_size();
            std::vector<std::shared_ptr<FakeQuantize>> upstream_fq;
            std::vector<int> input_index;
            for (auto i = 0; i < num_inputs; i++) {
                std::shared_ptr<Parameter> input = nullptr;
                // support several sequences:
                // 1. input --> Concat
                // 2. input --> Reshape --> Concat
                // 3. input --> FakeQuantize --> Reshape --> Concat
                // 4. input --> FakeQuantize --> Concat
                auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(concat->get_input_node_shared_ptr(i));
                if (reshape) {
                    auto fq = std::dynamic_pointer_cast<FakeQuantize>(reshape->get_input_node_shared_ptr(0));
                    if (fq) {
                        input = std::dynamic_pointer_cast<Parameter>(fq->get_input_node_shared_ptr(0));
                    } else {
                        input = std::dynamic_pointer_cast<Parameter>(reshape->get_input_node_shared_ptr(0));
                    }
                } else {
                    // ------------ experiment to prohibit transform with certain patterns ------------------
                    //auto fq = std::dynamic_pointer_cast<FakeQuantize>(concat->get_input_node_shared_ptr(i));
                    //if (fq) {
                    //    input = std::dynamic_pointer_cast<Parameter>(fq->get_input_node_shared_ptr(0));
                    //} else {
                    //    input = std::dynamic_pointer_cast<Parameter>(concat->get_input_node_shared_ptr(i));
                    //}
                }
                if (input) {
                    input_index.push_back(i);
                }
                auto fq = FindFqUpstream(concat->input_value(i));
                if (fq) {
                    upstream_fq.push_back(fq);
                }
            }
            if ((num_inputs == 2) && (input_index.size() > 0)) {
                OutputVector upstream0, upstream1;
                auto input0_shape = concat->input_value(0).get_shape();
                auto input1_shape = concat->input_value(1).get_shape();
                auto dims0 = ReduceDims(input0_shape, axis + 1);
                auto dims1 = ReduceDims(input1_shape, axis + 1);
                auto new_reshape0 = std::make_shared<ngraph::opset1::Reshape>(concat->input_value(0),
                    op::Constant::create(ngraph::element::i64,Shape{2},{dims0[0], dims0[1]})->output(0),false);
                auto new_reshape1 = std::make_shared<ngraph::opset1::Reshape>(concat->input_value(1),
                    op::Constant::create(ngraph::element::i64,Shape{2},{dims1[0], dims1[1]})->output(0),false);
                std::vector<float> padding0(dims1[0] * dims0[1], 1.0f);
                std::vector<float> padding1(dims0[0] * dims1[1], 1.0f);
                auto pad0_const = Constant::create(ngraph::element::f32, ov::Shape{dims1[0], dims0[1]}, padding0);
                auto pad1_const = Constant::create(ngraph::element::f32, ov::Shape{dims0[0], dims1[1]}, padding1);
                if (upstream_fq.size() > 0) {
                    size_t levels = 65535;
                    auto auto_broadcast = ov::op::AutoBroadcastType::NUMPY;
                    auto fq0_dim = pad0_const->output(0).get_shape().size();
                    auto fq1_dim = pad1_const->output(0).get_shape().size();
                    auto fq0_shape = (fq0_dim == 1) ? ov::Shape{1} : ((fq0_dim==2) ? ov::Shape{1,1} : ((fq0_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
                    auto fq1_shape = (fq1_dim == 1) ? ov::Shape{1} : ((fq1_dim==2) ? ov::Shape{1,1} : ((fq1_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
                    auto fq0_type = pad0_const->output(0).get_element_type();
                    auto fq1_type = pad1_const->output(0).get_element_type();
                    auto input_low_data = -1.0f;
                    auto input_high_data = 1.0f;
                    auto output_low_data = -1.0f;
                    auto output_high_data = 1.0f;
                    auto input0_low = std::make_shared<Constant>(fq0_type, fq0_shape, input_low_data);
                    auto input0_high = std::make_shared<Constant>(fq0_type, fq0_shape, input_high_data);
                    auto output0_low = std::make_shared<Constant>(fq0_type, fq0_shape, output_low_data);
                    auto output0_high = std::make_shared<Constant>(fq0_type, fq0_shape, output_high_data);
                    auto pad0_fq = std::make_shared<FakeQuantize>(pad0_const->output(0), input0_low->output(0), input0_high->output(0), output0_low->output(0), output0_high->output(0), levels, auto_broadcast);
                    auto input1_low = std::make_shared<Constant>(fq1_type, fq1_shape, input_low_data);
                    auto input1_high = std::make_shared<Constant>(fq1_type, fq1_shape, input_high_data);
                    auto output1_low = std::make_shared<Constant>(fq1_type, fq1_shape, output_low_data);
                    auto output1_high = std::make_shared<Constant>(fq1_type, fq1_shape, output_high_data);
                    auto pad1_fq = std::make_shared<FakeQuantize>(pad1_const->output(0), input1_low->output(0), input1_high->output(0), output1_low->output(0), output1_high->output(0), levels, auto_broadcast);
                    upstream0.push_back(new_reshape0->output(0));
                    upstream0.push_back(pad0_fq->output(0));
                    upstream1.push_back(pad1_fq->output(0));
                    upstream1.push_back(new_reshape1->output(0));
                } else {
                    upstream0.push_back(new_reshape0->output(0));
                    upstream0.push_back(pad0_const->output(0));
                    upstream1.push_back(pad1_const->output(0));
                    upstream1.push_back(new_reshape1->output(0));
                }
                auto new_concat0 = std::make_shared<ngraph::opset1::Concat>(upstream0, 0);
                auto new_concat1 = std::make_shared<ngraph::opset1::Concat>(upstream1, 0);
                auto new_multiply = std::make_shared<ov::op::v1::Multiply>(new_concat0->output(0), new_concat1->output(0), AutoBroadcastType::NONE);
                ov::Output<ov::Node> upstream = new_multiply->output(0);
                if (upstream_fq.size() > 0) {
                    auto levels = upstream_fq[0]->get_levels();
                    auto auto_broadcast = upstream_fq[0]->get_auto_broadcast();
                    auto input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[0]->input_value(1).get_node_shared_ptr());
                    auto input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[0]->input_value(2).get_node_shared_ptr());
                    auto output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[0]->input_value(3).get_node_shared_ptr());
                    auto output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[0]->input_value(4).get_node_shared_ptr());
                    auto input_low_data = *input_low->get_data_ptr<float>();
                    auto input_high_data = *input_high->get_data_ptr<float>();
                    auto output_low_data = *output_low->get_data_ptr<float>();
                    auto output_high_data = *output_high->get_data_ptr<float>();
                    for (auto i = 1; i < upstream_fq.size(); i++) {
                        input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[i]->input_value(1).get_node_shared_ptr());
                        input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[i]->input_value(2).get_node_shared_ptr());
                        output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[i]->input_value(3).get_node_shared_ptr());
                        output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(upstream_fq[i]->input_value(4).get_node_shared_ptr());
                        //if (input_low_data > *input_low->get_data_ptr<float>()) {
                            input_low_data *= fabs(*input_low->get_data_ptr<float>());
                        //}
                        //if (input_high_data < *input_high->get_data_ptr<float>()) {
                            input_high_data *= fabs(*input_high->get_data_ptr<float>());
                        //}
                        //if (output_low_data > *output_low->get_data_ptr<float>()) {
                            output_low_data *= fabs(*output_low->get_data_ptr<float>());
                        //}
                        //if (output_high_data < *output_high->get_data_ptr<float>()) {
                            output_high_data *= fabs(*output_high->get_data_ptr<float>());
                        //}
                    }
                    auto fq_dim = new_multiply->output(0).get_shape().size();
                    auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
                    auto fq_type = new_multiply->output(0).get_element_type();
                    auto new_input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
                    auto new_input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
                    auto new_output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
                    auto new_output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
                    auto new_multiply_fq = std::make_shared<FakeQuantize>(new_multiply->output(0), new_input_low->output(0), new_input_high->output(0), 
                        new_output_low->output(0), new_output_high->output(0), levels, auto_broadcast);
                    upstream = new_multiply_fq->output(0);
                }

                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(upstream,
                    op::Constant::create(ngraph::element::i64, Shape{concat->output(0).get_shape().size()}, concat->output(0).get_shape())->output(0),false);
                ngraph::replace_node_update_name(concat, new_reshape);
                is_graph_modfied = true;
            }

            continue;

        } else {

            OutputVector upstream;
            for (size_t i = 0; i < concat->input_values().size(); i++) {
                auto input_shape = concat->input_value(i).get_shape();
                auto dims = ReduceDims(input_shape, axis);
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(concat->input_value(i),
                    op::Constant::create(ngraph::element::i64,Shape{2},{dims[0], dims[1]})->output(0),false);
                auto new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0),
                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                upstream.push_back(new_transpose->output(0));
            }
            auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 0);
            auto new_transpose = std::make_shared<op::Transpose>(new_concat->output(0),
                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{output_shape.size()}, output_shape)->output(0),false);
            ngraph::replace_node_update_name(concat, new_reshape);
            is_graph_modfied = true;
            continue;                    
        }
    }
    return is_graph_modfied;
}
