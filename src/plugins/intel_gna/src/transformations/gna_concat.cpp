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

using namespace ngraph;
using namespace op;

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
        if (trivial_concat) {
            continue;   // concat is already supported natively by GNA
        } else {

            OutputVector upstream;
            for (size_t i = 0; i < concat->input_values().size(); i++) {
                auto input_shape = concat->input_value(i).get_shape();
                size_t dim0 = 1;
                for (size_t j = 0; j < (size_t)axis; j++) {
                    dim0 = dim0 * input_shape[j];
                }
                size_t dim1 = 1;
                for (size_t j = (size_t)axis; j < input_shape.size(); j++) {
                    dim1 = dim1 * input_shape[j];
                }
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(concat->input_value(i),
                    op::Constant::create(ngraph::element::i64,Shape{2},{dim0, dim1})->output(0),false);
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
