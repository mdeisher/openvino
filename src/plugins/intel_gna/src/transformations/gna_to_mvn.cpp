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

#include "gna_to_mvn.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

// Converts pytorch normalization pattern into MVN layer
// Pattern:  Unknown --> ReduceMean --> Subtract --> 

bool ngraph::pass::GnaToMvn::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto reducemean = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(node);
        if (reducemean == nullptr) {
            continue;
        }

        // Ugly code to match our pattern
        std::shared_ptr<ngraph::opset1::Subtract> subtract = nullptr;
        const Output<Node>& input = reducemean->input_value(0);
        auto children = reducemean->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        subtract = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(children.begin()->get_node()->shared_from_this());
        if (subtract == nullptr) {
            continue;
        }
        children = input.get_target_inputs();
        if (children.size() != 2) {
            continue;
        }
        bool found = true;
        for (auto p = children.begin(); p != children.end(); p++) {
            auto child_node = p->get_node()->shared_from_this();
            auto s_ptr = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(child_node);
            auto r_ptr = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(child_node);
            if (s_ptr) {
                if (s_ptr != subtract) {
                    found = false;
                }
            }
            if (r_ptr) {
                if (r_ptr != reducemean) {
                    found = false;
                }
            }
        }
        if (!found) {
            continue;
        }

        float epsilon = 0.0f;
        auto eps_mode = op::MVNEpsMode::INSIDE_SQRT;
        auto axes = reducemean->get_input_node_shared_ptr(1);
        auto normalize_variance = false;
        auto mvn = std::make_shared<op::v6::MVN>(input, axes, normalize_variance, epsilon, eps_mode);

        ngraph::replace_node_update_name(subtract, mvn);
        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
