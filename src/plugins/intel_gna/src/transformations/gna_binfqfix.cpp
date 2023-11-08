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

#include "gna_binfqfix.hpp"
#include "gna_helper.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::GnaBinaryFqFix::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto fq = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize> (node);
        if (nullptr == fq) {
            continue;
        }

        const Output<Node>& input = fq->input_value(0);
        auto input_shape = input.get_shape();
        auto levels = fq->get_levels();
        auto auto_broadcast = fq->get_auto_broadcast();
        auto input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(1).get_node_shared_ptr());
        auto input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(2).get_node_shared_ptr());
        auto output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(3).get_node_shared_ptr());
        auto output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(4).get_node_shared_ptr());
        auto fq_dim = input.get_shape().size();
        auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
        auto fq_type = input.get_element_type();
        auto input_low_data = input_low->get_data_ptr<float>();
        auto input_high_data = input_high->get_data_ptr<float>();
        auto output_low_data = output_low->get_data_ptr<float>();
        auto output_high_data = output_high->get_data_ptr<float>();
        auto scale_factor = (levels - 1) / (*output_high_data - *output_low_data);

        auto fq_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(input.get_node_shared_ptr());

        if (fq_const) {
            const float* data_ptr = fq_const->get_data_ptr<float>();
            size_t data_size = 1;
            for (size_t i = 0; i < input_shape.size(); i++) {
                data_size *= input_shape[i];
            }
            bool is_binary = true;
            for (size_t i = 0; i < data_size; i++) {
                if ((data_ptr[i] != 0.0) && (data_ptr[i] != 1.0)) {
                    is_binary = false;
                    break;
                }
            }

            if (is_binary) {
                size_t new_levels = 3;
                auto new_input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
                auto new_input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
                auto new_output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
                auto new_output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
                auto new_scale_factor = (new_levels - 1) / (*output_high_data - *output_low_data);
                auto new_fq = std::make_shared<FakeQuantize>(input, new_input_low->output(0), new_input_high->output(0), 
                    new_output_low->output(0), new_output_high->output(0), new_levels, auto_broadcast);
                ngraph::replace_node_update_name(fq, new_fq);
                is_graph_modfied = true;
            }
        }
    }

    return is_graph_modfied;
}
