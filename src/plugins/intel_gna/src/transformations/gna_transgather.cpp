// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This transformation for ReduceMean can be used while we wait for AvgPool support in the GNA plugin.
// Once AvgPool support is ready they the plugin can already convert ReduceMean to a pattern involving AvgPool.

#include "gna_transgather.hpp"

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "gna_helper.hpp"

using namespace ov::opset12;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

// Implements Transpose --> Reshape --> Reshape --> Gather patterns

GnaTransGatherTransformation::GnaTransGatherTransformation() {
    MATCHER_SCOPE(GnaTransGatherTransformation);

    auto transpose_pattern = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::any_input()});
    auto reshape1_pattern = pattern::wrap_type<Reshape>({transpose_pattern, pattern::any_input()});
    auto reshape2_pattern = pattern::wrap_type<Reshape>({reshape1_pattern, pattern::any_input()});
    auto gather_pattern = pattern::wrap_type<Gather>({reshape2_pattern, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto transpose = std::dynamic_pointer_cast<Transpose>(pattern_map.at(transpose_pattern).get_node_shared_ptr());
        auto reshape1 = std::dynamic_pointer_cast<Reshape>(pattern_map.at(reshape1_pattern).get_node_shared_ptr());
        auto reshape2 = std::dynamic_pointer_cast<Reshape>(pattern_map.at(reshape2_pattern).get_node_shared_ptr());
        auto gather = std::dynamic_pointer_cast<ov::op::v8::Gather>(pattern_map.at(gather_pattern).get_node_shared_ptr());

        const Output<Node>& parent = transpose->input_value(0);
        auto input_shape = parent.get_shape();
        if (input_shape[0] != 1) {
            return false;
        }
        auto output_shape = transpose->output(0).get_shape();
        const Output<Node>& transpose_order = transpose->input_value(1);
        auto transpose_order_dim = transpose_order.get_shape().size();
        if (transpose_order_dim != 1) {
            return false;
        }
        auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
        if (!const_with_order_values) {
            return false;
        }
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
        if ((order[0] != 0) || (order[1] != 2) || (order[2] != 1) || (order[3] != 3)) {
            return false;
        }
        auto gather_axis = gather->get_axis();
        if (gather_axis != 0) {
            return false;
        }
        auto gather_indices = gather->input_value(1);
        auto const_with_gather_indices = std::dynamic_pointer_cast<ngraph::opset1::Constant>(gather_indices.get_node_shared_ptr());
        if (!const_with_gather_indices) {
            return false;
        }
        int64_t index;
        if (const_with_gather_indices->get_output_element_type(0) == ov::element::i8) {
            index = *(const_with_gather_indices->get_data_ptr<int8_t>());
        } else if (const_with_gather_indices->get_output_element_type(0) == ov::element::i32) {
            index = *(const_with_gather_indices->get_data_ptr<int32_t>());
        } else {
            index = *(const_with_gather_indices->get_data_ptr<int64_t>());
        }
        auto gather_in_shape = gather->input_value(0).get_shape();
        auto gather_out_shape = gather->output(0).get_shape();
        // this is coded for a very specific case
        if ((gather_out_shape.size() != 1) || (index < 0) || (index >= gather_in_shape[0])) {
            return false;
        }

        auto new_reshape = std::make_shared<Reshape>(parent,
            Constant::create(element::i64, Shape{2}, {input_shape[1], input_shape[2] * input_shape[3]})->output(0),false);
        auto new_transpose = std::make_shared<Transpose>(new_reshape->output(0),
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1, 0}));
        new_reshape = std::make_shared<Reshape>(new_transpose->output(0),
            Constant::create(element::i64, Shape{2}, {input_shape[2], input_shape[3] * input_shape[1]})->output(0),false);
        
        uint64_t uindex = (uint64_t)index;
        auto new_split = std::make_shared<ngraph::opset1::StridedSlice>(new_reshape->output(0),
            ngraph::opset1::Constant::create(ngraph::element::i64,ngraph::Shape{2},{uindex, 0ull}),  // begin slice index
            ngraph::opset1::Constant::create(ngraph::element::i64,ngraph::Shape{2},{uindex + 1, 0ull}),  // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1ull, 1ull}),  // strides
                std::vector<int64_t>{0, 1},  // begin mask
                std::vector<int64_t>{0, 1},  // end mask
                std::vector<int64_t>{0, 0},  // new axis mask
                std::vector<int64_t>{0, 0},  // shrink axis mask
                std::vector<int64_t>{0, 0}); // ellipsis mask

        new_reshape = std::make_shared<Reshape>(new_split->output(0),
            Constant::create(element::i64, Shape{2}, {input_shape[3], input_shape[1]})->output(0),false);
        new_transpose = std::make_shared<Transpose>(new_reshape->output(0),
            Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1, 0}));
        new_reshape = std::make_shared<Reshape>(new_transpose->output(0),
            Constant::create(element::i64, Shape{1}, {input_shape[1] * input_shape[3]})->output(0),false);

        replace_output_update_name(gather, new_reshape);
        return true;
    };
    
    auto m = std::make_shared<pattern::Matcher>(gather_pattern, matcher_name);
    this->register_matcher(m, callback);
}

