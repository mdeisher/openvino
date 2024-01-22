// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This transformation for ReduceMean can be used while we wait for AvgPool support in the GNA plugin.
// Once AvgPool support is ready they the plugin can already convert ReduceMean to a pattern involving AvgPool.

#include "gna_reducemean.hpp"

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


using namespace ov::opset12;
using namespace ov::pass;

GnaReduceMeanTransformation::GnaReduceMeanTransformation() {
    MATCHER_SCOPE(ReduceMeanTransformation);

    auto reducemean_pattern = pattern::wrap_type<ReduceMean>({pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reducemean = pattern_map.at(reducemean_pattern).get_node_shared_ptr();

        const Output<Node>& parent = reducemean->input_value(0);
        ov::Shape reducemean_in_shape = parent.get_shape();
        ov::Shape reducemean_out_shape = reducemean->get_output_shape(0);
        ov::Output<ov::Node> reduction_axes = reducemean->input_value(1);
        auto axis_const = std::dynamic_pointer_cast<Constant>(reduction_axes.get_node_shared_ptr());
        auto axis_shape = axis_const->get_output_shape(0);
        auto axis_dim = axis_shape[0];
        auto axis_const_data = axis_const->get_data_ptr();
        auto axis = (int64_t*)axis_const_data;

        // This technique is limited to reducing axes of at most 1024 (combined total)
        // Only simple reductions are supported (TO DO:  transpose unsupported cases)
        if (((axis_dim == 2) && (axis[0] == reducemean_in_shape.size() - 2) && (axis[1] == reducemean_in_shape.size() - 1)) ||
            ((axis_dim == 1) && (axis[0] == reducemean_in_shape.size() - 1)) ||
            ((axis_dim == 1) && (reducemean_in_shape[reducemean_in_shape.size() - 1] == 1) && (axis[0] == reducemean_in_shape.size() - 2))) {
            size_t N = 1, C = 1, H = 1, W = 1;
            if (axis_dim == 2) {
                W = (reducemean_in_shape.size() == 4) ? reducemean_in_shape[0] * reducemean_in_shape[1] : reducemean_in_shape[0];
                C = reducemean_in_shape[reducemean_in_shape.size() - 1] * reducemean_in_shape[reducemean_in_shape.size() - 2];
            } else {
                for (size_t i = 0; i < reducemean_in_shape.size(); i++) {
                    if ((int64_t)i < axis[0]) {
                        W *= reducemean_in_shape[i];
                    }
                }
                C = reducemean_in_shape[axis[0]];
            }
            auto reshape1 = std::make_shared<Reshape>(parent, Constant::create(element::i64, Shape{4}, {N, H, W, C})->output(0), false);
            auto transpose1 = std::make_shared<Transpose>(reshape1->output(0), Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            std::vector<float> weight(C, 1.0f / C);
            auto weight_const = Constant::create(ngraph::element::f32, Shape{1, C, 1, 1}, weight);
            auto conv = std::make_shared<Convolution>(transpose1->output(0),weight_const->output(0), Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
            auto transpose2 = std::make_shared<Transpose>(conv->output(0),
                Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape2 = std::make_shared<Reshape>(transpose2->output(0), Constant::create(element::i64, 
                Shape{reducemean_out_shape.size()}, reducemean_out_shape)->output(0), false);

            replace_node(reducemean, reshape2);

            return true;

        } else {

            return false;
        }
    };
    
    auto m = std::make_shared<pattern::Matcher>(reducemean_pattern, matcher_name);
    this->register_matcher(m, callback);
}

GnaReduceSumTransformation::GnaReduceSumTransformation() {
    MATCHER_SCOPE(ReduceSumTransformation);

    auto reducesum_pattern = pattern::wrap_type<ReduceSum>({pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reducesum = pattern_map.at(reducesum_pattern).get_node_shared_ptr();

        const Output<Node>& parent = reducesum->input_value(0);
        ov::Shape reducesum_in_shape = parent.get_shape();
        ov::Shape reducesum_out_shape = reducesum->get_output_shape(0);
        ov::Output<ov::Node> reduction_axes = reducesum->input_value(1);
        auto axis_const = std::dynamic_pointer_cast<Constant>(reduction_axes.get_node_shared_ptr());
        auto axis_shape = axis_const->get_output_shape(0);
        auto axis_dim = axis_shape[0];
        auto axis_const_data = axis_const->get_data_ptr();
        auto axis = (int64_t*)axis_const_data;

        // This technique is limited to reducing axes of at most 1024 (combined total)
        // Only simple reductions are supported (TO DO:  transpose unsupported cases)
        if (((axis_dim == 2) && (axis[0] == reducesum_in_shape.size() - 2) && (axis[1] == reducesum_in_shape.size() - 1)) ||
            ((axis_dim == 1) && (axis[0] == reducesum_in_shape.size() - 1)) ||
            ((axis_dim == 1) && (reducesum_in_shape[reducesum_in_shape.size() - 1] == 1) && (axis[0] == reducesum_in_shape.size() - 2))) {
            size_t N = 1, C = 1, H = 1, W = 1;
            if (axis_dim == 2) {
                W = (reducesum_in_shape.size() == 4) ? reducesum_in_shape[0] * reducesum_in_shape[1] : reducesum_in_shape[0];
                C = reducesum_in_shape[reducesum_in_shape.size() - 1] * reducesum_in_shape[reducesum_in_shape.size() - 2];
            } else {
                for (size_t i = 0; i < reducesum_in_shape.size(); i++) {
                    if ((int64_t)i < axis[0]) {
                        W *= reducesum_in_shape[i];
                    }
                }
                C = reducesum_in_shape[axis[0]];
            }
            auto reshape1 = std::make_shared<Reshape>(parent, Constant::create(element::i64, Shape{4}, {N, H, W, C})->output(0), false);
            auto transpose1 = std::make_shared<Transpose>(reshape1->output(0), Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            std::vector<float> weight(C, 1.0f);
            auto weight_const = Constant::create(ngraph::element::f32, Shape{1, C, 1, 1}, weight);
            auto conv = std::make_shared<Convolution>(transpose1->output(0),weight_const->output(0), Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
            auto transpose2 = std::make_shared<Transpose>(conv->output(0),
                Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape2 = std::make_shared<Reshape>(transpose2->output(0), Constant::create(element::i64, 
                Shape{reducesum_out_shape.size()}, reducesum_out_shape)->output(0), false);

            replace_node(reducesum, reshape2);

            return true;

        } else if (((axis_dim == 1) && (axis[0] == reducesum_in_shape.size() - 3))) {

            auto C = reducesum_in_shape[reducesum_in_shape.size() - 3];
            auto H = reducesum_in_shape[reducesum_in_shape.size() - 2];
            auto W = reducesum_in_shape[reducesum_in_shape.size() - 1];
            auto reshape1 = std::make_shared<Reshape>(parent,Constant::create(element::i64, Shape{4}, {1ull, 1ull, C, H * W})->output(0),false);
            //auto transpose1 = std::make_shared<Transpose>(reshape1->output(0), Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            std::vector<float> weight(C, 1.0f);
            auto weight_const = Constant::create(ngraph::element::f32, Shape{1, 1, C, 1}, weight);
            auto conv = std::make_shared<Convolution>(reshape1->output(0),weight_const->output(0), Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
            //auto transpose2 = std::make_shared<Transpose>(conv->output(0),Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape2 = std::make_shared<Reshape>(conv->output(0), Constant::create(element::i64, 
                Shape{reducesum_out_shape.size()}, reducesum_out_shape)->output(0), false);

            replace_node(reducesum, reshape2);

            return true;

        } else {

            return false;
        }
    };
    
    auto m = std::make_shared<pattern::Matcher>(reducesum_pattern, matcher_name);
    this->register_matcher(m, callback);
}
