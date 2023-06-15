// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_put_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {
Output<Node> generate_zeros_with_convertlike(const Output<Node> sizes, const Output<Node> tensor_of_type) {
    auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto zeros = std::make_shared<v3::Broadcast>(const_0, sizes);
    return std::make_shared<v1::ConvertLike>(zeros, tensor_of_type);
}
}  // namespace

AtenIndexPutReplacer::AtenIndexPutReplacer() {
    auto index_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto index_op = cast_fw_node(m.get_match_root(), "aten::index_put_");
        if (!index_op) {
            return false;
        }
        NodeVector rt_copy_from{index_op};
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
        auto const_max_int = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});

        auto input = index_op->input_value(0);
        auto input_shape = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto indices = index_op->input_value(1);
        auto values = index_op->input_value(2);
        auto acc_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(index_op->input_value(3).get_node_shared_ptr());
        if (!acc_const) {
            add_exception_to_fw_node(index_op, "aten::index_put_: non constant accumulate input is not supported.");
            return false;
        }
        bool accumulate = acc_const->cast_vector<bool>()[0];

        int64_t indices_list_len;
        OutputVector indices_inputs;
        if (auto listconstruct = cast_fw_node(indices.get_node_shared_ptr(), "prim::ListConstruct")) {
            rt_copy_from.push_back(listconstruct);
            indices_inputs = listconstruct->input_values();
            indices_list_len = indices_inputs.size();
        } else {
            auto indices_partial_shape = indices.get_partial_shape();
            if (!indices_partial_shape.rank().is_static()) {
                // "We support only indices with static rank."
                add_exception_to_fw_node(index_op, "aten::index_put_: dynamic rank for indices is not supported.");
                return false;
            }
            auto indices_first_dim = indices_partial_shape[0];
            if (!indices_first_dim.is_static()) {
                // We support only lists of tensors with static number of elements.
                add_exception_to_fw_node(index_op,
                                         "aten::index_put_: dynamic dynamic number of indices is not supported.");
                return false;
            }
            indices_list_len = indices_first_dim.get_length();
            auto split = std::make_shared<v1::Split>(indices, const_0, indices_list_len);
            indices_inputs = split->outputs();
        }

        if (indices_list_len == 0) {
            copy_runtime_info(rt_copy_from, values.get_node_shared_ptr());
            replace_node(index_op, values.get_node_shared_ptr());
            return true;
        }

        auto const_indices_list_len = v0::Constant::create(element::i32, Shape{1}, {indices_list_len});

        std::shared_ptr<Node> broadcast_index_shape;
        Output<Node> index;
        if (indices_list_len > 1) {
            index = indices_inputs[0];
            for (int i = 1; i < indices_list_len; i++) {
                index = std::make_shared<v1::Add>(index, indices_inputs[i]);
            }
            broadcast_index_shape = std::make_shared<v3::ShapeOf>(index, element::i32);
            OutputVector indices_list;
            for (int i = 0; i < indices_list_len; i++) {
                auto broadcast = std::make_shared<v3::Broadcast>(indices_inputs[i], broadcast_index_shape);
                auto unsqueeze = std::make_shared<v0::Unsqueeze>(broadcast, const_neg_1);

                // change negative indices to positive indices
                auto const_i = v0::Constant::create(element::i32, Shape{}, {i});
                auto dim_i = std::make_shared<v8::Gather>(input_shape, const_i, const_0);
                auto dim_i_correct_type = std::make_shared<v1::ConvertLike>(dim_i, index);
                auto unsqueeze_add = std::make_shared<v1::Add>(unsqueeze, dim_i_correct_type);
                auto unsqueeze_add_mod = std::make_shared<v1::Mod>(unsqueeze_add, dim_i_correct_type);

                indices_list.push_back(unsqueeze_add_mod);
            }
            index = std::make_shared<v0::Concat>(indices_list, -1);
        } else {
            index = indices_inputs[0];
            // change negative indices to positive indices
            auto dim_0 = (std::make_shared<v8::Gather>(input_shape, const_0, const_0));
            auto dim_0_correct_type = (std::make_shared<v1::ConvertLike>(dim_0, index));
            index = std::make_shared<v1::Add>(index, dim_0_correct_type);
            index = std::make_shared<v1::Mod>(index, dim_0_correct_type);

            broadcast_index_shape = std::make_shared<v3::ShapeOf>(index, element::i32);
            index = std::make_shared<v0::Unsqueeze>(index, const_neg_1);
        }

        auto sub_data_shape = std::make_shared<v8::Slice>(input_shape, const_indices_list_len, const_max_int, const_1);
        auto values_shape = std::make_shared<v0::Concat>(OutputVector{broadcast_index_shape, sub_data_shape}, 0);
        values = std::make_shared<v3::Broadcast>(values, values_shape);
        values = std::make_shared<v1::ConvertLike>(values, input);

        std::shared_ptr<ov::Node> result;
        if (accumulate) {
            auto zeros = generate_zeros_with_convertlike(input_shape, input);
            auto scatter = std::make_shared<v3::ScatterNDUpdate>(zeros, index, values);
            result = std::make_shared<v1::Add>(input, scatter);
        } else {
            result = std::make_shared<v3::ScatterNDUpdate>(input, index, values);
        }
        copy_runtime_info(rt_copy_from, result);
        replace_node(index_op, result);
        result->set_friendly_name(index_op->get_friendly_name());
        return true;
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(index_op, "ov::frontend::pytorch::pass::AtenIndexPutReplacer");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
