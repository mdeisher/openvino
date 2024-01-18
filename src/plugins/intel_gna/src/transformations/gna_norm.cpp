// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This transformation for ReduceMean can be used while we wait for AvgPool support in the GNA plugin.
// Once AvgPool support is ready they the plugin can already convert ReduceMean to a pattern involving AvgPool.

#include "gna_norm.hpp"

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "gna_helper.hpp"

using namespace ov::opset12;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

// Implements normalization patterns using Convolutions
// Pattern:  --> ReduceMean --------\
//               --------------> Subtract --> Multiply 

GnaNormTransformation::GnaNormTransformation() {
    MATCHER_SCOPE(GnaNormTransformation);

    auto reshape1_pattern = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto transpose_pattern = pattern::wrap_type<Transpose>({reshape1_pattern, pattern::any_input()});
    auto reducemean1_pattern = pattern::wrap_type<AvgPool>({reshape1_pattern});
    auto reducemean2_pattern = pattern::wrap_type<AvgPool>({transpose_pattern});
    auto reducemean_pattern = std::make_shared<pattern::op::Or>(OutputVector{reducemean1_pattern, reducemean2_pattern});
    auto reshape2_pattern = pattern::wrap_type<Reshape>({reducemean_pattern, pattern::any_input()});
    auto reshape3_pattern = pattern::wrap_type<Reshape>({reshape2_pattern, pattern::any_input()});
    auto reshape2fq_pattern = pattern::wrap_type<FakeQuantize>({reshape2_pattern, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto reshape3fq_pattern = pattern::wrap_type<FakeQuantize>({reshape3_pattern, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto multiply1a_pattern = pattern::wrap_type<Multiply>({reshape2_pattern, pattern::any_input()});
    auto multiply1b_pattern = pattern::wrap_type<Multiply>({reshape3_pattern, pattern::any_input()});
    auto multiply1c_pattern = pattern::wrap_type<Multiply>({reshape2fq_pattern, pattern::any_input()});
    auto multiply1d_pattern = pattern::wrap_type<Multiply>({reshape3fq_pattern, pattern::any_input()});
    auto multiply1_pattern = std::make_shared<pattern::op::Or>(OutputVector{multiply1a_pattern, multiply1b_pattern, multiply1c_pattern, multiply1d_pattern});
    auto add_pattern = pattern::wrap_type<Add>({pattern::any_input(), multiply1_pattern});
    auto addfq_pattern = pattern::wrap_type<FakeQuantize>({add_pattern, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto multiply2a_pattern = pattern::wrap_type<Multiply>({add_pattern, pattern::any_input()});
    auto multiply2b_pattern = pattern::wrap_type<Multiply>({addfq_pattern, pattern::any_input()});
    auto multiply2_pattern = std::make_shared<pattern::op::Or>(OutputVector{multiply2a_pattern, multiply2b_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        bool use_fq = false;

        auto reshape2 = pattern_map.at(reshape2_pattern).get_node_shared_ptr();
        auto reducemean = reshape2->input_value(0).get_node_shared_ptr();
        auto children = reshape2->output(0).get_target_inputs();
        auto reshape3 = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (reshape3) {
            children = reshape3->output(0).get_target_inputs();
        }
        auto reshapefq = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto multiply1 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        if (reshapefq) {
            children = reshapefq->output(0).get_target_inputs();
            multiply1 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
            use_fq = true;
        }
        auto add = pattern_map.at(add_pattern).get_node_shared_ptr();
        children = add->output(0).get_target_inputs();
        auto addfq = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        if (addfq) {
            children = addfq->output(0).get_target_inputs();
            use_fq = true;
        } else {
            children = add->output(0).get_target_inputs();
        }
        auto multiply2 = std::dynamic_pointer_cast<Multiply>(children.begin()->get_node()->shared_from_this());
        children = multiply2->output(0).get_target_inputs();
        auto multiply2fq = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto transpose_in = std::dynamic_pointer_cast<Transpose>(reducemean->input_value(0).get_node_shared_ptr());
        if (multiply2fq) {
            children = multiply2fq->output(0).get_target_inputs();
            use_fq = true;
        }
        auto transpose_out = std::dynamic_pointer_cast<Transpose>(children.begin()->get_node()->shared_from_this());
        ov::Shape transpose_out_shape;
        Output<Node>& parent = reducemean->input_value(0);
        if (transpose_in) {
            if (!Is2DTranspose(transpose_in)) {
                return false;
            }
            if (transpose_out) {
                parent = transpose_in->input_value(0);
                transpose_out_shape = transpose_out->get_output_shape(0);
            }
        }
        ov::Shape reducemean_in_shape = parent.get_shape();
        ov::Shape reducemean_out_shape = reducemean->get_output_shape(0);
        auto avgpool = std::dynamic_pointer_cast<AvgPool>(reducemean);
        auto pool_shape = avgpool->get_kernel();
        if ((pool_shape[0] > 1) && (pool_shape[1] > 1)) {  // if pool is not a reduction abort
            return false;
        }
        size_t axis_dim = (reducemean_in_shape.size() - 2) + ((pool_shape[0] == 1) ? 1 : 0);
        const ov::Shape multiply2_out_shape = multiply2->get_output_shape(0);
        auto multiply2_const = std::dynamic_pointer_cast<Constant>(multiply2->input_value(1).get_node_shared_ptr());
        auto multiply2_const_fq = std::dynamic_pointer_cast<FakeQuantize>(multiply2->input_value(1).get_node_shared_ptr());
        if (multiply2_const_fq) {
            multiply2_const = std::dynamic_pointer_cast<Constant>(multiply2_const_fq->input_value(0).get_node_shared_ptr());
            use_fq = true;
        }
        if (multiply2_const == nullptr) {
            return false;
        }
        auto mul_weight_shape = multiply2_const->get_output_shape(0);
        size_t num_weights = 1;
        for (auto i = 0; i < mul_weight_shape.size(); i++) {
            num_weights *= mul_weight_shape[i];
        }
        if (num_weights > 1) {
            return false;
        }
        const float* mul_weight_ptr = multiply2_const->get_data_ptr<float>();
        float mul_weight = *mul_weight_ptr;

        // Reshape the problem to 2D
        ov::Shape new_shape;
        auto new_axis_dim = axis_dim;
        for (auto i = 0; i < reducemean_in_shape.size(); i++) {
            if ((reducemean_in_shape[i] == 1) && (i <= axis_dim)) {
                new_axis_dim--;
            } else {
                new_shape.push_back(reducemean_in_shape[i]);
            }
        }
        if ((new_shape.size() != 2) || (new_axis_dim != 0)) {
            return false;
        }
        if (transpose_in) {
            new_axis_dim = (new_axis_dim == 0) ? 1 : 0;
        }

        bool new_method = true;

        if (new_method && (new_shape[1] == 1)) {

            size_t N = 8; // number of kernels
            size_t C = new_shape[0];
            size_t H = 1;
            while (C > 1024) {  // 1024 is the maximum GNA input channels
                if (C % 2 == 0) {
                    H *= 2;
                    C /= 2;
                } else {
                    break;
                }
            }

            std::vector<float> avg_weights(N * H * C, -mul_weight / new_shape[0]);
            std::vector<float> identity(N * N, 0.0f);
            for (size_t i = 0; i < N; i++) {
                identity[i * N] = mul_weight;
            }
            auto avg_weights_const = InsertWeights(Shape{N, H, 1, C}, avg_weights, use_fq);
            auto identity_const = InsertWeights(Shape{N, 1, N, 1}, identity, use_fq);

            auto new_reshape = std::make_shared<Reshape>(parent,
                Constant::create(ngraph::element::i64, Shape{4}, {1ull, H, 1ull, C})->output(0),false);
            auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape->output(0), avg_weights_const->output(0),
                Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
            conv->set_friendly_name("NormAvg");
            ov::Output<ov::Node> upstream = conv->output(0); 
            if (reshapefq) {
                auto levels = reshapefq->get_levels();
                auto auto_broadcast = reshapefq->get_auto_broadcast();
                auto input_low = std::dynamic_pointer_cast<ngraph::op::Constant>(reshapefq->input_value(1).get_node_shared_ptr());
                auto input_high = std::dynamic_pointer_cast<ngraph::op::Constant>(reshapefq->input_value(2).get_node_shared_ptr());
                auto output_low = std::dynamic_pointer_cast<ngraph::op::Constant>(reshapefq->input_value(3).get_node_shared_ptr());
                auto output_high = std::dynamic_pointer_cast<ngraph::op::Constant>(reshapefq->input_value(4).get_node_shared_ptr());
                auto fq_dim = parent.get_shape().size();
                auto fq_shape = (fq_dim == 1) ? ov::Shape{1} : ((fq_dim==2) ? ov::Shape{1,1} : ((fq_dim==3) ? ov::Shape{1,1,1} : ov::Shape{1,1,1,1}));
                auto fq_type = parent.get_element_type();
                auto input_low_data = mul_weight * *input_low->get_data_ptr<float>();
                auto input_high_data = mul_weight * *input_high->get_data_ptr<float>();
                auto output_low_data = mul_weight * *output_low->get_data_ptr<float>();
                auto output_high_data = mul_weight * *output_high->get_data_ptr<float>();
                auto new_input_low = std::make_shared<Constant>(fq_type, fq_shape, input_low_data);
                auto new_input_high = std::make_shared<Constant>(fq_type, fq_shape, input_high_data);
                auto new_output_low = std::make_shared<Constant>(fq_type, fq_shape, output_low_data);
                auto new_output_high = std::make_shared<Constant>(fq_type, fq_shape, output_high_data);
                auto conv_fq = std::make_shared<FakeQuantize>(conv->output(0), new_input_low->output(0), new_input_high->output(0), 
                    new_output_low->output(0), new_output_high->output(0), levels, auto_broadcast);
                upstream = conv_fq->output(0);
            }
            new_reshape = std::make_shared<Reshape>(parent,
                Constant::create(ngraph::element::i64, Shape{4}, {1ull, H, C, 1ull})->output(0),false);
            conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(new_reshape->output(0), identity_const->output(0), upstream,
                Strides{1, N},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
            conv->set_friendly_name("NormBroadcast");
            upstream = conv->output(0);
            std::shared_ptr<ov::Node> outnode = multiply2;
            if (multiply2fq) {
                upstream = InsertOutputFQ(upstream, multiply2fq);
                outnode = multiply2fq;
            }
            new_reshape = std::make_shared<Reshape>(upstream,
                Constant::create(ngraph::element::i64, Shape{multiply2_out_shape.size()}, multiply2_out_shape)->output(0), false);

            replace_node_update_name(outnode, new_reshape);
            return true;

        } else if (new_shape[1] == 1) {  // problem is 1D

            size_t N = 1; // number of kernels
            size_t C = new_shape[0];
            size_t H = 1;
            while (C > 1024) {  // 1024 is the maximum GNA input channels
                if (C % 2 == 0) {
                    H *= 2;
                    C /= 2;
                } else {
                    break;
                }
            }

            std::vector<float> avg_weights(N * H * C, 1.0f / new_shape[0]);
            std::vector<float> avg_broadcast(N * new_shape[0], 0.0f);
            for (size_t i = 0; i < new_shape[0]; i++) {
                avg_broadcast[i * N] = 1.0f;
            }
            auto avg_weights_const = InsertWeights(Shape{N, H, 1, C}, avg_weights, use_fq);
            auto avg_broadcast_const = InsertWeights(Shape{new_shape[0], 1, 1, N}, avg_broadcast, use_fq);

            auto reshape_4d = std::make_shared<Reshape>(parent,
                Constant::create(ngraph::element::i64, Shape{4}, {1ull, H, 1ull, C})->output(0),false);
            auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_weights_const->output(0),
                Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
            conv->set_friendly_name("NormAvg");
            if (reshapefq) {
                auto conv_fq = CopyFQ(conv, reshapefq);
                reshape_4d = std::make_shared<Reshape>(conv_fq->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, N})->output(0),false);
            } else {
                reshape_4d = std::make_shared<Reshape>(conv->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, N})->output(0),false);
            }
            conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_broadcast_const->output(0),
                Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
            conv->set_friendly_name("NormBroadcast");
            auto reshape_in = std::make_shared<Reshape>(parent, Constant::create(element::i64, Shape{2}, {1ull, new_shape[0]})->output(0), false);
            std::shared_ptr<ov::op::v1::Reshape> reshape_sum = nullptr;
            if (reshapefq) {
                auto conv_fq = CopyFQ(conv, reshapefq);
                reshape_sum = std::make_shared<Reshape>(conv_fq->output(0), 
                    Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0]})->output(0),false);
            } else {
                reshape_sum = std::make_shared<Reshape>(conv->output(0), 
                    Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0]})->output(0),false);
            }
            OutputVector parts;
            parts.push_back(reshape_in->output(0));
            parts.push_back(reshape_sum->output(0));
            auto concat = std::make_shared<Concat>(parts, 0);

            ov::Output<ov::Node> upstream = concat->output(0);
            if (addfq) {
                upstream = InsertOutputFQ(concat->output(0), addfq);
            }

#define NUM_K 32
            auto transpose = std::make_shared<Transpose>(upstream, Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
            size_t num_K = NUM_K;
            while ((2 * new_shape[0]) % num_K != 0) {  // reduce parallelism if sizes don't match
                num_K = num_K / 2;
            }
            if (num_K != NUM_K) {
                printf("Warning:  failed to optimize parallel in norm\n");
            }
            auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose->output(0),
                Constant::create(element::i64, Shape{4}, {1ull, new_shape[0] / num_K, 2 * num_K, 1ull})->output(0),false);
            std::vector<float> weight(num_K * 2 * num_K, 0.0);
            for (size_t i = 0; i < num_K; i++) {
                weight[i * num_K * 2 + i * 2 + 0] = mul_weight;
                weight[i * num_K * 2 + i * 2 + 1] = -mul_weight;
            }
            auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, 2 * num_K, 1}, weight);
            if (multiply2_const_fq) {
                auto weight_fq = CopyFQ(weight_const, multiply2_const_fq);
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0),weight_fq->output(0), 
                    Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
            } else {
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0),weight_const->output(0), 
                    Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
            }
            std::shared_ptr<ov::Node> outnode = multiply2;
            upstream = conv->output(0);
            if (multiply2fq) {
                upstream = InsertOutputFQ(upstream, multiply2fq);
                outnode = multiply2fq;
            }
            auto subtract_mean = std::make_shared<ov::op::v1::Reshape>(upstream,
                Constant::create(element::i64, Shape{2}, {1ull, new_shape[0]})->output(0),false);
            subtract_mean->set_friendly_name("NormSubMean");
            if (multiply2_out_shape.size() == 2) {
                replace_node(outnode, subtract_mean);
            } else if (multiply2_out_shape.size() == 3) {
                reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                    Constant::create(element::i64, Shape{3}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2]})->output(0),false);
                replace_node(outnode, reshape);
            } else {
                reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                    Constant::create(element::i64, Shape{4}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2],multiply2_out_shape[3]})->output(0),false);
                replace_node(outnode, reshape);
            }
            return true;
        
        } else {

            if (new_axis_dim == 0) {

                auto H_padded = new_shape[0];
                if ((H_padded % 8) != 0) {
                    H_padded += 8 - (H_padded % 8);
                }
                std::vector<float> avg_weights(new_shape[0], 1.0f / new_shape[0]);
                std::vector<float> avg_broadcast(H_padded, 1.0f);
                auto avg_weights_const = Constant::create(ngraph::element::f32, Shape{1, new_shape[0], 1, 1}, avg_weights);
                auto avg_broadcast_const = Constant::create(ngraph::element::f32, Shape{H_padded, 1, 1, 1}, avg_broadcast);

                auto reshape_4d = std::make_shared<Reshape>(parent,
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, new_shape[0], new_shape[1], 1ull})->output(0),false);
                auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_weights_const->output(0),
                    Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
                conv->set_friendly_name("NormAvg");
                reshape_4d = std::make_shared<Reshape>(conv->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, new_shape[1], 1ull})->output(0),false);
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_broadcast_const->output(0),
                    Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
                conv->set_friendly_name("NormBroadcast");
                auto upstream = conv->output(0);
                if (new_shape[0] != H_padded) {
                    auto reshape = std::make_shared<Reshape>(upstream, Constant::create(element::i64, Shape{2}, {new_shape[1], H_padded})->output(0), false);
                    auto transpose = std::make_shared<Transpose>(reshape->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
                    std::vector<size_t> split_lengths;
                    split_lengths.push_back(new_shape[0]);
                    split_lengths.push_back(H_padded - new_shape[0]);
                    const auto split_lengths_const = Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data());
                    auto split = std::make_shared<VariadicSplit>(transpose->output(0), Constant::create(ov::element::i64, ov::Shape{}, {0}), split_lengths_const);
                    upstream = split->output(0);
                }
                auto reshape_in = std::make_shared<Reshape>(parent, Constant::create(element::i64, Shape{2}, {1ull, new_shape[0]*new_shape[1]})->output(0), false);
                auto reshape_sum = std::make_shared<Reshape>(upstream, Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0]*new_shape[1]})->output(0),false);
                OutputVector parts;
                parts.push_back(reshape_in->output(0));
                parts.push_back(reshape_sum->output(0));
                auto concat = std::make_shared<Concat>(parts, 0);

    #define NUM_K 32
                auto transpose = std::make_shared<Transpose>(concat->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
                size_t num_K = NUM_K;
                while ((2 * new_shape[0] * new_shape[1]) % num_K != 0) {  // reduce parallelism if sizes don't match
                    num_K = num_K / 2;
                }
                if (num_K != NUM_K) {
                    printf("Warning:  failed to optimize parallel in norm\n");
                }
                auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose->output(0),
                    Constant::create(element::i64, Shape{4}, {1ull, new_shape[0] * new_shape[1] / num_K, 2 * num_K, 1ull})->output(0),false);
                std::vector<float> weight(num_K * 2 * num_K, 0.0);
                for (size_t i = 0; i < num_K; i++) {
                    weight[i * num_K * 2 + i * 2 + 0] = mul_weight;
                    weight[i * num_K * 2 + i * 2 + 1] = -mul_weight;
                }
                auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, 2 * num_K, 1}, weight);
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0),weight_const->output(0), 
                    Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
                auto subtract_mean = std::make_shared<ov::op::v1::Reshape>(conv->output(0),
                    Constant::create(element::i64, Shape{2}, {1ull, new_shape[0] * new_shape[1]})->output(0),false);
                subtract_mean->set_friendly_name("NormSubMean");
                if (multiply2_out_shape.size() == 2) {
                    replace_node(multiply2, subtract_mean);
                } else if (multiply2_out_shape.size() == 3) {
                    reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                        Constant::create(element::i64, Shape{3}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2]})->output(0),false);
                    replace_node(multiply2, reshape);
                } else {
                    reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                        Constant::create(element::i64, Shape{4}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2],multiply2_out_shape[3]})->output(0),false);
                    replace_node(multiply2, reshape);
                }

            } else {

                std::vector<float> avg_weights(new_shape[1], 1.0f / new_shape[1]);
                std::vector<float> avg_broadcast(new_shape[1], 1.0f);
                auto avg_weights_const = Constant::create(ngraph::element::f32, Shape{1, 1, new_shape[1], 1}, avg_weights);
                auto avg_broadcast_const = Constant::create(ngraph::element::f32, Shape{new_shape[1], 1, 1, 1}, avg_broadcast);

                auto reshape_4d = std::make_shared<Reshape>(parent,
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, new_shape[0], new_shape[1], 1ull})->output(0),false);
                auto conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_weights_const->output(0),
                    Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
                conv->set_friendly_name("NormAvg");
                reshape_4d = std::make_shared<Reshape>(conv->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {1ull, new_shape[0], 1ull, 1ull})->output(0),false);
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape_4d->output(0), avg_broadcast_const->output(0),
                    Strides{1, 1},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},ov::op::PadType::VALID);
                conv->set_friendly_name("NormBroadcast");
                auto upstream = conv->output(0);
                auto reshape_in = std::make_shared<Reshape>(parent, Constant::create(element::i64, Shape{2}, {1ull, new_shape[0]*new_shape[1]})->output(0), false);
                auto reshape_sum = std::make_shared<Reshape>(upstream, Constant::create(ngraph::element::i64, Shape{2}, {1ull, new_shape[0]*new_shape[1]})->output(0),false);
                OutputVector parts;
                parts.push_back(reshape_in->output(0));
                parts.push_back(reshape_sum->output(0));
                auto concat = std::make_shared<Concat>(parts, 0);

    #define NUM_K 32
                auto transpose = std::make_shared<Transpose>(concat->output(0), Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
                size_t num_K = NUM_K;
                while ((2 * new_shape[0] * new_shape[1]) % num_K != 0) {  // reduce parallelism if sizes don't match
                    num_K = num_K / 2;
                }
                if (num_K != NUM_K) {
                    printf("Warning:  failed to optimize parallel in norm\n");
                }
                auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose->output(0),
                    Constant::create(element::i64, Shape{4}, {1ull, new_shape[0] * new_shape[1] / num_K, 2 * num_K, 1ull})->output(0),false);
                std::vector<float> weight(num_K * 2 * num_K, 0.0);
                for (size_t i = 0; i < num_K; i++) {
                    weight[i * num_K * 2 + i * 2 + 0] = mul_weight;
                    weight[i * num_K * 2 + i * 2 + 1] = -mul_weight;
                }
                auto weight_const = Constant::create(ngraph::element::f32, Shape{num_K, 1, 2 * num_K, 1}, weight);
                conv = std::make_shared<ov::intel_gna::op::GNAConvolution>(reshape->output(0),weight_const->output(0), 
                    Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1}, ov::op::PadType::VALID );
                upstream = conv->output(0);
                if (transpose_out) {
                    auto reshape = std::make_shared<ov::op::v1::Reshape>(upstream,
                        Constant::create(element::i64, Shape{2}, {new_shape[0], new_shape[1]})->output(0),false);
                    auto transpose = std::make_shared<Transpose>(reshape->output(0), 
                        Constant::create(ov::element::Type_t::i64, ov::Shape{2}, {1,0}));
                    upstream = transpose->output(0);
                }
                auto subtract_mean = std::make_shared<ov::op::v1::Reshape>(upstream,
                    Constant::create(element::i64, Shape{2}, {1ull, new_shape[0] * new_shape[1]})->output(0),false);
                subtract_mean->set_friendly_name("NormSubMean");
                if (transpose_out) {
                    if (transpose_out_shape.size() == 2) {
                        replace_node(transpose_out, subtract_mean);
                    } else if (transpose_out_shape.size() == 3) {
                        reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                            Constant::create(element::i64, Shape{3}, {transpose_out_shape[0], transpose_out_shape[1],transpose_out_shape[2]})->output(0),false);
                        replace_node(transpose_out, reshape);
                    } else {
                        reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                            Constant::create(element::i64, Shape{4}, {transpose_out_shape[0], transpose_out_shape[1],transpose_out_shape[2],transpose_out_shape[3]})->output(0),false);
                        replace_node(transpose_out, reshape);
                    }
                } else {
                    if (multiply2_out_shape.size() == 2) {
                        replace_node(multiply2, subtract_mean);
                    } else if (multiply2_out_shape.size() == 3) {
                        reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                            Constant::create(element::i64, Shape{3}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2]})->output(0),false);
                        replace_node(multiply2, reshape);
                    } else {
                        reshape = std::make_shared<ov::op::v1::Reshape>(subtract_mean->output(0), 
                            Constant::create(element::i64, Shape{4}, {multiply2_out_shape[0], multiply2_out_shape[1],multiply2_out_shape[2],multiply2_out_shape[3]})->output(0),false);
                        replace_node(multiply2, reshape);
                    }
                }
            }

            return true;
        }
    };
    
    auto m = std::make_shared<pattern::Matcher>(multiply2_pattern, matcher_name);
    this->register_matcher(m, callback);
}

