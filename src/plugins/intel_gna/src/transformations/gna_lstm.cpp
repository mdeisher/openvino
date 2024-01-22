// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/ngraph/itt.hpp"
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset4.hpp>
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <transformations/utils/utils.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "gna_lstm.hpp"

using namespace ov::opset1;
using namespace ov::opset4;
using namespace ov::pass;


bool GnaOldLstmDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto lstm_cell_v1 = std::dynamic_pointer_cast<ov::opset1::LSTMCell>(node);
        auto lstm_cell_v4 = std::dynamic_pointer_cast<ov::opset4::LSTMCell>(node);
        if ((nullptr == lstm_cell_v1) && (nullptr == lstm_cell_v4)) {
            continue;
        }

        const ov::Output<ov::Node>& X = (lstm_cell_v1) ? lstm_cell_v1->input_value(0) : lstm_cell_v4->input_value(0);
        const ov::Output<ov::Node>& H_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(1) : lstm_cell_v4->input_value(1);
        const ov::Output<ov::Node>& C_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(2) : lstm_cell_v4->input_value(2);
        const ov::Output<ov::Node>& W = (lstm_cell_v1) ? lstm_cell_v1->input_value(3) : lstm_cell_v4->input_value(3);
        const ov::Output<ov::Node>& R = (lstm_cell_v1) ? lstm_cell_v1->input_value(4) : lstm_cell_v4->input_value(4);
        const ov::Output<ov::Node>& bias = (lstm_cell_v1) ? lstm_cell_v1->input_value(5) : lstm_cell_v4->input_value(5);
        auto clip = (lstm_cell_v1) ? lstm_cell_v1->get_clip() : lstm_cell_v4->get_clip();

        // Xt*(W^T)
        auto Xt_W = std::make_shared<ov::op::v0::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        auto Ht_R = std::make_shared<ov::op::v0::MatMul>(H_t, R, false, true);
        // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
        auto add = std::make_shared<ov::op::v1::Add>(Ht_R, bias);
        auto XHB = std::make_shared<ov::op::v1::Add>(Xt_W, add);

        auto axis_node = ov::op::v0::Constant::create(element::u64, Shape{}, {1});
        auto split = std::make_shared<ov::op::v1::Split>(XHB, axis_node, 4);
        Output<Node> f = split->output(0);
        Output<Node> i = split->output(1);
        Output<Node> c = split->output(2);
        Output<Node> o = split->output(3);

        if (clip > 0.f) {
            auto clamp_f = std::make_shared<ov::op::v0::Clamp>(f, -clip, clip);
            auto clamp_i = std::make_shared<ov::op::v0::Clamp>(i, -clip, clip);
            auto clamp_c = std::make_shared<ov::op::v0::Clamp>(c, -clip, clip);
            auto clamp_o = std::make_shared<ov::op::v0::Clamp>(o, -clip, clip);
            f = clamp_f;
            i = clamp_i;
            c = clamp_c;
            o = clamp_o;
            if (lstm_cell_v1) {
                ov::copy_runtime_info(lstm_cell_v1, {clamp_f, clamp_i, clamp_c, clamp_o});
            } else {
                ov::copy_runtime_info(lstm_cell_v4, {clamp_f, clamp_i, clamp_c, clamp_o});
            }
        }

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
        auto f_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], f) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], f);
        auto i_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], i) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], i);
        auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], c) : ov::op::util::activation(lstm_cell_v4->get_activations()[1], c);
        auto o_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], o) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], o);

        // Ct = ft (.) Ct-1 + it (.) ct
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(f_t, C_t);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(i_t, c_t);
        auto out_C = std::make_shared<ov::op::v1::Add>(mul1, mul2);

        // H = ot (.) h(Ct)
        auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C) : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
        auto out_H = std::make_shared<ov::op::v1::Multiply>(o_t, hC);

        if (lstm_cell_v1) {
            out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
            out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
            ov::copy_runtime_info(lstm_cell_v1,
                {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
            ov::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
        } else {
            out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
            out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
            ov::copy_runtime_info(lstm_cell_v4,
                {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
            ov::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
        }
        is_graph_modfied = true;
    };

    return is_graph_modfied;
}

bool GnaOldStackedLstmDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto lstm_cell_v1 = std::dynamic_pointer_cast<ov::opset1::LSTMCell>(node);
        auto lstm_cell_v4 = std::dynamic_pointer_cast<ov::opset4::LSTMCell>(node);
        if ((nullptr == lstm_cell_v1) && (nullptr == lstm_cell_v4)) {
            continue;
        }

        const ov::Output<ov::Node>& X = (lstm_cell_v1) ? lstm_cell_v1->input_value(0) : lstm_cell_v4->input_value(0);
        const ov::Output<ov::Node>& H_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(1) : lstm_cell_v4->input_value(1);
        const ov::Output<ov::Node>& C_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(2) : lstm_cell_v4->input_value(2);
        const ov::Output<ov::Node>& W = (lstm_cell_v1) ? lstm_cell_v1->input_value(3) : lstm_cell_v4->input_value(3);
        const ov::Output<ov::Node>& R = (lstm_cell_v1) ? lstm_cell_v1->input_value(4) : lstm_cell_v4->input_value(4);
        const ov::Output<ov::Node>& bias = (lstm_cell_v1) ? lstm_cell_v1->input_value(5) : lstm_cell_v4->input_value(5);
        auto clip = (lstm_cell_v1) ? lstm_cell_v1->get_clip() : lstm_cell_v4->get_clip();

        OutputVector parts;
        parts.push_back(X);
        parts.push_back(H_t);
        auto XHt = std::make_shared<ov::op::v0::Concat>(parts, 1);

        auto W_shape = W.get_shape();
        auto R_shape = R.get_shape();
        if (W_shape[0] != R_shape[0]) {
            break;
        }
        auto W_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(W.get_node_shared_ptr());
        auto R_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(R.get_node_shared_ptr());
        if ((W_const == nullptr) || (R_const == nullptr)) {
            break;
        }
        auto W_data = W_const->get_data_ptr<float>();
        auto R_data = R_const->get_data_ptr<float>();
        std::vector<float> WR_vector(W_shape[0] * (W_shape[1] + R_shape[1]), 0.0f);
        float* WR_data = WR_vector.data();
        for (auto i = 0; i < W_shape[0]; i++) {
            for (auto j = 0; j < W_shape[1]; j++) {
                WR_data[i * (W_shape[1] + R_shape[1]) + j] = W_data[i * W_shape[1] + j];
            }
            for (auto j = 0; j < R_shape[1]; j++) {
                WR_data[i * (W_shape[1] + R_shape[1]) + W_shape[1] + j] = R_data[i * R_shape[1] + j];
            }
        }
        auto WR_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{W_shape[0], W_shape[1] + R_shape[1]}, WR_data);
        auto WR = WR_const->output(0);

        // [Xt / H_t]*([W R]^T)
        auto XHt_WR = std::make_shared<ov::op::v0::MatMul>(XHt, WR, false, true);
        // Xt*(W^T)
        //auto Xt_W = std::make_shared<ov::op::v0::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        //auto Ht_R = std::make_shared<ov::op::v0::MatMul>(H_t, R, false, true);
        // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
        //auto add = std::make_shared<ov::op::v1::Add>(Ht_R, bias);
        //auto XHB = std::make_shared<ov::op::v1::Add>(Xt_W, add);
        auto XHB= std::make_shared<ov::op::v1::Add>(XHt_WR, bias);

        auto axis_node = ov::op::v0::Constant::create(element::u64, Shape{}, {1});
        auto split = std::make_shared<ov::op::v1::Split>(XHB, axis_node, 4);
        Output<Node> f = split->output(0);
        Output<Node> i = split->output(1);
        Output<Node> c = split->output(2);
        Output<Node> o = split->output(3);

        if (clip > 0.f) {
            auto clamp_f = std::make_shared<ov::op::v0::Clamp>(f, -clip, clip);
            auto clamp_i = std::make_shared<ov::op::v0::Clamp>(i, -clip, clip);
            auto clamp_c = std::make_shared<ov::op::v0::Clamp>(c, -clip, clip);
            auto clamp_o = std::make_shared<ov::op::v0::Clamp>(o, -clip, clip);
            f = clamp_f;
            i = clamp_i;
            c = clamp_c;
            o = clamp_o;
            if (lstm_cell_v1) {
                ov::copy_runtime_info(lstm_cell_v1, {clamp_f, clamp_i, clamp_c, clamp_o});
            } else {
                ov::copy_runtime_info(lstm_cell_v4, {clamp_f, clamp_i, clamp_c, clamp_o});
            }
        }

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
        auto f_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], f) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], f);
        auto i_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], i) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], i);
        auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], c) : ov::op::util::activation(lstm_cell_v4->get_activations()[1], c);
        auto o_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], o) : ov::op::util::activation(lstm_cell_v4->get_activations()[0], o);
        // Ct = ft (.) Ct-1 + it (.) ct
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(f_t, C_t);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(i_t, c_t);
        auto out_C = std::make_shared<ov::op::v1::Add>(mul1, mul2);

        // H = ot (.) h(Ct)
        auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C) : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
        auto out_H = std::make_shared<ov::op::v1::Multiply>(o_t, hC);

        if (lstm_cell_v1) {
            out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
            out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
            //ov::copy_runtime_info(lstm_cell_v1,
            //    {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
            ov::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
        } else {
            out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
            out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
            //ov::copy_runtime_info(lstm_cell_v4,
            //    {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
            ov::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
        }
        is_graph_modfied = true;
    };

    return is_graph_modfied;
}


bool gna_optimized = true;
bool use_memory_layers = false;

bool GnaLstmDecompositionSlower::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto lstm_cell_v1 = std::dynamic_pointer_cast<ov::opset1::LSTMCell>(node);
        auto lstm_cell_v4 = std::dynamic_pointer_cast<ov::opset4::LSTMCell>(node);
        if ((nullptr == lstm_cell_v1) && (nullptr == lstm_cell_v4)) {
            continue;
        }

        const ov::Output<ov::Node>& X = (lstm_cell_v1) ? lstm_cell_v1->input_value(0) : lstm_cell_v4->input_value(0);
        const ov::Output<ov::Node>& H_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(1) : lstm_cell_v4->input_value(1);
        const ov::Output<ov::Node>& C_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(2) : lstm_cell_v4->input_value(2);
        const ov::Output<ov::Node>& W = (lstm_cell_v1) ? lstm_cell_v1->input_value(3) : lstm_cell_v4->input_value(3);
        const ov::Output<ov::Node>& R = (lstm_cell_v1) ? lstm_cell_v1->input_value(4) : lstm_cell_v4->input_value(4);
        const ov::Output<ov::Node>& bias = (lstm_cell_v1) ? lstm_cell_v1->input_value(5) : lstm_cell_v4->input_value(5);
        auto clip = (lstm_cell_v1) ? lstm_cell_v1->get_clip() : lstm_cell_v4->get_clip();


        if (gna_optimized && (clip == 0.0f)) {

            ov::OutputVector upstream;
            upstream.push_back(X);
            auto R_const = std::dynamic_pointer_cast<Constant>(R.get_node_shared_ptr());
            auto W_const = std::dynamic_pointer_cast<Constant>(W.get_node_shared_ptr());
            auto bias_const = std::dynamic_pointer_cast<Constant>(bias.get_node_shared_ptr());
            auto R_shape = R_const->output(0).get_shape();
            auto W_shape = W_const->output(0).get_shape();
            auto X_shape = X.get_shape();
            auto N = X_shape[0];
            auto num_cells = R_shape[0] / 4;
            const float* R_ptr = R_const->get_data_ptr<float>();
            const float* W_ptr = W_const->get_data_ptr<float>();
            const float* bias_ptr = bias_const->get_data_ptr<float>();
            std::vector<float> R_fio(3 * num_cells * R_shape[1], 0.0f);
            std::vector<float> R_c(num_cells * R_shape[1], 0.0f);
            std::vector<float> W_fio(3 * num_cells * W_shape[1], 0.0f);
            std::vector<float> W_c(num_cells * W_shape[1], 0.0f);
            std::vector<float> bias_fio(3 * num_cells, 0.0f);
            std::vector<float> bias_c(num_cells, 0.0f);
            float* R_fio_ptr = R_fio.data();
            float* R_c_ptr = R_c.data();
            float* W_fio_ptr = W_fio.data();
            float* W_c_ptr = W_c.data();
            float* bias_fio_ptr = bias_fio.data();
            float* bias_c_ptr = bias_c.data();
            // reorder weights to reduce elementwise operations
            auto num_R = num_cells * R_shape[1];
            auto num_W = num_cells * W_shape[1];
            auto num_bias = num_cells;
            memcpy(R_fio_ptr + 0 * num_R, R_ptr + 0 * num_R, num_R * sizeof(float));  // f
            memcpy(W_fio_ptr + 0 * num_W, W_ptr + 0 * num_W, num_W * sizeof(float));  // f
            memcpy(bias_fio_ptr + 0 * num_bias, bias_ptr + 0 * num_bias, num_bias * sizeof(float));  // f
            memcpy(R_fio_ptr + 1 * num_R, R_ptr + 1 * num_R, num_R * sizeof(float));  // i
            memcpy(W_fio_ptr + 1 * num_W, W_ptr + 1 * num_W, num_W * sizeof(float));  // i
            memcpy(bias_fio_ptr + 1 * num_bias, bias_ptr + 1 * num_bias, num_bias * sizeof(float));  // i
            memcpy(R_c_ptr + 0 * num_R, R_ptr + 2 * num_R, num_R * sizeof(float));   // c
            memcpy(W_c_ptr + 0 * num_W, W_ptr + 2 * num_W, num_W * sizeof(float));   // c
            memcpy(bias_c_ptr + 0 * num_bias, bias_ptr + 2 * num_bias, num_bias * sizeof(float));    // c
            memcpy(R_fio_ptr + 2 * num_R, R_ptr + 3 * num_R, num_R * sizeof(float));  // o
            memcpy(W_fio_ptr + 2 * num_W, W_ptr + 3 * num_W, num_W * sizeof(float));  // o
            memcpy(bias_fio_ptr + 2 * num_bias, bias_ptr + 3 * num_bias, num_bias * sizeof(float));  // o
            auto R_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells, R_shape[1]}, R_fio);
            auto R_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells, R_shape[1]}, R_c);
            auto W_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells, W_shape[1]}, W_fio);
            auto W_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells, W_shape[1]}, W_c);
            auto bias_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells}, bias_fio);
            auto bias_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells}, bias_c);
            // Xt*(W^T)
            auto Xt_W_fio = std::make_shared<ov::opset4::MatMul>(X, W_fio_const->output(0), false, true);
            auto Xt_W_c = std::make_shared<ov::opset4::MatMul>(X, W_c_const->output(0), false, true);
            // Ht-1*(R^T)
            const std::string h_var_name(H_t.get_node()->get_friendly_name() + "-mem");
            auto h_var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, h_var_name});
            if (use_memory_layers) {
                auto readvalue_h = std::make_shared<ov::opset11::ReadValue>(H_t, h_var);
                upstream[0] = readvalue_h->output(0);
            } else {
                upstream[0] = H_t;
            }
            auto Ht_R_fio = std::make_shared<ov::opset4::MatMul>(upstream[0], R_fio_const->output(0), false, true);
            auto Ht_R_c = std::make_shared<ov::opset4::MatMul>(H_t, R_c_const->output(0), false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            auto add_fio = std::make_shared<ov::opset4::Add>(Ht_R_fio, bias_fio_const->output(0));
            auto add_c = std::make_shared<ov::opset4::Add>(Ht_R_c, bias_c_const->output(0));
            auto XHB_fio = std::make_shared<ov::opset4::Add>(Xt_W_fio, add_fio);
            auto XHB_c = std::make_shared<ov::opset4::Add>(Xt_W_c, add_c);
            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto fio_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], XHB_fio)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], XHB_fio);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], XHB_c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], XHB_c);

            const auto split_lengths_const = ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {2 * num_cells, num_cells});
            auto split1 = std::make_shared<ov::opset1::VariadicSplit>(fio_t, 
                ov::opset4::Constant::create(ov::element::i64, ov::Shape{}, {1}), split_lengths_const);
            ov::Output<ov::Node> fi_t = split1->output(0);
            ov::Output<ov::Node> o_t = split1->output(1);

            auto fi_t_1D = std::make_shared<ov::opset4::Reshape>(fi_t, ov::opset4::Constant::create(element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);
            const std::string c_var_name(C_t.get_node()->get_friendly_name() + "-mem");
            auto c_var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, c_var_name});
            if (use_memory_layers) {
                auto readvalue_c = std::make_shared<ov::opset11::ReadValue>(C_t, c_var);
                upstream[0] = readvalue_c->output(0);
            } else {
                upstream[0] = C_t;
            }
            ov::OutputVector Cc_t;
            Cc_t.push_back(upstream[0]);
            Cc_t.push_back(c_t);
            auto concat2 = std::make_shared<ov::opset4::Concat>(Cc_t, 1);
            auto concat2_1D = std::make_shared<ov::opset4::Reshape>(concat2->output(0), ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul12_1D = std::make_shared<ov::opset4::Multiply>(fi_t_1D->output(0), concat2_1D->output(0));
            auto mul12 = std::make_shared<ov::opset4::Reshape>(mul12_1D, ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {N, 2 *  num_cells})->output(0), false);
            auto axis_node2 = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split2 = std::make_shared<ov::opset4::Split>(mul12, axis_node2, 2);
            auto out_C = std::make_shared<ov::opset4::Add>(split2->output(0), split2->output(1));

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
            }

            if (use_memory_layers) {
                auto assign_c = std::make_shared<ov::opset11::Assign>(out_C, c_var);
                auto assign_h = std::make_shared<ov::opset11::Assign>(out_H, h_var);
                ov::SinkVector sinks;
                sinks.push_back(assign_c);
                sinks.push_back(assign_h);
                f->add_sinks(sinks);
            }

            if (lstm_cell_v1) {
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;

        } else {

            // Xt*(W^T)
            auto Xt_W = std::make_shared<ov::opset4::MatMul>(X, W, false, true);
            // Ht-1*(R^T)
            auto Ht_R = std::make_shared<ov::opset4::MatMul>(H_t, R, false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            auto add = std::make_shared<ov::opset4::Add>(Ht_R, bias);
            auto XHB = std::make_shared<ov::opset4::Add>(Xt_W, add);

            auto axis_node = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split = std::make_shared<ov::opset4::Split>(XHB, axis_node, 4);
            ov::Output<ov::Node> f = split->output(0);
            ov::Output<ov::Node> i = split->output(1);
            ov::Output<ov::Node> c = split->output(2);
            ov::Output<ov::Node> o = split->output(3);

            if (clip > 0.f) {
                auto clamp_f = std::make_shared<ov::opset4::Clamp>(f, -clip, clip);
                auto clamp_i = std::make_shared<ov::opset4::Clamp>(i, -clip, clip);
                auto clamp_c = std::make_shared<ov::opset4::Clamp>(c, -clip, clip);
                auto clamp_o = std::make_shared<ov::opset4::Clamp>(o, -clip, clip);
                f = clamp_f;
                i = clamp_i;
                c = clamp_c;
                o = clamp_o;
                if (lstm_cell_v1) {
                    ngraph::copy_runtime_info(lstm_cell_v1, {clamp_f, clamp_i, clamp_c, clamp_o});
                } else {
                    ngraph::copy_runtime_info(lstm_cell_v4, {clamp_f, clamp_i, clamp_c, clamp_o});
                }
            }

            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto f_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], f)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], f);
            auto i_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], i)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], i);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], c);
            auto o_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], o)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], o);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul1 = std::make_shared<ov::opset4::Multiply>(f_t, C_t);
            auto mul2 = std::make_shared<ov::opset4::Multiply>(i_t, c_t);
            auto out_C = std::make_shared<ov::opset4::Add>(mul1, mul2);

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v1, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v4, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;
        }
    };

    return is_graph_modfied;
}

bool GnaLstmDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto lstm_cell_v1 = std::dynamic_pointer_cast<ov::opset1::LSTMCell>(node);
        auto lstm_cell_v4 = std::dynamic_pointer_cast<ov::opset4::LSTMCell>(node);
        if ((nullptr == lstm_cell_v1) && (nullptr == lstm_cell_v4)) {
            continue;
        }

        const ov::Output<ov::Node>& X = (lstm_cell_v1) ? lstm_cell_v1->input_value(0) : lstm_cell_v4->input_value(0);
        const ov::Output<ov::Node>& H_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(1) : lstm_cell_v4->input_value(1);
        const ov::Output<ov::Node>& C_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(2) : lstm_cell_v4->input_value(2);
        const ov::Output<ov::Node>& W = (lstm_cell_v1) ? lstm_cell_v1->input_value(3) : lstm_cell_v4->input_value(3);
        const ov::Output<ov::Node>& R = (lstm_cell_v1) ? lstm_cell_v1->input_value(4) : lstm_cell_v4->input_value(4);
        const ov::Output<ov::Node>& bias = (lstm_cell_v1) ? lstm_cell_v1->input_value(5) : lstm_cell_v4->input_value(5);
        auto clip = (lstm_cell_v1) ? lstm_cell_v1->get_clip() : lstm_cell_v4->get_clip();

        if (gna_optimized && (clip == 0.0f)) {

            ov::OutputVector upstream;
            upstream.push_back(X);
            auto R_const = std::dynamic_pointer_cast<Constant>(R.get_node_shared_ptr());
            auto W_const = std::dynamic_pointer_cast<Constant>(W.get_node_shared_ptr());
            auto bias_const = std::dynamic_pointer_cast<Constant>(bias.get_node_shared_ptr());
            auto R_shape = R_const->output(0).get_shape();
            auto W_shape = W_const->output(0).get_shape();
            auto X_shape = X.get_shape();
            auto N = X_shape[0];
            auto num_cells = R_shape[0] / 4;
            const float* R_ptr = R_const->get_data_ptr<float>();
            const float* W_ptr = W_const->get_data_ptr<float>();
            const float* bias_ptr = bias_const->get_data_ptr<float>();
            std::vector<float> R_fio(3 * num_cells * R_shape[1], 0.0f);
            std::vector<float> R_c(num_cells * R_shape[1], 0.0f);
            std::vector<float> W_fio(3 * num_cells * W_shape[1], 0.0f);
            std::vector<float> W_c(num_cells * W_shape[1], 0.0f);
            std::vector<float> bias_fio(3 * num_cells, 0.0f);
            std::vector<float> bias_c(num_cells, 0.0f);
            float* R_fio_ptr = R_fio.data();
            float* R_c_ptr = R_c.data();
            float* W_fio_ptr = W_fio.data();
            float* W_c_ptr = W_c.data();
            float* bias_fio_ptr = bias_fio.data();
            float* bias_c_ptr = bias_c.data();
            // reorder weights to reduce elementwise operations
            auto num_R = num_cells * R_shape[1];
            auto num_W = num_cells * W_shape[1];
            auto num_bias = num_cells;
            memcpy(R_fio_ptr + 0 * num_R, R_ptr + 0 * num_R, num_R * sizeof(float));  // f
            memcpy(W_fio_ptr + 0 * num_W, W_ptr + 0 * num_W, num_W * sizeof(float));  // f
            memcpy(bias_fio_ptr + 0 * num_bias, bias_ptr + 0 * num_bias, num_bias * sizeof(float));  // f
            memcpy(R_fio_ptr + 1 * num_R, R_ptr + 1 * num_R, num_R * sizeof(float));  // i
            memcpy(W_fio_ptr + 1 * num_W, W_ptr + 1 * num_W, num_W * sizeof(float));  // i
            memcpy(bias_fio_ptr + 1 * num_bias, bias_ptr + 1 * num_bias, num_bias * sizeof(float));  // i
            memcpy(R_c_ptr + 0 * num_R, R_ptr + 2 * num_R, num_R * sizeof(float));   // c
            memcpy(W_c_ptr + 0 * num_W, W_ptr + 2 * num_W, num_W * sizeof(float));   // c
            memcpy(bias_c_ptr + 0 * num_bias, bias_ptr + 2 * num_bias, num_bias * sizeof(float));    // c
            memcpy(R_fio_ptr + 2 * num_R, R_ptr + 3 * num_R, num_R * sizeof(float));  // o
            memcpy(W_fio_ptr + 2 * num_W, W_ptr + 3 * num_W, num_W * sizeof(float));  // o
            memcpy(bias_fio_ptr + 2 * num_bias, bias_ptr + 3 * num_bias, num_bias * sizeof(float));  // o
            auto R_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells, R_shape[1]}, R_fio);
            auto R_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells, R_shape[1]}, R_c);
            auto W_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells, W_shape[1]}, W_fio);
            auto W_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells, W_shape[1]}, W_c);
            auto bias_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{3 * num_cells}, bias_fio);
            auto bias_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{num_cells}, bias_c);

            OutputVector parts;
            parts.push_back(X);
            parts.push_back(H_t);
            auto XHt = std::make_shared<ov::op::v0::Concat>(parts, 1);

            auto R_fio_shape = R_fio_const->output(0).get_shape();
            auto W_fio_shape = W_fio_const->output(0).get_shape();
            std::vector<float> WR_fio_vector(W_fio_shape[0] * (W_fio_shape[1] + R_fio_shape[1]), 0.0f);
            float* WR_fio = WR_fio_vector.data();
            for (auto i = 0; i < W_fio_shape[0]; i++) {
                for (auto j = 0; j < W_fio_shape[1]; j++) {
                    WR_fio[i * (W_fio_shape[1] + R_fio_shape[1]) + j] = W_fio[i * W_fio_shape[1] + j];
                }
                for (auto j = 0; j < R_fio_shape[1]; j++) {
                    WR_fio[i * (W_fio_shape[1] + R_fio_shape[1]) + W_fio_shape[1] + j] = R_fio[i * R_fio_shape[1] + j];
                }
            }
            auto WR_fio_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{W_fio_shape[0], W_fio_shape[1] + R_fio_shape[1]}, WR_fio);

            auto R_c_shape = R_c_const->output(0).get_shape();
            auto W_c_shape = W_c_const->output(0).get_shape();
            std::vector<float> WR_c_vector(W_c_shape[0] * (W_c_shape[1] + R_c_shape[1]), 0.0f);
            float* WR_c = WR_c_vector.data();
            for (auto i = 0; i < W_c_shape[0]; i++) {
                for (auto j = 0; j < W_c_shape[1]; j++) {
                    WR_c[i * (W_c_shape[1] + R_c_shape[1]) + j] = W_c[i * W_c_shape[1] + j];
                }
                for (auto j = 0; j < R_c_shape[1]; j++) {
                    WR_c[i * (W_c_shape[1] + R_c_shape[1]) + W_c_shape[1] + j] = R_c[i * R_c_shape[1] + j];
                }
            }
            auto WR_c_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{W_c_shape[0], W_c_shape[1] + R_c_shape[1]}, WR_c);

            // [Xt / H_t]*([W R]^T)
            auto XHt_WR_fio = std::make_shared<ov::op::v0::MatMul>(XHt, WR_fio_const->output(0), false, true);
            auto XHt_WR_c = std::make_shared<ov::op::v0::MatMul>(XHt, WR_c_const->output(0), false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            auto XHB_fio = std::make_shared<ov::op::v1::Add>(XHt_WR_fio->output(0), bias_fio_const->output(0));
            auto XHB_c = std::make_shared<ov::op::v1::Add>(XHt_WR_c->output(0), bias_c_const->output(0));

            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto fio_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], XHB_fio)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], XHB_fio);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], XHB_c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], XHB_c);

            const auto split_lengths_const = ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {2 * num_cells, num_cells});
            auto split1 = std::make_shared<ov::opset1::VariadicSplit>(fio_t, 
                ov::opset4::Constant::create(ov::element::i64, ov::Shape{}, {1}), split_lengths_const);
            ov::Output<ov::Node> fi_t = split1->output(0);
            ov::Output<ov::Node> o_t = split1->output(1);

            auto fi_t_1D = std::make_shared<ov::opset4::Reshape>(fi_t, ov::opset4::Constant::create(element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);
            ov::OutputVector Cc_t;
            Cc_t.push_back(C_t);
            Cc_t.push_back(c_t);
            auto concat2 = std::make_shared<ov::opset4::Concat>(Cc_t, 1);
            auto concat2_1D = std::make_shared<ov::opset4::Reshape>(concat2->output(0), ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul12_1D = std::make_shared<ov::opset4::Multiply>(fi_t_1D->output(0), concat2_1D->output(0));
            auto mul12 = std::make_shared<ov::opset4::Reshape>(mul12_1D, ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {N, 2 *  num_cells})->output(0), false);
            auto axis_node2 = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split2 = std::make_shared<ov::opset4::Split>(mul12, axis_node2, 2);
            auto out_C = std::make_shared<ov::opset4::Add>(split2->output(0), split2->output(1));

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;

        } else {

            // Xt*(W^T)
            auto Xt_W = std::make_shared<ov::opset4::MatMul>(X, W, false, true);
            // Ht-1*(R^T)
            auto Ht_R = std::make_shared<ov::opset4::MatMul>(H_t, R, false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            auto add = std::make_shared<ov::opset4::Add>(Ht_R, bias);
            auto XHB = std::make_shared<ov::opset4::Add>(Xt_W, add);

            auto axis_node = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split = std::make_shared<ov::opset4::Split>(XHB, axis_node, 4);
            ov::Output<ov::Node> f = split->output(0);
            ov::Output<ov::Node> i = split->output(1);
            ov::Output<ov::Node> c = split->output(2);
            ov::Output<ov::Node> o = split->output(3);

            if (clip > 0.f) {
                auto clamp_f = std::make_shared<ov::opset4::Clamp>(f, -clip, clip);
                auto clamp_i = std::make_shared<ov::opset4::Clamp>(i, -clip, clip);
                auto clamp_c = std::make_shared<ov::opset4::Clamp>(c, -clip, clip);
                auto clamp_o = std::make_shared<ov::opset4::Clamp>(o, -clip, clip);
                f = clamp_f;
                i = clamp_i;
                c = clamp_c;
                o = clamp_o;
                if (lstm_cell_v1) {
                    ngraph::copy_runtime_info(lstm_cell_v1, {clamp_f, clamp_i, clamp_c, clamp_o});
                } else {
                    ngraph::copy_runtime_info(lstm_cell_v4, {clamp_f, clamp_i, clamp_c, clamp_o});
                }
            }

            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto f_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], f)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], f);
            auto i_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], i)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], i);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], c);
            auto o_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], o)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], o);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul1 = std::make_shared<ov::opset4::Multiply>(f_t, C_t);
            auto mul2 = std::make_shared<ov::opset4::Multiply>(i_t, c_t);
            auto out_C = std::make_shared<ov::opset4::Add>(mul1, mul2);

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v1, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v4, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;
        }
    };

    return is_graph_modfied;
}

bool GnaStackedLstmDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto lstm_cell_v1 = std::dynamic_pointer_cast<ov::opset1::LSTMCell>(node);
        auto lstm_cell_v4 = std::dynamic_pointer_cast<ov::opset4::LSTMCell>(node);
        if ((nullptr == lstm_cell_v1) && (nullptr == lstm_cell_v4)) {
            continue;
        }

        const ov::Output<ov::Node>& X = (lstm_cell_v1) ? lstm_cell_v1->input_value(0) : lstm_cell_v4->input_value(0);
        const ov::Output<ov::Node>& H_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(1) : lstm_cell_v4->input_value(1);
        const ov::Output<ov::Node>& C_t = (lstm_cell_v1) ? lstm_cell_v1->input_value(2) : lstm_cell_v4->input_value(2);
        const ov::Output<ov::Node>& W = (lstm_cell_v1) ? lstm_cell_v1->input_value(3) : lstm_cell_v4->input_value(3);
        const ov::Output<ov::Node>& R = (lstm_cell_v1) ? lstm_cell_v1->input_value(4) : lstm_cell_v4->input_value(4);
        const ov::Output<ov::Node>& bias = (lstm_cell_v1) ? lstm_cell_v1->input_value(5) : lstm_cell_v4->input_value(5);
        auto clip = (lstm_cell_v1) ? lstm_cell_v1->get_clip() : lstm_cell_v4->get_clip();

        if (gna_optimized && (clip == 0.0f)) {

            auto R_const = std::dynamic_pointer_cast<Constant>(R.get_node_shared_ptr());
            auto W_const = std::dynamic_pointer_cast<Constant>(W.get_node_shared_ptr());
            auto bias_const = std::dynamic_pointer_cast<Constant>(bias.get_node_shared_ptr());
            auto R_shape = R_const->output(0).get_shape();
            auto W_shape = W_const->output(0).get_shape();
            auto X_shape = X.get_shape();
            auto N = X_shape[0];
            auto num_cells = R_shape[0] / 4;
            OutputVector parts;
            parts.push_back(X);
            parts.push_back(H_t);
            auto XHt = std::make_shared<ov::op::v0::Concat>(parts, 1);

            auto W_data = W_const->get_data_ptr<float>();
            auto R_data = R_const->get_data_ptr<float>();
            std::vector<float> WR_vector(W_shape[0] * (W_shape[1] + R_shape[1]), 0.0f);
            float* WR_data = WR_vector.data();
            for (auto i = 0; i < W_shape[0]; i++) {
                for (auto j = 0; j < W_shape[1]; j++) {
                    WR_data[i * (W_shape[1] + R_shape[1]) + j] = W_data[i * W_shape[1] + j];
                }
                for (auto j = 0; j < R_shape[1]; j++) {
                    WR_data[i * (W_shape[1] + R_shape[1]) + W_shape[1] + j] = R_data[i * R_shape[1] + j];
                }
            }
            auto WR_const = ov::opset4::Constant::create(ngraph::element::f32, ov::Shape{W_shape[0], W_shape[1] + R_shape[1]}, WR_data);
            auto WR = WR_const->output(0);

            // [Xt / H_t]*([W R]^T)
            auto XHt_WR = std::make_shared<ov::op::v0::MatMul>(XHt, WR, false, true);
            // Xt*(W^T)
            // auto Xt_W_fio = std::make_shared<ov::opset4::MatMul>(X, W_fio_const->output(0), false, true);
            // auto Xt_W_c = std::make_shared<ov::opset4::MatMul>(X, W_c_const->output(0), false, true);
            // Ht-1*(R^T)
            // auto Ht_R_fio = std::make_shared<ov::opset4::MatMul>(H_t, R_fio_const->output(0), false, true);
            // auto Ht_R_c = std::make_shared<ov::opset4::MatMul>(H_t, R_c_const->output(0), false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            // auto add_fio = std::make_shared<ov::opset4::Add>(Ht_R_fio, bias_fio_const->output(0));
            // auto add_c = std::make_shared<ov::opset4::Add>(Ht_R_c, bias_c_const->output(0));
            auto XHB = std::make_shared<ov::op::v1::Add>(XHt_WR, bias);
            const auto split0_lengths_const = ov::opset4::Constant::create(ov::element::i64, ov::Shape{4}, {num_cells, num_cells, num_cells, num_cells});
            auto split0 = std::make_shared<ov::opset1::VariadicSplit>(XHB, 
                ov::opset4::Constant::create(ov::element::i64, ov::Shape{}, {1}), split0_lengths_const);
            parts.clear();
            parts.push_back(split0->output(0));
            parts.push_back(split0->output(1));
            parts.push_back(split0->output(3));
            auto XHB_fio = std::make_shared<ov::op::v0::Concat>(parts, 1);
            auto XHB_c = split0->output(2);
            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto fio_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], XHB_fio)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], XHB_fio);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], XHB_c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], XHB_c);

            const auto split1_lengths_const = ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {2 * num_cells, num_cells});
            auto split1 = std::make_shared<ov::opset1::VariadicSplit>(fio_t, 
                ov::opset4::Constant::create(ov::element::i64, ov::Shape{}, {1}), split1_lengths_const);
            ov::Output<ov::Node> fi_t = split1->output(0);
            ov::Output<ov::Node> o_t = split1->output(1);

            auto fi_t_1D = std::make_shared<ov::opset4::Reshape>(fi_t, ov::opset4::Constant::create(element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);
            ov::OutputVector Cc_t;
            Cc_t.push_back(C_t);
            Cc_t.push_back(c_t);
            auto concat2 = std::make_shared<ov::opset4::Concat>(Cc_t, 1);
            auto concat2_1D = std::make_shared<ov::opset4::Reshape>(concat2->output(0), ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {1ull, 2 * N *  num_cells})->output(0), false);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul12_1D = std::make_shared<ov::opset4::Multiply>(fi_t_1D->output(0), concat2_1D->output(0));
            auto mul12 = std::make_shared<ov::opset4::Reshape>(mul12_1D, ov::opset4::Constant::create(ov::element::i64, ov::Shape{2}, {N, 2 *  num_cells})->output(0), false);
            auto axis_node2 = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split2 = std::make_shared<ov::opset4::Split>(mul12, axis_node2, 2);
            auto out_C = std::make_shared<ov::opset4::Add>(split2->output(0), split2->output(1));

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
            }

            if (lstm_cell_v1) {
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;

        } else {

            // Xt*(W^T)
            auto Xt_W = std::make_shared<ov::opset4::MatMul>(X, W, false, true);
            // Ht-1*(R^T)
            auto Ht_R = std::make_shared<ov::opset4::MatMul>(H_t, R, false, true);
            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
            auto add = std::make_shared<ov::opset4::Add>(Ht_R, bias);
            auto XHB = std::make_shared<ov::opset4::Add>(Xt_W, add);

            auto axis_node = ov::opset4::Constant::create(ov::element::u64, ov::Shape{}, {1});
            auto split = std::make_shared<ov::opset4::Split>(XHB, axis_node, 4);
            ov::Output<ov::Node> f = split->output(0);
            ov::Output<ov::Node> i = split->output(1);
            ov::Output<ov::Node> c = split->output(2);
            ov::Output<ov::Node> o = split->output(3);

            if (clip > 0.f) {
                auto clamp_f = std::make_shared<ov::opset4::Clamp>(f, -clip, clip);
                auto clamp_i = std::make_shared<ov::opset4::Clamp>(i, -clip, clip);
                auto clamp_c = std::make_shared<ov::opset4::Clamp>(c, -clip, clip);
                auto clamp_o = std::make_shared<ov::opset4::Clamp>(o, -clip, clip);
                f = clamp_f;
                i = clamp_i;
                c = clamp_c;
                o = clamp_o;
                if (lstm_cell_v1) {
                    ngraph::copy_runtime_info(lstm_cell_v1, {clamp_f, clamp_i, clamp_c, clamp_o});
                } else {
                    ngraph::copy_runtime_info(lstm_cell_v4, {clamp_f, clamp_i, clamp_c, clamp_o});
                }
            }

            // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            auto f_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], f)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], f);
            auto i_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], i)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], i);
            auto c_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[1], c)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[1], c);
            auto o_t = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[0], o)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[0], o);

            // Ct = ft (.) Ct-1 + it (.) ct
            auto mul1 = std::make_shared<ov::opset4::Multiply>(f_t, C_t);
            auto mul2 = std::make_shared<ov::opset4::Multiply>(i_t, c_t);
            auto out_C = std::make_shared<ov::opset4::Add>(mul1, mul2);

            // H = ot (.) h(Ct)
            auto hC = (lstm_cell_v1) ? ov::op::util::activation(lstm_cell_v1->get_activations()[2], out_C)
                : ov::op::util::activation(lstm_cell_v4->get_activations()[2], out_C);
            auto out_H = std::make_shared<ov::opset4::Multiply>(o_t, hC);

            if (lstm_cell_v1) {
                out_H->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v1->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v1, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v1, {out_H->output(0), out_C->output(0)});
            } else {
                out_H->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".0");
                out_C->set_friendly_name(lstm_cell_v4->get_friendly_name() + ".1");
                ngraph::copy_runtime_info(lstm_cell_v4, {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
                ngraph::replace_node(lstm_cell_v4, {out_H->output(0), out_C->output(0)});
            }

            is_graph_modfied = true;
        }
    };

    return is_graph_modfied;
}
