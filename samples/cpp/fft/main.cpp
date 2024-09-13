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

#include <fstream>
#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/serialize.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace op;

#define DEFAULT_N 8
#define VERBOSE false

/**********************************************************/
/* fft.c                                                  */
/* (c) Douglas L. Jones                                   */
/* University of Illinois at Urbana-Champaign             */
/* January 19, 1992                                       */
/*                                                        */
/*   fft: in-place radix-2 DIT DFT of a complex input     */
/*                                                        */
/*   input:                                               */
/* n: length of FFT: must be a power of two               */
/* m: n = 2**m                                            */
/*   input/output                                         */
/* x: double array of length n with real part of data     */
/* y: double array of length n with imag part of data     */
/*                                                        */
/*   Permission to copy and use this program is granted   */
/*   under a Creative Commons "Attribution" license       */
/*   http://creativecommons.org/licenses/by/1.0/          */
/**********************************************************/
void fft(uint32_t n, uint32_t log_n, float x_real[], float x_imag[]) {
    uint32_t i, j, k, n1, n2;
    float c, s, e, a, t1, t2;

    j = 0; /* bit-reverse */
    n2 = n / 2;
    for (i = 1; i < n - 1; i++) {
        n1 = n2;
        while (j >= n1) {
            j = j - n1;
            n1 = n1 / 2;
        }
        j = j + n1;

        if (i < j) {
            t1 = x_real[i];
            x_real[i] = x_real[j];
            x_real[j] = t1;
            t1 = x_imag[i];
            x_imag[i] = x_imag[j];
            x_imag[j] = t1;
        }
    }

    n1 = 0; /* FFT */
    n2 = 1;

    for (i = 0; i < log_n; i++) {
        n1 = n2;
        n2 = n2 + n2;
        e = -6.283185307179586f / n2;
        a = 0.0;

        for (j = 0; j < n1; j++) {
            c = cos(a);
            s = sin(a);
            a = a + e;

            for (k = j; k < n; k = k + n2) {
                t1 = c * x_real[k + n1] - s * x_imag[k + n1];
                t2 = s * x_real[k + n1] + c * x_imag[k + n1];
                x_real[k + n1] = x_real[k] - t1;
                x_imag[k + n1] = x_imag[k] - t2;
                x_real[k] = x_real[k] + t1;
                x_imag[k] = x_imag[k] + t2;
            }
        }
        // dump intermediate output
        //printf("Stage %d Output:\n", i);
        //for (k = 0; k < n; k++) {
        //    printf("%f + j %f\n", x_real[k], x_imag[k]);
        //}
    }

    return;
}

void fft_twiddle(uint32_t n, uint32_t m, std::vector<float> &w_real, std::vector<float> &w_imag) {

    uint32_t i, j, k, n1, n2;
    float c, s, e, a;

    memset(w_real.data(), 0, n * n * sizeof(float));
    memset(w_imag.data(), 0, n * n * sizeof(float));

    n1 = 0;
    n2 = 1;

    for (i = 0; i <= m; i++) {
        n1 = n2;
        n2 = n2 + n2;
        e = -6.283185307179586f / n2;
        a = 0.0;

        if (i == m) {

            for (j = 0; j < n1; j++) {
                c = cos(a);
                s = sin(a);
                a = a + e;
                for (k = j; k < n; k = k + n2) {
                    w_real[k * n + k] = 1.0f;
                    w_real[(k + n1) * n + k] = 1.0f;
                    w_real[k * n + k + n1] = c;
                    w_imag[k * n + k + n1] = s;
                    w_real[(k + n1) * n + k + n1] = -c;
                    w_imag[(k + n1) * n + k + n1] = -s;
                }
            }
        }
    }

    return;
}

void bit_reverse_columns(std::vector<float> &matrix, uint32_t num_rows, uint32_t num_cols) {
    uint32_t j = 0; /* bit-reverse */
    uint32_t n2 = num_cols / 2;
    for (uint32_t i = 1; i < num_cols - 1; i++) {
        uint32_t n1 = n2;
        while (j >= n1) {
            j = j - n1;
            n1 = n1 / 2;
        }
        j = j + n1;

        if (i < j) {
            for (uint32_t k = 0; k < num_rows; k++) {
                float temp = matrix[k * num_cols + i];
                matrix[k * num_cols + i] = matrix[k * num_cols + j];
                matrix[k * num_cols + j] = temp;
            }
        }
    }
}

// Create a neural network graph that implements the Fast Fourier Transform
//
// NOTE:  The approach here is designed to leverage GNA sparse matrix operations.
//        It is undoubtedly not the most efficient design for CPU, GPU, or VPU.
//
std::shared_ptr<ov::Node> SparseAffineFft(Output<Node> parent) {
    OutputVector data_real, data_imag;
    data_real.push_back(parent);
    data_imag.push_back(parent);  // dummy initializer - will be overwritten
    auto input_shape = parent.get_shape();
    size_t W = input_shape[0];
    size_t logW = (size_t)log2(W);
    std::vector<float> twiddle_real(W * W, 0.0f);
    std::vector<float> twiddle_imag(W*W, 0.0f);
    for (uint32_t i = 0; i < logW; i++) {
        fft_twiddle(W, i, twiddle_real, twiddle_imag);
        if (i == 0) {  // for FFT of real signal we can eliminate the imaginary part of the calcs
            bit_reverse_columns(twiddle_real, W, W);  // bit reverse the input order for DIT FFT
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{W, W}, twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto matmul_real1 = std::make_shared<op::MatMul>(twiddle_real_const, data_real[0], false, false);
            matmul_real1->set_friendly_name("matmul_real1");
            data_real[0] = matmul_real1->output(0);
        } else if (i == 1) {  // for FFT of real signal we can eliminate some imaginary part of the calcs
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{W, W}, twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto twiddle_imag_const = op::Constant::create(ngraph::element::f32, Shape{W, W}, twiddle_imag);
            twiddle_imag_const->set_friendly_name("twiddle_imag_const");
            auto matmul_real1 = std::make_shared<op::MatMul>(twiddle_real_const, data_real[0], false, false);
            matmul_real1->set_friendly_name("matmul_real1");
            auto matmul_imag2 = std::make_shared<op::MatMul>(twiddle_imag_const, data_real[0], false, false);
            matmul_imag2->set_friendly_name("matmul_imag2");
            data_real[0] = matmul_real1->output(0);
            data_imag[0] = matmul_imag2->output(0);
        } else {
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{W, W}, twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto twiddle_imag_const = op::Constant::create(ngraph::element::f32, Shape{W, W}, twiddle_imag);
            twiddle_imag_const->set_friendly_name("twiddle_imag_const");
            auto matmul_real1 = std::make_shared<op::MatMul>(twiddle_real_const, data_real[0], false, false);
            matmul_real1->set_friendly_name("matmul_real1");
            auto matmul_real2 = std::make_shared<op::MatMul>(twiddle_imag_const, data_imag[0], false, false);
            matmul_real2->set_friendly_name("matmul_real2");
            auto matmul_imag1 = std::make_shared<op::MatMul>(twiddle_real_const, data_imag[0], false, false);
            matmul_imag1->set_friendly_name("matmul_imag1");
            auto matmul_imag2 = std::make_shared<op::MatMul>(twiddle_imag_const, data_real[0], false, false);
            matmul_imag2->set_friendly_name("matmul_imag2");
            auto sub_real = std::make_shared<op::v1::Subtract>(matmul_real1->output(0), matmul_real2->output(0));
            sub_real->set_friendly_name("sub_real");
            auto add_imag = std::make_shared<op::v1::Add>(matmul_imag1->output(0), matmul_imag2->output(0));
            add_imag->set_friendly_name("add_imag");
            data_real[0] = sub_real->output(0);
            data_imag[0] = add_imag->output(0);
        }
    }
    auto output_real = std::make_shared<ngraph::opset1::Reshape>(data_real[0],
        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
    output_real->set_friendly_name("output_real");
    auto output_imag = std::make_shared<ngraph::opset1::Reshape>(data_imag[0],
        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
    output_imag->set_friendly_name("output_imag");
    OutputVector parts;
    parts.push_back(output_real->output(0));
    parts.push_back(output_imag->output(0));
    auto output_2d = std::make_shared<ngraph::opset1::Concat>(parts, 1);
    output_2d->set_friendly_name("output_2d");

    return(output_2d);
}

// Create a neural network graph that implements the Fast Fourier Transform
//
// NOTE:  The approach here is designed to leverage GNA 2D convolution operations.
//        It is undoubtedly not the most efficient design for CPU, GPU, or VPU.
//        And it is still O(N^2) but the hope is that the parallelism in the GNA
//        Convolution operator may make the total cycles competitive.
//
std::shared_ptr<ov::Node> ConvolutionFft(Output<Node> parent) {
    OutputVector data_real, data_imag;
    data_real.push_back(parent);
    data_imag.push_back(parent);  // dummy initializer - will be overwritten
    auto input_shape = parent.get_shape();
    size_t W = input_shape[0];
    size_t logW = (size_t)log2(W);
    size_t num_kernels = 1;
    size_t num_columns = 1;
    std::vector<float> twiddle_real(W * W, 0.0f);
    std::vector<float> twiddle_imag(W * W, 0.0f);
    for (uint32_t i = 0; i < logW; i++) {
        num_kernels *= 2;
        num_columns *= 2;
        std::vector<float> partial_twiddle_real(num_kernels * num_columns, 0.0f);
        std::vector<float> partial_twiddle_imag(num_kernels * num_columns, 0.0f);
        fft_twiddle(W, i, twiddle_real, twiddle_imag);
        // extract block of twiddle factor matrix to be used as convolution kernels
        for (uint32_t j = 0; j < num_kernels; j++) {
            for (uint32_t k = 0; k < num_columns; k++) {
                partial_twiddle_real[j * num_columns + k] = twiddle_real[j * W + k];
                partial_twiddle_imag[j * num_columns + k] = twiddle_imag[j * W + k];
            }
        }
        if (i == 0) {  // for FFT of real signal we can eliminate the imaginary part of the calcs
            bit_reverse_columns(twiddle_real, W, W);  // bit reverse the input order for DIT FFT
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{W, 1, 1, W}, twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto reshape_1 = std::make_shared<ngraph::opset1::Reshape>(data_real[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, W, 1ull})->output(0),false);
            reshape_1->set_friendly_name("reshape_1");
            auto transpose_1 = std::make_shared<op::Transpose>(reshape_1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_1->set_friendly_name("transpose_1");
            auto conv = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                twiddle_real_const->output(0),Strides{1, W},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv->set_friendly_name("conv");
            auto transpose_2 = std::make_shared<op::Transpose>(conv->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_2->set_friendly_name("transpose_2");
            data_real[0] = transpose_2->output(0);
        } else if (i == 1) {  // for FFT of real signal we can eliminate some imaginary part of the calcs
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{num_kernels, 1, 1, num_columns}, partial_twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto twiddle_imag_const = op::Constant::create(ngraph::element::f32, Shape{num_kernels, 1, 1, num_columns}, partial_twiddle_imag);
            twiddle_imag_const->set_friendly_name("twiddle_imag_const");
            auto reshape_real = std::make_shared<ngraph::opset1::Reshape>(data_real[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W/num_columns, num_columns, 1ull})->output(0),false);
            reshape_real->set_friendly_name("reshape_real");
            auto transpose_1 = std::make_shared<op::Transpose>(reshape_real->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_1->set_friendly_name("transpose_1");
            auto conv_real1 = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                twiddle_real_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_real1->set_friendly_name("conv_real1");
            auto transpose_2 = std::make_shared<op::Transpose>(conv_real1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_2->set_friendly_name("transpose_2");
            auto transpose_3 = std::make_shared<op::Transpose>(reshape_real->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_3->set_friendly_name("transpose_3");
            auto conv_imag2 = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                twiddle_imag_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_imag2->set_friendly_name("conv_imag2");
            auto transpose_4 = std::make_shared<op::Transpose>(conv_imag2->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_4->set_friendly_name("transpose_4");
            data_real[0] = transpose_2->output(0);
            data_imag[0] = transpose_4->output(0);
        } else {
            auto twiddle_real_const = op::Constant::create(ngraph::element::f32, Shape{num_kernels, 1, 1, num_columns}, partial_twiddle_real);
            twiddle_real_const->set_friendly_name("twiddle_real_const");
            auto twiddle_imag_const = op::Constant::create(ngraph::element::f32, Shape{num_kernels, 1, 1, num_columns}, partial_twiddle_imag);
            twiddle_imag_const->set_friendly_name("twiddle_imag_const");
            auto reshape_real = std::make_shared<ngraph::opset1::Reshape>(data_real[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W/num_columns, num_columns, 1ull})->output(0),false);
            reshape_real->set_friendly_name("reshape_real");
            auto reshape_imag = std::make_shared<ngraph::opset1::Reshape>(data_imag[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W/num_columns, num_columns, 1ull})->output(0),false);
            reshape_imag->set_friendly_name("reshape_imag");
            auto transpose_1 = std::make_shared<op::Transpose>(reshape_real->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_1->set_friendly_name("transpose_1");
            auto conv_real1 = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                twiddle_real_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_real1->set_friendly_name("conv_real1");
            auto transpose_2 = std::make_shared<op::Transpose>(conv_real1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_2->set_friendly_name("transpose_2");
            auto transpose_3 = std::make_shared<op::Transpose>(reshape_imag->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_3->set_friendly_name("transpose_3");
            auto conv_real2 = std::make_shared<opset1::Convolution>(transpose_3->output(0),
                twiddle_imag_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_real2->set_friendly_name("conv_real2");
            auto transpose_4 = std::make_shared<op::Transpose>(conv_real2->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_4->set_friendly_name("transpose_4");
            auto transpose_5 = std::make_shared<op::Transpose>(reshape_imag->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_5->set_friendly_name("transpose_5");
            auto conv_imag1 = std::make_shared<opset1::Convolution>(transpose_5->output(0),
                twiddle_real_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_imag1->set_friendly_name("conv_imag1");
            auto transpose_6 = std::make_shared<op::Transpose>(conv_imag1->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_6->set_friendly_name("transpose_6");
            auto transpose_7 = std::make_shared<op::Transpose>(reshape_real->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            transpose_7->set_friendly_name("transpose_7");
            auto conv_imag2 = std::make_shared<opset1::Convolution>(transpose_7->output(0),
                twiddle_imag_const->output(0),Strides{1, num_columns},CoordinateDiff{0, 0},CoordinateDiff{0, 0},Strides{1, 1},PadType::VALID);
            conv_imag2->set_friendly_name("conv_imag2");
            auto transpose_8 = std::make_shared<op::Transpose>(conv_imag2->output(0),
                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            transpose_8->set_friendly_name("transpose_8");
            auto reshape_real1 = std::make_shared<ngraph::opset1::Reshape>(transpose_2->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
            auto reshape_real2 = std::make_shared<ngraph::opset1::Reshape>(transpose_4->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
            auto reshape_imag1 = std::make_shared<ngraph::opset1::Reshape>(transpose_6->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
            auto reshape_imag2 = std::make_shared<ngraph::opset1::Reshape>(transpose_8->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
            auto sub_real = std::make_shared<op::v1::Subtract>(reshape_real1->output(0), reshape_real2->output(0));
            sub_real->set_friendly_name("sub_real");
            auto add_imag = std::make_shared<op::v1::Add>(reshape_imag1->output(0), reshape_imag2->output(0));
            add_imag->set_friendly_name("add_imag");
            data_real[0] = sub_real->output(0);
            data_imag[0] = add_imag->output(0);
        }
    }
    auto output_real = std::make_shared<ngraph::opset1::Reshape>(data_real[0],
        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
    output_real->set_friendly_name("output_real");
    auto output_imag = std::make_shared<ngraph::opset1::Reshape>(data_imag[0],
        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, W})->output(0),false);
    output_imag->set_friendly_name("output_imag");
    OutputVector parts;
    parts.push_back(output_real->output(0));
    parts.push_back(output_imag->output(0));
    auto output_2d = std::make_shared<ngraph::opset1::Concat>(parts, 1);
    output_2d->set_friendly_name("output_2d");

    return(output_2d);
}

std::shared_ptr<Function> createNgraphFunctionCustomer(size_t W, size_t logW) {
    SinkVector sinks;
    auto paramNode = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, W}}));
    paramNode->set_friendly_name("input");
    auto input1_2d = std::make_shared<ngraph::opset1::Reshape>(paramNode->output(0),
        op::Constant::create(ngraph::element::i64, Shape{2}, {W, 1ull})->output(0),false);

    //auto output_2d = SparseAffineFft(input1_2d->output(0));
    auto output_2d = ConvolutionFft(input1_2d->output(0));

    auto result_full = std::make_shared<op::Result>(output_2d->output(0));
    result_full->set_friendly_name("result_full");

    std::shared_ptr<ngraph::Function> fnPtr =
        std::make_shared<ngraph::Function>(result_full, ngraph::ParameterVector{paramNode}, "fft");
    fnPtr->add_sinks(sinks);
    fnPtr->add_results({result_full});

    return fnPtr;
}

int main(int argc, char* argv[]) {
    size_t N = DEFAULT_N;
    size_t LOGN = (size_t)log2(N);

    if (argc > 1) {
        N = atoi(argv[1]);
        LOGN = (size_t)log2(N);
    }

    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // --------------------------- 1. Load inference engine -------------------------------------
    std::cout << "Loading Inference Engine" << std::endl;
    Core ie;

    std::vector<float> real(N, 0.0);
    std::vector<float> imag(N, 0.0);
    std::vector<float> twiddle_real(N * N, 0.0f);
    std::vector<float> twiddle_imag(N * N, 0.0f);
    std::default_random_engine gen;
    std::normal_distribution<float> dist(0.0, 1.0);

    FILE* fp = fopen("fft_input.csv", "w");
    printf("Input:\n");
    for (uint32_t i = 0; i < N; i++) {
        real[i] = (float)dist(gen);
        printf("%f + j %f\n", real[i], imag[i]);
        if (fp) {
            fprintf(fp, "%e", real[i]);
            if (i < N - 1) {
                fprintf(fp, ",");
            }
        }
    }
    if (fp) {
        fprintf(fp, "\n");
        fclose(fp);
    }

    fft(N, LOGN, (float*)real.data(), (float*)imag.data());

    if (VERBOSE) {
        for (uint32_t i = 0; i < LOGN; i++) {
            fft_twiddle(8, i, twiddle_real, twiddle_imag);
            printf("Stage %d:\n", i);
            for (uint32_t j = 0; j < N; j++) {
                for (uint32_t k = 0; k < N; k++) {
                    printf("%4.3f + j %4.3f, ", twiddle_real[j * N + k], twiddle_imag[j * N + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    fp = fopen("fft_output.csv", "w");
    printf("Output:\n");
    for (uint32_t i = 0; i < N; i++) {
        printf("%f + j %f\n", real[i], imag[i]);
        if (fp) {
            fprintf(fp, "%e,", real[i]);
        }
    }
    if (fp) {
        for (uint32_t i = 0; i < N; i++) {
            fprintf(fp, "%e", imag[i]);
            if (i < N - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
    }

    //--------------------------- 2. Create network using ngraph function -----------------------------------
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>("fft.xml", "fft.bin", ngraph::pass::Serialize::Version::IR_V10);
    const auto& pass_config = manager.get_pass_config();
    manager.run_passes(createNgraphFunctionCustomer(N, LOGN));

    return 0;
}
