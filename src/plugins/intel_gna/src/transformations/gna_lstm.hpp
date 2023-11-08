// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include <transformations_visibility.hpp>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"


namespace ov {
namespace pass {

class GnaLstmDecomposition : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("GnaLstmDecomposition", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace ov
