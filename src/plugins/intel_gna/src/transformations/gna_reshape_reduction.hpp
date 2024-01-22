// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GnaReshapeFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaReshapeFuse", "0");
    GnaReshapeFuse();
};

class GnaReshapeToSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaReshapeToSqueeze", "0");
    GnaReshapeToSqueeze();
};

class GnaReshapeToUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaReshapeToUnsqueeze", "0");
    GnaReshapeToUnsqueeze();
};

class GnaReshapeReduction : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaReshapeReduction", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov