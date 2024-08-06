// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GnaRewriteNormTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaRewriteNormTransformation", "0");
    GnaRewriteNormTransformation();
};

class GnaNormTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaNormTransformation", "0");
    GnaNormTransformation();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
