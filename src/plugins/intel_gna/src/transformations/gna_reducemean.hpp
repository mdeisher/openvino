// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

// To call this transformation include the header and call transformation manager:
// ov::pass::Manager m;
// m.register_pass<ov::intel_gna::pass::TestTransformation>();
// m.run_passes(model);
class GnaReduceMeanTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaReduceMeanTransformation", "0");
    GnaReduceMeanTransformation();
};

class GnaReduceSumTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaReduceSumTransformation", "0");
    GnaReduceSumTransformation();
};

}  // namespace pass
}  // namespace ov
