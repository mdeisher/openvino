// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

// To call this transformation include the header and call transformation manager:
// ov::pass::Manager m;
// m.register_pass<ov::intel_gna::pass::TestTransformation>();
// m.run_passes(model);
class GnaMhaTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaTransformation", "0");
    GnaMhaTransformation();
};

class GnaMhaFqTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaFqTransformation", "0");
    GnaMhaFqTransformation();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
