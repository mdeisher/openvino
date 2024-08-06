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

class GnaMhaQKVTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaQKVTransformation", "0");
    GnaMhaQKVTransformation();
};

class GnaMhaQKVFqTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaQKVFqTransformation", "0");
    GnaMhaQKVFqTransformation();
};

class GnaMhaSelfTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaSelfTransformation", "0");
    GnaMhaSelfTransformation();
};

class GnaMhaSelfFqTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaSelfFqTransformation", "0");
    GnaMhaSelfFqTransformation();
};

class GnaMhaSplitSelfTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaSplitSelfTransformation", "0");
    GnaMhaSplitSelfTransformation();
};

class GnaMhaSplitSelfFqTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GnaMhaSplitSelfFqTransformation", "0");
    GnaMhaSplitSelfFqTransformation();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
