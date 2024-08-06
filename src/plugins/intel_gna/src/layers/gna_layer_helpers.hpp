// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_layer_info.hpp"

namespace ov {
namespace intel_gna {
namespace layer_utils {

/**
 * @brief retrievs blob from const layer connected to certain layer
 * @param input
 * @param idx
 */
inline InferenceEngine::Blob::Ptr getParamFromInputAsBlob(InferenceEngine::CNNLayerPtr input, size_t idx) {
    if (input->insData.size() <= idx) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx << "input";
    }
    auto iLayerData = input->insData[idx].lock();
    if (!iLayerData) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                         << ", input: cannot dereference data weak-pointer";
    }
    auto iLayer = getCreatorLayer(iLayerData).lock();
    if (!iLayer) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                         << ", input: cannot dereference creator layer weak-pointer";
    }
    if (!LayerInfo(iLayer).isConst()) {
        //THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
        //                                 << ", input: expected to be of type const, but was: " << iLayer->type;
        if (iLayer->type == "FakeQuantize") {
            auto prevLayerData = iLayer->insData[0].lock();
            if (!prevLayerData) {
                THROW_GNA_LAYER_EXCEPTION(iLayer)
                    << "cannot get data from " << 0 << ", iLayer: cannot dereference data weak-pointer";
            }
            auto prevLayer = getCreatorLayer(prevLayerData).lock();
            if (!prevLayer) {
                THROW_GNA_LAYER_EXCEPTION(iLayer)
                    << "cannot get data from " << 0 << ", input: cannot dereference creator layer weak-pointer";
            }
            iLayer = prevLayer;
        }
    }

    if (!iLayer->blobs.count("custom")) {
        THROW_GNA_LAYER_EXCEPTION(iLayer) << "cannot get custom blob";
    }

    return iLayer->blobs["custom"];
}

}  // namespace layer_utils
}  // namespace intel_gna
}  // namespace ov
