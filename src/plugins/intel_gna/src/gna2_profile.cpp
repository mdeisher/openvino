/* ============================================================================
 *
 * Copyright 2022 Intel Corporation All Rights Reserved.
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

#include "gna2_profile.hpp"
#include "gna2_to_ir.hpp"

void GnaPrintOpName(Gna2OperationType type) {
    switch (type) { 
        case Gna2OperationTypeNone:
            printf("Nop");
            break;
        case Gna2OperationTypeConvolution:
            printf("Conv");
            break;
        case Gna2OperationTypeCopy:
            printf("Copy");
            break;
        case Gna2OperationTypeFullyConnectedAffine:
            printf("Affine");
            break;
        case Gna2OperationTypeElementWiseAffine:
            printf("Diag");
            break;
        case Gna2OperationTypeGmm:
            printf("Gmm");
            break;
        case Gna2OperationTypeRecurrent:
            printf("Rnn");
            break;
        case Gna2OperationTypeTransposition:
            printf("Transpose");
            break;
        case Gna2OperationTypeThreshold:
            printf("Thresh");
            break;
        case Gna2OperationTypeConvolutionDWSC:
            printf("Dwsc");
            break;
        case Gna2OperationTypeEnhancedCopy:
            printf("eCopy");
            break;
        default:
            printf("Unknown");
    }
}

void GnaPrintModelError(Gna2Model* model) {
    Gna2ModelError error;
    Gna2ModelGetLastError(&error);
    uint32_t i = error.Source.OperationIndex;
    uint32_t j = error.Source.OperandIndex;
    uint32_t k = error.Source.ParameterIndex;
    uint32_t m = error.Source.ShapeDimensionIndex;
    printf("ModelCreate failed:\n");
    printf("\tReason: %u\n", error.Reason);
    printf("\tValue: %lld\n", error.Value);
    printf("\tType: %u\n", error.Source.Type);
    printf("\tOperationIndex: %d\n", i);
    printf("\tOperandIndex: %d\n", j);
    printf("\tParameterIndex: %d\n", k);
    printf("\tShapeDimensionIndex: %d (%s)\n", m, DimensionsToString((Gna2Shape*)&(model->Operations[i].Operands[j]->Shape)).c_str());
    printf("\n");
}

void GnaProfileAbort(int32_t flag, char *msg) {
    if (flag) {
        printf("\nError %d %s\n\n", flag, msg);
        exit(-1); 
    }                               
}

void GnaProfilerRun(uint32_t gna_device_id, Gna2Model* model, uint32_t num_infer) {
    uint32_t gna_model_id;

    Gna2Status gna_status = Gna2ModelCreate(gna_device_id, model, &gna_model_id);
    if (gna_status) {
        GnaPrintModelError(model);
    } else {
        Gna2AccelerationMode accel_mode = Gna2AccelerationModeAuto;
        uint32_t timeout_msec = 10000;
        uint32_t request_config_id;
        enum Gna2InstrumentationPoint perf_counter_id[2];
        uint64_t perf_counter[2], perf_counter_min[2], perf_counter_max[2], perf_counter_sum[2];
        uint32_t perf_config_id;
        perf_counter[0] = perf_counter[1] = 0;
        perf_counter_id[0] = Gna2InstrumentationPointHwTotal;
        perf_counter_id[1] = Gna2InstrumentationPointHwStall;
        gna_status = Gna2RequestConfigCreate(gna_model_id, &request_config_id);
        GnaProfileAbort(gna_status, "Gna2RequestConfigCreate failed!");
        gna_status = Gna2RequestConfigSetAccelerationMode(request_config_id, accel_mode);
        GnaProfileAbort(gna_status, "Gna2RequestConfigSetAccelerationMode failed!");
        gna_status = Gna2InstrumentationConfigCreate(2, perf_counter_id, perf_counter, &perf_config_id);
        GnaProfileAbort(gna_status, "Gna2InstrumentationConfigCreate failed!");
        gna_status = Gna2InstrumentationConfigSetUnit(perf_config_id, Gna2InstrumentationUnitCycles);
        GnaProfileAbort(gna_status, "Gna2InstrumentationConfigSetUnit failed!");
        gna_status = Gna2InstrumentationConfigAssignToRequestConfig(perf_config_id, request_config_id);
        GnaProfileAbort(gna_status, "Gna2InstrumentationConfigAssignToRequestConfig failed!");
        uint32_t sat_count = 0;
        for (uint32_t i = 0; i < num_infer; i++) {
            uint32_t request_id;
            gna_status = Gna2RequestEnqueue(request_config_id, &request_id);
            GnaProfileAbort(gna_status, "Gna2RequestEnqueue failed!");
            gna_status = Gna2RequestWait(request_id, timeout_msec);
            GnaProfileAbort(((gna_status!=Gna2StatusSuccess)&&(gna_status!=Gna2StatusWarningArithmeticSaturation)), "Gna2RequestWait failed!");
            sat_count += ((gna_status == Gna2StatusWarningArithmeticSaturation) ? 1 : 0);
            if (i == 0) {
                perf_counter_min[0] = perf_counter_max[0] = perf_counter_sum[0] = perf_counter[0];
                perf_counter_min[1] = perf_counter_max[1] = perf_counter_sum[1] = perf_counter[1];
            } else {
                if (perf_counter[0] > perf_counter_max[0]) {
                    perf_counter_max[0] = perf_counter[0];
                    perf_counter_max[1] = perf_counter[1];
                } else if (perf_counter[0] < perf_counter_min[0]) {
                    perf_counter_min[0] = perf_counter[0];
                    perf_counter_min[1] = perf_counter[1];
                }
                perf_counter_sum[0] += perf_counter[0];
                perf_counter_sum[1] += perf_counter[1];
            }
        }
        printf("%llu,%llu,%f,", perf_counter_min[0], perf_counter_max[0], (float)perf_counter_sum[0]/num_infer);
        printf("%llu,%llu,%f\n", perf_counter_min[1], perf_counter_max[1], (float)perf_counter_sum[1]/num_infer);
        gna_status = Gna2RequestConfigRelease(request_config_id);
        GnaProfileAbort(gna_status, "Gna2RequestConfigRelease failed!");
        if (sat_count > 0) {
            printf("\tSaturation occured in %u / %u frames!\n", sat_count, num_infer);
        }
        gna_status = Gna2InstrumentationConfigRelease(perf_config_id);
        GnaProfileAbort(gna_status, "Gna2InstrumentationConfigRelease failed!");
    }

    gna_status = Gna2ModelRelease(gna_model_id);
    GnaProfileAbort(gna_status, "Gna2ModelRelease failed!");
}

void GnaProfiler(Gna2Model* model) {
    uint32_t num_infer = 100;
    uint32_t gna_device_id = 0;
    uint32_t num_layers = model->NumberOfOperations;
    Gna2Operation* pFirstOperation = model->Operations;

    printf("GNA Profiler:  Model has % d layer%s\n", model->NumberOfOperations, (model->NumberOfOperations>1) ? "s" : "");
    printf("Collecting perf counters over %u runs...\n", num_infer);
    printf("Operation_Profiled,MinTotal,MaxTotal,AvgTotal,MinStall,MaxStall,AvgStall\n");

    Gna2Status gna_status = Gna2DeviceOpen(gna_device_id);
    GnaProfileAbort(gna_status, "Gna2DeviceOpen failed!");

    printf("FullModel,");
    GnaProfilerRun(gna_device_id, model, num_infer);

    for (uint32_t i = 0; i < num_layers; i++) {
        model->NumberOfOperations = 1;
        model->Operations = pFirstOperation + i;
        GnaPrintOpName(model->Operations->Type);
        printf(",");
        GnaProfilerRun(gna_device_id, model, num_infer);
    }

    model->NumberOfOperations = num_layers;
    model->Operations = pFirstOperation;

    gna_status = Gna2DeviceClose(0);
    GnaProfileAbort(gna_status, "Gna2DeviceClose failed!");
}