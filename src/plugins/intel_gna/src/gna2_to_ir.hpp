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

#pragma once

#include <vector>
#include <string>
#include <gna2-model-api.h>

#define ALIGN(memSize, pad) (((long long unsigned int)((memSize) + pad - 1) / pad) * pad)
#define ALIGN64(memSize)    ALIGN(memSize, 64)

typedef enum type_gna_buffer_indices {
    IN_IDX = 0,
    OUT_IDX = 1,
    WGT_IDX = 2,
    BIAS_IDX = 3,
    ACT_IDX = 4,
    SCL_IDX = 5
} gna_buffer_indices;

typedef struct {
	uint8_t* start;
	uint8_t* end;
} gna_memory_region;

typedef struct {
	std::string name;
	std::string value;
} gna_attribute;

typedef struct {
    uint32_t levels;
    float max;
    float min;
    float mean;
    double scale_factor;
} gna_quant_data;

typedef struct gna_port_type {
	std::string name;
    Gna2Shape shape;
	Gna2DataType type;
	std::string type_name;
	uint32_t operand_index;
	gna_memory_region region;
	uint32_t num_elements;
    gna_quant_data quant;  // data used for quantization (if available)
	bool is_constant = false;  // applies to inputs only
	uint32_t ext_id = 0;  // identifier of staging buffer (if available)
	uint32_t ext_size = 0; // size of original ngraph buffer (if available)
} gna_port;

typedef struct gna_node_type {
	std::string name;
	Gna2OperationType type;
	std::string type_name;
	std::vector<gna_attribute> attribute;
	std::vector<gna_port> input;
	std::vector<gna_port> output;
	std::string ext_name;  // corresponding ngraph node name (if available)
    std::string ext_friendly_name;  // corresponding ngraph node name (if available)
    uint32_t ext_id = 0;  // corresponding ngraph node id (if available)
    std::string ext_type;  // corresponding ngraph node type (if available)
} gna_node;

typedef struct {
    uint32_t node_index;
    uint32_t port_index;
} gna_port_location;

typedef struct {
	uint32_t from_layer;
	uint32_t from_port;
	uint32_t to_layer;
	uint32_t to_port;
} gna_edge;

void PrintNode(gna_node node);
std::string DimensionsToString(Gna2Shape *shape);
uint32_t GnaDataNumBytes(Gna2DataType data_type);
uint32_t GnaShapeNumElements(Gna2Shape shape);
gna_memory_region GetRegion(const Gna2Tensor* tensor);
gna_port GnaPort(const Gna2Tensor *tensor, std::string name, uint32_t idx);
gna_attribute GnaAttribute(std::string name, std::string value);
void ReplaceGnaAttribute(gna_node& node, std::string name, std::string value);
std::string GetGnaTypeName(Gna2OperationType type);
std::string GetGnaPortName(gna_buffer_indices index);
void BuildGnaNodeList(std::vector<gna_node> & node_list, Gna2Model model);
gna_edge GnaEdge(uint32_t from_layer, uint32_t from_port, uint32_t to_layer, uint32_t to_port);
bool BufferOverwrite(gna_memory_region src, gna_memory_region dest);
void BuildGnaEdgeList(std::vector<gna_edge>& edge_list, std::vector<gna_node> node_list);
bool GetConnectedNodeIndices(uint32_t &node_out_idx, uint32_t &port_out_idx, uint32_t node_in_idx, uint32_t port_in_idx, std::vector<gna_node> node_list);
void GnaWriteXml(char* filename, Gna2Model model);
void GnaWriteXml(char* filename, Gna2Model model, std::vector<void*> inputs);
void GnaWriteXml(char* filename, std::vector<gna_node> node);
