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

void PrintNode(gna_node node) {
    printf("%s\n", node.name.c_str());
    for (uint32_t i = 0; i < node.input.size(); i++) {
        printf("\tin%d: %s\n", i, node.input[i].name.c_str());
        printf("\t\t%p - %p\n", node.input[i].region.start, node.input[i].region.end);
    }
    for (uint32_t i = 0; i < node.output.size(); i++) {
        printf("\tout%d: %s\n", i, node.output[i].name.c_str());
        printf("\t\t%p - %p\n", node.output[i].region.start, node.output[i].region.end);
    }
}

std::string DimensionsToString(Gna2Shape *shape) {
    // Returns a string with the Gna2Shape dimensions separated by commas
	std::string result;
	if (shape) {
		result = std::to_string(shape->Dimensions[0]);
		for (uint32_t i = 1; i < shape->NumberOfDimensions; i++) {
			result = result + "," + std::to_string(shape->Dimensions[i]);
		}
	}
	else {
		result = "none";
	}
	return result;
}

uint32_t GnaDataNumBytes(Gna2DataType data_type) {
    // Returns the number of bytes per element for a given Gna2DataType
	switch (data_type) {
		case Gna2DataTypeInt8:
			return 1;
		case Gna2DataTypeInt16:
			return 2;
		case Gna2DataTypeInt32:
			return 4;
		case Gna2DataTypeInt64:
			return 8;
		case Gna2DataTypeUint8:
			return 1;
		case Gna2DataTypeUint16:
			return 2;
		case Gna2DataTypeUint32:
			return 4;
		case Gna2DataTypeUint64:
			return 8;
        case Gna2DataTypePwlSegment:
            return 8;
        case Gna2DataTypeCompoundBias:
            return 8;
        default:
            fprintf(stderr, "Illegal Gna2DataType!\n");
	}

	return 0;
}

uint32_t GnaShapeNumElements(Gna2Shape shape) {
    // Returns the total number of elements in a Gna2Shape
	uint32_t num_elements = 1;
	for (uint32_t i = 0; i < shape.NumberOfDimensions; i++) {
		num_elements *= shape.Dimensions[i];
	}
	return (num_elements);
}

gna_memory_region GetRegion(const Gna2Tensor* tensor) {
    // Returns the start and end of a Gna2Tensor data memory
	gna_memory_region region;
	region.start = (uint8_t*)tensor->Data;
	region.end = (uint8_t*)tensor->Data + GnaShapeNumElements(tensor->Shape) * GnaDataNumBytes(tensor->Type);
	return region;
}

gna_port GnaPort(const Gna2Tensor *tensor, std::string name, uint32_t idx) {
    // Returns a gna_port structure populated from the specified tensor port
	gna_port port;

	port.name = name;
	port.shape = tensor->Shape;
	port.type = tensor->Type;
	port.type_name = "I" + std::to_string(8 * GnaDataNumBytes(port.type));
	port.operand_index = idx;
	port.region = GetRegion(tensor);
	port.num_elements = GnaShapeNumElements(tensor->Shape);

	return port;
}

gna_attribute GnaAttribute(std::string name, std::string value) {
    // Returns a gna_attribute structure given a name and value
	gna_attribute attribute;

	attribute.name = name;
	attribute.value = value;

	return attribute;
}

void ReplaceGnaAttribute(gna_node &node, std::string name, std::string value) {
    // Replaces the value of a given gna_node attribute if it exists, otherwise creates it
    bool found = false;
    for (uint32_t i = 0; i < node.attribute.size(); i++) {
        if (node.attribute[i].name == name) {
            node.attribute[i].value = value;
            found = true;
            break;
        }
    }
    if (!found) {
        node.attribute.push_back(GnaAttribute(name, value));
    }
}

std::string GetGnaTypeName(Gna2OperationType type) {
    // Returns a string given a Gna2OperationType
    std::string type_name;
    if (type == Gna2OperationTypeConvolution) {
        type_name = "GnaConvolution";
    } else if (type == Gna2OperationTypeCopy) {
        type_name = "GnaCopy";
    } else if (type == Gna2OperationTypeFullyConnectedAffine) {
        type_name = "GnaAffine";
    } else if (type == Gna2OperationTypeElementWiseAffine) {
        type_name = "GnaDiagonalAffine";
    } else if (type == Gna2OperationTypeTransposition) {
        type_name = "GnaTranspose";
    }
    return type_name;
}

std::string GetGnaPortName(gna_buffer_indices index) {
    // Returns a string given an operand index
    std::string port_name;
    if (index == IN_IDX) {
        port_name = "input";
    } else if (index == OUT_IDX) {
        port_name = "output";
    } else if (index == WGT_IDX) {
        port_name = "weights";
    } else if (index == BIAS_IDX) {
        port_name = "bias";
    } else if (index == ACT_IDX) {
        port_name = "activation";
    } else if (index == SCL_IDX) {
        port_name = "rowscale";
    }
    return port_name;
}

gna_attribute AddressToAttribute(std::string name, uint8_t* addr) {
    // Returns a gna_attribute given a name and memory address
    gna_attribute attr;
    attr.name = name;
    attr.value = std::to_string((long long unsigned int)addr);
    return attr;
}

void BuildGnaNodeList(std::vector<gna_node>& node_list, Gna2Model model) {
    // Given a Gna2Model, populates a vector of gna_node structures
	for (uint32_t i = 0; i < model.NumberOfOperations; i++) {
		Gna2Operation op = model.Operations[i];
		gna_node node;
        gna_attribute attr;
		node.type_name = GetGnaTypeName(op.Type);
		node.name = node.type_name + "_" + std::to_string(i);
		node.type = op.Type;
		node.input.push_back(GnaPort(op.Operands[0], "input", 0));
		node.attribute.push_back(GnaAttribute("inputShape", DimensionsToString((Gna2Shape*)&op.Operands[0]->Shape)));
        attr = AddressToAttribute("inputBegin", node.input[0].region.start);
        node.attribute.push_back(attr);
        attr = AddressToAttribute("inputEnd", node.input[0].region.end);
        node.attribute.push_back(attr);
		node.output.push_back(GnaPort(op.Operands[1], "output", 1));
		node.attribute.push_back(GnaAttribute("outputShape", DimensionsToString((Gna2Shape*)&op.Operands[1]->Shape)));
        attr = AddressToAttribute("outputBegin", node.output[0].region.start);
        node.attribute.push_back(attr);
        attr = AddressToAttribute("outputEnd", node.output[0].region.end);
        node.attribute.push_back(attr);
		if (op.Type == Gna2OperationTypeConvolution) {
			node.input.push_back(GnaPort(op.Operands[2], "weights", 2));
			node.attribute.push_back(GnaAttribute("kernelShape", DimensionsToString((Gna2Shape*)&op.Operands[2]->Shape)));
			if (op.Operands[3]) {
				node.input.push_back(GnaPort(op.Operands[3], "bias", 3));
				node.attribute.push_back(GnaAttribute("biasShape", DimensionsToString((Gna2Shape*)&op.Operands[3]->Shape)));
			}
			if (op.Operands[4]) {
				node.input.push_back(GnaPort(op.Operands[4], "activation", 4));
			}
			node.attribute.push_back(GnaAttribute("has_activation", (op.Operands[4]) ? "yes" : "no"));
			if (op.NumberOfParameters > 0) {
				Gna2Shape* stride = (Gna2Shape*)op.Parameters[0];
				node.attribute.push_back(GnaAttribute("convolutionStride", DimensionsToString(stride)));
			}
			if (model.Operations[i].NumberOfParameters > 1) {
				Gna2BiasMode* biasMode = (Gna2BiasMode*)op.Parameters[1];
                if (biasMode != NULL) {
					node.attribute.push_back(GnaAttribute("biasMode", std::to_string((uint32_t)*biasMode)));
				}
			}
			if (model.Operations[i].NumberOfParameters > 2) {
				Gna2PoolingMode* poolingMode = (Gna2PoolingMode*)op.Parameters[2];
                if (poolingMode != NULL) {
					node.attribute.push_back(GnaAttribute("poolingMode", std::to_string((uint32_t)*poolingMode)));
				}
			}
			if (model.Operations[i].NumberOfParameters > 3) {
				Gna2Shape* poolingWindow = (Gna2Shape*)op.Parameters[3];
                if (poolingWindow != NULL) {
					node.attribute.push_back(GnaAttribute("poolingWindow", DimensionsToString(poolingWindow)));
				}
			}
			if (model.Operations[i].NumberOfParameters > 4) {
				Gna2Shape* poolingStride = (Gna2Shape*)op.Parameters[4];
                if (poolingStride != NULL) {
					node.attribute.push_back(GnaAttribute("poolingStride", DimensionsToString(poolingStride)));
				}
			}
			if (model.Operations[i].NumberOfParameters > 5) {
				Gna2Shape* zeroPadding = (Gna2Shape*)op.Parameters[5];
                if (zeroPadding != NULL) {
					node.attribute.push_back(GnaAttribute("zeroPadding", DimensionsToString(zeroPadding)));
				}
			}

		} else if (model.Operations[i].Type == Gna2OperationTypeCopy) {
			if (op.NumberOfParameters > 0) {
				Gna2Shape* shape = (Gna2Shape*)op.Parameters[0];
				node.attribute.push_back(GnaAttribute("shape", DimensionsToString(shape)));
			}

		} else if (model.Operations[i].Type == Gna2OperationTypeFullyConnectedAffine) {
			node.input.push_back(GnaPort(op.Operands[2], "weights", 2));
			node.attribute.push_back(GnaAttribute("weightShape", DimensionsToString((Gna2Shape*)&op.Operands[2]->Shape)));
			node.input.push_back(GnaPort(op.Operands[3], "bias", 3));
			node.attribute.push_back(GnaAttribute("biasShape", DimensionsToString((Gna2Shape*)&op.Operands[3]->Shape)));
			if (op.Operands[4]) {
				node.input.push_back(GnaPort(op.Operands[4], "activation", 4));
			}
			node.attribute.push_back(GnaAttribute("has_activation", (op.Operands[4]) ? "yes" : "no"));
			if (op.Operands[5]) {
				node.input.push_back(GnaPort(op.Operands[5], "row_multipliers", 5));
			}
			node.attribute.push_back(GnaAttribute("has_row_multipliers", (op.Operands[5]) ? "yes" : "no"));
			if (op.NumberOfParameters > 0) {
				Gna2BiasMode* biasMode = (Gna2BiasMode*)op.Parameters[0];
                if (biasMode != NULL) {
					node.attribute.push_back(GnaAttribute("biasMode", std::to_string((uint32_t)*biasMode)));
                }
			}
			if (op.NumberOfParameters > 1) {
				uint32_t* biasVectorIndex = (uint32_t*)op.Parameters[1];
				node.attribute.push_back(GnaAttribute("biasVectorIndex", std::to_string(*biasVectorIndex)));
			}

		} else if (model.Operations[i].Type == Gna2OperationTypeElementWiseAffine) {
			node.input.push_back(GnaPort(op.Operands[2], "weights", 2));
			node.attribute.push_back(GnaAttribute("weightShape", DimensionsToString((Gna2Shape*)&op.Operands[2]->Shape)));
			node.input.push_back(GnaPort(op.Operands[3], "bias", 3));
			node.attribute.push_back(GnaAttribute("biasShape", DimensionsToString((Gna2Shape*)&op.Operands[3]->Shape)));
			if (op.Operands[4]) {
				node.input.push_back(GnaPort(op.Operands[4], "activation", 4));
			}
			node.attribute.push_back(GnaAttribute("has_activation", (op.Operands[4]) ? "yes" : "no"));

		}
		node_list.push_back(node);
	}
}

gna_edge GnaEdge(uint32_t from_layer, uint32_t from_port, uint32_t to_layer, uint32_t to_port) {
    // Returns a gna_edge structure given connection information
	gna_edge edge;

	edge.from_layer = from_layer;
	edge.from_port = from_port;
	edge.to_layer = to_layer;
	edge.to_port = to_port;

	return edge;
}

bool BufferOverwrite(gna_memory_region src, gna_memory_region dest) {
    // Determine if dst buffer would be overwritten by src
    bool overwrite = false;
	// is first element of source inside destination region?
    if ((src.start >= dest.start) && (src.start < dest.end)) {
        overwrite = true;
    }
	// is last element of source within destination region?
    if (((src.end - 1) >= dest.start) && ((src.end - 1) < dest.end)) {
        overwrite = true;
    }
    return overwrite;
}

bool ContainedIn(gna_memory_region small, gna_memory_region big) {
	// Determine if "small" region is contained in "big" region
    bool is_contained_in = false;
    if ((small.start >= big.start) && (small.end <= big.end)) {
        is_contained_in= true;
    }
    return is_contained_in;
}

gna_memory_region IntersectionRegion(gna_memory_region region1, gna_memory_region region2) {
    // Return intersection of two memory regions
    gna_memory_region intersection = {0, 0}; // null intersection

    if (region1.start <= region2.start) {
        if (region1.end >= region2.start) {
            if (region1.end >= region2.end) {
                // region1:  |------------|
                // region2:      |-----|
                intersection.start = region2.start;
                intersection.end = region2.end;
            } else {
                // region1:  |------------|
                // region2:      |-----------|
                intersection.start = region2.start;
                intersection.end = region1.end;
            }
        }
    } else {
        if (region2.end >= region1.end) {
            // region1:   |------|
            // region2: |------------|
            intersection.start = region1.start;
            intersection.end = region1.end;
        } else if (region2.end >= region1.start) {
            // region1:      |------------|
            // region2: |-----------|
            intersection.start = region1.start;
            intersection.end = region2.end;
        }
    }

    return intersection;
}

std::vector<gna_memory_region> UnionSet(std::vector<gna_memory_region> set) {
    // Return the union of set of memory regions
    std::vector<gna_memory_region> union_set = set;
    bool merged = true;
    while (merged) {
        merged = false;
        std::vector<gna_memory_region> temp_set; 
        gna_memory_region union_region = union_set[0];
        for (uint32_t i = 1; i < union_set.size(); i++) {
            gna_memory_region region2 = union_set[i - 1];
            if (union_region.start <= region2.start) {
                if (union_region.end >= region2.start) {
                    if (union_region.end >= region2.end) {
                        // union_region:  |------------|
                        // region2:           |-----|
                        merged = true;
                    } else {
                        // union_region:  |------------|
                        // region2:           |-----------|
                        union_region.end = region2.end;
                        merged = true;
                    }
                } else {
                    // union_region:  |------------|
                    // region2:                       |-----|
                    temp_set.push_back(region2);  // not mergeable
                }
            } else {
                if (region2.end >= union_region.end) {
                    // union_region:  |------------|
                    // region2:     |----------------|
                    union_region.start = region2.start;
                    union_region.end = region2.end;
                    merged = true;
                } else if (region2.end >= union_region.start) {
                    // union_region:  |------------|
                    // region2:     |-----------|
                    union_region.start = region2.start;
                    merged = true;
                } else {
                    // union_region:  |------------|
                    // region2: |---|
                    temp_set.push_back(region2);  // not mergeable
                }
            }
        }
        temp_set.push_back(union_region);
        union_set = temp_set;
    }

    return union_set;
}

bool SetCoversRegion(std::vector<gna_memory_region> set, gna_memory_region region) {
    // Determine if a set of regions completely covers a given memory region
    std::vector<gna_memory_region> intersection;
    bool covered = true;
	for (uint32_t i = 0; i < set.size(); i++) {
        gna_memory_region overlap;
        if (set[i].start <= region.start) {
            overlap.start = region.start;
            if (set[i].end > region.start) {
                overlap.end = (set[i].end <= region.end) ? set[i].end : region.end;
            } else {
                overlap.end = region.start + 1;
            }
        } else {
            overlap.start = (set[i].start < region.end) ? set[i].start : region.end;
            overlap.end = (set[i].end <= region.end) ? set[i].end : region.end;
        }
        intersection.push_back(overlap);
	}

	return covered;
}



void BuildGnaEdgeListByOutput(std::vector<gna_edge>& edge_list, std::vector<gna_node> node_list) {
    // Creates a vector of gna_edge structures given a vector of gna_node structures
    for (uint32_t i = 0; i < node_list.size(); i++) {
        PrintNode(node_list[i]);
        for (uint32_t j = 0; j < node_list[i].output.size(); j++) {
            bool edge_found = false;
            bool overwrite = false;
			// search ahead in execution order until another node overwrites the output buffer
            for (uint32_t k = i + 1; k < node_list.size(); k++) {
                for (uint32_t l = 0; l < node_list[k].input.size(); l++) {
                    if (BufferOverwrite(node_list[i].output[j].region, node_list[k].input[l].region)) {
                        edge_list.push_back(GnaEdge(i, j, k, l));
                        printf("Edge %s %s --> %s %s\n", node_list[i].name.c_str(), node_list[i].output[j].name.c_str(),
                               node_list[k].name.c_str(), node_list[k].input[l].name.c_str());
                        edge_found = true;
                    }
                }
                for (uint32_t m = 0; m < node_list[k].output.size(); m++) {
                    if (BufferOverwrite(node_list[k].output[m].region, node_list[i].output[j].region)) {
                        overwrite = true;
                    }
                }
                if (overwrite) {
                    //break;
                }
            }
			// final outputs will not be connected to any other node
            if (!edge_found) {
                printf("node %d %s output not connected to another node\n", i, node_list[i].name.c_str());
            }
        }
    }
}

std::vector<gna_node> BuildInputList(std::vector<gna_node> node_list, std::vector<void*> graph_inputs) {
    // Return a list of inputs given the node list and list of graph input addresses
    std::vector<gna_node> inputs;
    for (uint32_t i = 0; i < node_list.size(); i++) {
        for (uint32_t j = 0; j < node_list[i].input.size(); j++) {
            gna_memory_region dst_region = node_list[i].input[j].region;
            bool found = false;
            // search for parents in node list
            for (uint32_t k = i; k > 0; k--) {
                for (uint32_t l = 0; l < node_list[k - 1].output.size(); l++) {
                    gna_memory_region src_region = node_list[k - 1].output[l].region;
                    if (BufferOverwrite(src_region, dst_region)) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
            // search for parents in input list
            if (!found) {
                for (uint32_t k = 0; k < inputs.size(); k++) {
                    gna_memory_region src_region = inputs[k].output[0].region;
                    if (BufferOverwrite(src_region, dst_region)) {
                        found = true;
                        break;
                    }
                }
            }
            // create dummy input node if needed
            // if (!found && (node_list[i].input[j].name == "input")) {  // leave out weights and biases
            if (!found) {  // include weights and biases
                gna_node node;
                gna_port output = node_list[i].input[j];
                gna_attribute attr;
                node.type = Gna2OperationTypeNone;
                node.type_name = node_list[i].input[j].name;
                node.name = node_list[i].input[j].name + std::to_string(i);
                output.name = "output";
                node.output.push_back(output);
                attr = AddressToAttribute("begin", node_list[i].input[j].region.start);
                node.attribute.push_back(attr);
                attr = AddressToAttribute("end", node_list[i].input[j].region.end);
                node.attribute.push_back(attr);
                // if node is not in graph input list then change name
                bool is_input = false;
                for (uint32_t k = 0; k < graph_inputs.size(); k++) {
                    if (output.region.start == graph_inputs[k]) { // TODO: check for overlap and not just start address
                        is_input = true;
                    }
                }
                if (!is_input && (node.type_name == "input")) {
                    node.type_name = "const_input";
                }
                inputs.push_back(node);
            }
        }
    }
    return inputs;
}

std::vector<gna_node> BuildOutputList(std::vector<gna_node> node_list) {
    // Return a list of outputs given the node list
    std::vector<gna_node> outputs;
    for (uint32_t i = 0; i < node_list.size(); i++) {
        for (uint32_t j = 0; j < node_list[i].output.size(); j++) {
            gna_memory_region src_region = node_list[i].output[j].region;
            bool found = false;
            // search for children
            for (uint32_t k = i + 1; k < node_list.size(); k++) {
                for (uint32_t l = 0; l < node_list[k].input.size(); l++) {
                    gna_memory_region dst_region = node_list[k].input[l].region;
                    if (BufferOverwrite(src_region, dst_region)) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
            if (!found) {
                gna_node node;
                gna_port input = node_list[i].output[j];
                gna_attribute attr;
                node.type = Gna2OperationTypeNone;
                node.type_name = "Output";
                node.name = "Output_" + std::to_string(i) + "_" + std::to_string(j);
                input.name = "input";
                node.input.push_back(input);
                attr = AddressToAttribute("begin", node_list[i].output[j].region.start);
                node.attribute.push_back(attr);
                attr = AddressToAttribute("end", node_list[i].output[j].region.end);
                node.attribute.push_back(attr);
                outputs.push_back(node);
            }
        }
    }
    return outputs;
}

void InsertDummyNodes(std::vector<gna_node>& node_list, std::vector<void*> graph_inputs) {
    // Insert non-GNA dummy nodes (inputs, outputs, splits, concats) into the node list
    std::vector<gna_node> inputs;
    std::vector<gna_node> outputs;
    std::vector<gna_node> concats;
    std::vector<gna_node> splits;
    std::vector<std::string> concat_before;
    std::vector<std::string> split_before;

    // find inputs and insert dummy input nodes into graph
    inputs = BuildInputList(node_list, graph_inputs);
    for (uint32_t i = 0; i < inputs.size(); i++) {
        node_list.insert(node_list.begin() + i, inputs[i]);
    }

    // find outputs and insert dummy output nodes into graph
    outputs = BuildOutputList(node_list);
    for (uint32_t i = 0; i < outputs.size(); i++) {
        node_list.insert(node_list.end(), outputs[i]);
    }

    // search for concats
    for (uint32_t i = 0; i < node_list.size(); i++) {
        //PrintNode(node_list[i]);
        for (uint32_t j = 0; j < node_list[i].input.size(); j++) {
            gna_memory_region dst_region = node_list[i].input[j].region;
            std::vector<gna_port_location> parent;
            std::vector<gna_memory_region> intersection;
            // find potential parent nodes
			for (uint32_t k = i; k > 0; k--) {
                for (uint32_t l = 0; l < node_list[k - 1].output.size(); l++) {
                    gna_memory_region src_region = node_list[k - 1].output[l].region;
                    if (BufferOverwrite(src_region, dst_region)) {
                        parent.push_back({k - 1, l});
                        intersection.push_back(IntersectionRegion(src_region, dst_region));
					}
				}
			}
            if ((intersection.size() > 0) && (!ContainedIn(dst_region, intersection[0]))) {  // concat found
                gna_node concat;
                gna_attribute attr;
                gna_port concat_output = node_list[i].input[j];
                std::vector<gna_memory_region> remaining_region;
                concat.type = Gna2OperationTypeNone;  // concats are not valid GNA layer types
                concat.type_name = "Concat/Merge";
                concat.name = "Concat_" + std::to_string(i) + "_" + std::to_string(j);
                concat_output.name = "output";
                attr = AddressToAttribute("outputBegin", concat_output.region.start);
                concat.attribute.push_back(attr);
                attr = AddressToAttribute("outputEnd", concat_output.region.end);
                concat.attribute.push_back(attr);
                attr.name = "outputShape";
                attr.value = DimensionsToString(&concat_output.shape);
                concat.attribute.push_back(attr);
                concat.output.push_back(concat_output);
                remaining_region.push_back(dst_region);
                for (uint32_t k = 0; k < intersection.size(); k++) { // reverse time order
                    std::vector<uint32_t> member_of_region;
                    for (uint32_t l = 0; l < remaining_region.size(); l++) {
                        gna_memory_region overlap = IntersectionRegion(intersection[k], remaining_region[l]);
                        if (overlap.end > overlap.start + 1) {  // overlap not empty
                            member_of_region.push_back(l);
                        }
                    }
                    if (member_of_region.size() > 1) {
                        // insert split of output parent[k] into member regions
                        gna_node split;
                        gna_port input = node_list[parent[k].node_index].output[parent[k].port_index];
                        split.type = Gna2OperationTypeNone; // splits are not valid GNA layer types
                        split.type_name = "Split";
                        split.name = "Split_" + std::to_string(i) + "_" + std::to_string(j);
                        input.name = "input";
                        split.input.push_back(input);
                        for (uint32_t l = 0; l < member_of_region.size(); l++) {
                            gna_port output;
                            output.region = IntersectionRegion(intersection[k], remaining_region[member_of_region[l]]);
                            output.is_constant = false;
                            output.type = node_list[parent[k].node_index].output[parent[k].port_index].type;
                            output.type_name = node_list[parent[k].node_index].output[parent[k].port_index].type_name;
                            output.num_elements = (uint32_t)(output.region.end - output.region.start) / GnaDataNumBytes(output.type);
                            output.name = "output";
                            split.output.push_back(output);
                            output.name = "input";
                            attr = AddressToAttribute("input"+std::to_string(concat.input.size())+"Begin", output.region.start);
                            concat.attribute.push_back(attr);
                            attr = AddressToAttribute("input"+std::to_string(concat.input.size())+"End", output.region.end);
                            concat.attribute.push_back(attr);
                            attr.name = "input" + std::to_string(concat.input.size()) + "Shape";
                            attr.value = DimensionsToString(&output.shape);
                            concat.attribute.push_back(attr);
                            concat.input.push_back(output);
                        }
                        splits.push_back(split);
                        split_before.push_back(node_list[i].name);

                    } else {
                        gna_port input = node_list[parent[k].node_index].output[parent[k].port_index];
                        input.name = "input";
                        attr = AddressToAttribute("input"+std::to_string(concat.input.size())+"Begin", input.region.start);
                        concat.attribute.push_back(attr);
                        attr = AddressToAttribute("input"+std::to_string(concat.input.size())+"End", input.region.end);
                        concat.attribute.push_back(attr);
                        attr.name = "input" + std::to_string(concat.input.size()) + "Shape";
                        attr.value = DimensionsToString(&input.shape);
                        concat.attribute.push_back(attr);
                        concat.input.push_back(input);
                    }
                    // remove overlap from remaining_region
                    for (uint32_t l = 0; l < member_of_region.size(); l++) {
                        gna_memory_region remaining = remaining_region[member_of_region[l]];
                        gna_memory_region overlap = IntersectionRegion(intersection[k], remaining_region[member_of_region[l]]);
                        if (overlap.start == remaining.start) {
                            if (overlap.end < remaining.end) {      // remove beginning of region
                                remaining_region[member_of_region[l]].start = overlap.end;
                            } else {                                // remove entire region
                                remaining_region.erase(remaining_region.begin() + member_of_region[l]);
                            }
                        } else if (overlap.end == remaining.end) {  // remove end of region
                            remaining_region[member_of_region[l]].end = overlap.start;
                        } else {                                    // remove middle of region, split region
                            gna_memory_region new_remaining;
                            new_remaining.start = overlap.end;
                            new_remaining.end = remaining_region[member_of_region[l]].end;
                            remaining_region[member_of_region[l]].end = overlap.start;
                            remaining_region.insert(remaining_region.begin() + member_of_region[l] + 1, new_remaining);
                        }
                    }
                    if (remaining_region.size() == 0) {
                        break;
                    } else if (k == intersection.size() - 1) { // concat input(s) not found -- probably implicit padding
                        for (uint32_t l = 0; l < remaining_region.size(); l++) {
                            gna_port input;
                            input.name = "input";
                            input.region = remaining_region[l];
                            input.type = Gna2DataTypeInt16;  // only works for GNA 3.0 and earlier
                            input.type_name = "I" + std::to_string(8 * GnaDataNumBytes(input.type));
                            input.is_constant = true;
                            input.num_elements = (uint32_t)(input.region.end - input.region.start) / GnaDataNumBytes(input.type);
                            concat.input.push_back(input);
                            // Need to create more dummy inputs here or perhaps make a second input
                            // pass after concat detection pass.
                        }
                    }
                }
                concats.push_back(concat);
                concat_before.push_back(node_list[i].name);
            }
		}
	}
    // insert dummy nodes
    for (uint32_t i = 0; i < splits.size(); i++) {
        uint32_t k = (uint32_t)node_list.size(); // insert at end of not found
        for (uint32_t j = 0; j < node_list.size(); j++) {
            if (node_list[j].name == split_before[i]) {
                k = j;
                break;
            }
        }
        node_list.insert(node_list.begin() + k, splits[i]);
    }
    for (uint32_t i = 0; i < concats.size(); i++) {
        uint32_t k = (uint32_t)node_list.size();  // insert at end of not found
        for (uint32_t j = 0; j < node_list.size(); j++) {
            if (node_list[j].name == concat_before[i]) {
                k = j;
                break;
            }
        }
        node_list.insert(node_list.begin() + k, concats[i]);
    }
    // find concat inputs and insert dummy input nodes into graph
    inputs.clear();
    inputs = BuildInputList(node_list, graph_inputs);
    for (uint32_t i = 0; i < inputs.size(); i++) {
        node_list.insert(node_list.begin() + i, inputs[i]);
    }
}

void BuildGnaEdgeList(std::vector<gna_edge>& edge_list, std::vector<gna_node> node_list) {
    // Creates a vector of gna_edge structures given a vector of gna_node structures
	for (uint32_t i = 0; i < node_list.size(); i++) {
        //PrintNode(node_list[i]);
		for (uint32_t j = 0; j < node_list[i].input.size(); j++) {
            gna_memory_region dst_region = node_list[i].input[j].region;
            bool found = false;
            // find potential parent nodes
			for (uint32_t k = i; k > 0; k--) {
                for (uint32_t l = 0; l < node_list[k - 1].output.size(); l++) {
                    gna_memory_region src_region = node_list[k - 1].output[l].region;
                    if (ContainedIn(dst_region, src_region)) {
						edge_list.push_back(GnaEdge(k-1, l, i, j));
                        //printf("Edge %s %s %d --> %s %s %d\n", node_list[k - 1].name.c_str(), node_list[k - 1].output[l].name.c_str(), l,
                        //       node_list[i].name.c_str(), node_list[i].input[j].name.c_str(), j);
                        found = true;
						break;
					}
				}
                if (found) {
                    break;
                }
			}
			// constants can be out of order
            for (uint32_t k = i + 1; k < node_list.size(); k++) {
                if ((node_list[k].output.size() > 0) && node_list[k].output[0].is_constant) {
                    if (ContainedIn(node_list[i].input[j].region, node_list[k].output[0].region)) {
                        edge_list.push_back(GnaEdge(k, 0, i, j));
                        //printf("Edge %s %s %d --> %s %s %d\n", node_list[k].name.c_str(), node_list[k].output[0].name.c_str(), 0,
                        //       node_list[i].name.c_str(), node_list[i].input[j].name.c_str(), j);
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
            if ((!found) && (j==0)) {
                printf("node %d %s input buffer not found\n", i, node_list[i].name.c_str());
            }
		}
	}
}

bool GetConnectedNodeIndices(uint32_t &node_out_idx, uint32_t &port_out_idx, uint32_t node_in_idx, uint32_t port_in_idx, std::vector<gna_node> node_list) {
    // Sets the parent node and port indices and returns true of the parent was found
    bool found = false;
    for (uint32_t i = node_in_idx; i > 0; i--) {
        for (uint32_t j = 0; j < node_list[i - 1].output.size(); j++) {
            if ((node_list[node_in_idx].input[port_in_idx].region.start >= node_list[i - 1].output[j].region.start) &&
                (node_list[node_in_idx].input[port_in_idx].region.start < node_list[i - 1].output[j].region.end)) {
                node_out_idx = i - 1;
                port_out_idx = j;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        // constants can be out of order
        for (uint32_t i = node_in_idx + 1; i < node_list.size(); i++) {
            if ((node_list[i].output.size() > 0) && node_list[i].output[0].is_constant) {
                if ((node_list[node_in_idx].input[port_in_idx].region.start >= node_list[i].output[0].region.start) &&
                    (node_list[node_in_idx].input[port_in_idx].region.start < node_list[i].output[0].region.end)) {
                    node_out_idx = i;
                    port_out_idx = 0;
                    found = true;
                    break;
                }
            }
        }
    }
    return found;
}

void GnaWriteXml(char* filename, Gna2Model model) {
    // Create an IR-XML-like representation of the pre-GNA graph for display in Netron
    std::vector<void*> inputs;  // empty list will be ignored
    GnaWriteXml(filename, model, inputs);
}

// This version can accept an input list from the OpenVINO GNA plugin.
// Having an input list clears up ambiguity of consts and true inputs.
// From the GNA plugin (e.g., GNAModelSerial::Export) just add this:
//    auto inputs = inputs_.Get()[0].ptrs;
//    GnaWriteXml("gna_model.xml", *gna2model_, inputs);
//
void GnaWriteXml(char* filename, Gna2Model model, std::vector<void*> inputs) {
	// Create an IR-XML-like representation of the pre-GNA graph for display in Netron
	FILE *fp = fopen(filename, "w");
	if (fp != NULL) {
		std::vector<gna_node> node;
		std::vector<gna_edge> edge;
		BuildGnaNodeList(node, model);
        InsertDummyNodes(node, inputs);
		BuildGnaEdgeList(edge, node);
		fprintf(fp, "<?xml version=\"1.0\"?>\n");
		fprintf(fp, "<net name=\"network\" version=\"10\">\n");
		fprintf(fp, "\t<layers>\n");
		for (uint32_t i = 0; i < node.size(); i++) {
			fprintf(fp, "\t\t<layer id=\"%u\" name=\"%s\" type=\"%s\" version=\"opset1\">\n",
				i, node[i].name.c_str(), node[i].type_name.c_str());
			fprintf(fp, "\t\t\t<data ");
			for (uint32_t j = 0; j < node[i].attribute.size(); j++) {
				fprintf(fp, "%s=\"%s\" ", node[i].attribute[j].name.c_str(), node[i].attribute[j].value.c_str());
			}
			fprintf(fp, "/>\n");
			fprintf(fp, "\t\t\t<input>\n");
			for (uint32_t j = 0; j < node[i].input.size(); j++) {
				fprintf(fp, "\t\t\t\t<port id=\"%u\" precision=\"%s\">\n", j, node[i].input[j].type_name.c_str());
				fprintf(fp, "\t\t\t\t\t<dim>%u</dim>\n", node[i].input[j].num_elements);
				fprintf(fp, "\t\t\t\t</port>\n");
			}
			fprintf(fp, "\t\t\t</input>\n");
			fprintf(fp, "\t\t\t<output>\n");
			for (uint32_t j = 0; j < node[i].output.size(); j++) {
				fprintf(fp, "\t\t\t\t<port id=\"%u\" precision=\"%s\">\n", j, node[i].output[j].type_name.c_str());
				fprintf(fp, "\t\t\t\t\t<dim>%u</dim>\n", node[i].output[j].num_elements);
				fprintf(fp, "\t\t\t\t</port>\n");
			}
			fprintf(fp, "\t\t\t</output>\n");
			fprintf(fp, "\t\t</layer>\n");
		}
		fprintf(fp, "\t</layers>\n");
		fprintf(fp, "\t<edges>\n");
		for (uint32_t i = 0; i < edge.size(); i++) {
			fprintf(fp, "\t\t<edge from-layer=\"%u\" from-port=\"%u\" to-layer=\"%u\" to-port=\"%u\"/>\n",
				edge[i].from_layer, edge[i].from_port, edge[i].to_layer, edge[i].to_port);
		}
		fprintf(fp, "\t</edges>\n");
		fprintf(fp, "</net>\n");
		fclose(fp);
	}
}

void GnaWriteXml(char* filename, std::vector<gna_node> node) {
    // Create an IR-XML-like representation of the pre-GNA graph for display in Netron
    FILE* fp = fopen(filename, "w");
    if (fp != NULL) {
        std::vector<gna_edge> edge;
        BuildGnaEdgeList(edge, node);
        fprintf(fp, "<?xml version=\"1.0\"?>\n");
        fprintf(fp, "<net name=\"network\" version=\"10\">\n");
        fprintf(fp, "\t<layers>\n");
        for (uint32_t i = 0; i < node.size(); i++) {
            fprintf(fp,
                    "\t\t<layer id=\"%u\" name=\"%s\" type=\"%s\" version=\"opset1\">\n",
                    i,
                    node[i].name.c_str(),
                    node[i].type_name.c_str());
            fprintf(fp, "\t\t\t<data ");
            for (uint32_t j = 0; j < node[i].attribute.size(); j++) {
                fprintf(fp, "%s=\"%s\" ", node[i].attribute[j].name.c_str(), node[i].attribute[j].value.c_str());
            }
            fprintf(fp, "/>\n");
            fprintf(fp, "\t\t\t<input>\n");
            for (uint32_t j = 0; j < node[i].input.size(); j++) {
                fprintf(fp, "\t\t\t\t<port id=\"%u\" precision=\"%s\">\n", j, node[i].input[j].type_name.c_str());
                fprintf(fp, "\t\t\t\t\t<dim>%u</dim>\n", node[i].input[j].num_elements);
                fprintf(fp, "\t\t\t\t</port>\n");
            }
            fprintf(fp, "\t\t\t</input>\n");
            fprintf(fp, "\t\t\t<output>\n");
            for (uint32_t j = 0; j < node[i].output.size(); j++) {
                fprintf(fp, "\t\t\t\t<port id=\"%u\" precision=\"%s\">\n", j, node[i].output[j].type_name.c_str());
                fprintf(fp, "\t\t\t\t\t<dim>%u</dim>\n", node[i].output[j].num_elements);
                fprintf(fp, "\t\t\t\t</port>\n");
            }
            fprintf(fp, "\t\t\t</output>\n");
            fprintf(fp, "\t\t</layer>\n");
        }
        fprintf(fp, "\t</layers>\n");
        fprintf(fp, "\t<edges>\n");
        for (uint32_t i = 0; i < edge.size(); i++) {
            fprintf(fp,
                    "\t\t<edge from-layer=\"%u\" from-port=\"%u\" to-layer=\"%u\" to-port=\"%u\"/>\n",
                    edge[i].from_layer,
                    edge[i].from_port,
                    edge[i].to_layer,
                    edge[i].to_port);
        }
        fprintf(fp, "\t</edges>\n");
        fprintf(fp, "</net>\n");
        fclose(fp);
    }
}

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