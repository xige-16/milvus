// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include "storage/PayloadStream.h"

namespace milvus::storage {

StorageType
ReadMediumType(PayloadInputStream* input_stream);

void
AddPayloadToArrowBuilder(std::shared_ptr<arrow::ArrayBuilder> builder, const Payload& payload);

void
AddOneStringToArrowBuilder(std::shared_ptr<arrow::ArrayBuilder> builder, const char* str, int str_size);

std::shared_ptr<arrow::ArrayBuilder>
CreateArrowBuilder(DataType data_type);

std::shared_ptr<arrow::ArrayBuilder>
CreateArrowBuilder(DataType data_type, int dim);

std::shared_ptr<arrow::Schema>
CreateArrowSchema(DataType data_type);

std::shared_ptr<arrow::Schema>
CreateArrowSchema(DataType data_type, int dim);

int64_t
GetPayloadSize(const Payload* payload);

const uint8_t*
GetRawValuesFromArrowArray(std::shared_ptr<arrow::Array> array, DataType data_type);

int
GetDimensionFromArrowArray(std::shared_ptr<arrow::Array> array, DataType data_type);

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim);
}  // namespace milvus::storage
