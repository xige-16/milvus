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

#include <iostream>
#include <memory>

#include "arrow/api.h"
#include "common/FieldMeta.h"
#include "common/Utils.h"
#include "common/VectorTrait.h"
#include "exceptions/EasyAssert.h"
#include "storage/Exception.h"

namespace milvus::storage {

using DataType = milvus::DataType;

// class FieldData {
// public:
//    explicit FieldData(const Payload& payload);
//
//    explicit FieldData(std::shared_ptr<arrow::Array> raw_data, DataType data_type);
//
//    explicit FieldData(const uint8_t* data, int length);
//
//    //    explicit FieldData(std::unique_ptr<uint8_t[]> data, int length, DataType data_type): data_(std::move(data)),
//    //    data_len_(length), data_type_(data_type) {}
//
//    ~FieldData() = default;
//
//    DataType
//    get_data_type() const {
//        return data_type_;
//    }
//
//    bool
//    get_bool_payload(int idx) const;
//
//    void
//    get_one_string_payload(int idx, char** cstr, int* str_size) const;
//
//    // get the bytes stream of the arrow array data
//    std::unique_ptr<Payload>
//    get_payload() const;
//
//    int
//    get_payload_length() const {
//        return array_->length();
//    }
//
//    int
//    get_data_size() const;
//
// private:
//    std::shared_ptr<arrow::Array> array_;
//    //    std::unique_ptr<uint8_t[]> data_;
//    //    int64_t data_len_;
//    DataType data_type_;
//};

class FieldDataBase {
 public:
    explicit FieldDataBase(DataType data_type) : data_type_(data_type) {
    }
    virtual ~FieldDataBase() = default;

    virtual void
    FillFieldData(const void* source, ssize_t element_count) = 0;

    virtual void
    FillFieldData(const std::shared_ptr<arrow::Array> array) = 0;

    virtual const void*
    Data() const = 0;

    virtual int
    GetNumRows() const = 0;

    virtual int64_t
    GetDataSize() const = 0;

    virtual int64_t
    GetDim() const = 0;

    DataType
    GetDataType() const {
        return data_type_;
    }

 protected:
    const DataType data_type_;
};

using FieldDataPtr = std::shared_ptr<FieldDataBase>;

template <typename Type, bool is_scalar = false>
class FieldDataImpl : public FieldDataBase {
 public:
    // constants
    using Chunk = FixedVector<Type>;
    FieldDataImpl(FieldDataImpl&&) = delete;
    FieldDataImpl(const FieldDataImpl&) = delete;

    FieldDataImpl&
    operator=(FieldDataImpl&&) = delete;
    FieldDataImpl&
    operator=(const FieldDataImpl&) = delete;

 public:
    explicit FieldDataImpl(ssize_t dim, DataType data_type) : FieldDataBase(data_type), dim_(is_scalar ? 1 : dim) {
    }

    void
    FillFieldData(const void* source, ssize_t element_count) override {
        AssertInfo(element_count % dim_ == 0, "invalid element count");
        if (element_count == 0) {
            return;
        }
        AssertInfo(field_data_.size() == 0, "no empty field vector");
        field_data_.resize(element_count);
        std::copy_n(static_cast<const Type*>(source), element_count, field_data_.data());
    }

    void
    FillFieldData(const std::shared_ptr<arrow::Array> array) override {
        AssertInfo(array != nullptr, "null arrow array");
        auto element_count = array->length() * dim_;
        if (element_count == 0) {
            return;
        }
        switch (data_type_) {
            case DataType::BOOL: {
                AssertInfo(array->type()->id() == arrow::Type::type::BOOL, "inconsistent data type");
                auto bool_array = std::dynamic_pointer_cast<arrow::BooleanArray>(array);
                FixedVector<bool> values(element_count);
                for (size_t index = 0; index < element_count; ++index) {
                    values[index] = bool_array->Value(index);
                }
                return FillFieldData(values.data(), element_count);
            }
            case DataType::INT8: {
                AssertInfo(array->type()->id() == arrow::Type::type::INT8, "inconsistent data type");
                auto int8_array = std::dynamic_pointer_cast<arrow::Int8Array>(array);
                return FillFieldData(int8_array->raw_values(), element_count);
            }
            case DataType::INT16: {
                AssertInfo(array->type()->id() == arrow::Type::type::INT16, "inconsistent data type");
                auto int16_array = std::dynamic_pointer_cast<arrow::Int16Array>(array);
                return FillFieldData(int16_array->raw_values(), element_count);
            }
            case DataType::INT32: {
                AssertInfo(array->type()->id() == arrow::Type::type::INT32, "inconsistent data type");
                auto int32_array = std::dynamic_pointer_cast<arrow::Int32Array>(array);
                return FillFieldData(int32_array->raw_values(), element_count);
            }
            case DataType::INT64: {
                AssertInfo(array->type()->id() == arrow::Type::type::INT64, "inconsistent data type");
                auto int64_array = std::dynamic_pointer_cast<arrow::Int64Array>(array);
                return FillFieldData(int64_array->raw_values(), element_count);
            }
            case DataType::FLOAT: {
                AssertInfo(array->type()->id() == arrow::Type::type::FLOAT, "inconsistent data type");
                auto float_array = std::dynamic_pointer_cast<arrow::FloatArray>(array);
                return FillFieldData(float_array->raw_values(), element_count);
            }
            case DataType::DOUBLE: {
                AssertInfo(array->type()->id() == arrow::Type::type::DOUBLE, "inconsistent data type");
                auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(array);
                return FillFieldData(double_array->raw_values(), element_count);
            }
            case DataType::STRING:
            case DataType::VARCHAR: {
                AssertInfo(array->type()->id() == arrow::Type::type::STRING, "inconsistent data type");
                auto string_array = std::dynamic_pointer_cast<arrow::StringArray>(array);
                std::vector<std::string> values(element_count);
                for (size_t index = 0; index < element_count; ++index) {
                    values[index] = string_array->GetString(index);
                }
                return FillFieldData(values.data(), element_count);
            }
            case DataType::VECTOR_FLOAT: {
                AssertInfo(array->type()->id() == arrow::Type::type::FIXED_SIZE_BINARY, "inconsistent data type");
                auto vector_array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(array);
                return FillFieldData(vector_array->raw_values(), element_count);
            }
            case DataType::VECTOR_BINARY: {
                AssertInfo(array->type()->id() == arrow::Type::type::FIXED_SIZE_BINARY, "inconsistent data type");
                auto vector_array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(array);
                return FillFieldData(vector_array->raw_values(), element_count);
            }
            default: {
                throw NotSupportedDataTypeException(GetName() + "::FillFieldData" + " not support data type " +
                                                    datatype_name(data_type_));
            }
        }
    }

    std::string
    GetName() const {
        return "FieldDataImpl";
    }

    const void*
    Data() const override {
        return get_values();
    }

    int
    GetNumRows() const override {
        auto len = length();
        AssertInfo(len % dim_ == 0, "field data size not aligned");
        return len / dim_;
    }

    int64_t
    GetDim() const override {
        return dim_;
    }

    int64_t
    GetDataSize() const override {
        return sizeof(Type) * length();
    }

    Type
    get_value(ssize_t offset) const {
        return field_data_[offset];
    }

    const Type*
    get_values() const {
        return field_data_.data();
    }

    int64_t
    length() const {
        return field_data_.size();
    }

 private:
    const ssize_t dim_;
    Chunk field_data_;
};

// template <>
// int64_t FieldDataImpl<std::string, true>::GetDataSize() const {
//    int64_t data_size = 0;
//    for (auto& str : field_data_) {
//        data_size += str.size();
//    }
//
//    return data_size;
//}

template <typename Type>
class FieldData : public FieldDataImpl<Type, true> {
 public:
    static_assert(IsScalar<Type> || std::is_same_v<Type, PkType>);
    explicit FieldData(DataType data_type) : FieldDataImpl<Type, true>::FieldDataImpl(1, data_type) {
    }
};

template <>
class FieldData<FloatVector> : public FieldDataImpl<float, false> {
 public:
    explicit FieldData(int64_t dim, DataType data_type) : FieldDataImpl<float, false>::FieldDataImpl(dim, data_type) {
    }
};

template <>
class FieldData<BinaryVector> : public FieldDataImpl<uint8_t, false> {
 public:
    explicit FieldData(int64_t dim, DataType data_type) : binary_dim_(dim), FieldDataImpl(dim / 8, data_type) {
        Assert(dim % 8 == 0);
    }

    int64_t
    GetDim() const {
        return binary_dim_;
    }

 private:
    int64_t binary_dim_;
};

}  // namespace milvus::storage
