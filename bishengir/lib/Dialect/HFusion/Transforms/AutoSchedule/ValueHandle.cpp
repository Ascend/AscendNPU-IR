//===- ValueHandle.cpp - Handles to Payload IR in Auto Schedule -*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class for holding handles to the payload IR
// during the auto schedule process.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {
static std::string stringifyNamedAttr(mlir::StringRef name,
                                      mlir::Attribute attr) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << name;
  if (attr) {
    os << " = ";
    attr.print(os);
  }
  return buffer;
}
} // namespace

namespace mlir {
namespace hfusion {

//===----------------------------------------------------------------------===//
// Implementation of ValueHandle
//===----------------------------------------------------------------------===//

Value ValueHandle::getImpl(Value matchTarget, OpBuilder &opBuilder) {
  llvm::report_fatal_error("Not implemented!");
}

Value ValueHandle::getImpl() { llvm::report_fatal_error("Not implemented!"); }

void ValueHandle::setStatus(HandleStatus s) { status_ = s; }

void ValueHandle::setHandle(Value v) {
  handle_ = v;
  status_ = HandleStatus::kValid;
}

void ValueHandle::invalidate() { status_ = HandleStatus::kInvalid; }

//===----------------------------------------------------------------------===//
// Implementation of ValueHandleFoldResult
//===----------------------------------------------------------------------===//

ValueHandleFoldResult::ValueHandleFoldResult(int64_t cst, MLIRContext *ctx)
    : PointerUnion<ValueHandle *, Attribute>(
          IntegerAttr::get(IntegerType::get(ctx, 64), cst)) {}

std::optional<int64_t> ValueHandleFoldResult::getConstInteger() {
  auto maybeAttr = this->dyn_cast<Attribute>();
  if (!maybeAttr)
    return std::nullopt;
  return getConstantIntValue(this->get<Attribute>());
}

std::optional<ValueHandle *> ValueHandleFoldResult::getValueHandle() {
  auto *maybeAttr = this->dyn_cast<ValueHandle *>();
  if (!maybeAttr)
    return std::nullopt;
  return this->get<ValueHandle *>();
}

//===----------------------------------------------------------------------===//
// Implementation of RegularValueHandle
//===----------------------------------------------------------------------===//

Value RegularValueHandle::getImpl() {
  if (this->status_ == HandleStatus::kValid)
    return handle_;
  llvm::report_fatal_error("Invalid handle!");
}

//===----------------------------------------------------------------------===//
// Implementation of NamedValueHandle
//===----------------------------------------------------------------------===//

Value NamedValueHandle::getImpl(Value matchTarget, OpBuilder &opBuilder) {
  if (this->status_ == HandleStatus::kValid)
    return handle_;

  if (this->status_ != HandleStatus::kNeedsRematch) {
    llvm::report_fatal_error("Invalid handle!");
    return Value{};
  }

  TypedValue<transform::TransformHandleTypeInterface> matchResult;
  switch (this->type_) {
  case (IdentifierType::kAttribute):
    matchResult =
        opBuilder
            .create<transform::MatchOp>(
                matchTarget.getLoc(), matchTarget, /*ops=*/ArrayAttr{},
                /*op_attrs=*/
                opBuilder.getDictionaryAttr(
                    ArrayRef<NamedAttribute>{opBuilder.getNamedAttr(
                        this->name_, opBuilder.getUnitAttr())}))
            .getResults();
    break;
  case (IdentifierType::kOperation):
    matchResult =
        opBuilder
            .create<transform::MatchOp>(
                matchTarget.getLoc(), matchTarget,
                /*ops=*/
                opBuilder.getArrayAttr({opBuilder.getStringAttr(this->name_)}),
                /*op_attrs=*/DictionaryAttr{})
            .getResults();
    break;
  default:
    llvm::report_fatal_error("Not implemented!");
    return Value();
  }

  if (this->needsReverse_) {
    matchResult = opBuilder.create<transform::ReverseOp>(
        matchResult.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/matchResult);
  }
  this->handle_ = matchResult;
  this->status_ = HandleStatus::kValid;
  return handle_;
}

//===----------------------------------------------------------------------===//
// Implementation of FuncArgHandle
//===----------------------------------------------------------------------===//

Value FuncArgHandle::getImpl(Value funcValue, OpBuilder &opBuilder) {
  if (this->status_ == HandleStatus::kValid)
    return handle_;

  if (this->status_ == HandleStatus::kNeedsRematch) {
    auto getFuncArgOp = opBuilder.create<transform::GetFuncArgumentOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        SmallVector<int64_t>{static_cast<int64_t>(this->funcArgNum_)},
        /*is_inverted=*/false);
    this->handle_ = getFuncArgOp;
    this->status_ = HandleStatus::kValid;
    return handle_;
  }
  llvm::report_fatal_error("Invalid handle!");
  return {};
}

//===----------------------------------------------------------------------===//
// Implementation of HandleRecord
//===----------------------------------------------------------------------===//

std::string HandleRecord::getAndRecordAttrName(StringRef oldAttrName) {
  auto iter = attributeCount_.find(oldAttrName);
  std::string newTag = oldAttrName.data();
  size_t count = 1;
  if (iter != attributeCount_.end()) {
    count = iter->second;
    newTag = newTag + "_" + std::to_string(count++);
  }
  // update old attr with new count
  attributeCount_.insert_or_assign(oldAttrName, count);
  return newTag;
}

void HandleRecord::resetAllHandles() {
  for (std::unique_ptr<ValueHandle> &h : allocatedHandles_) {
    llvm::TypeSwitch<ValueHandle *, void>(h.get())
        .Case([](NamedValueHandle *h) {
          h->setStatus(HandleStatus::kNeedsRematch);
        })
        .Case(
            [](RegularValueHandle *h) { h->setStatus(HandleStatus::kInvalid); })
        .Case(
            [](FuncArgHandle *h) { h->setStatus(HandleStatus::kNeedsRematch); })
        .Default([](ValueHandle *) { llvm::report_fatal_error("Not implemented!"); });
  }
}

void HandleRecord::clear() {
  for (std::unique_ptr<ValueHandle> &h : allocatedHandles_)
    h.reset();

  allocatedHandles_.clear();
  attributeCount_.clear();
  attribute2ValueHandle_.clear();
}

void setStatusTo(ValueHandles &vhs, HandleStatus status) {
  llvm::for_each(vhs, [&status](ValueHandle *vh) { vh->setStatus(status); });
}

std::optional<NamedValueHandle *>
HandleRecord::tryFetchRecordImpl(const detail::Identifier &identifier) {
  auto uniqueIdentifier = identifier.getUniqueIdentifier();
  if (!attribute2ValueHandle_.contains(uniqueIdentifier))
    return {};

  return attribute2ValueHandle_.at(uniqueIdentifier);
}

namespace detail {

StringRef OperationIdentifier::getUniqueIdentifier() const { return getName(); }

AttributeIdentifier::AttributeIdentifier(StringRef name)
    : Identifier(IdentifierType::kAttribute) {
  requiredAttrs_.insert({name, Attribute()});
  uniqueIdentifier_ = name;
}

AttributeIdentifier::AttributeIdentifier(StringRef name, Attribute attr)
    : Identifier(IdentifierType::kAttribute) {
  requiredAttrs_.insert({name, attr});
  uniqueIdentifier_ = stringifyNamedAttr(name, attr);
}

AttributeIdentifier::AttributeIdentifier(
    const DenseMap<StringRef, Attribute> &requiredAttrs,
    const DenseMap<StringRef, Attribute> &optionalAttrs)
    : Identifier(IdentifierType::kAttribute) {
  requiredAttrs_ = requiredAttrs;
  optionalAttrs_ = optionalAttrs;
  std::string identifier = "required_";
  for (auto required : requiredAttrs)
    identifier += stringifyNamedAttr(required.getFirst(), required.getSecond());
  identifier += "_optional";
  for (auto optional : requiredAttrs)
    identifier += stringifyNamedAttr(optional.getFirst(), optional.getSecond());
  uniqueIdentifier_ = identifier;
}

DictionaryAttr AttributeIdentifier::getAttrs(OpBuilder &builder,
                                             bool required) const {
  SmallVector<NamedAttribute> attrList;
  auto targetAttrs = required ? requiredAttrs_ : optionalAttrs_;
  for (auto attr : targetAttrs)
    attrList.push_back(builder.getNamedAttr(
        attr.getFirst(),
        attr.getSecond() ? attr.getSecond() : builder.getUnitAttr()));
  return DictionaryAttr::get(builder.getContext(), attrList);
}

StringRef AttributeIdentifier::getUniqueIdentifier() const {
  return uniqueIdentifier_;
}

} // namespace detail

} // namespace hfusion
} // namespace mlir
