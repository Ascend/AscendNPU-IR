//===- VectorizeAttrs.cpp - Vectorizer metadata --------------------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>

using namespace mlir;
using namespace mlir::hivm;

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.cpp.inc"

void HIVMVectorizeDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.cpp.inc"
      >();
}

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeDialect.cpp.inc"

Attribute GroupsAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess() || parser.parseLSquare())
    return {};

  SmallVector<DenseI64ArrayAttr> groups;
  if (succeeded(parser.parseOptionalRSquare())) {
    if (parser.parseGreater())
      return {};
    return GroupsAttr::get(parser.getContext(), groups);
  }

  do {
    DenseI64ArrayAttr group;
    if (parser.parseAttribute(group) || !group)
      return {};
    groups.push_back(group);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare() || parser.parseGreater())
    return {};
  auto emitError = [&] {
    return parser.emitError(parser.getCurrentLocation());
  };
  if (failed(GroupsAttr::verify(emitError, groups)))
    return {};
  return GroupsAttr::get(parser.getContext(), groups);
}

void GroupsAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  for (size_t index = 0; index < getGroups().size(); ++index) {
    if (index)
      printer << ", ";
    printer.printAttribute(getGroups()[index]);
  }
  printer << "]>";
}

LogicalResult
GroupsAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<DenseI64ArrayAttr> groups) {
  for (DenseI64ArrayAttr group : groups) {
    if (group.empty())
      return emitError() << "partition group must not be empty";
    for (int64_t ordinal : group.asArrayRef())
      if (ordinal < 0)
        return emitError() << "partition ordinal must be non-negative";
  }
  return success();
}

Attribute PlanAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess() || parser.parseKeyword("cost") || parser.parseEqual())
    return {};
  FloatAttr cost;
  if (parser.parseAttribute(cost))
    return {};
  if (parser.parseComma() || parser.parseKeyword("groups") ||
      parser.parseEqual())
    return {};
  GroupsAttr groups;
  if (parser.parseAttribute(groups))
    return {};
  if (parser.parseGreater())
    return {};
  auto emitError = [&] {
    return parser.emitError(parser.getCurrentLocation());
  };
  if (failed(PlanAttr::verify(emitError, cost, groups)))
    return {};
  return PlanAttr::get(parser.getContext(), cost, groups);
}

void PlanAttr::print(AsmPrinter &printer) const {
  printer << "<cost = ";
  printer.printAttribute(getCost());
  printer << ", groups = ";
  printer.printAttribute(getGroups());
  printer << ">";
}

LogicalResult
PlanAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                 FloatAttr cost, GroupsAttr groups) {
  if (!cost)
    return emitError() << "plan cost must be a float attribute";
  if (!llvm::isa<Float64Type>(cost.getType()))
    return emitError() << "plan cost must be f64";
  if (!groups)
    return emitError() << "plan groups must be a groups attribute";
  auto value = cost.getValueAsDouble();
  if (!std::isfinite(value))
    return emitError() << "plan cost must be finite";
  if (value < 0.0)
    return emitError() << "plan cost must be non-negative";
  return success();
}
