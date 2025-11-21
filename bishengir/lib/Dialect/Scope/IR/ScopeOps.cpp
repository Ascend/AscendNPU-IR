//===- ScopeOps.cpp --- Implementation of Scope dialect operations --------===//
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

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::scope;

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  FunctionType functionType;
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  // Parse the body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  ScopeOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ScopeOp::print(OpAsmPrinter &p) {
  p << " : ";
  p.printFunctionalType(SmallVector<Type>(), SmallVector<Type>());
  p << " ";

  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs());
}