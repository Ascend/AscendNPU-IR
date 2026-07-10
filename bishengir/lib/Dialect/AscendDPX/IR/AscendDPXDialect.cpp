//===- AscendDPXDialect.cpp - AscendDPX dialect implementation ------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::ascend_dpx;

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.cpp.inc"

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AscendDPXDialect
//===----------------------------------------------------------------------===//

void ascend_dpx::AscendDPXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CallScalarOp - CallOpInterface
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable
mlir::ascend_dpx::CallScalarOp::getCallableForCallee() {
  return getCalleeAttr();
}

void mlir::ascend_dpx::CallScalarOp::setCalleeFromCallable(
    mlir::CallInterfaceCallable callee) {
  setCalleeAttr(cast<FlatSymbolRefAttr>(callee.get<SymbolRefAttr>()));
}

mlir::Operation::operand_range
mlir::ascend_dpx::CallScalarOp::getArgOperands() {
  return getCallArgs();
}

mlir::MutableOperandRange
mlir::ascend_dpx::CallScalarOp::getArgOperandsMutable() {
  return getCallArgsMutable();
}

//===----------------------------------------------------------------------===//
// CallScalarOp - SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascend_dpx::CallScalarOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto *callable = symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!callable)
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid function";

  auto fnIface = dyn_cast<FunctionOpInterface>(callable);
  if (!fnIface)
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a function";

  auto argTypes = fnIface.getArgumentTypes();
  auto resTypes = fnIface.getResultTypes();
  auto args = getCallArgs();

  if (argTypes.size() != args.size())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
    if (args[i].getType() != argTypes[i])
      return emitOpError("operand type mismatch: expected ")
             << argTypes[i] << " but got " << args[i].getType()
             << " for operand " << i;

  if (resTypes.size() != getResultTypes().size())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = resTypes.size(); i != e; ++i)
    if (getResult(i).getType() != resTypes[i])
      return emitOpError("result type mismatch: expected ")
             << resTypes[i] << " but got " << getResult(i).getType()
             << " for result " << i;

  return success();
}
