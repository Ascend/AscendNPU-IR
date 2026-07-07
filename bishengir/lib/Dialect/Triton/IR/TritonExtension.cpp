//===- TritonExtension.cpp - BishengIR Triton dialect extension -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/IR/TritonExtension.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/Triton/IR/TritonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CallScalarOp — CallOpInterface
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable mlir::triton::CallScalarOp::getCallableForCallee() {
  return getCalleeAttr();
}

void mlir::triton::CallScalarOp::setCalleeFromCallable(
    mlir::CallInterfaceCallable callee) {
  setCalleeAttr(cast<FlatSymbolRefAttr>(callee.get<SymbolRefAttr>()));
}

mlir::Operation::operand_range mlir::triton::CallScalarOp::getArgOperands() {
  return getCallArgs();
}

mlir::MutableOperandRange mlir::triton::CallScalarOp::getArgOperandsMutable() {
  return getCallArgsMutable();
}

//===----------------------------------------------------------------------===//
// CallScalarOp — SymbolUserOpInterface
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::triton::CallScalarOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
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

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Dialect extension registration
//===----------------------------------------------------------------------===//

void bishengir::registerTritonDialectExtension(
    mlir::DialectRegistry &registry) {
  // addOperations() is protected; use RegisteredOperationName::insert directly.
  registry.addExtension(+[](mlir::MLIRContext *,
                            mlir::triton::TritonDialect *dialect) {
    mlir::RegisteredOperationName::insert<mlir::triton::CallScalarOp>(*dialect);
  });
}
