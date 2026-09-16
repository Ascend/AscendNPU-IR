//===- SymbolOps.cpp --- Implementation of Symbol dialect operations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::symbol;

//===----------------------------------------------------------------------===//
// SymbolicIntOp
//===----------------------------------------------------------------------===//

void SymbolicIntOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getSymbolName());
}

ParseResult SymbolicIntOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr symbol;
  SmallVector<OpAsmParser::UnresolvedOperand> intSymbols;
  AffineMapAttr intExpressions;
  Type resultType;

  if (parser.parseSymbolName(symbol))
    return failure();

  result.getOrAddProperties<SymbolicIntOp::Properties>().symbol_name =
      FlatSymbolRefAttr::get(symbol);

  NamedAttrList attrs;
  // optional [...] {affine_map}
  if (succeeded(parser.parseOptionalLSquare()) &&
      (parser.parseOperandList(intSymbols) || parser.parseRSquare() ||
       parser.parseComma() ||
       parser.parseAttribute(intExpressions,
                             getIntExpressionsAttrName(result.name), attrs)))
    return failure();

  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  if (parser.parseColonType(resultType))
    return failure();

  if (parser.resolveOperands(intSymbols,
                             parser.getBuilder().getType<IndexType>(),
                             result.operands))
    return failure();

  result.addTypes(resultType);
  result.attributes = attrs;
  return success();
}

void SymbolicIntOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypeRange result, FlatSymbolRefAttr symbolName,
                          int64_t minVal, int64_t maxVal) {
  build(odsBuilder, odsState, result, symbolName,
        odsBuilder.getI64IntegerAttr(minVal),
        odsBuilder.getI64IntegerAttr(maxVal), ValueRange(),
        AffineMapAttr::get(AffineMap::get(0, 0, odsBuilder.getContext())));
}

void SymbolicIntOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          FlatSymbolRefAttr symbolName) {
  int64_t minValue = 0;
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  build(odsBuilder, odsState, odsBuilder.getIndexType(), symbolName, minValue,
        maxValue);
}

void SymbolicIntOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          FlatSymbolRefAttr symbolName, ValueRange intSymbols,
                          AffineMapAttr intExpressions) {
  int64_t minValue = 0;
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  build(odsBuilder, odsState, odsBuilder.getIndexType(), symbolName,
        odsBuilder.getI64IntegerAttr(minValue),
        odsBuilder.getI64IntegerAttr(maxValue), intSymbols, intExpressions);
}

// Use a custom printer here to avoid the AffineMap from getting hoisted
// when printed. This makes it so the AffineMap is printed inline with the op.
void SymbolicIntOp::print(OpAsmPrinter &p) {
  FlatSymbolRefAttr symbolAttrStr = getSymbolNameAttr();
  p << " " << symbolAttrStr;

  auto intExpressions = getIntExpressions();
  if (intExpressions.has_value() && !intExpressions->getValue().isEmpty()) {
    p << " [";
    llvm::interleaveComma(getIntSymbols(), p);
    p << "], "
      << "affine_map<" << intExpressions->getValue() << ">";
  }

  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getIntExpressionsAttrName(getOperation()->getName()),
                       getSymbolNameAttrName(getOperation()->getName())});
  p << " : " << getResult().getType();
}

LogicalResult SymbolicIntOp::verify() {
  for (auto symbol : getIntSymbols()) {
    Operation *definingOp = symbol.getDefiningOp();
    // TODO: add canonicalize-like pattern so that it doesn't accept
    // symbolic_int as IntSymbols
    if (!isa_and_nonnull<SymbolicIntOp, tensor::DimOp, arith::IndexCastOp>(
            definingOp))
      return emitOpError() << "int symbol must be produced by valid operations";
  }

  auto intExpressions = getIntExpressions();
  if (!intExpressions.has_value())
    return success();

  AffineMap affineMap = intExpressions->getAffineMap();
  if (affineMap.getNumDims() != 0)
    return emitOpError() << "the affine map should only contain symbols";

  auto numSymbolsInMap = affineMap.getNumSymbols();
  if (getIntSymbols().size() != numSymbolsInMap)
    return emitOpError() << "number of int symbols " << getIntSymbols().size()
                         << " doesn't match with affine map "
                         << numSymbolsInMap;

  // Verify that the map only produces one result.
  if (affineMap.getNumResults() > 1)
    return emitOpError() << "mapping must not produce more than one value";

  return success();
}

//===----------------------------------------------------------------------===//
// BindSymbolicShapeOp
//===----------------------------------------------------------------------===//

ParseResult BindSymbolicShapeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  SmallVector<OpAsmParser::UnresolvedOperand> shapeSymbols;
  AffineMapAttr shapeExpressions;
  Type operandType;

  if (parser.parseOperand(operand) || parser.parseComma() ||
      parser.parseLSquare() || parser.parseOperandList(shapeSymbols) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseAttribute(shapeExpressions,
                            getShapeExpressionsAttrName(result.name),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType)) {
    return failure();
  }

  if (parser.resolveOperand(operand, operandType, result.operands) ||
      parser.resolveOperands(shapeSymbols,
                             parser.getBuilder().getType<IndexType>(),
                             result.operands)) {
    return failure();
  }

  return success();
}

// Use a custom printer here to avoid the AffineMap from getting hoisted
// when printed. This makes it so the AffineMap is printed inline with the op.
void BindSymbolicShapeOp::print(OpAsmPrinter &p) {
  p << " " << getOperand() << ", [";
  llvm::interleaveComma(getShapeSymbols(), p);
  p << "], "
    << "affine_map<" << getShapeExpressions().getValue() << ">";
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getShapeExpressionsAttrName(getOperation()->getName())});
  p << " : " << getOperand().getType();
}

LogicalResult BindSymbolicShapeOp::verify() {
  if (getShapeSymbols().empty())
    return emitOpError() << "requires non-empty shapeSymbols";

  if (getShapeExpressions().getAffineMap().getNumSymbols() !=
      getShapeSymbols().size())
    return emitOpError() << "number of shape symbols doesn't match the number "
                            "of symbols in the affine.map";

  return success();
}
