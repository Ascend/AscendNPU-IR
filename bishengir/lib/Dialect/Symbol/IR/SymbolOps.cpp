//===- SymbolOps.cpp --- Implementation of Symbol dialect operations -----===//
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
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Utils/Util.h"
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
                          FlatSymbolRefAttr symbolName, ValueRange intSymbols) {
  int64_t minValue = 0;
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  SmallVector<AffineExpr> symbolResults;
  for (size_t i = 0; i < intSymbols.size(); ++i) {
    symbolResults.push_back(odsBuilder.getAffineSymbolExpr(i));
  }
  AffineMap identityMap = AffineMap::get(0, intSymbols.size(), symbolResults,
                                         odsBuilder.getContext());
  AffineMapAttr identityMapAttr = AffineMapAttr::get(identityMap);
  build(odsBuilder, odsState, odsBuilder.getIndexType(), symbolName,
        odsBuilder.getI64IntegerAttr(minValue),
        odsBuilder.getI64IntegerAttr(maxValue), intSymbols, identityMapAttr);
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
    Operation *defOp = symbol.getDefiningOp();
    // TODO: add canonicalize-like pattern so that it doesn't accept
    // symbolic_int as IntSymbols
    if (!isa_and_nonnull<SymbolicIntOp, tensor::DimOp>(defOp) &&
        !mlir::utils::isArithOp(defOp))
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

// fold unit map from another symbolic_int op, e.g.
// optimize:
//   %S1 = symbol.symbolic_int @S1 [%S0], affine_map<()[s0] -> (s0)>
//   use(%S1)
// to:
//   use(%S0)
OpFoldResult SymbolicIntOp::fold(FoldAdaptor adaptor) {
  SmallVector<Value> symbols = this->getIntSymbols();
  if (symbols.size() != 1) {
    return {};
  }
  Value input = symbols.front();
  auto inputSymbol = input.getDefiningOp<symbol::SymbolicIntOp>();
  if (!inputSymbol) {
    return {};
  }
  auto affineMapMaybe = getIntExpressions();
  if (!affineMapMaybe.has_value()) {
    return {};
  }

  // make sure the affine_map matches pattern: `affine_map<()[s0] -> (s0)>`
  auto affineMap = affineMapMaybe->getAffineMap();
  if (affineMap.getNumDims() != 0 || affineMap.getNumResults() != 1 ||
      !isa<AffineSymbolExpr>(affineMap.getResult(0))) {
    return {};
  }
  return input;
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
  Value operand = getOperand();
  auto shapedType = cast<ShapedType>(operand.getType());
  int64_t numResults = getShapeExpressions().getAffineMap().getNumResults();
  if (static_cast<int64_t>(numResults) != shapedType.getRank()) {
    return emitOpError() << "number of results doesn't match the rank of "
                            "binded operand shape";
  }
  if (getShapeSymbols().empty())
    return emitOpError() << "requires non-empty shapeSymbols";

  if (getShapeExpressions().getAffineMap().getNumSymbols() !=
      getShapeSymbols().size())
    return emitOpError() << "number of shape symbols doesn't match the number "
                            "of symbols in the affine.map";

  return success();
}
