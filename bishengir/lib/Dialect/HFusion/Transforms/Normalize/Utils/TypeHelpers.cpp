//===- TypeHelpers.cpp -----------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

bool isI1ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(1);
}

bool isI8ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(8);
}

bool isI16ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(16);
}

bool isF16ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isF16();
}

template <typename srcType>
bool isElemType(Type valueType) {
  if constexpr (std::is_same_v<bool, srcType>) {
    return isI1ElemType(valueType);
  }
  if constexpr (std::is_same_v<int8_t, srcType>) {
    return isI8ElemType(valueType);
  }
  if constexpr (std::is_same_v<int16_t, srcType>) {
    return isI16ElemType(valueType);
  }
  if constexpr (std::is_same_v<float, srcType>) {
    return isF16ElemType(valueType);
  }
  return false;
}

bool hasI1ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isI1ElemType(v.getType()); });
}

bool hasI8ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isI8ElemType(v.getType()); });
}

[[maybe_unused]] bool hasI16ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI16ElemType(v.getType()); });
}

bool hasF16ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isF16ElemType(v.getType()); });
}

[[maybe_unused]] bool allI1ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI1ElemType(v.getType()); });
}

bool allI8ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI8ElemType(v.getType()); });
}

bool allI16ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI16ElemType(v.getType()); });
}

template <typename srcType, typename targetType, typename>
SmallVector<Value> normalizeSrcToTargetType(PatternRewriter &rewriter,
                                            const SmallVector<Value> &values,
                                            bool unsignedSrc) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<srcType>(v.getType())) {
      result.push_back(v);
      continue;
    }

    Type dstType = rewriter.getType<targetType>();
    Value castResult = unsignedSrc ? castTo(rewriter, v, dstType,
                                            hfusion::TypeFn::cast_unsigned)
                                   : castTo(rewriter, v, dstType);
    result.push_back(castResult);
  }
  return result;
}

arith::CmpFPredicate getCmpFloatPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return arith::CmpFPredicate::OEQ;
  case arith::CmpIPredicate::ne:
    return arith::CmpFPredicate::ONE;
  case arith::CmpIPredicate::slt:
    return arith::CmpFPredicate::OLT;
  case arith::CmpIPredicate::sle:
    return arith::CmpFPredicate::OLE;
  case arith::CmpIPredicate::sgt:
    return arith::CmpFPredicate::OGT;
  case arith::CmpIPredicate::sge:
    return arith::CmpFPredicate::OGE;
  case arith::CmpIPredicate::ult:
    return arith::CmpFPredicate::OLT;
  case arith::CmpIPredicate::ule:
    return arith::CmpFPredicate::OLE;
  case arith::CmpIPredicate::ugt:
    return arith::CmpFPredicate::OGT;
  case arith::CmpIPredicate::uge:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unexpected arith::CmpIPredicate");
}

Operation *cloneArithOp(PatternRewriter &rewriter, Location loc,
                        Operation *bodyOp, IRMapping &mapper) {
  const DenseMap<Value, Value> &valueMap = mapper.getValueMap();
  Value oldLhs = bodyOp->getOperand(0);
  Value oldRhs = bodyOp->getOperand(1);
  Value lhs = valueMap.at(oldLhs);
  Value rhs = valueMap.at(oldRhs);
  if (isa<arith::AddFOp>(bodyOp) || isa<arith::AddIOp>(bodyOp)) {
    auto newAddf = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    return newAddf;
  }
  if (isa<arith::MulFOp>(bodyOp) || isa<arith::MulIOp>(bodyOp)) {
    auto newMulf = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
    return newMulf;
  }
  if (isa<arith::SubFOp>(bodyOp) || isa<arith::SubIOp>(bodyOp)) {
    auto newSubf = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
    return newSubf;
  }
  if (auto cmpi = dyn_cast<arith::CmpIOp>(bodyOp)) {
    auto pred = getCmpFloatPredicate(cmpi.getPredicate());
    auto cmpf = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    return cmpf;
  }
  if (auto cmpf = dyn_cast<arith::CmpFOp>(bodyOp)) {
    auto newCmpf =
        rewriter.create<arith::CmpFOp>(loc, cmpf.getPredicate(), lhs, rhs);
    return newCmpf;
  }
  if (isa<arith::DivFOp>(bodyOp) || isa<arith::DivSIOp>(bodyOp) ||
      isa<arith::DivUIOp>(bodyOp)) {
    auto newDivf = rewriter.create<arith::DivFOp>(loc, lhs, rhs);
    return newDivf;
  }
  if (isa<arith::MaximumFOp>(bodyOp) || isa<arith::MaxSIOp>(bodyOp) ||
      isa<arith::MaxUIOp>(bodyOp)) {
    auto newMaxf = rewriter.create<arith::MaximumFOp>(loc, lhs, rhs);
    return newMaxf;
  }
  if (isa<arith::MinimumFOp>(bodyOp) || isa<arith::MinSIOp>(bodyOp) ||
      isa<arith::MinUIOp>(bodyOp)) {
    auto newMinf = rewriter.create<arith::MinimumFOp>(loc, lhs, rhs);
    return newMinf;
  }
  llvm::report_fatal_error("unsupported body op to map");
}

Operation *mapReduceBodyOpToFloat(PatternRewriter &rewriter, Location loc,
                                  Operation *bodyOp, Type srcType,
                                  IRMapping &mapper) {
  if (isa<linalg::YieldOp>(bodyOp)) {
    return rewriter.clone(*bodyOp, mapper);
  }
  if (auto select = dyn_cast<arith::SelectOp>(bodyOp)) {
    Value cond = mapper.lookup(select.getCondition());
    Value trueValue = mapper.lookup(select.getTrueValue());
    Value falseValue = mapper.lookup(select.getFalseValue());
    auto newSelect = rewriter.create<arith::SelectOp>(
        loc, trueValue.getType(), cond, trueValue, falseValue);
    return newSelect;
  }
  // simply clone op with no f16 or i8 operand
  assert(bodyOp->getNumOperands() == 2 && "only support binary arith op");
  Value oldLhs = bodyOp->getOperand(0);
  Value oldRhs = bodyOp->getOperand(1);
  if (srcType == rewriter.getI8Type() && !isI8ElemType(oldLhs.getType()) &&
      !isI8ElemType(oldRhs.getType())) {
    return rewriter.clone(*bodyOp, mapper);
  }
  if (srcType == rewriter.getF16Type() && !isF16ElemType(oldLhs.getType()) &&
      !isF16ElemType(oldRhs.getType())) {
    return rewriter.clone(*bodyOp, mapper);
  }

  // convert arith op from srcType to targetType
  return cloneArithOp(rewriter, loc, bodyOp, mapper);
}

Operation *createNewReduceOp(linalg::ReduceOp op, PatternRewriter &rewriter,
                             Type srcType, Type targetType,
                             SmallVector<Value> &newInputs,
                             SmallVector<Value> &newInits) {
  bool isF16ToF32 = false;
  if (targetType == rewriter.getF32Type() && srcType == rewriter.getF16Type()) {
    isF16ToF32 = true;
  }

  IRMapping mapper;
  for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
    mapper.map(operand, newInputs[idx]);
  }
  for (const auto &[idx, operand] : llvm::enumerate(op.getInits())) {
    mapper.map(operand, newInits[idx]);
  }

  Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
  // change f16 result types to targetType
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    ShapedType shapedType = dyn_cast_or_null<ShapedType>(res.getType());
    bool isTargetType =
        isF16ToF32 ? isF16ElemType(shapedType) : isI8ElemType(shapedType);
    if (!shapedType || !isTargetType) {
      continue;
    }
    auto srcShapedType = shapedType.clone(targetType);
    newOp->getResult(idx).setType(srcShapedType);
  }

  // create reduce op inner region with srcType changed to targetType
  Region &newRegion = newOp->getRegions().front();
  Block *newBlock = rewriter.createBlock(&newRegion);
  rewriter.setInsertionPointToStart(newBlock);

  Block *block = &op.getRegion().front();
  for (BlockArgument bbArg : block->getArguments()) {
    // change op region block srcType arg using targetType
    Type argType = bbArg.getType();
    bool isSrcType = isF16ToF32 ? argType.isF16() : argType.isInteger(8);
    Type newArgType = (isSrcType ? targetType : argType);
    mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
  }

  Location loc = newRegion.getLoc();
  for (Operation &bodyOp : *block) {
    // change op within region to targetType.
    Operation *newBodyOp =
        mapReduceBodyOpToFloat(rewriter, loc, &bodyOp, srcType, mapper);
    mapper.map(bodyOp.getResults(), newBodyOp->getResults());
  }
  rewriter.setInsertionPointAfter(newOp);
  return newOp;
}

void replaceI8ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                           const SmallVector<Value> &newResults,
                                           PatternRewriter &rewriter,
                                           bool enableOverflow,
                                           bool isUnsigned) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI8ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult =
        castTo(rewriter, newResult, rewriter.getI8Type(),
               hfusion::RoundMode::TRUNC, std::nullopt, enableOverflow, false,
               isUnsigned ? hfusion::TypeFn::cast_unsigned
                          : hfusion::TypeFn::cast_signed);
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

void replaceI16ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                const SmallVector<Value> &newResults,
                                PatternRewriter &rewriter) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI16ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value overflowResult =
        hfusion::OverflowProcess(rewriter, newResult, rewriter.getI16Type());
    Value castResult = castTo(rewriter, overflowResult, rewriter.getI16Type());
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

// Explicit template instantiations
template bool mlir::hfusion::isElemType<signed char>(Type);
template SmallVector<Value> mlir::hfusion::normalizeSrcToTargetType<float, Float32Type, std::enable_if<true, void>>(PatternRewriter &, const SmallVector<Value> &, bool);
template SmallVector<Value> mlir::hfusion::normalizeSrcToTargetType<short, Float32Type, std::enable_if<true, void>>(PatternRewriter &, const SmallVector<Value> &, bool);
template SmallVector<Value> mlir::hfusion::normalizeSrcToTargetType<bool, Float16Type, std::enable_if<true, void>>(PatternRewriter &, const SmallVector<Value> &, bool);
template SmallVector<Value> mlir::hfusion::normalizeSrcToTargetType<signed char, Float32Type, std::enable_if<true, void>>(PatternRewriter &, const SmallVector<Value> &, bool);
template SmallVector<Value> mlir::hfusion::normalizeSrcToTargetType<signed char, Float16Type, std::enable_if<true, void>>(PatternRewriter &, const SmallVector<Value> &, bool);
} // namespace mlir::hfusion