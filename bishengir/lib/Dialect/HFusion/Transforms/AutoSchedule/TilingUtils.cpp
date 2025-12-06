//===- TilingUtils.h -- Utilities for Auto Schedule Tiling ------*- C++ -*-===//
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
// This file implements tiling utilties for auto scheduler.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tiling-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// By convention, tiling key is the first tiling data.
constexpr size_t kTilingKeyPos = 0;

Value evaluateAffineExpr(AffineExpr e, const SmallVector<OpFoldResult> &symbols,
                         OpBuilder &opBuilder) {
  return affine::makeComposedAffineApply(opBuilder, opBuilder.getUnknownLoc(),
                                         e, symbols)
      ->getResult(0);
}

} // namespace

NamedAttribute hfusion::getTilingDataAttr(OpBuilder &opBuilder) {
  return hacc::createHACCKernelArgAttr(opBuilder.getContext(),
                                       hacc::KernelArgType::kTilingData);
}

NamedAttribute hfusion::getTilingKeyAttr(OpBuilder &opBuilder) {
  return hacc::createHACCKernelArgAttr(opBuilder.getContext(),
                                       hacc::KernelArgType::kTilingKey);
}

//===----------------------------------------------------------------------===//
// Expr Implementation
//===----------------------------------------------------------------------===//

Expr Expr::operator+(int64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs + rhs;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator+(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs + rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::operator-(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs - rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::floorDiv(uint64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs.floorDiv(rhs);
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::floorDiv(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs.floorDiv(rhs);
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::operator*(int64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs * rhs;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator*(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs * rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::alignTo(uint64_t align) {
  MLIRContext *ctx = getContext();
  assert(align != 0u && "Align can't be 0.");
  AffineExpr alignExpr = getAffineConstantExpr(align, ctx);
  AffineExpr self = getAffineSymbolExpr(0, ctx);
  AffineExpr result = ((self + alignExpr) - 1).floorDiv(alignExpr) * alignExpr;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::alignDown(uint64_t align) {
  MLIRContext *ctx = getContext();
  assert(align != 0u && "Align can't be 0.");
  AffineExpr alignExpr = getAffineConstantExpr(align, ctx);
  AffineExpr self = getAffineSymbolExpr(0, ctx);
  AffineExpr result = self.floorDiv(alignExpr) * alignExpr;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator>(int64_t cst) {
  Expr vExpr = builder_->createConstExpr(cst);
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sgt, this->v_, vExpr.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator>(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sgt, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator<(int64_t cst) {
  Expr vExpr = builder_->createConstExpr(cst);
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::slt, this->v_, vExpr.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator<(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::slt, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator==(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::eq, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator==(int64_t cst) {
  Expr vExpr = builder_->createConstExpr(cst);
  return this->operator==(vExpr);
}

Expr Expr::operator<=(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sle, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator>=(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sge, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr hfusion::max(Expr lhs, Expr rhs) {
#ifndef NDEBUG
  StmtExprBuilder &builder = lhs.getBuilder();

  Value result = builder.create<arith::MaxSIOp>(
      lhs.getMaterializedValue().getLoc(), lhs.getMaterializedValue(),
      rhs.getMaterializedValue());

  return Expr(result, ExprKind::kRegular, &builder);
#else
  return select(lhs >= rhs, lhs, rhs);
#endif
}

Expr hfusion::max(Expr lhs, int64_t rhs) {
  StmtExprBuilder &builder = lhs.getBuilder();
  Expr rhsExpr = builder.createConstExpr(rhs);
  Value result = builder.create<arith::MaxSIOp>(
      lhs.getMaterializedValue().getLoc(), lhs.getMaterializedValue(),
      rhsExpr.getMaterializedValue());
  return Expr(result, ExprKind::kRegular, &builder);
}

Expr hfusion::min(Expr lhs, Expr rhs) {
  StmtExprBuilder &builder = lhs.getBuilder();
  Value result = builder.create<arith::MinSIOp>(
      lhs.getMaterializedValue().getLoc(), lhs.getMaterializedValue(),
      rhs.getMaterializedValue());
  return Expr(result, ExprKind::kRegular, &builder);
}

Expr hfusion::select(Expr condition, Expr trueValue, Expr falseValue) {
  StmtExprBuilder &builder = condition.getBuilder();
  return condition * trueValue +
         (builder.createConstExpr(1) - condition) * falseValue;
}

Expr hfusion::select(Expr condition, Expr trueValue, int64_t falseValue) {
  StmtExprBuilder &builder = condition.getBuilder();
  return select(condition, trueValue, builder.createConstExpr(falseValue));
}

//===----------------------------------------------------------------------===//
// StmtExprBuilder Implementation
//===----------------------------------------------------------------------===//

Expr StmtExprBuilder::createConstExpr(int64_t cst) {
  return Expr(
      evaluateAffineExpr(mlir::getAffineConstantExpr(cst, this->getContext()),
                         /*symbols=*/{}, *this),
      ExprKind::kRegular, this);
}

Expr StmtExprBuilder::createDimSymbolExpr(Value tensorValue, size_t dimIdx) {
  assert(tilingInfo_);
  assert(isa<ShapedType>(tensorValue.getType()) &&
         "source value must be shaped type!");
  auto dimValue =
      this->create<tensor::DimOp>(this->getUnknownLoc(), tensorValue, dimIdx)
          ->getOpResult(0);
  return DimSymbol(dimValue, this);
}

Operation *StmtExprBuilder::recursivelyCloneOp(Operation *op,
                                               IRMapping &mapper) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    recursivelyCloneOp(defOp, mapper);
  }
  Operation *newOp = this->clone(*op, mapper);
  mapper.map(op->getResults(), newOp->getResults());
  return newOp;
}

Expr StmtExprBuilder::createDimSymbolExpr(size_t tensorIdx, size_t dimIdx) {
  assert(tilingInfo_);
  Value hostTilingArg = tilingInfo_->getHostTilingFuncArg(tensorIdx);
  Value maybeReshapedInput = this->kernelInfo_->inputValues[tensorIdx];
  if (isa<BlockArgument>(maybeReshapedInput)) {
    // create symbol from tensor arg directly
    return createDimSymbolExpr(hostTilingArg, dimIdx);
  }
  // create symbol from reshape op result
  SmallVector<Operation *> reshapeTrace =
      hfusion::getReshapeOrSliceOpProduceTrace(maybeReshapedInput);
  assert(!reshapeTrace.empty() && "reshape trace must not be empty");

#ifndef NDEBUG
  Value reshapeProducer = hfusion::getReshapeOrSliceSource(reshapeTrace.back());
  assert(isa<BlockArgument>(reshapeProducer) &&
         "src of reshape op should be block argument");
#endif

  Value hostTilingV = hostTilingArg;
  Expr expr;
  // replace reshape op one by one from arg to users
  for (Operation *reshapeOp : llvm::reverse(reshapeTrace)) {
    Value reshapeSrc = hfusion::getReshapeOrSliceSource(reshapeOp);
    // clone reshape op and replace src with value in host tiling func
    IRMapping mapper;
    mapper.map(reshapeSrc, hostTilingV);
    Operation *clonedReshapeOp = recursivelyCloneOp(reshapeOp, mapper);
    Value clonedReshapeV = clonedReshapeOp->getResult(0);
    expr = createDimSymbolExpr(clonedReshapeV, dimIdx);
    // update current host tiling value using the result reshaped value
    hostTilingV = clonedReshapeV;
  }
  return expr;
}

Expr StmtExprBuilder::createExpr(Value val, int64_t idx, IRMapping &mapper) {
  auto dimValue = cast<ShapedType>(val.getType()).getShape()[idx];
  if (!ShapedType::isDynamic(dimValue)) {
    return createConstExpr(dimValue);
  }
  return createDimSymbolExpr(recursivelyCloneValue(val, mapper), idx);
}

Value StmtExprBuilder::recursivelyCloneValue(Value val, IRMapping &mapper) {
  if (mapper.contains(val))
    return mapper.getValueMap().at(val);
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto argNumber = blockArg.getArgNumber();
    Value kernelArg = kernelInfo_->getKernelFuncArg(argNumber);
    // create symbol from reshape op result
    Value hostTilingArg = tilingInfo_->getHostTilingFuncArg(argNumber);
    mapper.map(kernelArg, hostTilingArg);
    return hostTilingArg;
  }

  auto *op = val.getDefiningOp();
  assert(op != nullptr && "Defining op not found");

  for (Value operand : op->getOperands()) {
    recursivelyCloneValue(operand, mapper);
  }
  auto resultPosition = cast<OpResult>(val).getResultNumber();
  Operation *newOp = this->clone(*op, mapper);
  mapper.map(op->getResults(), newOp->getResults());
  return newOp->getResult(resultPosition);
}

SmallVector<Expr> StmtExprBuilder::createDimSymbolExprs(size_t tensorIdx,
                                                        size_t startDim,
                                                        size_t endDim) {
  return llvm::map_to_vector(
      llvm::to_vector(llvm::seq<size_t>(startDim, endDim)),
      [this, &tensorIdx](size_t idx) -> Expr {
        return this->createDimSymbolExpr(
            /*tensorIdx=*/tensorIdx, /*dimIdx=*/idx);
      });
}

SmallVector<Expr> hfusion::tiling::getAccumulatedDims(SmallVector<Expr> dims) {
  assert(!dims.empty());
  SmallVector<Expr> accumulatedDims = {dims.front()};
  for (const auto &dim : llvm::drop_begin(dims)) {
    auto accumulatedValue = accumulatedDims.back() * dim;
    accumulatedDims.push_back(accumulatedValue);
  }
  return accumulatedDims;
}

//===----------------------------------------------------------------------===//
// StmtExprBuilder Implementation
//===----------------------------------------------------------------------===//

CallStmt StmtExprBuilder::createCallStmt(FlatSymbolRefAttr funcName,
                                         SmallVector<Value> operands) {
  // Check whether the function exists in the module.
  Operation *symbol = this->module_.lookupSymbol(funcName.getAttr());
  if (!symbol) {
    assert(false && "module has no such function");
    return {};
  }

  // Check whether the operands type is consistent with the function definition.
  auto funcOp = mlir::dyn_cast<func::FuncOp>(symbol);

  FunctionType funcType = funcOp.getFunctionType();
  if (operands.size() != funcType.getNumInputs()) {
    assert(false && "Argument count mismatch for function: ");
    return {};
  }

  for (size_t i = 0; i < operands.size(); ++i) {
    if (operands[i].getType() != funcType.getInput(i)) {
      assert(false && "Operand type mismatch at index ");
      return {};
    }
  }

  // create func.call op
  auto callOp = this->create<func::CallOp>(
      this->getUnknownLoc(), funcType.getResults(), funcName, operands);
  return CallStmt(this, callOp);
}

CallStmt StmtExprBuilder::createExternCallStmt(FlatSymbolRefAttr funcName,
                                               SmallVector<Value> operands,
                                               StringAttr externLibraryPath) {
  // If function is not present, create extern declaration based on the operand
  // type
  auto originalInsertionPoint = this->saveInsertionPoint();

  this->setInsertionPointToEnd(this->module_.getBody());
  if (!this->module_.lookupSymbol(funcName.getAttr())) {
    SmallVector<Type> operandTypes;
    for (auto operand : operands) {
      operandTypes.push_back(operand.getType());
    }
    auto funcType = FunctionType::get(this->getContext(), operandTypes, {});
    auto funcOp = this->create<func::FuncOp>(this->getUnknownLoc(),
                                             funcName.getValue(), funcType);
    auto libPathAttr =
        StringAttr::get(this->getContext(), externLibraryPath.getValue());
    funcOp->setAttr(hacc::ExternalFunctionPathAttr::name, libPathAttr);

    // extern functions are host function
    hacc::utils::setHost(funcOp);
    funcOp.setPrivate();
  }
  this->restoreInsertionPoint(originalInsertionPoint);

  auto *symbol = this->module_.lookupSymbol(funcName.getAttr());
  if (!symbol) {
    return {};
  }

  auto funcOp = mlir::dyn_cast<func::FuncOp>(symbol);
  if (!funcOp) {
    return {};
  }

  auto callOp =
      this->create<func::CallOp>(this->getUnknownLoc(), funcOp, operands);
  return CallStmt(this, callOp);
}

void StmtExprBuilder::createConstraintVerification(
    const Expr &predicate, llvm::StringRef errorMessage) {
  Location loc = predicate.getMaterializedValue().getLoc();
  Value predicateValue = this->create<arith::IndexCastUIOp>(
      loc, this->getI1Type(), predicate.getMaterializedValue());
  this->create<cf::AssertOp>(loc, predicateValue, errorMessage);
}

//===----------------------------------------------------------------------===//
// TilingData Implementation
//===----------------------------------------------------------------------===//

bool TilingData::isConst() const {
  return std::holds_alternative<int64_t>(data_);
}

Expr *TilingData::getExpr() const {
  assert(!isConst());
  return std::get<std::unique_ptr<Expr>>(data_).get();
}

int64_t TilingData::getConst() const {
  assert(isConst());
  return std::get<int64_t>(data_);
}

void TilingData::setData(int64_t newData) { data_ = TilingDataTy(newData); }

void TilingData::setData(Expr &&newData) {
  data_ = TilingDataTy(std::make_unique<Expr>(newData));
}

void TilingData::setHeuristicValueForKey(TilingKey key, int64_t hint) {
  if (this->isConst() && this->getConst() != hint)
    emitWarning(UnknownLoc(), "setting a heuristic tiling value that is "
                              "inconsistent with the constant tiling data");

  if (heuristicForKey_.contains(key))
    emitWarning(UnknownLoc(), "Overwriting existing heuristic");

  heuristicForKey_[key] = hint;
}

std::optional<int64_t>
TilingData::getHeuristicValueForKey(TilingKey key) const {
  if (heuristicForKey_.contains(key))
    return heuristicForKey_.at(key);

  return std::nullopt;
}

void TilingData::resetHeuristics() { heuristicForKey_.clear(); }

//===----------------------------------------------------------------------===//
// TilingCases Implementation
//===----------------------------------------------------------------------===//

LogicalResult TilingCases::addKey(TilingKey caseKey) {
  return success(cases.insert(caseKey));
}

//===----------------------------------------------------------------------===//
// TilingInfo Implementation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> TilingInfo::evaluateTilingComputation(
    TilingComputeFn fn, KernelInfo *kernelInfo, StmtExprBuilder *builder) {
  FailureOr<std::pair<TilingCases, TilingStruct>> result =
      fn(kernelInfo, builder);
  if (failed(result))
    return {};

  std::tie(this->caseKeys_, this->struct_) = std::move(*result);
  SmallVector<Value> returns;
  for (std::unique_ptr<TilingData> &data : struct_) {
    Value exprValue = data->getExpr()->getMaterializedValue();
    Value castedValue = castIndexTo(exprValue, data->getType(), *builder);
    returns.push_back(castedValue);
  }
  return returns;
}

SmallVector<TilingData *> TilingInfo::getTilingStruct() {
  return llvm::map_to_vector(struct_,
                             [](TilingDataPtr &td) { return td.get(); });
}

BlockArgument TilingInfo::getHostTilingFuncArg(size_t idx) {
  if (idx >= hostTilingFunc_.getNumArguments()) {
    llvm::report_fatal_error("idx for host tiling func arg out of bound");
  }
  return hostTilingFunc_.getArgument(idx);
}

bool TilingInfo::isTilingFullyStatic() {
  return llvm::all_of(struct_, [](TilingDataPtr &td) { return td->isConst(); });
}

void TilingInfo::recordKernelFunc(TilingKey key, func::FuncOp f) {
  tilingKey2Kernel_.insert({key, f});
}

DenseMap<TilingKey, func::FuncOp> TilingInfo::getTilingKey2KernelMap() {
  return tilingKey2Kernel_;
}

LogicalResult TilingInfo::trySimplifyTilingFunc() {
  // Simplify host tiling func
  PassManager pm(hostTilingFunc_->getContext());
  pm.addPass(memref::createResolveRankedShapeTypeResultDimsPass());
  CanonicalizerOptions options;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.addPass(createCSEPass());

  if (failed(pm.run(hostTilingFunc_)))
    return failure();

  // Get return value and see if it's constant
  func::ReturnOp returnOp = nullptr;
  hostTilingFunc_.walk([&returnOp](func::ReturnOp op) { returnOp = op; });

  for (auto [returnVal, tilingDataPtr] :
       llvm::zip_equal(returnOp->getOperands(), this->struct_)) {
    std::optional<int64_t> maybeConst =
        getConstantIntValue(getAsOpFoldResult(returnVal));
    if (!maybeConst.has_value())
      continue;
    tilingDataPtr->setData(maybeConst.value());
  }
  return success();
}

void TilingInfo::pruneTilingExcept(int64_t keepKey) {
  caseKeys_.removeIf(keepKey);
}

TilingData *TilingInfo::getTilingData(unsigned idx) const {
  assert(idx < size());
  return struct_[idx].get();
}

TilingData *TilingInfo::getTilingKey() const {
  return getTilingData(kTilingKeyPos);
}

//===----------------------------------------------------------------------===//
// TilingStruct Implementation
//===----------------------------------------------------------------------===//

TilingStruct::TilingStruct(size_t size) {
  data_ = SmallVector<TilingDataPtr>(size);
}

void TilingStruct::push_back(TilingData &&tilingData) {
  data_.push_back(std::make_unique<TilingData>(std::move(tilingData)));
}

TilingDataPtr &TilingStruct::operator[](size_t index) {
  assert(index < data_.size());
  return data_[index];
}

const TilingDataPtr &TilingStruct::operator[](size_t index) const {
  assert(index < data_.size());
  return data_[index];
}