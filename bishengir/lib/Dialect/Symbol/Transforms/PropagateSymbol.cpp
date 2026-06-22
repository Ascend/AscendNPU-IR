//===- PropagateSymbol.cpp --------- Propagate Symbol Pass ----------------===//
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
// This file implements a pass to propagate symbols
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Symbol/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "propagate-symbol"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace symbol {
#define GEN_PASS_DEF_PROPAGATESYMBOL
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"

namespace {

// TODO: replace the code implementation here after merging the full
// implementation of SymbolManager
class SymbolManager {
public:
  static FlatSymbolRefAttr getUniqueSymbolAttr(MLIRContext *ctx) {
    return FlatSymbolRefAttr::get(ctx, SymbolManager::getUniqueSymbolName());
  }

  static std::string getUniqueSymbolName() {
    return "S" + std::to_string(symbolIdx++);
  }

private:
  static int symbolIdx;
};
int SymbolManager::symbolIdx = 0;

mlir::AffineMapAttr createAffineFromShape(ArrayRef<int64_t> shape,
                                          mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::AffineExpr, 4> exprs;
  int64_t dynSymbolNum = 0;
  for (auto dim : shape) {
    if (dim == mlir::ShapedType::kDynamic) {
      exprs.push_back(mlir::getAffineSymbolExpr(dynSymbolNum, ctx));
      dynSymbolNum++;
    } else {
      exprs.push_back(mlir::getAffineConstantExpr(dim, ctx));
    }
  }
  auto affineMap = mlir::AffineMap::get(0, dynSymbolNum, exprs, ctx);
  return mlir::AffineMapAttr::get(affineMap);
}

// create BindSymbolicShapeOp for op with ReifyRankedShapedTypeOpInterface
// this pattern will bind reified tensor.dim on dynamic dims:
// input 1:
//   %add = linalg.elemwise_binary
//          ins(%0, %1 : tensor<?x640xf16>, tensor<?x640xf16>)
//          outs(%2 : tensor<?x640xf16>)
// output 1:
//   %dim = tensor.dim %0, %c0
//   %add = linalg.elemwise_binary ins(%0, %1) outs(%2)
//   symbol.bind_symbolic_shape %add, [%dim], affine_map<()[s0] -> (s0, 640)>
//
// if reified shape is affine.apply, this pattern will create new SymbolicIntOp
// with affine and bind it
// input2:
//   %concat = tensor.concat dim(0) %0, %1 : (tensor<?x8xf16>, tensor<?x8xf16>)
// output2:
//   %dim0 = tensor.dim %0, %c0
//   %dim1 = tensor.dim %1, %c0
//   %op =
//   %concat = ...
//   symbol.bind_symbolic_shape %concat, [%op], affine_map<()[s0] -> (s0, 8)>
struct BindReifyResultShape
    : public OpInterfaceRewritePattern<ReifyRankedShapedTypeOpInterface> {
  using OpInterfaceRewritePattern<
      ReifyRankedShapedTypeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ReifyRankedShapedTypeOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return isa<symbol::BindSymbolicShapeOp>(user);
        })) {
      return rewriter.notifyMatchFailure(op, "already bind symbolic shape op");
    }

    ReifiedRankedShapedTypeDims outputShapes;
    if (failed(reifyResultShapes(rewriter, op, outputShapes))) {
      return rewriter.notifyMatchFailure(op, "fail to reify result shapes");
    }

    // bind reified output shape for each op result
    SmallVector<symbol::BindSymbolicShapeOp> bindOps;
    for (const auto &[outputShape, outputValue] :
         llvm::zip(outputShapes, op->getResults())) {
      auto shapedType = dyn_cast<ShapedType>(outputValue.getType());
      if (!shapedType) {
        LDBG("not support bind non ShapedType: " << shapedType);
        continue;
      }
      auto bindMaybe = bindReifiedResultShape(outputShape, outputValue,
                                              op->getLoc(), rewriter);
      if (bindMaybe.has_value()) {
        bindOps.push_back(bindMaybe.value());
      }
    }

    if (bindOps.empty()) {
      return rewriter.notifyMatchFailure(op, "no dynamic value to bind");
    }
    return success();
  }

  std::optional<symbol::BindSymbolicShapeOp>
  bindReifiedResultShape(const SmallVector<OpFoldResult> &outputShape,
                         Value outputValue, Location loc,
                         PatternRewriter &rewriter) const {
    SmallVector<Value> bindValues;
    for (const OpFoldResult &ofr : outputShape) {
      LDBG("reified output shape: " << ofr);
      if (ofr.is<Attribute>()) {
        continue;
      }
      Operation *reifyOp = ofr.get<Value>().getDefiningOp();
      if (!reifyOp) {
        continue;
      }
      if (auto dimOp = dyn_cast<tensor::DimOp>(reifyOp)) {
        bindValues.push_back(dimOp.getResult());
        continue;
      }
      if (auto affineOp = dyn_cast<affine::AffineApplyOp>(reifyOp)) {
        bindValues.push_back(handleAffineOp(affineOp, rewriter));
        continue;
      }
      if (auto symbolIntOp = dyn_cast<symbol::SymbolicIntOp>(reifyOp)) {
        bindValues.push_back(symbolIntOp.getResult());
        continue;
      }
      if (mlir::utils::isArithOp(reifyOp)) {
        bindValues.push_back(reifyOp->getResult(0));
        continue;
      }
      llvm::report_fatal_error("unsupported reify op type");
    }

    MLIRContext *ctx = getContext();
    auto outputType = cast<ShapedType>(outputValue.getType());
    auto shapeExpr = createAffineFromShape(outputType.getShape(), ctx);
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(outputValue);
    return rewriter.create<symbol::BindSymbolicShapeOp>(loc, outputValue,
                                                        bindValues, shapeExpr);
  }

  Value handleAffineOp(affine::AffineApplyOp affineOp,
                       PatternRewriter &rewriter) const {
    // create new SymbolicIntOp based on reified affine op
    MLIRContext *ctx = getContext();
    auto symbolAttr = SymbolManager::getUniqueSymbolAttr(ctx);
    return rewriter.create<symbol::SymbolicIntOp>(
        affineOp->getLoc(), symbolAttr, affineOp->getOperands(),
        affineOp.getMapAttr());
  }
};

// compute the dynIndex for dynamic dims given the index for all dims
std::optional<int64_t> getIndexForDynamicDim(ArrayRef<int64_t> shapes,
                                             int64_t index) {
  int64_t size = static_cast<int64_t>(shapes.size());
  if (index >= size || !ShapedType::isDynamic(shapes[index])) {
    return std::nullopt;
  }

  int64_t dynIndex = 0;
  for (int64_t i = 0; i < size && i < index; ++i) {
    if (ShapedType::isDynamic(shapes[i])) {
      dynIndex++;
    }
  }
  return dynIndex;
}

// propagate symbols by replacing tensor.dim with the symbol it binds to.
// input:
//   %S0 = symbol.symbolic_int @S0
//   symbol.bind_symbolic_shape %arg0, [%S0], affine_map<()[s0] -> (s0, 640)>
//   %dim = tensor.dim %arg0, %c0
//   %empty = tensor.empty(%dim) : tensor<?x640xf16>
//   %add = linalg.elemwise_binary ins(%arg0, %arg1) outs(%empty)
//   symbol.bind_symbolic_shape %add, [%dim], affine_map<()[s0] -> (s0, 640)>
// output:
//   ...
//   %empty = tensor.empty(%S0) : tensor<?x640xf16>
//   %add = linalg.elemwise_binary ins(%arg0, %arg1) outs(%empty)
//   symbol.bind_symbolic_shape %add, [%S0], affine_map<()[s0] -> (s0, 640)>
class PropagateSymbolByTensorDim : public OpRewritePattern<tensor::DimOp> {
public:
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const final {
    Value dimSrc = op.getSource();
    auto shapedType = dyn_cast<ShapedType>(dimSrc.getType());
    if (!shapedType) {
      return failure();
    }

    auto constIndex = op.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    if (!constIndex) {
      return rewriter.notifyMatchFailure(
          op, "only support dim index to be constant int");
    }
    int64_t index = constIndex.value();

    auto bindOp = utils::getBindSymbolUser(dimSrc);
    if (!bindOp.has_value()) {
      return rewriter.notifyMatchFailure(op,
                                         "no symbol bind for tensor.dim src");
    }

    // replace cur tensor.dim with symbol corresponding to the dim index
    SmallVector<Value> shapeSymbols = bindOp->getShapeSymbols();
    auto dynIndexMaybe = getIndexForDynamicDim(shapedType.getShape(), index);
    if (!dynIndexMaybe.has_value()) {
      return rewriter.notifyMatchFailure(op, "invalid index for dynamic dim");
    }
    auto dynIndex = dynIndexMaybe.value();
    auto *curSymbolOp = shapeSymbols[dynIndex].getDefiningOp();
    if (!curSymbolOp) {
      return rewriter.notifyMatchFailure(op, "no symbol op found");
    }
    if (op == curSymbolOp) {
      // to avoid the case when the bind symbol is cur tensor.dim op.
      // it can happen if tensor.dim use shape from tensor op other than arg.
      return rewriter.notifyMatchFailure(
          op, "should replace with a different symbol op");
    }
    rewriter.replaceOp(op, shapeSymbols[dynIndex]);
    return success();
  }
};

// Symbolize dynamic shape dims by creating symbolic_int op, to replace arith
// index op used for init dynamic shaped tensor
template <typename OpType>
class SymbolizeDynamicShapeTensor : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    // collect operands of index type representing dynamic dims
    SmallVector<Value> indexOperands;
    if constexpr (std::is_same_v<OpType, tensor::EmptyOp>) {
      indexOperands = op->getOperands();
    } else if constexpr (std::is_same_v<OpType, tensor::ExpandShapeOp>) {
      auto expand = cast<tensor::ExpandShapeOp>(op);
      SmallVector<OpFoldResult> outputs = expand.getMixedOutputShape();
      SmallVector<int64_t> statics;
      dispatchIndexOpFoldResults(outputs, indexOperands, statics);
    } else if constexpr (std::is_same_v<OpType, tensor::ExtractSliceOp>) {
      auto slice = cast<tensor::ExtractSliceOp>(op);
      SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
      SmallVector<int64_t> statics;
      dispatchIndexOpFoldResults(sizes, indexOperands, statics);
    } else {
      llvm::report_fatal_error("unsupported tensor op type");
    }

    // filter arith index operands
    SmallVector<Operation *> indexOps;
    for (Value operand : indexOperands) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !mlir::utils::isArithOp(defOp)) {
        continue;
      }
      indexOps.push_back(defOp);
    }
    if (indexOps.empty()) {
      return failure();
    }

    // replace arith index operand with new symbol int op
    for (Operation *curIndexOp : indexOps) {
      auto symbolAttr =
          SymbolManager::getUniqueSymbolAttr(rewriter.getContext());
      rewriter.setInsertionPointAfter(curIndexOp);
      Value curIndexValue = curIndexOp->getResult(0);
      auto symbolOp = rewriter.create<symbol::SymbolicIntOp>(
          curIndexOp->getLoc(), symbolAttr, ValueRange{curIndexValue});
      rewriter.replaceAllUsesExcept(curIndexValue, symbolOp.getResult(),
                                    symbolOp);
    }
    return success();
  }
};

// Unify all symbols in `symbolsToUnify` using the topmost symbol in kernel
LogicalResult unifySymbols(ArrayRef<Value> symbolsToUnify,
                           PatternRewriter &rewriter) {
  // remove duplicate symbols
  DenseSet<Value> uniqSymbols{symbolsToUnify.begin(), symbolsToUnify.end()};
  if (uniqSymbols.size() <= 1) {
    return failure();
  }

  // sort symbols to based on operation order in kernel (top to down)
  SmallVector<Value> symbols = llvm::to_vector(uniqSymbols);
  llvm::sort(symbols, [](Value lhs, Value rhs) {
    Operation *lhsOp = lhs.getDefiningOp();
    Operation *rhsOp = rhs.getDefiningOp();
    assert(lhsOp != nullptr && rhsOp != nullptr);
    assert(lhsOp->getBlock() == rhsOp->getBlock() &&
           "lhs and rhs op must be in the same block");
    return lhsOp->isBeforeInBlock(rhsOp);
  });

  // unify symbols using the topmost symbol
  for (size_t i = 1; i < symbols.size(); ++i) {
    rewriter.replaceAllUsesWith(symbols[i], symbols[0]);
  }
  return success();
}

std::optional<Value> getBindSymbolForDim(Value value, int64_t dim) {
  // get binded symbol for `value` of corresponding `dim`
  auto shapedType = llvm::cast<ShapedType>(value.getType());
  auto dynIndex = getIndexForDynamicDim(shapedType.getShape(), dim);
  assert(dynIndex.has_value());
  auto bindOp = utils::getAnyUserOfType<BindSymbolicShapeOp>(value);
  if (!bindOp.has_value())
    return std::nullopt;
  auto val = bindOp->getShapeSymbols()[dynIndex.value()];
  return val;
}

SmallVector<Value> getBindSymbolForDim(ArrayRef<Value> values, int64_t dim) {
  SmallVector<Value> result;
  for (Value value : values) {
    auto symbolMaybe = getBindSymbolForDim(value, dim);
    if (symbolMaybe.has_value()) {
      result.push_back(symbolMaybe.value());
    }
  }
  return result;
}

LogicalResult unifySymbolsForDim(ArrayRef<Value> values, int64_t dim,
                                 PatternRewriter &rewriter) {
  // collect symbols from `values` of corresponding `dim`
  SmallVector<Value> symbols;
  for (Value value : values) {
    auto symbol = getBindSymbolForDim(value, dim);
    if (symbol.has_value()) {
      symbols.push_back(symbol.value());
    }
  }
  return unifySymbols(symbols, rewriter);
}

/// Unfiy op with same operands and results by merging symbols of all values
template <typename OpType>
class UnifySameOperandsAndResultShape : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0) {
      return failure();
    }

    // make sure all operands and results to be unified are the same shape
    SmallVector<Value> values;
    values.append(SmallVector<Value>{op->getOperands()});
    values.append(SmallVector<Value>{op->getResults()});
    auto shapedTypes = getShapedTypes(values);
    if (shapedTypes.empty() || shapedTypes.size() != values.size()) {
      return rewriter.notifyMatchFailure(
          op, "operands and results should be shaped types");
    }

    if (!hasSameShape(shapedTypes)) {
      return rewriter.notifyMatchFailure(
          op, "operands and results should have the same shape");
    }

    // unify symbols for each dim
    auto shapedType = shapedTypes.front();
    bool changed = false;
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      if (!shapedType.isDynamicDim(dim)) {
        continue;
      }
      LogicalResult unifyResult = unifySymbolsForDim(values, dim, rewriter);
      if (succeeded(unifyResult)) {
        changed = true;
      }
    }
    return success(changed);
  }

  SmallVector<ShapedType> getShapedTypes(ArrayRef<Value> values) const {
    SmallVector<ShapedType> result;
    for (Value value : values) {
      auto shapedType = dyn_cast<ShapedType>(value.getType());
      if (shapedType) {
        result.push_back(shapedType);
      }
    }
    return result;
  }

  bool hasSameShape(ArrayRef<ShapedType> shapedTypes) const {
    if (shapedTypes.empty()) {
      return false;
    }
    ArrayRef<int64_t> shape = shapedTypes.front().getShape();
    return llvm::all_of(shapedTypes, [&shape](ShapedType type) {
      return type.getShape() == shape;
    });
  }
};

/// Unify op with partial equal dims between inputs and inits/results, by
/// merging the symbols for these equal dims in corresponding inputs and
/// inits/results values.
/// Constraint:
/// 1. all inits/results should have the same shape
/// 2. all inputs should have the same shape
template <typename OpType>
class UnifyPartialEquivalentDims : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value> inputs = getInputs(op);
    SmallVector<Value> outputs = getOutputs(op);
    if (inputs.empty() || outputs.empty()) {
      return rewriter.notifyMatchFailure(op, "cannot get inputs and outputs");
    }

    if (!hasSameShape(inputs) || !hasSameShape(outputs)) {
      return rewriter.notifyMatchFailure(
          op, "inputs or outputs should have same shape");
    }

    auto outputType = llvm::cast<ShapedType>(outputs[0].getType());
    SmallVector<int64_t> map = getDimMap(op);

    int64_t outputRank = outputType.getRank();
    bool changed = false;
    for (int64_t dim = 0; dim < outputRank; ++dim) {
      if (!outputType.isDynamicDim(dim) || map[dim] == -1) {
        continue;
      }
      auto outputSymbols = getBindSymbolForDim(outputs, dim);
      auto inputSymbols = getBindSymbolForDim(inputs, map[dim]);
      if (outputSymbols.empty() || inputSymbols.empty()) {
        continue;
      }
      SmallVector<Value> symbols;
      symbols.append(outputSymbols);
      symbols.append(inputSymbols);
      LogicalResult unifyResult = unifySymbols(symbols, rewriter);
      if (succeeded(unifyResult)) {
        changed = true;
      }
    }
    return success(changed);
  }

  bool hasSameShape(ArrayRef<Value> values) const {
    assert(!values.empty() && "values cannot be empty");
    auto firstType = dyn_cast<ShapedType>(values[0].getType());
    if (!firstType) {
      return false;
    }
    for (Value value : llvm::drop_begin(values)) {
      auto curType = dyn_cast<ShapedType>(value.getType());
      if (curType.getShape() != firstType.getShape()) {
        return false;
      }
    }
    return true;
  }

  SmallVector<Value> getInputs(OpType op) const {
    SmallVector<Value> inputs;
    if constexpr (std::is_same_v<linalg::BroadcastOp, OpType>) {
      inputs = {op.getInput()};
    } else if constexpr (std::is_same_v<linalg::ReduceOp, OpType> ||
                         std::is_same_v<tensor::ConcatOp, OpType>) {
      inputs = op.getInputs();
    } else {
      llvm::report_fatal_error("unsupported op type for UnifyPartialEquivalentDims");
    }
    return inputs;
  }

  SmallVector<Value> getOutputs(OpType op) const {
    SmallVector<Value> outputs;
    if constexpr (std::is_same_v<linalg::BroadcastOp, OpType>) {
      outputs.push_back(op.getInit());
      outputs.push_back(op->getResult(0));
    } else if constexpr (std::is_same_v<linalg::ReduceOp, OpType>) {
      SmallVector<Value> inits = op.getInits();
      SmallVector<Value> results = op->getResults();
      outputs.append(inits);
      outputs.append(results);
    } else if constexpr (std::is_same_v<tensor::ConcatOp, OpType>) {
      outputs.push_back(op.getResult());
    } else {
      llvm::report_fatal_error("unsupported op type for UnifyPartialEquivalentDims");
    }
    return outputs;
  }

  SmallVector<int64_t> getDimMap(linalg::BroadcastOp op) const {
    DenseSet<int64_t> brcDims(op.getDimensions().begin(),
                              op.getDimensions().end());
    auto dstType = llvm::cast<ShapedType>(op.getInit().getType());
    int64_t cnt = 0;
    int64_t rank = dstType.getRank();
    SmallVector<int64_t> map(rank, -1);
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (brcDims.contains(dim)) {
        cnt -= 1;
        continue;
      }
      map[dim] = dim + cnt;
    }
    return map;
  }

  SmallVector<int64_t> getDimMap(linalg::ReduceOp op) const {
    DenseSet<int64_t> reduceDims(op.getDimensions().begin(),
                                 op.getDimensions().end());
    auto srcType = llvm::cast<ShapedType>(op.getInputs()[0].getType());
    int64_t rank = srcType.getRank();
    SmallVector<int64_t> map;
    for (int64_t i = 0; i < rank; ++i) {
      if (reduceDims.contains(i)) {
        continue;
      }
      map.push_back(i);
    }
    return map;
  }

  SmallVector<int64_t> getDimMap(tensor::ConcatOp op) const {
    int64_t rank = op.getResultType().getRank();
    SmallVector<int64_t> map = llvm::to_vector(llvm::seq<int64_t>(0, rank));
    map[op.getDim()] = -1;
    return map;
  }
};

class FoldSymbolToTensorEmpty
    : public OpRewritePattern<symbol::BindSymbolicShapeOp> {
public:
  using OpRewritePattern<symbol::BindSymbolicShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(symbol::BindSymbolicShapeOp bindOp,
                                PatternRewriter &rewriter) const final {
    Value bindSrc = bindOp.getOperand();
    auto empty = bindSrc.getDefiningOp<tensor::EmptyOp>();
    if (!empty) {
      return failure();
    }
    SmallVector<Value> operands = empty->getOperands();
    SmallVector<Value> symbols = bindOp.getShapeSymbols();
    if (operands == symbols) {
      return failure();
    }

    auto newEmpty = rewriter.create<tensor::EmptyOp>(
        empty->getLoc(), empty->getResultTypes(), symbols);
    rewriter.replaceOp(empty, newEmpty);
    return success();
  }
};

class PropagateSymbolPass
    : public impl::PropagateSymbolBase<PropagateSymbolPass> {
public:
  explicit PropagateSymbolPass() : PropagateSymbolBase() {}
  void runOnOperation() final;
};

// create init symbolic_int and bind_symbolic_shape for func arguments
void initSymbolForFuncArgs(func::FuncOp func) {
  OpBuilder builder(func);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&func.getRegion().front());

  Operation &firstOp = func.getFunctionBody().getBlocks().front().front();
  Location loc = firstOp.getLoc();
  MLIRContext *ctx = func.getContext();

  for (BlockArgument ba : func.getArguments()) {
    auto shapedType = dyn_cast<ShapedType>(ba.getType());
    if (!shapedType || shapedType.hasStaticShape()) {
      continue;
    }

    if (utils::getBindSymbolUser(ba).has_value()) {
      // avoid bind same argument multiple times
      continue;
    }

    SmallVector<Value> symbolValues;
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      if (!shapedType.isDynamicDim(dim)) {
        continue;
      }
      auto symbolAttr = SymbolManager::getUniqueSymbolAttr(ctx);
      Value symbol = builder.create<symbol::SymbolicIntOp>(loc, symbolAttr);
      symbolValues.push_back(symbol);
    }

    auto shapeExpr = createAffineFromShape(shapedType.getShape(), ctx);
    builder.create<symbol::BindSymbolicShapeOp>(loc, ba, symbolValues,
                                                shapeExpr);
  }
}

void PropagateSymbolPass::runOnOperation() {
  func::FuncOp func = getOperation();
  initSymbolForFuncArgs(func);
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<SymbolizeDynamicShapeTensor<tensor::EmptyOp>>(ctx);
  patterns.add<SymbolizeDynamicShapeTensor<tensor::ExpandShapeOp>>(ctx);
  patterns.add<SymbolizeDynamicShapeTensor<tensor::ExtractSliceOp>>(ctx);

  patterns.add<BindReifyResultShape>(ctx);
  patterns.add<PropagateSymbolByTensorDim>(ctx);
  patterns.add<FoldSymbolToTensorEmpty>(ctx);

  patterns.add<UnifySameOperandsAndResultShape<linalg::ElemwiseBinaryOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<hfusion::ElemwiseBinaryOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<linalg::ElemwiseUnaryOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<hfusion::ElemwiseUnaryOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<hfusion::CastOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<hfusion::LoadOp>>(ctx);
  patterns.add<UnifySameOperandsAndResultShape<hfusion::StoreOp>>(ctx);

  patterns.add<UnifyPartialEquivalentDims<linalg::BroadcastOp>>(ctx);
  patterns.add<UnifyPartialEquivalentDims<linalg::ReduceOp>>(ctx);
  patterns.add<UnifyPartialEquivalentDims<tensor::ConcatOp>>(ctx);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> createPropagateSymbolPass() {
  return std::make_unique<PropagateSymbolPass>();
}

} // namespace symbol
} // namespace mlir