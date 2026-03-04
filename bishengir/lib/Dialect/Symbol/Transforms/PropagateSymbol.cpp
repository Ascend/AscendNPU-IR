//===- PropagateSymbol.cpp --------- Propagate Symbol Pass ----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to propagate symbols
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Symbol/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
      // TODO: support reify op of arith index type
      llvm_unreachable("unsupported reify op type");
    }

    if (bindValues.empty()) {
      return std::nullopt;
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
  if (index >= ssize_t(shapes.size()) ||
      !ShapedType::isDynamic(shapes[index])) {
    return std::nullopt;
  }

  int64_t dynIndex = 0;
  for (int64_t i = 0; i < ssize_t(shapes.size()) && i < index; ++i) {
    if (ShapedType::isDynamic(shapes[i])) {
      dynIndex++;
    }
  }
  return dynIndex;
}

// propagate symbols by replacing tensor.dim with the symbol it binds to
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
    auto dynIndex = getIndexForDynamicDim(shapedType.getShape(), index);
    if (!dynIndex.has_value()) {
      return failure();
    }
    rewriter.replaceOp(op, shapeSymbols[dynIndex.value()]);
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

  patterns.add<BindReifyResultShape>(ctx);
  patterns.add<PropagateSymbolByTensorDim>(ctx);

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