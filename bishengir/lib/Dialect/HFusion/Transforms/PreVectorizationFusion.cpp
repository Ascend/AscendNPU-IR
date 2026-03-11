//===- PreVectorizationFusion.cpp --- Fuse and generalize elemwise ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PREVECTORIZATIONFUSION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-pre-vectorization-fusion"

using namespace mlir;
using namespace mlir::hfusion;

static thread_local bool archisAscend950{false};

static void generalizeBroadcastOp(PatternRewriter &rewriter,
                                  linalg::BroadcastOp op) {
  auto broadcastDims = op.getDimensions();
  auto resultType = cast<RankedTensorType>(op.getInit().getType());
  SmallVector<AffineExpr> inputExprs;
  SmallVector<AffineExpr> resultExprs;
  auto ctx = rewriter.getContext();
  for (int64_t i = 0; i < resultType.getRank(); i++) {
    resultExprs.push_back(mlir::getAffineDimExpr(i, ctx));
    if (llvm::find(broadcastDims, i) != broadcastDims.end())
      inputExprs.push_back(mlir::getAffineConstantExpr(0, ctx));
    else
      inputExprs.push_back(resultExprs.back());
  }

  Value src = hfusion_conversion_utils::createExpandShapeOp(
      op, rewriter, op.getInput(), resultType);

  SmallVector<AffineMap> indexingMaps{
      AffineMap::get(resultType.getRank(), 0, inputExprs, ctx),
      AffineMap::get(resultType.getRank(), 0, resultExprs, ctx)};

  auto nestedBody = [](OpBuilder &nestedBuilder, Location nestedLoc,
                       ValueRange blockArgs) {
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
  };

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      op, op->getResultTypes(), ValueRange{src}, ValueRange{op.getInit()},
      indexingMaps,
      SmallVector<utils::IteratorType>(resultType.getRank(),
                                       utils::IteratorType::parallel),
      nestedBody);
}

FailureOr<linalg::GenericOp> generalizeGatherOp(RewriterBase &rewriter,
                                                hfusion::GatherOp op) {
  if (!op.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(op, "not pure tensor semantics");

  Value src = op.getSrc();
  Value index = op.getIndex();
  Value init = op.getInit();

  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto idxTy = dyn_cast<RankedTensorType>(index.getType());
  auto outTy = dyn_cast<RankedTensorType>(init.getType());
  if (!srcTy || !idxTy || !outTy)
    return rewriter.notifyMatchFailure(op, "expects ranked tensors");

  if (srcTy.getRank() != 1 || idxTy.getRank() != 1 || outTy.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "only supports 1D src/index/out");

  if (static_cast<int64_t>(op.getAxis()) != 0)
    return rewriter.notifyMatchFailure(op,
                                       "only supports axis=0 for 1D gather");

  if (!idxTy.isDynamicDim(0) && !outTy.isDynamicDim(0) &&
      idxTy.getDimSize(0) != outTy.getDimSize(0))
    return rewriter.notifyMatchFailure(op, "out dim must match index dim");

  MLIRContext *ctx = rewriter.getContext();
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(/*rank=*/1, ctx);
  SmallVector<AffineMap> indexingMaps{idMap, idMap};
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel};

  linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      /*resultTensorTypes=*/TypeRange{outTy},
      /*inputs=*/ValueRange{index},
      /*outputs=*/ValueRange{init}, indexingMaps, iterators,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // args[0] = indexElem (i32), args[1] = outElem (i32)
        Value idxElem = args[0];

        Value idx =
            b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), idxElem);

        // 1D table: src[idx]
        Value val = b.create<tensor::ExtractOp>(loc, src, ValueRange{idx});

        b.create<linalg::YieldOp>(loc, val);
      });

  rewriter.replaceOp(op, genericOp->getResults());
  return genericOp;
}

namespace {
struct PreVectorizationFusionPass
    : public impl::PreVectorizationFusionBase<PreVectorizationFusionPass> {
public:
  PreVectorizationFusionPass(const mlir::PreVectorizationFusionOptions &options)
      : impl::PreVectorizationFusionBase<PreVectorizationFusionPass>(options) {}

  void runOnOperation() override;
};

struct HFusionGeneralizationPatterns
    : public OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
  using OpInterfaceRewritePattern<
      mlir::linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Skip AIC temporary. Move VF to HIVM
    if (isInCubeScope(op)) {
      return failure();
    }
    // Load/Store ops will be converted to DMAs
    // Cast ops stays as it is to prevent fusion
    if (isa<hfusion::LoadOp, hfusion::StoreOp>(op) ||
        op->hasAttr(utils::simtLabel) ||
        // TODO: Handle single element fillop
        hfusion::isMatmulOps(op) || isa<hfusion::ReduceWithIndexOp>(op) ||
        hfusion::opCanFuseIntoMatmul(op) || isa<linalg::TransposeOp>(op))
      return failure();
    if (hfusion::isSingleElementLinalgOp(op) && isa<linalg::FillOp>(op)){
      Type elemType = op.getDpsInputs()[0].getType();
      // For the FP8 data type, we need to preserve the path from linalg.fill to linalg.generic, so that in AutoVectorizeV2 we can lower
      // arith.constant 0.000000e+00 : f8E4M3FN
      // into
      // arith.constant dense<0.000000e+00> : vector<256xf8E4M3FN>.
      // This allows us to use bitcast to avoid generating FP8 constants, which are not accepted in the LLVM IR received in CCEC.
      if(!elemType.isFloat8E4M3FN() && !elemType.isFloat8E5M2())
        return failure();
    }
    // Handle broadcastOp to be expand + linalg.generic when
    // the input is from CollapseShapeOp
    if (auto brcOp = dyn_cast<linalg::BroadcastOp>(op.getOperation());
        brcOp && brcOp.getInput().getDefiningOp<tensor::CollapseShapeOp>()) {
      generalizeBroadcastOp(rewriter, brcOp);
      return success();
    }
    if (auto g = dyn_cast<hfusion::GatherOp>(op.getOperation())) {
      if (failed(generalizeGatherOp(rewriter, g)))
        return failure();
      return success();
    }
    return generalizeNamedOp(rewriter, op);
  }
};

bool isYieldGeneric(linalg::GenericOp &operandOp, OpOperand *operand) {
  // The body of linalg has yield only
  auto &block = operandOp.getRegion().front();
  if (!llvm::hasSingleElement(block))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield)
    return false;
  // Support parallel type only.
  for (auto it : operandOp.getIteratorTypesArray()) {
    if (!linalg::isParallelIterator(it))
      return false;
  }
  return true;
}

bool ElemwiseOpFuseControlFn(OpOperand *operand) {
  // Scenerio 1: Add folding with reshape by expansion patterns.
  auto producerOp = operand->get().getDefiningOp();
  if (!producerOp)
    return false;
  if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(producerOp)) {
    if (hfusion::isMatmulOps(linalgOp))
      return false;
  }
  auto producerUsers =
      llvm::make_filter_range(producerOp->getUsers(), [&](Operation *user) {
        return !isa<annotation::MarkOp>(user);
      });
  if (!llvm::hasSingleElement(producerUsers)) {
    // multi consumers && constant support fusion
    auto B = std::begin(producerUsers), E = std::end(producerUsers);
    if (B != E && isFillOp(producerOp)) {
      return true;
    }
    return false;
  }
  // Scenerio 2: Avoid Combining the genericOp B with its operand
  // definition A which is genericOp either, if the A is the ancestor of B.
  // Because sinking innerloop invariant op A into for loop may cause extra
  // computation.
  // 1. Only process combination between genericOp
  auto producerGen = dyn_cast_or_null<linalg::GenericOp>(producerOp);
  if (producerGen == nullptr)
    return true;
  auto consumerOp = operand->getOwner();
  auto consumerGen = llvm::dyn_cast_or_null<linalg::GenericOp>(consumerOp);
  if (consumerGen == nullptr)
    return true;
  // 2. current vstu/vldu are not support 1bit type, skip combination
  // may bring error instructions
  // TODO: this restriction may remove in future.
  auto type = dyn_cast_or_null<RankedTensorType>(operand->get().getType());
  if (type && type.getElementType().isInteger(1)) {
    return true;
  }
  // 3. Keep conbination with some simple cases may help performance.
  if (isYieldGeneric(producerGen, operand)) {
    return true;
  }
  // 4. consumer cannot be producer's ancestor, then if the following
  // is correct, meaning producer & consumer are not under the same for scope.
  auto consumerFor = consumerOp->getParentOfType<scf::ForOp>();
  if (!consumerFor)
    return true;
  return consumerFor->isAncestor(producerOp);
}

static void populateFusionPatterns(RewritePatternSet &patterns) {
  // Add elementwise op fusion patterns.
  linalg::populateElementwiseOpsFusionPatterns(patterns,
                                               ElemwiseOpFuseControlFn);
}

template <typename OpTy>
inline constexpr bool IsMatmulVariantV =
    std::is_same_v<OpTy, linalg::MatmulOp> ||
    std::is_same_v<OpTy, linalg::MatmulTransposeAOp> ||
    std::is_same_v<OpTy, linalg::MatmulTransposeBOp> ||
    std::is_same_v<OpTy, linalg::BatchMatmulOp>;

template <typename MatmulOpTy,
          typename = std::enable_if_t<IsMatmulVariantV<MatmulOpTy>>>
struct HFusionMatmulDecomposePatterns : public OpRewritePattern<MatmulOpTy> {
  using OpRewritePattern<MatmulOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatmulOpTy op,
                                PatternRewriter &rewriter) const override {
    // TODO: Skip AIC temporary. Move VF to HIVM
    if (isInCubeScope(op)) {
      return failure();
    }

    Value mmadOutput = op.getDpsInitOperand(0)->get();
    auto blockArg = dyn_cast_if_present<BlockArgument>(mmadOutput);
    if (blockArg) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parentOp)) {
        return failure();
      }
      auto scfForOp =
          dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (scfForOp) {
        OpOperand *iterArgOperand = scfForOp.getTiedLoopInit(blockArg);
        if (!iterArgOperand) {
          return failure();
        }
        Value initVal = iterArgOperand->get();
        if (hfusion::isZeroOrEmptyTensor(initVal)) {
          return failure();
        }
      }
    }
    if (Operation *emptyOp = mmadOutput.getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }
    if (Operation *fillOp = mmadOutput.getDefiningOp<linalg::FillOp>()) {
      return failure();
    }
    auto elementType = getElementTypeOrSelf(op->getResult(0));
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    auto zero = rewriter.create<arith::ConstantOp>(op->getLoc(), zeroAttr);
    auto newMmadInit =
        mlir::utils::createEmptyOp(rewriter, op->getLoc(), op->getResult(0));
    auto fillInit =
        rewriter
            .create<linalg::FillOp>(op->getLoc(), TypeRange(newMmadInit),
                                    ValueRange{zero}, ValueRange(newMmadInit))
            ->getResult(0);
    auto newMmad = rewriter
                       .create<MatmulOpTy>(op->getLoc(), op->getResultTypes(),
                                           op.getDpsInputs(), fillInit)
                       ->getResult(0);
    auto newAddEmpty =
        mlir::utils::createEmptyOp(rewriter, op->getLoc(), op->getResult(0));
    auto newAddOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::add,
            ValueRange{newMmad, op.getDpsInitOperand(0)->get()},
            ValueRange(newAddEmpty))
            ->getResult(0);
    rewriter.replaceOp(op, newAddOp);
    return success();
  }
};

struct FlattenIndirectLoadMask
    : public OpRewritePattern<hfusion::IndirectLoadOp> {
public:
  using OpRewritePattern<hfusion::IndirectLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::IndirectLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value mask = op.getMask();
    auto maskType = cast<RankedTensorType>(mask.getType());
    if (!maskType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "maskType must be static shape");

    auto fillOp = dyn_cast<linalg::FillOp>(mask.getDefiningOp());
    if (!fillOp) {
      return rewriter.notifyMatchFailure(op,
                                         "mask is not defined by linalg.fill");
    }
    auto shape = maskType.getShape();
    if (shape.size() <= 1)
      return rewriter.notifyMatchFailure(op, "mask shape <= 1");
    Type elementType = maskType.getElementType();
    // if (!isI8ElemType(maskType)) {
    //   return rewriter.notifyMatchFailure(op, "mask is not i8");
    // }

    rewriter.setInsertionPoint(fillOp);
    int64_t totalElements = 1;
    for (int64_t dim : shape) {
      totalElements *= dim;
    }
    auto newEmptyType = RankedTensorType::get({totalElements}, elementType);
    Value newEmpty = rewriter.create<tensor::EmptyOp>(
        loc, newEmptyType.getShape(), elementType);

    Value fillValue = fillOp.getInputs()[0];
    auto newFill = rewriter.create<linalg::FillOp>(loc, ValueRange{fillValue},
                                                   ValueRange{newEmpty});

    auto maybeReassociation =
        getReassociationIndicesForReshape(newEmptyType, maskType);
    if (!maybeReassociation.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "failed to create reassociation for tensor.expand_shape");
    }
    auto expandedFillOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, maskType, newFill.getResult(0), maybeReassociation.value());

    rewriter.replaceOp(fillOp, expandedFillOp);

    return success();
  }
};

struct ExtractInlinePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newMaps;
    bool changed = false;

    unsigned numLoops = op.getNumLoops();
    auto oldMaps = op.getIndexingMapsArray();
    auto ctx = rewriter.getContext();

    for (auto it : llvm::enumerate(op.getDpsInputs())) {
      Value input = it.value();
      AffineMap oldMap = oldMaps[it.index()];

      // input from tensor.extract
      auto extractOp = input.getDefiningOp<tensor::ExtractOp>();
      if (!extractOp) {
        newInputs.push_back(input);
        newMaps.push_back(oldMap);
        continue;
      }

      Value srcTensor = extractOp.getTensor();
      auto srcType = dyn_cast<RankedTensorType>(srcTensor.getType());
      if (!srcType) {
        newInputs.push_back(input);
        newMaps.push_back(oldMap);
        continue;
      }

      int64_t rank = srcType.getRank();

      // A: Rank-0 Tensor (tensor<f32>)
      if (rank == 0) {
        newInputs.push_back(srcTensor);
        // Map: (d0, d1, ...) -> ()
        newMaps.push_back(AffineMap::get(numLoops, 0, ctx));
        changed = true;
      }
      // B: Rank-1 Tensor & length = 1 (tensor<1xf32>)
      else if (rank == 1 && srcType.getDimSize(0) == 1) {
        newInputs.push_back(srcTensor);
        auto zeroExpr = rewriter.getAffineConstantExpr(0);
        newMaps.push_back(AffineMap::get(numLoops, 0, {zeroExpr}, ctx));
        changed = true;
      } else {
        newInputs.push_back(input);
        newMaps.push_back(oldMap);
      }
    }

    if (!changed)
      return failure();

    for (auto it : llvm::enumerate(op.getDpsInits())) {
      newMaps.push_back(oldMaps[op.getDpsInputs().size() + it.index()]);
    }

    auto newOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), op.getResultTypes(), newInputs, op.getDpsInits(), newMaps,
        op.getIteratorTypesArray());

    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

static void populateMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<HFusionMatmulDecomposePatterns<linalg::MatmulOp>>(
      patterns.getContext());
  patterns.add<HFusionMatmulDecomposePatterns<linalg::MatmulTransposeAOp>>(
      patterns.getContext());
  patterns.add<HFusionMatmulDecomposePatterns<linalg::MatmulTransposeBOp>>(
      patterns.getContext());
  patterns.add<HFusionMatmulDecomposePatterns<linalg::BatchMatmulOp>>(
      patterns.getContext());
}

/// Before conversion:
/// ```mlir
//    %70 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<()
//    -> ()>], iterator_types = []} ins(%c0_i32 : i32) outs(%69 : tensor<i32>)
/// ```
/// After conversion:
/// ```mlir
///   "%70 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>],
///   iterator_types = ["parallel"]} outs(%69 : tensor<1xi32>)
/// ```
struct ZeroDimToOneDimGenericPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!(hfusion::isZeroElementLinalgOp(op) && hfusion::isFillOp(op))) {
      return failure();
    }
    if (op.getNumLoops() != 0) {
      return failure();
    }
    if (op.getNumDpsInits() != 1) {
      return failure();
    }

    auto output = op.getDpsInitOperand(0)->get();
    auto outputType = output.getType().cast<RankedTensorType>();
    if (outputType.getRank() != 0) {
      return failure();
    }

    auto newOutputType =
        RankedTensorType::get({1}, outputType.getElementType());

    Value emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
        op->getLoc(), newOutputType, mlir::ValueRange{});
    Value newOutput =
        mlir::utils::createEmptyOp(rewriter, op->getLoc(), emptyTensor);
    auto ctx = rewriter.getContext();
    auto inputMap = AffineMap::get(1, 0, {}, ctx);
    auto outputMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, ctx);
    SmallVector<AffineMap> newMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> newIteratorTypes = {
        utils::IteratorType::parallel};
    auto newOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), newOutputType, op.getDpsInputs(), ValueRange{newOutput},
        newMaps, newIteratorTypes);
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());
    mlir::Value zeroIndex =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    mlir::Value extractedResult = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), newOp.getResult(0), ValueRange{zeroIndex});

    mlir::Value finalResult = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(), outputType, ValueRange{extractedResult});
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

static void
populatePreVectorizationFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<HFusionGeneralizationPatterns>(patterns.getContext());
  patterns.add<ExtractInlinePattern>(patterns.getContext());
  patterns.add<ZeroDimToOneDimGenericPattern>(patterns.getContext());
  populateMatmulPatterns(patterns);
  populateFusionPatterns(patterns);
  annotation::MarkOp::getCanonicalizationPatterns(patterns,
                                                  patterns.getContext());
  patterns.getContext()
      ->getLoadedDialect<linalg::LinalgDialect>()
      ->getCanonicalizationPatterns(patterns);
}

bool isCopyFromGM(memref::CopyOp copyOp) {
  Value src = copyOp.getSource();
  return util::isFromFunctionArg(src);
}

void InsertPadConstMark(Operation *moduleOp) {
  // add pad value mark
  moduleOp->walk([&](Operation *op) {
    // if it is one load copy
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (!isCopyFromGM(copyOp))
        return WalkResult::skip();
      auto dst = copyOp.getTarget();
      auto maybeAlloc = utils::tracebackMemRefToAlloc(dst);
      for (auto *user : maybeAlloc.value()->getUsers()) {
        if (auto fillOp = llvm::dyn_cast<linalg::FillOp>(user)) {
          // check if op for load-padding-value
          Value inputVal = fillOp.getDpsInputs()[0];
          if (inputVal.getType().isIntOrFloat()) {
            Value val_alloc = fillOp.getDpsInits()[0];
            OpBuilder b(op);
            ArrayAttr keysVec = b.getStrArrayAttr({utils::padConst});
            SmallVector<Value> valuesVec = {inputVal};
            b.setInsertionPoint(fillOp);
            b.create<annotation::MarkOp>(fillOp->getLoc(), val_alloc, valuesVec,
                                         keysVec);
          }
          break;
        }
      }
    }
    return WalkResult::advance();
  });
}

// Replace linalg.fill with empty tensor when its user is linalg.reduce which
// statisfies shouldUseTileReductionUsingForV2 pattern.
// before:
// %2 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> 
//      tensor<64xf32> 
// %reduced = linalg.reduce ins(%1 : tensor<64x128xf32>) 
//      outs(%2 : tensor<64xf32>) dimensions = [1]
// after:
// %2 = tensor.empty() : tensor<64xf32>
// %reduced = linalg.reduce ins(%1 : tensor<64x128xf32>) 
//      outs(%2 : tensor<64xf32>) dimensions = [1]
void EmptifyReduceInit(Operation *op, IRRewriter &rewriter) {
  op->walk([&](linalg::ReduceOp reduceOp) {
    if (!(hfusion::shouldUseTileReductionUsingForV2(rewriter, reduceOp)))
      return;
    Value initValue = reduceOp.getInits()[0];
    auto initOp = initValue.getDefiningOp();
    if (!initOp || !mlir::hfusion::isFillOp(initOp))
      return;
    RankedTensorType outType = initValue.getType().cast<RankedTensorType>();
    rewriter.setInsertionPoint(reduceOp);
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        reduceOp.getLoc(), outType.getShape(), outType.getElementType());
    rewriter.modifyOpInPlace(reduceOp, [&]() {
      reduceOp.getInitsMutable()[0].assign(emptyOp.getResult());
    });
    if (initOp->use_empty())
      rewriter.eraseOp(initOp);
  });
}

void PreVectorizationFusionPass::runOnOperation() {
  Operation *op = getOperation();
  archisAscend950 =
      hacc::utils::isAscend950(getOperation()->getParentOfType<ModuleOp>());
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  OpBuilder builder(context);

  InsertPadConstMark(op);

  // Clone linalg.fill for every linalg.matmul to avoid cases like:
  //
  // %ret = linalg.fll
  // linalg.matmul ins(...) outs(%ret)
  // ...
  // some_vector_op(%rest)
  // TODO: properly fix this
  op->walk([&builder](linalg::MatmulOp matmul) {
    Operation *mmInit = matmul.getDpsInitOperand(0)->get().getDefiningOp();
    if (llvm::isa_and_nonnull<linalg::FillOp>(mmInit)) {
      IRMapping mapping;
      builder.setInsertionPoint(mmInit);
      auto newOp = builder.clone(*mmInit, mapping);
      matmul.getDpsInitOperand(0)->assign(newOp->getResult(0));
    }
  });

  {
    RewritePatternSet preprocessPatterns(context);
    preprocessPatterns.add<FlattenIndirectLoadMask>(context);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(preprocessPatterns)))) {
      signalPassFailure();
    }
  }

  if (archisAscend950 && this->enableTritonCompile) {
    IRRewriter rewriter(&getContext());
    EmptifyReduceInit(op, rewriter);
  }

  populatePreVectorizationFusionPatterns(patterns);
  // Use TopDownTraversal for compile time reasons
  GreedyRewriteConfig grc;
  grc.useTopDownTraversal = true;
  (void)applyPatternsGreedily(op, std::move(patterns), grc);
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createPreVectorizationFusionPass(
    const PreVectorizationFusionOptions &options) {
  return std::make_unique<PreVectorizationFusionPass>(options);
}
