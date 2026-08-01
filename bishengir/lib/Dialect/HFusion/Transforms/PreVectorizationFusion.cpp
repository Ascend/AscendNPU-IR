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
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Analysis/VFFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_PREVECTORIZATIONFUSION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-pre-vectorization-fusion"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::analysis;

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
    if (scope::utils::isInCubeScope(op)) {
      return failure();
    }
    // Load/Store ops will be converted to DMAs
    // Cast ops stays as it is to prevent fusion
    if (isa<hfusion::LoadOp, hfusion::StoreOp>(op) ||
        hivm::util::isSIMTVF(op) ||
        // TODO: Handle single element fillop
        hfusion::isMatmulOps(op) || isa<hfusion::ReduceWithIndexOp>(op) ||
        hfusion::opCanFuseIntoMatmul(op) || isa<linalg::TransposeOp>(op))
      return failure();
    if (hfusion::isSingleElementLinalgOp(op) && isa<linalg::FillOp>(op)) {
      Type elemType = op.getDpsInputs()[0].getType();
      // For the FP8 data type, we need to preserve the path from linalg.fill to
      // linalg.generic, so that in AutoVectorizeV2 we can lower arith.constant
      // 0.000000e+00 : f8E4M3FN into arith.constant dense<0.000000e+00> :
      // vector<256xf8E4M3FN>. This allows us to use bitcast to avoid generating
      // FP8 constants, which are not accepted in the LLVM IR received in CCEC.
      // I1 processed the same way, it prevents creation of
      // unsupported memref.get_global.
      if (!elemType.isFloat8E4M3FN() && !elemType.isFloat8E5M2() &&
          !elemType.isInteger(1))
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

// The GeneralizeMulextPattern is a transformation pattern that
// converts Hfusion::MulExtOp (a custom multiply-extended operation)
// into a linalg.generic operation that performs element-wise
// multiplication with extended precision using VMULL.
struct GeneralizeMulextPattern : public OpRewritePattern<hfusion::MulExtOp> {
  using OpRewritePattern<hfusion::MulExtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::MulExtOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultTypes = op->getResultTypes();
    Type lowType = resultTypes[0];
    Type highType = resultTypes[1];
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    int64_t rank = lhsType.getRank();

    // Create index mapping
    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes;
    auto ctx = rewriter.getContext();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

    for (int64_t i = 0; i < rank; ++i) {
      iteratorTypes.push_back(utils::IteratorType::parallel);
    }

    for (int64_t i = 0; i < 4; ++i) {
      indexingMaps.push_back(identityMap);
    }

    // Create empty output tensor
    auto shape = lhsType.getShape();
    Value lowEmpty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), shape, getElementTypeOrSelf(lowType));
    Value highEmpty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), shape, getElementTypeOrSelf(highType));

    auto lhsElemType = getElementTypeOrSelf(lhsType);
    bool isSigned = lhsElemType.isSignlessInteger() || lhsElemType.isSignedInteger();

    // create linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      TypeRange{lowType, highType},
      ValueRange{lhs, rhs},
      ValueRange{lowEmpty, highEmpty},
      indexingMaps,
      iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value lhsElem = args[0];
        Value rhsElem = args[1];
        Value low, high;
        if (isSigned) {
          auto mulExt = b.create<arith::MulSIExtendedOp>(
              loc, b.getI32Type(), b.getI32Type(), lhsElem, rhsElem);
          low = mulExt.getLow();
          high = mulExt.getHigh();
        } else {
          auto mulExt = b.create<arith::MulUIExtendedOp>(
              loc, b.getI32Type(), b.getI32Type(), lhsElem, rhsElem);
          low = mulExt.getLow();
          high = mulExt.getHigh();              
        }
        // use arith.mulsi_extended
        b.create<linalg::YieldOp>(loc, ValueRange{low, high});
      });
    rewriter.replaceOp(op, genericOp->getResults());
    return success();
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

static unsigned getElementwiseFusionSize(linalg::GenericOp genericOp) {
  unsigned size = 0;
  for (Operation &op : genericOp.getBody()->without_terminator()) {
    // Count scalar compute ops in the region as a proxy for fused op count.
    if (!isa<linalg::YieldOp>(op))
      ++size;
  }
  return std::max(1u, size);
}

// issue 1174
static bool flowsToCubeCopy(mlir::Operation *startOp, int maxDepth = 6) {
  llvm::SmallVector<std::pair<mlir::Operation *, int>, 8> worklist;
  worklist.push_back({startOp, 0});
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;

  while (!worklist.empty()) {
    auto [curr, depth] = worklist.pop_back_val();
    if (depth >= maxDepth || !visited.insert(curr).second) {
      continue;
    }

    for (auto user : curr->getUsers()) {
      if (llvm::isa<hivm::CopyOp>(user)) {
        return true;
      }

      if (llvm::isa<linalg::LinalgOp, tensor::ExpandShapeOp,
                    tensor::CollapseShapeOp, linalg::TransposeOp>(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }
  return false;
}

bool ElemwiseOpFuseControlFn(OpOperand *operand, int maxFusedElementwiseOps,
                             const VFStackInfoBuilder &stackInfoBuilder) {
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
  auto consumerOp = operand->getOwner();
  // Ensure no vector stack overflow if fused.
  if (!stackInfoBuilder.fitsStack({producerOp, consumerOp}))
    return false;
  auto producerGen = dyn_cast_or_null<linalg::GenericOp>(producerOp);
  if (producerGen == nullptr)
    return true;
  auto consumerGen = llvm::dyn_cast_or_null<linalg::GenericOp>(consumerOp);
  if (consumerGen == nullptr)
    return true;
  if (maxFusedElementwiseOps > 0) {
    // TODO: maybe we not needed after we have the VF stack limit check.
    unsigned fusedSize = getElementwiseFusionSize(producerGen) +
                         getElementwiseFusionSize(consumerGen);
    if (fusedSize > static_cast<unsigned>(maxFusedElementwiseOps))
      return false;
  }
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
  // 5. prevent fusion across synchronization boundaries.
  if (producerOp->getBlock() == consumerOp->getBlock()) {
    // Determine the scanning range: instructions located between producer and
    // consumer.
    Operation *start = producerOp;
    Operation *end = consumerOp;

    // Ensure the scanning order is correct for the iterator.
    if (end->isBeforeInBlock(start))
      std::swap(start, end);

    // Traverse all instructions between start and end ops.
    bool hasSyncBarrier = false;
    for (auto it = start->getNextNode(); it != end && it != nullptr; it = it->getNextNode()) {
      // If a synchronization barrier is found, prohibit fusion to avoid data
      // races.
      if (isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
              hivm::CreateSyncBlockLockOp, hivm::SyncBlockLockOp,
              hivm::SyncBlockUnlockOp>(it)) {
        hasSyncBarrier = true;
        break;
      }
    }

    if (hasSyncBarrier) {
      if (!flowsToCubeCopy(consumerOp)) {
        return false;
      }
    }
  }

  return consumerFor->isAncestor(producerOp);
}

static void populateFusionPatterns(RewritePatternSet &patterns,
                                   int maxFusedElementwiseOps,
                                   bool enableVFStackLimit) {
  VFStackInfoBuilder stackInfoBuilder(enableVFStackLimit);
  linalg::ControlFusionFn controlFn = [maxFusedElementwiseOps,
                                       stackInfoBuilder](OpOperand *operand) {
    return ElemwiseOpFuseControlFn(operand, maxFusedElementwiseOps,
                                   stackInfoBuilder);
  };
  // Add elementwise op fusion patterns.
  linalg::populateElementwiseOpsFusionPatterns(patterns, controlFn);
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
    if (scope::utils::isInCubeScope(op)) {
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
    if (mmadOutput.getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }
    if (mmadOutput.getDefiningOp<linalg::FillOp>()) {
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
    Location addLoc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(newMmad.getType());
    int64_t rank = tensorType.getRank();
    bool isFloat = isa<FloatType>(getElementTypeOrSelf(tensorType));
    SmallVector<AffineMap> addMaps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> addIterators(rank,
                                                  utils::IteratorType::parallel);
    auto newAddGeneric = rewriter.create<linalg::GenericOp>(
        addLoc, newAddEmpty.getType(),
        ValueRange{newMmad, op.getDpsInitOperand(0)->get()},
        ValueRange{newAddEmpty}, addMaps, addIterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value addResult;
          if (isFloat)
            addResult = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          else
            addResult = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          b.create<linalg::YieldOp>(nestedLoc, addResult);
        });
    rewriter.replaceOp(op, newAddGeneric->getResult(0));
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
    for (auto userIt : llvm::enumerate(op.getOperands())) {
      Value userIn = userIt.value();
      if (isa<MemRefType>(userIn.getType())) {
        return failure();
      }
    }
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

struct ExpandShapeToImplicitBrcInGenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    unsigned numLoops = op.getNumLoops();
    auto oldMaps = op.getIndexingMapsArray();
    auto ctx = rewriter.getContext();
    SmallVector<Value> newInputs = llvm::to_vector(op.getDpsInputs());
    SmallVector<AffineMap> newMaps(oldMaps.begin(),
                                   oldMaps.begin() + op.getNumDpsInputs());

    for (auto it : llvm::enumerate(op.getDpsInputs())) {
      Value input = it.value();
      AffineMap oldMap = oldMaps[it.index()];
      // input from tensor.expand_shape

      auto expandShapeOp = input.getDefiningOp<tensor::ExpandShapeOp>();
      if (!expandShapeOp || isa_and_nonnull<tensor::CollapseShapeOp>(
                                expandShapeOp.getSrc().getDefiningOp()))
        continue;
      // To replace input with expand_shape src, the following conditions must
      // be met:
      //  - The expand_shape only inserts unit dims (an inserted unit dim that
      //  is indexed by a constant-0). Each reassociation group of result dims
      //  maps to exactly one source dim, and within a group at most one dim
      //  carries the source extent while the rest are unit dims.
      auto resShape = cast<RankedTensorType>(input.getType()).getShape();
      auto reassoc = expandShapeOp.getReassociationIndices();
      bool hasInlineBrcAxes = false;
      for (auto [i, result] : llvm::enumerate(oldMap.getResults())) {
        auto constExpr = dyn_cast<AffineConstantExpr>(result);
        // 0 needs to map shape 1
        if (constExpr && constExpr.getValue() == 0 && resShape[i] == 1) {
          hasInlineBrcAxes = true;
          break;
        }
      }
      if (!hasInlineBrcAxes)
        continue;

      // Collapse each reassociation group into a single map result.
      bool isPureUnitExpand = true;
      SmallVector<AffineExpr> newMapResults;
      newMapResults.reserve(reassoc.size());
      for (const auto &group : reassoc) {
        AffineExpr extentExpr = getAffineConstantExpr(0, ctx);
        unsigned numExtentDims = 0;
        for (int64_t d : group) {
          if (resShape[d] != 1) {
            extentExpr = oldMap.getResult(d);
            ++numExtentDims;
          }
        }
        // A group that splits one source dim into several non-unit dims is a
        // genuine reshape, not a pure unit expansion; leave it for the standard
        // reshape-fusion patterns and bail out for this operand. For example,
        // 6xi32 -> 2x3xi32.
        if (numExtentDims > 1) {
          isPureUnitExpand = false;
          break;
        }
        newMapResults.push_back(extentExpr);
      }
      if (!isPureUnitExpand)
        continue;

      newInputs[it.index()] = expandShapeOp.getSrc();
      newMaps[it.index()] = AffineMap::get(numLoops, 0, newMapResults, ctx);
      changed = true;
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
struct ZeroDimToOneDimGenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    bool isFill =
        hfusion::isZeroElementLinalgOp(op) && hfusion::isFillOp(op);
    if (!isFill && op.getNumLoops() != 0) {
      return failure();
    }
    if (op.getNumDpsInits() != 1) {
      return failure();
    }

    auto output = op.getDpsInitOperand(0)->get();
    auto outputType = mlir::cast<RankedTensorType>(output.getType());
    if (outputType.getRank() != 0) {
      return failure();
    }

    // For non-fill rank-0 generics, bail if any tensor input is rank > 0
    // (e.g. constant-indexed). Check before creating any ops to avoid leaving
    // orphaned ops in the IR (applyPatternsGreedily does not roll back).
    if (!isFill) {
      for (Value input : op.getDpsInputs()) {
        auto tensorType = dyn_cast<RankedTensorType>(input.getType());
        if (tensorType && tensorType.getRank() > 0)
          return failure();
      }
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
    SmallVector<utils::IteratorType> newIteratorTypes = {
        utils::IteratorType::parallel};

    linalg::GenericOp newOp;
    if (isFill) {
      // Original path: scalar inputs use broadcast map (d0)->().
      SmallVector<AffineMap> newMaps = {inputMap, outputMap};
      newOp = rewriter.create<linalg::GenericOp>(
          op.getLoc(), newOutputType, op.getDpsInputs(), ValueRange{newOutput},
          newMaps, newIteratorTypes);
    } else {
      // Rank-0 tensor inputs: expand to rank-1 with identity map (d0)->(d0).
      // Passing a rank-0 tensor with broadcast map (d0)->() crashes
      // FoldScalarOrSplatConstant in the same greedy-apply cycle.
      SmallVector<Value> newInputs;
      SmallVector<AffineMap> newMaps;
      for (Value input : op.getDpsInputs()) {
        auto tensorType = dyn_cast<RankedTensorType>(input.getType());
        if (tensorType && tensorType.getRank() == 0) {
          auto newType =
              RankedTensorType::get({1}, tensorType.getElementType());
          Value expanded = rewriter.create<tensor::ExpandShapeOp>(
              op.getLoc(), newType, input,
              SmallVector<ReassociationIndices>{});
          newInputs.push_back(expanded);
          newMaps.push_back(outputMap);
        } else {
          newInputs.push_back(input);
          newMaps.push_back(inputMap);
        }
      }
      newMaps.push_back(outputMap);
      newOp = rewriter.create<linalg::GenericOp>(
          op.getLoc(), newOutputType, newInputs, ValueRange{newOutput},
          newMaps, newIteratorTypes);
    }
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

static void populatePreVectorizationFusionPatterns(RewritePatternSet &patterns,
                                                   int maxFusedElementwiseOps,
                                                   bool enableVFStackLimit) {
  patterns.add<GeneralizeMulextPattern>(patterns.getContext());
  patterns.add<HFusionGeneralizationPatterns>(patterns.getContext());
  patterns.add<ExtractInlinePattern>(patterns.getContext());
  patterns.add<ExpandShapeToImplicitBrcInGenericPattern>(patterns.getContext());
  patterns.add<ZeroDimToOneDimGenericPattern>(patterns.getContext());
  populateMatmulPatterns(patterns);
  populateFusionPatterns(patterns, maxFusedElementwiseOps, enableVFStackLimit);
  annotation::MarkOp::getCanonicalizationPatterns(patterns,
                                                  patterns.getContext());
  if (auto *linalgDialect = patterns.getContext()
                                ->getLoadedDialect<linalg::LinalgDialect>()) {
    linalgDialect->getCanonicalizationPatterns(patterns);
  }
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
      if (!maybeAlloc)
        return WalkResult::skip();
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
    populateEmptifyReduceInitPatterns(patterns);
  }

  populatePreVectorizationFusionPatterns(patterns, maxFusedElementwiseOps,
                                         enableVFStackLimit);
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
