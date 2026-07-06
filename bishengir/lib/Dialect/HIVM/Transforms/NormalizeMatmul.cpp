//===- NormalizeMatmul.cpp - normalize hivm matmul op.---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEMATMUL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-normalize-matmul"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

bool isSatisfiedBrcForPerChannel(hivm::VBrcOp brcOp,
                                        Operation *hookOp = nullptr);
namespace {

constexpr StringLiteral kAlreadySetRealMKN = "already_set_real_mkn";
constexpr StringLiteral kNormalizedInL0C = "normalized_in_L0C";
constexpr StringLiteral kNormalizedInitOrBias = "normalized_init_or_bias";
constexpr StringLiteral kMayNotExec = "may_not_exec";

/// Hint that we only need to pad the k-dimension for Dot.
constexpr llvm::StringLiteral kDotPadOnlyK = "dot_pad_only_k";

bool isNotWritten(Operation *val) {
  return llvm::none_of(val->getUses(), [&val](OpOperand &user) {
    Operation *maybeWriter = user.getOwner();
    auto maybeCopyOpInterface = dyn_cast<CopyOpInterface>(maybeWriter);
    return maybeCopyOpInterface &&
           maybeCopyOpInterface.getTarget().getDefiningOp() == val;
  });
}

Operation *getSingleWriter(Operation *val) {
  for (OpOperand &user : val->getUses()) {
    Operation *maybeWriter = user.getOwner();
    if (!isa<CopyOpInterface>(maybeWriter))
      continue;

    Operation *dstOp =
        cast<CopyOpInterface>(maybeWriter).getTarget().getDefiningOp();
    if (dstOp != val)
      continue;

    return maybeWriter;
  }
  return nullptr;
}

static bool isConstZero(Value v) {
  if (!v)
    return false;

  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_PosZeroFloat()) ||
        matchPattern(v, m_NegZeroFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_Zero())) {
      return true;
    }
  }
  return false;
}

/// Optimize padding on L1 if we're given the hint.
bool tryOptimizePad(Operation *maybeLoadOp, Value mmadSource,
                    PatternRewriter &rewriter) {
  if (isa<BlockArgument>(mmadSource))
    return false;

  bool padKOnly =
      utils::getAnnotateOpWithAttr(mmadSource, kDotPadOnlyK).has_value();
  if (!padKOnly)
    return false;

  auto loadOp = dyn_cast_if_present<hivm::LoadOp>(maybeLoadOp);
  if (!loadOp)
    return false;

  if (!isConstZero(loadOp.getPadValue()))
    return false;

  LDBG("removing pad for load op: " << *loadOp);
  rewriter.modifyOpInPlace(loadOp, [&loadOp]() {
    loadOp.setPadModeAttr(hivm::PadModeAttr());
    loadOp.getPadValueMutable().clear();
    loadOp.getLeftPaddingNumMutable().clear();
    loadOp.getRightPaddingNumMutable().clear();
    loadOp.setInitOutBuffer(false);
    loadOp.getInitConditionMutable().clear();
  });
  return true;
}

struct NormalizeMatmulPass
    : public impl::NormalizeMatmulBase<NormalizeMatmulPass> {
  using Base::Base;
  void runOnOperation() override;
};

SmallVector<Value> getShapeFromMixedSizes(ArrayRef<OpFoldResult> mixedSizes,
                                          Location loc,
                                          PatternRewriter &rewriter) {
  SmallVector<Value> sizes;
  for (OpFoldResult size : mixedSizes) {
    sizes.push_back(mlir::getValueOrCreateConstantIndexOp(rewriter, loc, size));
  }
  return sizes;
}

// If value is from memref, we get shape from memref.subview or memref.alloc.
// If value is from tensor, we get shape from value directly.
FailureOr<SmallVector<Value>>
getRealShapeFromMemrefOrTensor(Value val, Location loc,
                               PatternRewriter &rewriter) {
  // Assume this is a memref.space_cast from a tight couple buffer.
  // Assume it must be a static shape.
  // TODO: Properly get the shape.
  if (auto toTensor =
          dyn_cast_if_present<bufferization::ToTensorOp>(val.getDefiningOp())) {
    if (auto memspace = dyn_cast_if_present<memref::MemorySpaceCastOp>(
            toTensor.getMemref().getDefiningOp())) {
      return getShapeFromMixedSizes(
          memref::getMixedSizes(rewriter, loc, memspace->getResult(0)), loc,
          rewriter);
    }
  }
  if (isa<RankedTensorType>(val.getType()) &&
      !val.getDefiningOp<bufferization::ToTensorOp>())
    return getShapeFromMixedSizes(tensor::getMixedSizes(rewriter, loc, val),
                                  loc, rewriter);

  FailureOr<memref::AllocOp> status = getMemRefAlloc(val);
  if (failed(status))
    return getShapeFromMixedSizes(tensor::getMixedSizes(rewriter, loc, val),
                                  loc, rewriter);

  memref::AllocOp rootAlloc = *(status);
  SmallVector<Operation *> candidateSubViews;
  // Find all SubViewOps that uses the root AllocOp.
  for (OpOperand &user : rootAlloc->getUses()) {
    if (auto target = dyn_cast<memref::SubViewOp>(user.getOwner())) {
      candidateSubViews.push_back(user.getOwner());
    }
  }
  // If there is no SubView, return Alloc's shape
  if (candidateSubViews.empty()) {
    return getValueListFromMixedTypeLists(
        rootAlloc.getDynamicSizes(), rootAlloc.getMemref().getType().getShape(),
        val.getLoc(), rewriter);
  }
  // Filter the SubViewOps that is NOT written into.
  candidateSubViews.erase(llvm::remove_if(candidateSubViews, isNotWritten),
                          candidateSubViews.end());
  if (candidateSubViews.size() != 1) {
    LDBG("candidate subview size : " << candidateSubViews.size());
    return rootAlloc.emitError("Don't support the case when the root alloc "
                               "is subview-ed and written to multiple times");
  }

  auto *writerOp = getSingleWriter(candidateSubViews.front());
  assert(writerOp != nullptr);
  // We can only optimize the padding on L1 if the users explicitly
  // give us the hint that they only want to pad the K-dimension.
  // Otherwise, if we only use the real M and real N to calculate mmad,
  // there will be dirty data on the non-padded region.
  bool optimized = tryOptimizePad(writerOp, val, rewriter);
  auto subview = dyn_cast<memref::SubViewOp>(*candidateSubViews.begin());
  if (!optimized) {
    LDBG("Using main block size for mmadL1");
    return getValueListFromMixedTypeLists(
        rootAlloc.getDynamicSizes(), rootAlloc.getMemref().getType().getShape(),
        val.getLoc(), rewriter);
  }
  LDBG("Using tail block size for mmadL1");
  assert(subview != nullptr);
  return getValueListFromMixedTypeLists(
      subview.getSizes(), subview.getStaticSizes(), val.getLoc(), rewriter);
}

template <typename T>
FailureOr<SmallVector<Value>> extractRealMKN(T op, PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  SmallVector<Value> mkn;
  size_t batchIndexBias = 0;
  if constexpr (std::is_same_v<T, hivm::BatchMmadL1Op>) {
    batchIndexBias = 1;
  }
  auto realMK = getRealShapeFromMemrefOrTensor(op.getA(), loc, rewriter);
  const int matrixSize = 2;
  if (failed(realMK) || (*realMK).size() != matrixSize + batchIndexBias) {
    return failure();
  }
  auto realKN = getRealShapeFromMemrefOrTensor(op.getB(), loc, rewriter);
  if (failed(realKN) || (*realKN).size() != matrixSize + batchIndexBias) {
    return failure();
  }
  // set m, k, n
  // TODO: m is set to be l1M for group gemm scenario (use kDotPadOnlyK),
  //       which should be enhanced.
  Value realM;
  if (op.getATranspose().has_value()) {
    realM = (*realMK)[1 + batchIndexBias];
  } else {
    realM = (*realMK)[0 + batchIndexBias];
  }

  if (utils::getAnnotateOpWithAttr(op.getA(), kDotPadOnlyK).has_value()) {
    auto cType = dyn_cast<RankedTensorType>(op.getC().getType());
    if (cType && cType.hasStaticShape()) {
      size_t l1MIdx = (std::is_same_v<T, hivm::BatchMmadL1Op>) ? 1 : 0;
      int64_t l1M = cType.getShape()[l1MIdx + batchIndexBias];
      realM = rewriter.create<arith::ConstantIndexOp>(loc, l1M);
    }
  }

  mkn.push_back(realM);
  if (op.getATranspose().has_value()) {
    mkn.push_back((*realMK)[0 + batchIndexBias]);
  } else {
    mkn.push_back((*realMK)[1 + batchIndexBias]);
  }
  if (op.getBTranspose().has_value()) {
    mkn.push_back((*realKN)[0 + batchIndexBias]);
  } else {
    mkn.push_back((*realKN)[1 + batchIndexBias]);
  }
  return mkn;
}

template <>
FailureOr<SmallVector<Value>> extractRealMKN(hivm::MmadMxL1Op op,
                                             PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto realMK = getRealShapeFromMemrefOrTensor(op.getA(), loc, rewriter);
  static constexpr size_t matrixSize = 2;
  if (failed(realMK) || (*realMK).size() != matrixSize) {
    return failure();
  }
  auto realKN = getRealShapeFromMemrefOrTensor(op.getB(), loc, rewriter);
  if (failed(realKN) || (*realKN).size() != matrixSize) {
    return failure();
  }

  // set m, k, n
  Value realM = op.getATranspose().has_value() ? (*realMK)[1] : (*realMK)[0];
  Value realK = op.getATranspose().has_value() ? (*realMK)[0] : (*realMK)[1];
  Value realN = op.getBTranspose().has_value() ? (*realKN)[0] : (*realKN)[1];
  return SmallVector<Value>{realM, realK, realN};
}

template <typename T>
struct SetRealMKNPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T mmadLikeOp,
                                PatternRewriter &rewriter) const override {
    if (mmadLikeOp->hasAttr(kAlreadySetRealMKN))
      return rewriter.notifyMatchFailure(mmadLikeOp, "Pattern already applied");

    auto mkn = extractRealMKN<T>(mmadLikeOp, rewriter);
    if (failed(mkn))
      return rewriter.notifyMatchFailure(mmadLikeOp, "Failed to extract mkn");

    Operation *op = mmadLikeOp.getOperation();
    // This pattern is intended to run only once. We clone the op and use
    // `GreedyRewriteStrictness::ExistingOps` to achieve this.
    auto newOp = rewriter.clone(*op);
    rewriter.modifyOpInPlace(newOp, [&newOp, &mkn]() {
      auto newMmadLikeOp = cast<T>(newOp);
      newMmadLikeOp.getRealMMutable().assign((*mkn)[0]);
      newMmadLikeOp.getRealKMutable().assign((*mkn)[1]);
      newMmadLikeOp.getRealNMutable().assign((*mkn)[2]);
    });
    rewriter.replaceOp(op, newOp);
    newOp->setAttr(kAlreadySetRealMKN, rewriter.getUnitAttr());
    return success();
  }
};

/// Input IR:
///
/// ```
/// %2 = ops // not 0 const
/// %3 = hivm.hir.mmadL1 ins(*)
///        outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
/// ```
///
/// is converted into:
/// ```
/// %2 = ops
/// %3 = tensor.empty() : tensor<16x32xf32>
/// %4 = hivm.hir.mmadL1 ins(*)
///        outs(%3 : tensor<16x32xf32>) -> tensor<16x32xf32>
/// %5 = hivm.hir.vadd ins(%2, %4: tensor<1x32xf32>) outs(%2 :
/// tensor<16x32xf32>)
/// ```
template <typename T>
LogicalResult decomposeMatmulWithElementwiseAdd(PatternRewriter &rewriter,
                                                T op) {
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getC());
  auto newMmad = cast<T>(rewriter.clone(*op.getOperation()));
  newMmad.getCMutable().assign(newMmadInit);
  Value constTrue = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
  newMmad.setInitCondition(constTrue);
  auto addInit = mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getC());
  auto addOp = rewriter.create<hivm::VAddOp>(
      op.getLoc(), TypeRange{newMmad.getResults()[0].getType()},
      ValueRange{newMmad.getResults()[0], op.getDpsInitOperand(0)->get()},
      ValueRange{addInit});

  rewriter.replaceOp(op, addOp.getResult());
  return success();
}

/// Input IR:
///
/// ```
/// %arg = ...
/// %cond = arith.cmpi (...)
/// %mmad = mmadL1 ins(%A, %B, %cond, ...) outs(%arg)
///
/// ```
///
/// is converted into:
/// ```
/// %arg = ...
/// %cond = arith.cmpi (...)
/// %tmp = tensor.empty
/// %mmad = mmadL1 ins(%A, %B, true, ...) outs(%tmp)
///
/// %res = scf.if %cond {
///   yield %mmad
/// } else {
///   %bias = add ins(%arg, %mmad)
///   yield %bias
/// }
/// ```
template <typename T>
LogicalResult
decomposeMatmulWithConditionalElementwiseAdd(PatternRewriter &rewriter, T op) {
  Location loc = op.getLoc();
  auto newMmadInit = mlir::utils::createEmptyOp(rewriter, loc, op.getC());
  auto newMmad = cast<T>(rewriter.clone(*op.getOperation()));
  newMmad.getCMutable().assign(newMmadInit);
  Value constTrue = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
  newMmad.setInitCondition(constTrue);

  auto ifOp = rewriter.create<scf::IfOp>(
      op->getLoc(), newMmad->getResultTypes(), op.getInitCondition(),
      /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    rewriter.create<scf::YieldOp>(op->getLoc(), newMmad->getResults());
  }
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    auto addInit = mlir::utils::createEmptyOp(rewriter, loc, op.getC());
    auto addOp = rewriter.create<hivm::VAddOp>(
        loc, TypeRange{newMmad.getResults()[0].getType()},
        ValueRange{newMmad.getResults()[0], op.getDpsInitOperand(0)->get()},
        ValueRange{addInit});
    rewriter.create<scf::YieldOp>(op->getLoc(), addOp->getResults());
  }
  rewriter.replaceOp(op, ifOp.getResults());
  return success();
}

inline Value getBiasInputForPerChannelAdd(Value v) {
  auto defOp = traceDefOp<hivm::VBrcOp>(v);
  assert(defOp.has_value());
  auto brcOp = cast<hivm::VBrcOp>(defOp.value());
  Value src = brcOp.getSrc();
  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>())
    src = extractMmadBiasFromPotentialUnitDimExpand(src);

  // Consider fp16 to fp32 inner conversion form l1ToBias
  if (auto castOp = src.getDefiningOp<hivm::VCastOp>())
    if (getElementTypeOrSelf(castOp.getSingleSrc().getType()).isF16() &&
        getElementTypeOrSelf(castOp.getSingleDst().getType()).isF32())
      src = castOp.getSingleSrc();

  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>())
    src = extractMmadBiasFromPotentialUnitDimExpand(src);

  return src;
}

/// Input IR:
///
/// ```
/// %alloc = memref.alloc() : memref<1x32xf32>
/// hivm.hir.load ins(%bias : memref<1x32xf32>) outs(%alloc: memref<1x32xf32>)
/// %1 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf32>
/// %2 = tensor.empty() : tensor<16x32xf32>
/// %3 = hivm.hir.vbrc ins(%1 : tensor<1x32xf32>) outs(%2 : tensor<16x32xf32>)
///        broadcast_dims = [0]
/// %4 = hivm.hir.mmadL1 ins(*) outs(%3 : tensor<16x32xf32>) ->
///        tensor<16x32xf32>
/// ```
///
/// is converted into
/// ```
/// %alloc = memref.alloc() : memref<1x32xf32>
/// hivm.hir.load ins(%bias : memref<1x32xf32>) outs(%alloc: memref<1x32xf32>)
/// %1 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf32>
/// %2 = tensor.empty() : tensor<16x32xf32>
/// %3 = hivm.hir.mmadL1 ins(*, bias = %1) outs(%2 : tensor<16x32xf32>) ->
///        tensor<16x32xf32>
/// ```
template <typename T, typename = std::enable_if_t<
                          !std::is_same_v<T, hivm::MmadMxL1Op>>>
LogicalResult decomposeMatmulWithPerChannelAdd(PatternRewriter &rewriter,
                                               T op) {
  auto perChannelValue = getBiasInputForPerChannelAdd(op.getC());
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getC());
  auto newMmad = cast<T>(rewriter.clone(*op.getOperation()));
  newMmad.getCMutable().assign(newMmadInit);
  newMmad.getPerChannelBiasMutable().assign(perChannelValue);
  Value constTrue = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
  // reset init flag to true
  newMmad.setInitCondition(constTrue);
  rewriter.replaceOp(op, newMmad);
  return success();
}

/// Input IR:
///
/// ```
/// %0 = tensor.empty() : tensor<16x128xf32>
/// %1 = scf.for %2 = lb to ub iter_args(%arg1 = %0) ->
///   (tensor<16x128xf32>) : i32 {
///  %2 = hivm.hir.mmadL1 ins(*) outs(%arg1 : tensor<16x128xf32>) ->
///         tensor<16x128xf32>
///  scf.yield ...
/// }
/// %2 = hivm.hir.vbrc ins(%bias : tensor<1x128xf32>)
///        outs(%5 : tensor<16x128xf32>)
///        broadcast_dims = [0] -> tensor<16x128xf32>
/// %3 = tensor.empty() : tensor<16x128xf32>
/// %4 = hivm.hir.vadd ins(%1, %2 : tensor<16x128xf32>, tensor<16x128xf32>)
///        outs(%3 : tensor<16x128xf32>) -> tensor<16x128xf32>
/// some_use(%4)
/// ```
///
/// is converted into
/// ```
/// %0 = tensor.empty() : tensor<16x128xf32>
/// %1 = scf.for %2 = lb to ub iter_args(%arg1 = %0) ->
///   (tensor<16x128xf32>) : i32 {
///    %2 = hivm.hir.mmadL1 ins(*, bias = %bias)
///           outs(%arg1 : tensor<16x128xf32>) -> tensor<16x128xf32>
///   scf.yield ...
/// }
/// some_use(%1)
/// ```
template <typename T, typename = std::enable_if_t<
                          !std::is_same_v<T, hivm::MmadMxL1Op>>>
LogicalResult
decomposeMatmulWithPostPerChannelAddWithSplitKAdd(PatternRewriter &rewriter, T op) {
  auto matmulOutput = op.getC();
  auto blockArg = dyn_cast_if_present<BlockArgument>(matmulOutput);
  assert(blockArg && "blockArg is not nullptr for split k");
  auto scfForOp =
      dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
  assert(scfForOp && "scfForOp is not nullptr for split k");
  Value scfRes = scfForOp->getResults()[blockArg.getArgNumber() - 1];
  auto addOp = cast<hivm::VAddOp>(*scfRes.getUsers().begin());
  int64_t brcInputIndex = -1;
  int64_t matmulInputIndex = -1;
  auto addInputs = addOp.getSrc();
  for (int64_t i = 0; i < static_cast<int64_t>(addInputs.size()); i++) {
    if (traceDefOp<hivm::VBrcOp>(addInputs[i]).has_value()) {
      brcInputIndex = i;
    } else if (traceDefOp<hivm::MmadL1Op>(addInputs[i]).has_value()) {
      matmulInputIndex = i;
    }
  }
  if (brcInputIndex == -1 || matmulInputIndex == -1) {
    return failure();
  }

  auto perChannelVal = getBiasInputForPerChannelAdd(addInputs[brcInputIndex]);
  op.getPerChannelBiasMutable().assign(perChannelVal);
  rewriter.replaceAllUsesWith(addOp->getResults()[0], scfRes);
  return success();
}

/// Input IR:
///
/// ```
/// %alloc = memref.alloc()
/// %32 = bufferization.to_tensor %alloc
/// %33 = tensor.empty()
/// %34 = hivm.hir.vcast ins(%32) outs(%33)
/// %35 = tensor.empty()
/// %expanded = tensor.expand_shape %34
/// %36 = vbrc %expanded: (1, n) %35 :(m, n)
/// %mat = for split k (%iterator = %36) {
///   %acc_mad = mmadL1 dst(%iterator)
///   yield %acc_mad
/// }
/// ```
///
/// is converted into
/// ```
/// %0 = tensor.empty() : tensor<16x128xf32>
/// %1 = scf.for %arg12 = %lb to ub iter_args(%arg1 = %0) ->
///   (tensor<16x128xf32>) : i32 {
///   %init = arith.cmpi eq, %arg12, %lb : i32
///   %2 = hivm.hir.mmadL1 ins(*, %init, *, bias = %bias)
///           outs(%arg1 : tensor<16x128xf32>) -> tensor<16x128xf32>
///   scf.yield ...
/// }
/// some_use(%1)
/// ```
template <typename T, typename = std::enable_if_t<
                          !std::is_same_v<T, hivm::MmadMxL1Op>>>
LogicalResult
decomposeMatmulWithMMInitPerChannelAddWithSplitK(PatternRewriter &rewriter, T op) {
  auto perChannelValue = getBiasInputForPerChannelAdd(op.getC());
  op.getPerChannelBiasMutable().assign(perChannelValue);

  auto matmulOutput = op.getC();
  auto blockArg = dyn_cast_if_present<BlockArgument>(matmulOutput);
  assert(blockArg && "blockArg is not nullptr for mm init per channel split k");
  auto scfForOp =
      dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
  assert(scfForOp && "scfForOp is not nullptr for mm init per channel split k");

  rewriter.setInsertionPoint(op);
  auto additionalCondition = rewriter.create<arith::CmpIOp>(
      op.getLoc(), arith::CmpIPredicate::eq, scfForOp.getLowerBound(),
      scfForOp.getInductionVar());
  op.setInitCondition(additionalCondition);

  rewriter.setInsertionPoint(scfForOp);
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getC());
  auto blockArgIdx = blockArg.getArgNumber() - 1;
  scfForOp.getInitArgsMutable()[blockArgIdx].assign(newMmadInit);
  return success();
}

bool hasDebugUse(Value val) {
  for (OpOperand &use : val.getUses()) {
    Operation *userOp = use.getOwner();
    if (isa<hivm::DebugOp>(userOp)) {
      return true;
    }
    for (Value result : userOp->getResults()) {
      if (hasDebugUse(result)) {
        return true;
      }
    }
  }
  return false;
}

// Find the outer res of scf.if and scf.for block
// The %arg should be the output of op if in scf.for
// The output only used by yield op
struct CCFInfo {
  Value inVal;
  Value outVal;
  BlockArgument blockArg;
  Operation *insertPointOp = nullptr;
  bool isFailure = false;
  bool mayNotExec = false;
  bool mayNotExecWithIf = false;

  static CCFInfo getFailure(CCFInfo &info) {
    info.isFailure = true;
    return info;
  }
};

static Operation *getAncestorInBlock(Operation *inner, const Block *block) {
  Operation *cur = inner;
  while (cur) {
    if (cur->getBlock() == block) {
      return cur;
    }
    cur = cur->getParentOp();
  }
  return nullptr;
}

CCFInfo getOutermostCCFInfo(Operation *op, CCFInfo info) {
  Operation *parentOp = op->getParentOp();
  if (!parentOp || !(isa<scf::ForOp>(parentOp) || isa<scf::IfOp>(parentOp)))
    return info;

  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    auto blockArg = dyn_cast_if_present<BlockArgument>(info.inVal);
    if (!blockArg || blockArg.getOwner() != forOp.getBody())
      return CCFInfo::getFailure(info);

    unsigned argIdx = blockArg.getArgNumber() - 1;
    // Relax single-use: allow uses that are scf.if else pass-throughs and the
    // op itself.
    for (OpOperand &use : blockArg.getUses()) {
      Operation *user = use.getOwner();
      // Allowed: the op itself (mmad or inner for/if that chains to mmad).
      auto userInblock = getAncestorInBlock(user, op->getBlock());
      if (userInblock == op)
        continue;
      // Allowed: scf.if else-yield that passes blockArg through unchanged.
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (yieldOp->getBlock() ==
            dyn_cast<scf::IfOp>(yieldOp->getParentOp()).elseBlock())
          continue;
      }
      return CCFInfo::getFailure(info);
    }
    // The res should only be used by yields in the for body.
    for (OpOperand &use : info.outVal.getUses()) {
      Operation *user = use.getOwner();
      auto yieldOp = dyn_cast<scf::YieldOp>(user);
      if (!yieldOp)
        return CCFInfo::getFailure(info);
      if (yieldOp->getBlock() != forOp.getBody())
        return CCFInfo::getFailure(info);
      if (use.getOperandNumber() != argIdx)
        return CCFInfo::getFailure(info);
    }

    IntegerAttr ubAttr, lbAttr;
    if (matchPattern(forOp.getUpperBound(), m_Constant(&ubAttr)) &&
        matchPattern(forOp.getLowerBound(), m_Constant(&lbAttr))) {
      if (ubAttr.getValue().sle(lbAttr.getValue())) {
        info.mayNotExec = true;
      }
    } else {
      info.mayNotExec = true;
    }
    info.blockArg = blockArg;
    info.inVal = forOp.getInitArgs()[argIdx];
    info.outVal = forOp->getResult(argIdx);
    info.insertPointOp = forOp;
  } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
    if (!ifOp.elseBlock())
      return CCFInfo::getFailure(info);
    auto thenYieldOp = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYieldOp = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    unsigned resultIdx = 0;
    bool found = false;
    if (op->getBlock() == ifOp.thenBlock()) {
      for (unsigned i = 0; i < thenYieldOp->getNumOperands(); ++i) {
        if (thenYieldOp->getOperand(i) == info.outVal) {
          resultIdx = i;
          found = true;
          break;
        }
      }
    } else {
      for (unsigned i = 0; i < elseYieldOp->getNumOperands(); ++i) {
        if (elseYieldOp->getOperand(i) == info.outVal) {
          resultIdx = i;
          found = true;
          break;
        }
      }
    }
    if (!found)
      return CCFInfo::getFailure(info);

    if (op->getBlock() == ifOp.thenBlock()) {
      if (elseYieldOp->getOperand(resultIdx) != info.inVal)
        return CCFInfo::getFailure(info);
    } else {
      if (thenYieldOp->getOperand(resultIdx) != info.inVal)
        return CCFInfo::getFailure(info);
    }

    if (!matchPattern(ifOp.getCondition(), m_One())) {
      info.mayNotExec = true;
      info.mayNotExecWithIf = true;
    }
    info.outVal = ifOp->getResult(resultIdx);
    info.insertPointOp = ifOp;
  }

  return getOutermostCCFInfo(parentOp, info);
}
template <typename T>
CCFInfo getResFromSingleUseChain(Operation *op) {
  CCFInfo initInfo;
  initInfo.inVal = cast<T>(op).getC();
  initInfo.outVal = op->getResult(0);
  initInfo.insertPointOp = op;
  return getOutermostCCFInfo(op, initInfo);
}

Value initCounter(PatternRewriter &rewriter, Operation &op) {
    rewriter.setInsertionPoint(&op);
    // Alloca + store 0 before the inner scf.for. Outer-loop body re-runs this
    // every outer iteration, so the counter resets per outer step.
    Location loc = op.getLoc();
    Value counterBuf = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getI32Type()));
    // Mark the alloca so downstream passes can recognize it as the
    // normalize-matmul iteration counter.
    counterBuf.getDefiningOp()->setAttr(kNormalizeMatmulCounterAttr, rewriter.getUnitAttr());
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto storeOp = rewriter.create<memref::StoreOp>(loc, zeroI32, counterBuf,
                                                    ValueRange{});
    storeOp->setAttr(
        hivm::TCoreTypeAttr::name,
        hivm::TCoreTypeAttr::get(rewriter.getContext(),
                                 hivm::TCoreType::CUBE_AND_VECTOR));
    return counterBuf;
}

template <typename T>
Value updateInitCondition(PatternRewriter &rewriter, T op, Value counterBuf) {
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();
  Value curCount =
      rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
  Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto firstIterCond = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, curCount, zeroI32);

  // In the same then-branch, right after matmul: counter += 1; store back.
  // Counter only advances on iterations where the scf.if condition fired,
  // which is exactly what the fallback below relies on.
  rewriter.setInsertionPointAfter(op);
  Value oneI32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value nextCount = rewriter.create<arith::AddIOp>(loc, curCount, oneI32);
  rewriter.create<memref::StoreOp>(loc, nextCount, counterBuf, ValueRange{});
  return firstIterCond;
}

template <typename T> Value updateInitTrue(PatternRewriter &rewriter, T op) {
  Value constTrue = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
  return constTrue;
}

template <typename T> Value getCounterBufFromInitCondition(T mmadLikeOp) {
  // Attempt to get the init condition value from the operation
  Value initCond = mmadLikeOp.getInitCondition();
  if (!initCond)
    return nullptr;

  // Trace back from initCondition to find the counter buffer
  auto cmpOp = initCond.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return nullptr;

  auto loadOp = cmpOp.getOperand(0).getDefiningOp<memref::LoadOp>();
  if (!loadOp)
    return nullptr;

  Value counterBuf = loadOp.getOperand(0);
  if (counterBuf.getDefiningOp<memref::AllocOp>() ||
      counterBuf.getDefiningOp<memref::AllocaOp>()) {
    return counterBuf;
  }

  return nullptr;
}

hivm::VAddOp createVadd(PatternRewriter &rewriter, Location loc, Type type,
                        Value operand_l, Value operand_r) {
  auto addInit = mlir::utils::createEmptyOp(rewriter, loc, operand_l);
  auto addOp = rewriter.create<hivm::VAddOp>(loc, TypeRange{type},
                                             ValueRange{operand_l, operand_r},
                                             ValueRange{addInit});
  return addOp;
}

// Set kNormalizedInL0C attribute with proper index for scf::ForOp, scf::IfOp,
// or UnitAttr for other operations. If the attribute already exists, append
// the new index to the existing list instead of overwriting.
void setNormalizedInL0CWithIndex(PatternRewriter &rewriter, Operation &insertPointOp,
                                  Value outerOutVal) {
  // skip the case that insertPointOp is mmad
  if (!isa<scf::ForOp, scf::IfOp>(insertPointOp))
    return;
  unsigned resultIdx = 0;
  for (Value result : insertPointOp.getResults()) {
    if (result == outerOutVal) {
      break;
    }
    resultIdx++;
  }

  // Check if attribute already exists
  SmallVector<Attribute> indices;
  if (auto existingAttr = insertPointOp.getAttrOfType<ArrayAttr>(kNormalizedInL0C)) {
    // Append existing indices
    for (Attribute attr : existingAttr) {
      indices.push_back(attr);
    }
  }

  // Add the new index (avoid duplicates)
  auto newIdxAttr = rewriter.getI32IntegerAttr(resultIdx);
  bool isDuplicate = false;
  for (Attribute attr : indices) {
    if (auto idxAttr = attr.dyn_cast<IntegerAttr>()) {
      if (idxAttr.getInt() == static_cast<int64_t>(resultIdx)) {
        isDuplicate = true;
        break;
      }
    }
  }
  if (!isDuplicate) {
    indices.push_back(newIdxAttr);
  }

  insertPointOp.setAttr(kNormalizedInL0C, rewriter.getArrayAttr(indices));
}

void setRemainInL0CAttr(PatternRewriter &rewriter, Value outerInVal) {
  if (Operation *defOp = outerInVal.getDefiningOp()) {
    defOp->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
  }
}

template <typename T>
void addTailFallback(PatternRewriter &rewriter, Operation &op, T mmad,
                     Value counterBuf, Value outerInVal, Value outerOutVal,
                     bool isAdd = false) {
  rewriter.setInsertionPointAfter(&op);
  Location loc = op.getLoc();
  Value postCount =
      rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
  Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value neverRan = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  postCount, zeroI32);

  auto fallbackIf = rewriter.create<scf::IfOp>(
      loc, mmad->getResultTypes(), neverRan, /*withElseRegion=*/true);

  // Then block
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(fallbackIf.thenBlock());
    if (auto brcOp = outerInVal.getDefiningOp<hivm::VBrcOp>()) {
      auto newVbrcOp =
          cast<hivm::VBrcOp>(rewriter.clone(*brcOp.getOperation()));
      rewriter.create<scf::YieldOp>(loc, ValueRange{newVbrcOp.getResult()});
    } else {
      rewriter.create<scf::YieldOp>(loc, ValueRange{outerInVal});
    }
  }

  // Else block
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(fallbackIf.elseBlock());
    if (isAdd) {
      auto addOp = createVadd(rewriter, loc, outerOutVal.getType(), outerInVal,
                              outerOutVal);
      rewriter.create<scf::YieldOp>(loc, ValueRange{addOp.getResults()[0]});
    } else {
      rewriter.create<scf::YieldOp>(loc, ValueRange{outerOutVal});
    }
  }

  // replace user of output of CCF
  rewriter.replaceUsesWithIf(outerOutVal, fallbackIf.getResult(0),
                             [&](OpOperand &operand) {
                               Operation *userOp = operand.getOwner();
                               return !fallbackIf->isAncestor(userOp);
                             });
}

// Check if previous L0C could be reuse
bool couldReuse(Value outerInVal) {
  // Check the user of previous L0C is only one
  if (!outerInVal.hasOneUse())
    return false;

  // check the user of previous L0C is mmadL1
  if (auto mmadOp = outerInVal.getDefiningOp<hivm::MmadL1Op>()) {
    return true;
  }

  // check the user of previous L0C is output from mmadL1 in CCF
  if (Operation *defOp = outerInVal.getDefiningOp()) {
    if(defOp->hasAttr(kMayNotExec) || !isa<scf::ForOp>(defOp))
      return false;
    // For scf::ForOp with multiple results, use ArrayAttr to specify indices
    // Find the corresponding result index
    unsigned resultIdx = 0;
    for (Value result : defOp->getResults()) {
      if (result == outerInVal) {
        break;
      }
      resultIdx++;
    }

    // Check if this specific result has the L0C normalized attribute
    if (auto normalizedAttr = defOp->getAttrOfType<ArrayAttr>(kNormalizedInL0C)) {
      if (normalizedAttr.empty()) {
        return false;
      }

      for (Attribute idxAttr : normalizedAttr) {
        if (auto idxInt = idxAttr.dyn_cast<IntegerAttr>()) {
          if (idxInt.getInt() == static_cast<int64_t>(resultIdx)
              ) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

struct BrcBiasInfo {
  MatmulBiasMode brcBiasMode = MatmulBiasMode::NoBias;
  Value perChannelValue;
  hivm::VAddOp addOp;
};

// Get the Bias mode
// PerChannelAdd: Vbrc(x, [0])
// ZeroInitNoAccumulation: Vbrc(0)
// ReuseL0C: prev_val (mmadL1) is confrimly executed and current mayNotExec is
//          false; TODO: mayNotExecWithIf criteria should be removed
// NoBias: empty ElementwiseAdd: x[n, m]
template <typename T> BrcBiasInfo getBrcBiasMode(CCFInfo ccfinfo, T op) {
  // refer to getMatmulLikeBiasMode
  Value outerInVal = ccfinfo.inVal;
  BrcBiasInfo info;
  if (auto brcOp = outerInVal.getDefiningOp<hivm::VBrcOp>()) {
    if (isSatisfiedBrcForPerChannel(brcOp)) {
      info.perChannelValue = getBiasInputForPerChannelAdd(outerInVal);
      info.brcBiasMode = MatmulBiasMode::PerChannelAdd;
      return info;
    }
    if (isConstZero(brcOp.getSrc())) {
      info.brcBiasMode = MatmulBiasMode::ZeroInitNoAccumulation;
      return info;
    }
  } else if (couldReuse(outerInVal) && (!ccfinfo.mayNotExecWithIf)) {
    info.brcBiasMode = MatmulBiasMode::ReuseL0C;
    return info;
  } else if (outerInVal.hasOneUse()) {
    if (auto addOp = dyn_cast<hivm::VAddOp>(*outerInVal.getUsers().begin())) {
      for (Value src : addOp.getSrc()) {
        if (auto brcOp = src.getDefiningOp<hivm::VBrcOp>()) {
          if (isSatisfiedBrcForPerChannel(brcOp)) {
            info.perChannelValue = getBiasInputForPerChannelAdd(src);
            info.brcBiasMode = MatmulBiasMode::PostPerChannelAddWithSplitK;
            return info;
          }
        }
      }
    }
  }
  auto emptyOps =
      traceDefOps<tensor::EmptyOp>(outerInVal,
                                   /*isSingleChain=*/false,
                                   /*traceMode=*/TraceResultMode::StrictSame);
  info.brcBiasMode = !emptyOps.empty() ? MatmulBiasMode::NoBias
                                       : MatmulBiasMode::ElementwiseAdd;

  return info;
}

// Add counter and if block in the tail for the case that mmad is probably not
// executed
template <typename T> constexpr bool isUnsupportedOpForNormalizeMmadCCF() {
  // TODO: remove when MmadMxL1Op is supported with bias.
  return std::is_same_v<T, hivm::MmadMxL1Op>;
}

template <typename T>
struct NormalizeMmadCCFPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // TODO: need to be reverted when Affinity GMM supported
    if constexpr (isUnsupportedOpForNormalizeMmadCCF<T>())
      return failure();
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    bool isDisableHfusionVectorize = false;
    if (moduleOp) {
      isDisableHfusionVectorize =
          moduleOp->hasAttr("hfusion.disableHfusionVectorize");
    }

    auto scopeOp = op->template getParentOfType<scope::ScopeOp>();
    if ((scopeOp && !scopeOp->hasAttr(hivm::MatmulLimitedInCubeAttr::name)) ||
        isDisableHfusionVectorize) {
      LDBG("Affinity pattern already applied");
      return rewriter.notifyMatchFailure(op,
                                         "Affinity pattern already applied");
    }

    if (op->hasAttr(kNormalizedInL0C)) {
      LDBG("Pattern already applied");
      return rewriter.notifyMatchFailure(op, "Pattern already applied");
    }

    if (!matchPattern(op.getInitConditionMutable().get(), m_Zero())) {
      LDBG("Init condition is not zero");
      return rewriter.notifyMatchFailure(op, "Init condition is not zero");
    }
    // Check if in CCF
    if (!op->template getParentOfType<scf::ForOp>() &&
        !op->template getParentOfType<scf::IfOp>()) {
      LDBG("Not in CCF");
    } else {
      LDBG("In CCF");
    }

    // get Mmad Pattern: isSingleUseChain blockOp
    auto ccfInfo = getResFromSingleUseChain<T>(op);
    // If upper user enssure matmul limited in cube, we need to set mayNotExec
    // to false to avoid adding if condition.
    if (auto parantop = op->getParentOp()) {
      if (parantop->hasAttr(hivm::MatmulLimitedInCubeAttr::name)) {
        ccfInfo.mayNotExec = false;
        ccfInfo.mayNotExecWithIf = false;
      }
    }

    Value outerInVal = ccfInfo.inVal;
    Value outerOutVal = ccfInfo.outVal;
    Operation *insertPointOp = ccfInfo.insertPointOp;
    BlockArgument blockArg = ccfInfo.blockArg;
    bool skipOptimize = ccfInfo.isFailure;
    bool mayNotExec = ccfInfo.mayNotExec;
    bool isUsingCounter = false;
    // get bias Info
    BrcBiasInfo biasInfo = getBrcBiasMode<T>(ccfInfo, op);
    LDBG("BiasMode:" << biasInfo.brcBiasMode);
    LDBG("skipOptimize:" << skipOptimize);
    LDBG("mayNotExec:" << mayNotExec);
    // create counter buffer
    Value counterBuf;
    if (!isa<T>(insertPointOp))
      counterBuf = initCounter(rewriter, *insertPointOp);

    // create new mmad op
    Value newInit = mlir::utils::createEmptyOp(
        rewriter, insertPointOp->getLoc(), outerInVal);
    auto tmpNewMmad = cast<T>(rewriter.clone(*op.getOperation()));

    // set init condition before mmad op
    Value initCondition;
    if (!isa<T>(insertPointOp)) {
      initCondition = updateInitCondition<T>(rewriter, op, counterBuf);
      isUsingCounter = true;
      if (!isa<scf::IfOp>(op->getParentOp())) {
        tmpNewMmad->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
        op->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
      }
      if (biasInfo.brcBiasMode == MatmulBiasMode::ReuseL0C) {
        op->setAttr(kNormalizedInL0C, rewriter.getUnitAttr());
        LDBG("ReuseL0C in for-loop or if: no need to decompose matmul");

        // Set reuse tag for the defining op of outerInVal
        setRemainInL0CAttr(rewriter, outerInVal);

        return success();
      }
      LDBG("initCondition using counter");
    } else if (biasInfo.brcBiasMode == MatmulBiasMode::ReuseL0C) {
      LDBG("initCondition always false");
      LDBG("ReuseL0C no need to decompose matmul");
      op->setAttr(kNormalizedInL0C, rewriter.getUnitAttr());

      // Set reuse tag for the defining op of outerInVal
      setRemainInL0CAttr(rewriter, outerInVal);

      return success();
    } else {
      rewriter.setInsertionPoint(insertPointOp);
      initCondition = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
      mayNotExec = false;
      LDBG("initCondition always true");
    }
    tmpNewMmad.setInitCondition(initCondition);

    // Normalize the outermost init arg to tensor.empty
    if (biasInfo.brcBiasMode != MatmulBiasMode::NoBias) {
      if (!isa<T>(insertPointOp) && blockArg) {
        auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (forOp) {
          auto blockArgIdx = blockArg.getArgNumber() - 1;
          forOp.getInitArgsMutable()[blockArgIdx].assign(newInit);
        } else {
          return rewriter.notifyMatchFailure(
              op, "expected the outermost init arg to be a block argument of a "
                  "for op");
        }
      } else {
        // not in the for loop
        tmpNewMmad.getCMutable().assign(newInit);
      }
    }

    // If it could be merged with bias
    if (biasInfo.brcBiasMode == MatmulBiasMode::PerChannelAdd ||
        biasInfo.brcBiasMode == MatmulBiasMode::PostPerChannelAddWithSplitK) {
      tmpNewMmad.getPerChannelBiasMutable().assign(biasInfo.perChannelValue);
      // remove vadd
      if (biasInfo.addOp) {
        rewriter.replaceAllUsesWith(biasInfo.addOp->getResults()[0],
                                    insertPointOp->getResults()[0]);
      }
      tmpNewMmad->setAttr(kNormalizedInitOrBias, rewriter.getUnitAttr());
    }
    tmpNewMmad->setAttr(kNormalizedInL0C, rewriter.getUnitAttr());
    setNormalizedInL0CWithIndex(rewriter, *insertPointOp, outerOutVal);

    if (mayNotExec)
      insertPointOp->setAttr(kMayNotExec, rewriter.getUnitAttr());

    // Add if block for the case that mmad is probably not executed
    if (biasInfo.brcBiasMode == MatmulBiasMode::ElementwiseAdd) {
      if (mayNotExec) {
        // generate vadd + yield
        addTailFallback<T>(rewriter, *insertPointOp, tmpNewMmad, counterBuf,
                           outerInVal, outerOutVal, true);
      } else {
        // generate vadd
        rewriter.setInsertionPointAfter(insertPointOp);
        Location loc = insertPointOp->getLoc();
        auto addOp =
            createVadd(rewriter, loc, insertPointOp->getResults()[0].getType(),
                       insertPointOp->getResults()[0], outerInVal);
        mlir::DominanceInfo domInfo(op->getParentOp());
        for (auto &use : llvm::make_early_inc_range(outerOutVal.getUses())) {
          Operation *userOp = use.getOwner();
          if (!domInfo.properlyDominates(addOp, userOp))
            continue;
          if (userOp == addOp)
            continue;
          rewriter.modifyOpInPlace(userOp,
                                   [&]() { use.set(addOp.getResult()[0]); });
        }
      }
      LDBG("Default: decompose matmul with elemwise add");
    } else if ((biasInfo.brcBiasMode ==
                MatmulBiasMode::ZeroInitNoAccumulation) &&
               (!isUsingCounter)) {
      LDBG("decompose matmul with  zero init no accumlation");
    } else {
      if (mayNotExec) {
        // generate yield
        addTailFallback<T>(rewriter, *insertPointOp, tmpNewMmad, counterBuf,
                           outerInVal, outerOutVal);
      }
      LDBG("decompose matmul with other cases");
    }

    // replace mmad op with the new one finally
    Operation *lastOperandDef = initCondition.getDefiningOp();
    for (auto operand : tmpNewMmad->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (defOp->getBlock() != lastOperandDef->getBlock())
        continue;
      if (lastOperandDef->isBeforeInBlock(defOp))
        lastOperandDef = defOp;
    }
    rewriter.setInsertionPointAfter(op);
    auto newMmad = cast<T>(rewriter.clone(*tmpNewMmad.getOperation()));
    rewriter.eraseOp(tmpNewMmad);
    rewriter.replaceOp(op, newMmad);

    return success();
  }
};

template <typename T>
struct DecomposeMatmulWithBiasPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kNormalizedInitOrBias) || op->hasAttr(kNormalizedInL0C)) {
      LDBG("Pattern already applied");
      return rewriter.notifyMatchFailure(op, "Pattern already applied");
    }

    MatmulBiasMode biasMode = op.getMatmulBiasMode();
    if (biasMode == MatmulBiasMode::NoBias) {
      LDBG("no bias");
      return rewriter.notifyMatchFailure(op, "no bias");
    }
    if (op.shouldDecomposeBiasByElementAdd() && op.isInitConstant(true)) {
      LDBG("no need to decompose matmul with elemwise add since init is true");
      return failure();
    }
    if (op.shouldDecomposeBiasByElementAdd() && op.isInitConstant(false)) {
      LDBG("decompose matmul with elemwise add");
      return decomposeMatmulWithElementwiseAdd<T>(rewriter, op);
    }
    if constexpr (!std::is_same_v<T, hivm::MmadMxL1Op>) {
      if (biasMode == MatmulBiasMode::PerChannelAdd) {
        LDBG("decompose matmul with per channel add");
        return decomposeMatmulWithPerChannelAdd<T>(rewriter, op);
      }
      if (biasMode == MatmulBiasMode::PostPerChannelAddWithSplitK) {
        LDBG("decompose matmul with post per channel add with split k add");
        return decomposeMatmulWithPostPerChannelAddWithSplitKAdd<T>(rewriter,
                                                                    op);
      }
      if (biasMode == MatmulBiasMode::MMInitPerChannelAddWithSplitK) {
        LDBG("decompose matmul with mm init per channel add with split k add");
        return decomposeMatmulWithMMInitPerChannelAddWithSplitK<T>(rewriter,
                                                                   op);
      }
    }

    Value mmadResult = op.getResults()[0];
    if (scope::utils::isInCubeScope(op) && hasDebugUse(mmadResult)) {
      return failure();
    }

    if (op.shouldDecomposeBiasByElementAdd() && !op.isInitConstant()) {
      LDBG("decompose matmul with elemwise add and non-const init");
      return decomposeMatmulWithConditionalElementwiseAdd<T>(rewriter, op);
    }

    return failure();
  }
};

void populateSetRealMKNPattern(RewritePatternSet &patterns) {
  patterns.add<SetRealMKNPattern<hivm::MmadL1Op>,
               SetRealMKNPattern<hivm::MmadMxL1Op>,
               SetRealMKNPattern<hivm::BatchMmadL1Op>>(patterns.getContext());
}

void populateNormalizeMatmulPattern(RewritePatternSet &patterns) {
  patterns.add<NormalizeMmadCCFPattern<hivm::MmadL1Op>,
               NormalizeMmadCCFPattern<hivm::BatchMmadL1Op>,
               DecomposeMatmulWithBiasPattern<hivm::MmadL1Op>,
               DecomposeMatmulWithBiasPattern<hivm::BatchMmadL1Op>,
               DecomposeMatmulWithBiasPattern<hivm::MmadMxL1Op>>(
      patterns.getContext());
}

void NormalizeMatmulPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  {
    RewritePatternSet patterns(context);
    populateSetRealMKNPattern(patterns);
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      signalPassFailure();
  }
  {
    RewritePatternSet patterns(context);
    populateNormalizeMatmulPattern(patterns);
    GreedyRewriteConfig config = GreedyRewriteConfig();
    // Enable `TopDownTraversal` to search more optimization, e.g.
    //
    // ```
    // %1 = mad
    // %2 = add ins (%1, ..)
    // %3 = mad outs(%2)
    // ```
    //
    // If top down, first mad and add will be optimized mmad with bias
    // ```
    // %2 = mad with bias (...)
    // %3 = mad outs(%2)
    // ```
    // then, no need to decompose the second mad to 'mad + add' anymore because
    // the mad result can be accumulated in L0C.
    //
    // But if it is BottomUpTraversal, the second mad will be decompose to
    // 'mad + add' and lose 'mad + mad' optimization.
    config.useTopDownTraversal = true;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeMatmulPass() {
  return std::make_unique<NormalizeMatmulPass>();
}

