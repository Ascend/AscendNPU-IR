//===- NormalizeMatmul.cpp - normalize hivm matmul op.---------------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
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

namespace mlir {
namespace hivm {
bool isSatisfiedBrcForPerChannel(hivm::VBrcOp brcOp,
                                 Operation *hookOp = nullptr);
} // namespace hivm
} // namespace mlir

namespace {

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
  // A3 lit expects `init_out_buffer = false` to remain; A5 expects the attr
  // omitted entirely (DefaultValuedOptional).
  bool isRegBased = false;
  for (Operation *p = loadOp; p; p = p->getParentOp()) {
    if (auto moduleOp = dyn_cast<ModuleOp>(p)) {
      if (hacc::utils::getTargetDevice(moduleOp).has_value()) {
        isRegBased = hacc::utils::isRegBasedArch(moduleOp);
        break;
      }
    }
  }
  rewriter.modifyOpInPlace(loadOp, [&loadOp, isRegBased]() {
    loadOp.setPadModeAttr(hivm::PadModeAttr());
    loadOp.getPadValueMutable().clear();
    loadOp.getLeftPaddingNumMutable().clear();
    loadOp.getRightPaddingNumMutable().clear();
    if (isRegBased)
      loadOp.removeInitOutBufferAttr();
    else
      loadOp.setInitOutBuffer(false);
    loadOp.getInitConditionMutable().clear();
  });
  return true;
}

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

struct NormalizeMatmulPass
    : public impl::NormalizeMatmulBase<NormalizeMatmulPass> {
  using Base::Base;
  void runOnOperation() override;
};

namespace membase {

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
  // set m,k,n
  if (op.getATranspose().has_value()) {
    mkn.push_back((*realMK)[1 + batchIndexBias]);
    mkn.push_back((*realMK)[0 + batchIndexBias]);
  } else {
    mkn.push_back((*realMK)[0 + batchIndexBias]);
    mkn.push_back((*realMK)[1 + batchIndexBias]);
  }
  if (op.getBTranspose().has_value()) {
    mkn.push_back((*realKN)[0 + batchIndexBias]);
  } else {
    mkn.push_back((*realKN)[1 + batchIndexBias]);
  }
  return mkn;
}

template <typename T>
struct SetRealMKNPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto mkn = extractRealMKN<T>(op, rewriter);
    if (failed(mkn) || alreadyExists(op, *mkn)) {
      return failure();
    }
    op.getRealMMutable().assign((*mkn)[0]);
    op.getRealKMutable().assign((*mkn)[1]);
    op.getRealNMutable().assign((*mkn)[2]);
    return success();
  }

private:
  bool alreadyExists(T op, ArrayRef<Value> mkn) const {
    return op.getRealM() == mkn[0] && op.getRealK() == mkn[1] &&
           op.getRealN() == mkn[2];
  }
};

/// Input IR:
///
/// ```
/// %0 = tensor.empty()
/// scf.for ... iter_args(%arg0 = %0)
/// %1 = hivm.hir.mmadL1 ins(*)
///        outs(%arg0 : tensor<16x32xf32>) -> tensor<16x32xf32>
/// ```
///
/// is converted into:
/// ```
/// %0 = tensor.empty()
/// %1 = hivm.vbrc (0, %0) -> tensor<16x32xf32>
/// scf.for ... iter_args(%arg0 = %1)
//  %2 = tensor.empty() : tensor<16x32xf32>
/// %3 = hivm.hir.mmadL1 ins(*)
///        outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
/// %4 = hivm.hir.vadd ins(%arg0, %3) outs(%arg0)
/// ```
template <typename T>
LogicalResult
decomposeMatmulWithCrossLoopElementwiseAdd(PatternRewriter &rewriter, T op) {
  Location loc = op.getLoc();

  auto maybePreLoopEmptyOp = traceDefOp<tensor::EmptyOp>(op.getC());
  assert(maybePreLoopEmptyOp.has_value());

  auto preLoopEmptyOp = cast<tensor::EmptyOp>(maybePreLoopEmptyOp.value());
  rewriter.setInsertionPointAfterValue(preLoopEmptyOp);

  auto mmadCType = op.getC().getType();
  auto elemType = getElementTypeOrSelf(mmadCType);
  auto constZero = utils::createConstantOp<int>(rewriter, loc, elemType, 0);
  auto zeroFillOp = rewriter.create<hivm::VBrcOp>(
      loc, TypeRange{mmadCType}, constZero, preLoopEmptyOp.getResult());
  rewriter.replaceAllUsesExcept(preLoopEmptyOp.getResult(),
                                zeroFillOp->getResults()[0], zeroFillOp);

  rewriter.setInsertionPointAfter(op);
  auto newMmadInit = mlir::utils::createEmptyOp(rewriter, loc, op.getC());
  auto newMmad = cast<T>(rewriter.clone(*op.getOperation()));
  newMmad.getCMutable().assign(newMmadInit);
  Value constTrue = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  newMmad.setInitCondition(constTrue);

  auto addInit = mlir::utils::createEmptyOp(rewriter, loc, op.getC());
  auto addOp = rewriter.create<hivm::VAddOp>(
      loc, TypeRange{newMmad.getResults()[0].getType()},
      ValueRange{newMmad.getResults()[0], op.getDpsInitOperand(0)->get()},
      ValueRange{addInit});

  rewriter.replaceOp(op, addOp.getResult());
  return success();
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
template <typename T>
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
template <typename T>
LogicalResult
decomposeMatmulWithPerChannelAddWithSplitKAdd(PatternRewriter &rewriter, T op) {
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

template <typename T>
struct DecomposeMatmulWithBiasPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    MatmulBiasMode biasMode = op.getMatmulBiasMode();
    if (biasMode == MatmulBiasMode::NoBias) {
      return rewriter.notifyMatchFailure(op, "no bias");
    }
    if (op.shouldDecomposeBiasByElementAdd()) {
      if (!op.isInitConstant(false)) {
        return failure();
      }
      LDBG("decompose matmul with elemwise add");
      return decomposeMatmulWithElementwiseAdd<T>(rewriter, op);
    }
    if (op.shouldDecomposeBiasByCrossLoopElementAdd()) {
      assert(op.isInitFirstLoopIter());
      LDBG("decompose matmul with crossloop elemwise add");
      return decomposeMatmulWithCrossLoopElementwiseAdd<T>(rewriter, op);
    }
    if (biasMode == MatmulBiasMode::PerChannelAdd) {
      LDBG("decompose matmul with per channel add");
      return decomposeMatmulWithPerChannelAdd<T>(rewriter, op);
    }
    if (biasMode == MatmulBiasMode::PerChannelAddWithSplitK) {
      LDBG("decompose matmul with per channel add with split k add");
      return decomposeMatmulWithPerChannelAddWithSplitKAdd<T>(rewriter, op);
    }
    return failure();
  }
};

void populateNormalizeMatmulPattern(RewritePatternSet &patterns) {
  patterns.add<DecomposeMatmulWithBiasPattern<hivm::MmadL1Op>,
               DecomposeMatmulWithBiasPattern<hivm::BatchMmadL1Op>>(
      patterns.getContext());
  patterns.add<SetRealMKNPattern<hivm::MmadL1Op>,
               SetRealMKNPattern<hivm::BatchMmadL1Op>>(patterns.getContext());
}

void runMemBasedNormalizeMatmul(func::FuncOp funcOp, MLIRContext *context) {
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
  (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
}

} // namespace membase

namespace regbase {

constexpr StringLiteral kAlreadySetRealMKN = "already_set_real_mkn";
constexpr StringLiteral kNormalizedInL0C = "normalized_in_L0C";
constexpr StringLiteral kNormalizedInitOrBias = "normalized_init_or_bias";
constexpr StringLiteral kMayNotExec = "may_not_exec";

// Set on an mmad whose CCF still owes a tail fallback. Emission is deferred to
// ReuseL0CAddIfPattern, which can see the CCF's consumer and so can tell
// whether a fallback is needed at all.
constexpr StringLiteral kDeferredTailFallback = "deferred_tail_fallback";

// Tags the i1 that records whether the L0C block was left dirty by an earlier
// CCF, so that a later CCF can reuse it instead of rebuilding the predicate.
constexpr StringLiteral kCounterPrevious = "counter_previous";

// Marker set by FoldFractalVtransposePattern to distinguish a real 2D transpose
// from a fractal layout indicator (nZ vs zN) in extractRealMKN.
constexpr StringLiteral kFractalVtransposeFolded = "fractal_vtranspose_folded";

FailureOr<SmallVector<Value>>
extractRealMKN(LocalMatmulLikeOpInterface matmulOp, PatternRewriter &rewriter) {
  auto loc = matmulOp.getLoc();
  SmallVector<Value> mkn;
  const bool isBatchMmad = isa<BatchMmadL1Op>(matmulOp.getOperation());
  const size_t batchIndexBias = isBatchMmad ? 1 : 0;
  auto realMK =
      getRealShapeFromMemrefOrTensor(matmulOp.getMatmulA(), loc, rewriter);
  const int matrixSize = 2;
  const int matrixFractalSize = 4;
  if (failed(realMK) ||
      ((*realMK).size() != matrixSize + batchIndexBias &&
       (*realMK).size() != matrixFractalSize + batchIndexBias)) {
    return failure();
  }
  auto realKN =
      getRealShapeFromMemrefOrTensor(matmulOp.getMatmulB(), loc, rewriter);
  if (failed(realKN) ||
      ((*realKN).size() != matrixSize + batchIndexBias &&
       (*realKN).size() != matrixFractalSize + batchIndexBias)) {
    return failure();
  }
  // set m, k, n
  // TODO: m is set to be l1M for group gemm scenario (use kDotPadOnlyK),
  //       which should be enhanced.
  Value realM;
  // On a Fractal->ND convert_layout operand the transpose flag encodes the
  // fractal layout (nZ vs zN) rather than a 2D transpose, so it must not swap
  // m and k. Once FoldFractalVtransposePattern has absorbed a real vtranspose
  // into the flag it also marks the op, and the flag means a 2D transpose again.
  auto isFractalSpaceTranspose = [&](Value v) -> bool {
    auto convert = v.getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return false;
    if (convert.getSrcLayout().getDataLayout() != hivm::DataLayout::Fractal ||
        !convert.getDstLayout().isNDLayout())
      return false;
    return !matmulOp->hasAttr(kFractalVtransposeFolded);
  };
  if ((*realMK).size() == matrixFractalSize + batchIndexBias) {
    // Fractal 4D shape
    if (matmulOp.isMatmulATransposed()) {
      realM = rewriter.create<arith::MulIOp>(
          loc, (*realMK)[0 + batchIndexBias],
          (*realMK)[3 + batchIndexBias]);
    } else {
      realM = rewriter.create<arith::MulIOp>(
          loc, (*realMK)[1 + batchIndexBias],
          (*realMK)[2 + batchIndexBias]);
    }
  } else {
    // Original 2D shape logic
    if (matmulOp.isMatmulATransposed() &&
        !isFractalSpaceTranspose(matmulOp.getMatmulA())) {
      realM = (*realMK)[1 + batchIndexBias];
    } else {
      realM = (*realMK)[0 + batchIndexBias];
    }
  }

  if (utils::getAnnotateOpWithAttr(matmulOp.getMatmulA(), kDotPadOnlyK)
          .has_value()) {
    auto cType = dyn_cast<RankedTensorType>(matmulOp.getMatmulC().getType());
    if (cType && cType.hasStaticShape()) {
      const size_t l1MIdx = isBatchMmad ? 1 : 0;
      int64_t l1M = cType.getShape()[l1MIdx + batchIndexBias];
      realM = rewriter.create<arith::ConstantIndexOp>(loc, l1M);
    }
  }

  mkn.push_back(realM);
  if ((*realMK).size() == matrixFractalSize + batchIndexBias) {
    if (matmulOp.isMatmulATransposed()) {
      mkn.push_back(rewriter.create<arith::MulIOp>(
          loc, (*realMK)[1 + batchIndexBias],
          (*realMK)[2 + batchIndexBias]));
    } else {
      mkn.push_back(rewriter.create<arith::MulIOp>(
          loc, (*realMK)[0 + batchIndexBias],
          (*realMK)[3 + batchIndexBias]));
    }
  } else {
    if (matmulOp.isMatmulATransposed() &&
        !isFractalSpaceTranspose(matmulOp.getMatmulA())) {
      mkn.push_back((*realMK)[0 + batchIndexBias]);
    } else {
      mkn.push_back((*realMK)[1 + batchIndexBias]);
    }
  }
  if ((*realKN).size() == matrixFractalSize + batchIndexBias) {
    if (matmulOp.isMatmulBTransposed()) {
      mkn.push_back(rewriter.create<arith::MulIOp>(
          loc, (*realKN)[1 + batchIndexBias],
          (*realKN)[2 + batchIndexBias]));
    } else {
      mkn.push_back(rewriter.create<arith::MulIOp>(
          loc, (*realKN)[0 + batchIndexBias],
          (*realKN)[3 + batchIndexBias]));
    }
  } else {
    if (matmulOp.isMatmulBTransposed() &&
        !isFractalSpaceTranspose(matmulOp.getMatmulB())) {
      mkn.push_back((*realKN)[0 + batchIndexBias]);
    } else {
      mkn.push_back((*realKN)[1 + batchIndexBias]);
    }
  }
  return mkn;
}

struct SetRealMKNPattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface mmadLikeOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = mmadLikeOp.getOperation();
    if (op->hasAttr(kAlreadySetRealMKN))
      return rewriter.notifyMatchFailure(op, "Pattern already applied");

    auto mkn = extractRealMKN(mmadLikeOp, rewriter);
    if (failed(mkn))
      return rewriter.notifyMatchFailure(op, "Failed to extract mkn");

    // This pattern is intended to run only once. We clone the op and use
    // `GreedyRewriteStrictness::ExistingOps` to achieve this.
    Operation *newOp = rewriter.clone(*op);
    cast<LocalMatmulLikeOpInterface>(newOp).setMatmulRealMKN(
        (*mkn)[0], (*mkn)[1], (*mkn)[2]);
    rewriter.replaceOp(op, newOp);
    newOp->setAttr(kAlreadySetRealMKN, rewriter.getUnitAttr());
    return success();
  }
};

static bool tracesToLocalMatmulLike(Value v) {
  return traceDefOp<MmadL1Op>(v).has_value() ||
         traceDefOp<BatchMmadL1Op>(v).has_value() ||
         traceDefOp<MmadMxL1Op>(v).has_value();
}

static LocalMatmulLikeOpInterface cloneLocalMatmulLikeOp(
    PatternRewriter &rewriter, LocalMatmulLikeOpInterface op) {
  return cast<LocalMatmulLikeOpInterface>(rewriter.clone(*op.getOperation()));
}

LogicalResult decomposeMatmulWithElementwiseAdd(
    PatternRewriter &rewriter, LocalMatmulLikeOpInterface op) {
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getMatmulC());
  auto newMmad = cloneLocalMatmulLikeOp(rewriter, op);
  newMmad.setMatmulC(newMmadInit);
  Value constTrue =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 1);
  newMmad.setInitCondition(constTrue);
  auto addInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getMatmulC());
  auto addOp = rewriter.create<hivm::VAddOp>(
      op.getLoc(), TypeRange{newMmad.getOperation()->getResult(0).getType()},
      ValueRange{newMmad.getOperation()->getResult(0), op.getMatmulC()},
      ValueRange{addInit});

  rewriter.replaceOp(op.getOperation(), addOp.getResult());
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
LogicalResult decomposeMatmulWithConditionalElementwiseAdd(
    PatternRewriter &rewriter, LocalMatmulLikeOpInterface op) {
  Location loc = op.getLoc();
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, loc, op.getMatmulC());
  auto newMmad = cloneLocalMatmulLikeOp(rewriter, op);
  newMmad.setMatmulC(newMmadInit);
  Value constTrue =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 1);
  newMmad.setInitCondition(constTrue);

  auto ifOp = rewriter.create<scf::IfOp>(
      loc, newMmad.getOperation()->getResultTypes(),
      op.getMatmulInitCondition(),
      /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    rewriter.create<scf::YieldOp>(loc, newMmad.getOperation()->getResults());
  }
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    auto addInit = mlir::utils::createEmptyOp(rewriter, loc, op.getMatmulC());
    auto addOp = rewriter.create<hivm::VAddOp>(
        loc,
        TypeRange{newMmad.getOperation()->getResult(0).getType()},
        ValueRange{newMmad.getOperation()->getResult(0), op.getMatmulC()},
        ValueRange{addInit});
    rewriter.create<scf::YieldOp>(loc, addOp->getResults());
  }
  rewriter.replaceOp(op.getOperation(), ifOp.getResults());
  return success();
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
LogicalResult decomposeMatmulWithPerChannelAdd(PatternRewriter &rewriter,
                                               LocalMatmulLikeOpInterface op) {
  auto perChannelValue = getBiasInputForPerChannelAdd(op.getMatmulC());
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getMatmulC());
  auto newMmad = cloneLocalMatmulLikeOp(rewriter, op);
  newMmad.setMatmulC(newMmadInit);
  newMmad.setPerChannelBias(perChannelValue);
  Value constTrue =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 1);
  // reset init flag to true
  newMmad.setInitCondition(constTrue);
  rewriter.replaceOp(op.getOperation(), newMmad.getOperation());
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
LogicalResult decomposeMatmulWithPostPerChannelAddWithSplitKAdd(
    PatternRewriter &rewriter, LocalMatmulLikeOpInterface op) {
  auto matmulOutput = op.getMatmulC();
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
    } else if (tracesToLocalMatmulLike(addInputs[i])) {
      matmulInputIndex = i;
    }
  }
  if (brcInputIndex == -1 || matmulInputIndex == -1) {
    return failure();
  }

  auto perChannelVal = getBiasInputForPerChannelAdd(addInputs[brcInputIndex]);
  op.setPerChannelBias(perChannelVal);
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
LogicalResult decomposeMatmulWithMMInitPerChannelAddWithSplitK(
    PatternRewriter &rewriter, LocalMatmulLikeOpInterface op) {
  auto perChannelValue = getBiasInputForPerChannelAdd(op.getMatmulC());
  op.setPerChannelBias(perChannelValue);

  auto matmulOutput = op.getMatmulC();
  auto blockArg = dyn_cast_if_present<BlockArgument>(matmulOutput);
  assert(blockArg && "blockArg is not nullptr for mm init per channel split k");
  auto scfForOp =
      dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
  assert(scfForOp && "scfForOp is not nullptr for mm init per channel split k");

  rewriter.setInsertionPoint(op.getOperation());
  auto additionalCondition = rewriter.create<arith::CmpIOp>(
      op.getLoc(), arith::CmpIPredicate::eq, scfForOp.getLowerBound(),
      scfForOp.getInductionVar());
  op.setInitCondition(additionalCondition);

  rewriter.setInsertionPoint(scfForOp);
  auto newMmadInit =
      mlir::utils::createEmptyOp(rewriter, op.getLoc(), op.getMatmulC());
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

    // If the forOp result has annotation.mark {matmul_at_least_once},
    // the matmul inside is guaranteed to execute at least once, so we
    // can override mayNotExec to false regardless of loop bounds.
    auto maybeMarkOp =
        utils::getAnnotateOpWithAttr(info.outVal, "matmul_at_least_once");
    if (maybeMarkOp.has_value()) {
      info.mayNotExec = false;
    }
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

    // The res should only be used by yields in the if block.
    for (OpOperand &use : info.outVal.getUses()) {
      Operation *user = use.getOwner();
      auto yieldOp = dyn_cast<scf::YieldOp>(user);
      if (!yieldOp)
        return CCFInfo::getFailure(info);
      if (yieldOp->getBlock() != op->getBlock())
        return CCFInfo::getFailure(info);
      if (use.getOperandNumber() != resultIdx)
        return CCFInfo::getFailure(info);
    }

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

CCFInfo getResFromSingleUseChain(LocalMatmulLikeOpInterface op) {
  CCFInfo initInfo;
  initInfo.inVal = op.getMatmulC();
  initInfo.outVal = op.getOperation()->getResult(0);
  initInfo.insertPointOp = op.getOperation();
  return getOutermostCCFInfo(op.getOperation(), initInfo);
}

Value initCounter(PatternRewriter &rewriter, Operation &op) {
  rewriter.setInsertionPoint(&op);
  // Alloca + store 0 before the inner scf.for. Outer-loop body re-runs this
  // every outer iteration, so the counter resets per outer step.
  Location loc = op.getLoc();
  Value counterBuf = rewriter.create<memref::AllocaOp>(
      loc, MemRefType::get({}, rewriter.getI32Type()));
  // Mark the alloca so downstream passes can recognize it as the
  // normalize-matmul iteration counter.
  counterBuf.getDefiningOp()->setAttr(kNormalizeMatmulCounterAttr,
                                      rewriter.getUnitAttr());
  Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto storeOp = rewriter.create<memref::StoreOp>(loc, zeroI32, counterBuf,
                                                  ValueRange{});
  storeOp->setAttr(
      hivm::TCoreTypeAttr::name,
      hivm::TCoreTypeAttr::get(rewriter.getContext(),
                               hivm::TCoreType::CUBE_AND_VECTOR));
  return counterBuf;
}

Value updateInitCondition(PatternRewriter &rewriter,
                        LocalMatmulLikeOpInterface op, Value counterBuf) {
  rewriter.setInsertionPoint(op.getOperation());
  Location loc = op.getLoc();
  Value curCount =
      rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
  Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto firstIterCond = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, curCount, zeroI32);

  // In the same then-branch, right after matmul: counter += 1; store back.
  // Counter only advances on iterations where the scf.if condition fired,
  // which is exactly what the fallback below relies on.
  rewriter.setInsertionPointAfter(op.getOperation());
  Value oneI32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value nextCount = rewriter.create<arith::AddIOp>(loc, curCount, oneI32);
  rewriter.create<memref::StoreOp>(loc, nextCount, counterBuf, ValueRange{});
  return firstIterCond;
}

hivm::VAddOp createVadd(PatternRewriter &rewriter, Location loc, Type type,
                        Value operandL, Value operandR) {
  auto addInit = mlir::utils::createEmptyOp(rewriter, loc, operandL);
  auto addOp = rewriter.create<hivm::VAddOp>(loc, TypeRange{type},
                                             ValueRange{operandL, operandR},
                                             ValueRange{addInit});
  return addOp;
}

unsigned getResultIndex(Operation &op, Value val) {
  unsigned idx = 0;
  for (Value result : op.getResults()) {
    if (result == val)
      break;
    idx++;
  }
  return idx;
}

// Set kNormalizedInL0C attribute with proper index for scf::ForOp, scf::IfOp,
// or UnitAttr for other operations. If the attribute already exists, append
// the new index to the existing list instead of overwriting.
void setNormalizedInL0CWithIndex(PatternRewriter &rewriter,
                                 Operation &insertPointOp, Value ccfOutVal) {
  // skip the case that insertPointOp is mmad
  if (!isa<scf::ForOp, scf::IfOp>(insertPointOp))
    return;
  unsigned resultIdx = getResultIndex(insertPointOp, ccfOutVal);

  // Check if attribute already exists
  SmallVector<Attribute> indices;
  if (auto existingAttr =
          insertPointOp.getAttrOfType<ArrayAttr>(kNormalizedInL0C)) {
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

void setRemainInL0CAttr(PatternRewriter &rewriter, Value ccfInVal) {
  if (Operation *defOp = ccfInVal.getDefiningOp()) {
    defOp->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
  }
}

// True if `val` is the result of a CCF op that this pass already normalized to
// accumulate in L0C, i.e. its index is listed in the op's kNormalizedInL0C.
bool isCCFOpResultInL0C(Operation *defOp, Value val) {
  if (!defOp || !isa<scf::ForOp, scf::IfOp>(defOp))
    return false;
  unsigned resultIdx = getResultIndex(*defOp, val);
  if (auto attr = defOp->getAttrOfType<ArrayAttr>(kNormalizedInL0C)) {
    for (Attribute a : attr) {
      if (auto idxAttr = mlir::dyn_cast<IntegerAttr>(a))
        if (idxAttr.getInt() == static_cast<int64_t>(resultIdx))
          return true;
    }
  }
  return false;
}

// Locate the iteration counter that `initCounter` placed in front of `ccfOp`
// for the CCF result at `idx`. Scanning stops at the previous CCF so that
// counters belonging to an earlier loop are not picked up by mistake.
Value findCounterBufFor(Operation *ccfOp, unsigned idx) {
  for (Operation *prev = ccfOp->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (auto allocaOp = dyn_cast<memref::AllocaOp>(prev)) {
      auto attr =
          allocaOp->getAttrOfType<IntegerAttr>(kNormalizeMatmulCounterAttr);
      if (attr && attr.getInt() == static_cast<int64_t>(idx))
        return allocaOp.getResult();
    }
    if (isa<scf::ForOp, scf::IfOp>(prev))
      break;
  }
  return Value();
}

// A following CCF or bare mmad clears L0C through its own init condition, so
// whatever this CCF leaves behind is never read and no zero fill is needed. The
// guard only matters when the chain ends here, e.g. at a store or a return.
bool needsTailFallback(Value ccfOutVal) {
  if (!ccfOutVal.hasOneUse())
    return true;
  Operation *user = *ccfOutVal.getUsers().begin();
  if (auto nextForOp = dyn_cast<scf::ForOp>(user)) {
    // Check if the init arg corresponding to ccfOutVal is in L0C
    // (kNormalizedInL0C records which result indices are in L0C)
    for (unsigned i = 0; i < nextForOp.getInitArgs().size(); ++i) {
      if (nextForOp.getInitArgs()[i] == ccfOutVal &&
          isCCFOpResultInL0C(user, nextForOp.getResult(i))) {
        return false; // confirmed: next CCF handles it
      }
    }
  }
  return !isa<hivm::MmadL1Op, hivm::BatchMmadL1Op, hivm::MmadMxL1Op>(user);
}

// Build an i1 telling whether the L0C block feeding `ccfInVal` still holds data
// that has to be cleared: true means dirty, false means a producer definitely
// wrote it. A constant is returned whenever the answer is known statically,
// which lets the callers drop the runtime check entirely.
Value getOrCreateCounterPrevious(PatternRewriter &rewriter, Value ccfInVal,
                                 Location loc) {
  Operation *defOp = ccfInVal.getDefiningOp();
  if (!defOp)
    return rewriter.create<arith::ConstantIntOp>(loc, 1, 1);

  if (isa<scf::ForOp, scf::IfOp>(defOp)) {
    const bool isForOp = isa<scf::ForOp>(defOp);

    // Not normalized: nothing accumulated in L0C, so treat it as dirty.
    if (!defOp->hasAttr(kNormalizedInL0C))
      return rewriter.create<arith::ConstantIntOp>(loc, 1, 1);

    // A for loop without kMayNotExec always ran, so its mmad wrote L0C.
    if (isForOp && !defOp->hasAttr(kMayNotExec))
      return rewriter.create<arith::ConstantIntOp>(loc, 0, 1);

    // Otherwise the producer may not have run and we need its counter.
    unsigned idx = getResultIndex(*defOp, ccfInVal);
    Value counterBuf = findCounterBufFor(defOp, idx);
    if (!counterBuf)
      return rewriter.create<arith::ConstantIntOp>(loc, 1, 1);

    // Reuse the predicate already built for this counter, if any.
    for (Operation *next = defOp->getNextNode(); next;
         next = next->getNextNode()) {
      if (auto andOp = dyn_cast<arith::AndIOp>(next)) {
        if (andOp->hasAttr(kCounterPrevious)) {
          for (Value operand : andOp->getOperands()) {
            if (auto cmpOp = operand.getDefiningOp<arith::CmpIOp>())
              if (auto loadOp = cmpOp.getLhs().getDefiningOp<memref::LoadOp>())
                if (loadOp.getMemRef() == counterBuf)
                  return andOp.getResult();
          }
        }
      }
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(next)) {
        if (cmpOp->hasAttr(kCounterPrevious))
          if (auto loadOp = cmpOp.getLhs().getDefiningOp<memref::LoadOp>())
            if (loadOp.getMemRef() == counterBuf)
              return cmpOp.getResult();
      }
      if (isa<scf::ForOp, scf::IfOp>(next))
        break;
    }

    // The L0C is dirty only if it was already dirty coming in *and* this
    // producer did not run. For an scf.if we cannot tell which block yields the
    // value, so stay conservative.
    Value counterPrevIn;
    if (isForOp) {
      auto forOp = cast<scf::ForOp>(defOp);
      if (idx < forOp.getInitArgs().size())
        counterPrevIn =
            getOrCreateCounterPrevious(rewriter, forOp.getInitArgs()[idx], loc);
      else
        counterPrevIn = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    } else {
      counterPrevIn = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    }
    if (matchPattern(counterPrevIn, m_Zero()))
      return rewriter.create<arith::ConstantIntOp>(loc, 0, 1);

    rewriter.setInsertionPointAfter(defOp);
    Value postCount =
        rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto neverRan = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, postCount, zeroI32);
    if (matchPattern(counterPrevIn, m_One())) {
      neverRan->setAttr(kCounterPrevious, rewriter.getUnitAttr());
      return neverRan.getResult();
    }
    auto counterPrevOut =
        rewriter.create<arith::AndIOp>(loc, counterPrevIn, neverRan);
    counterPrevOut->setAttr(kCounterPrevious, rewriter.getUnitAttr());
    return counterPrevOut.getResult();
  }

  // A bare mmad always writes L0C.
  if (isa<hivm::MmadL1Op, hivm::BatchMmadL1Op, hivm::MmadMxL1Op>(defOp))
    return rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
  return rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
}

void addTailFallback(PatternRewriter &rewriter, Operation &op,
                     LocalMatmulLikeOpInterface mmad, Value counterBuf,
                     Value ccfInVal, Value ccfOutVal, bool isAdd = false) {
  rewriter.setInsertionPointAfter(&op);
  Location loc = op.getLoc();
  Value postCount =
      rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
  Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value neverRan = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  postCount, zeroI32);

  auto fallbackIf = rewriter.create<scf::IfOp>(
      loc, mmad.getOperation()->getResultTypes(), neverRan,
      /*withElseRegion=*/true);

  // Then block
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(fallbackIf.thenBlock());
    if (auto brcOp = ccfInVal.getDefiningOp<hivm::VBrcOp>()) {
      auto newVbrcOp =
          cast<hivm::VBrcOp>(rewriter.clone(*brcOp.getOperation()));
      rewriter.create<scf::YieldOp>(loc, ValueRange{newVbrcOp->getResult(0)});
    } else {
      rewriter.create<scf::YieldOp>(loc, ValueRange{ccfInVal});
    }
  }

  // Else block
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(fallbackIf.elseBlock());
    if (isAdd) {
      auto addOp =
          createVadd(rewriter, loc, ccfOutVal.getType(), ccfInVal, ccfOutVal);
      rewriter.create<scf::YieldOp>(loc, ValueRange{addOp.getResults()[0]});
    } else {
      rewriter.create<scf::YieldOp>(loc, ValueRange{ccfOutVal});
    }
  }

  // replace user of output of CCF
  rewriter.replaceUsesWithIf(ccfOutVal, fallbackIf.getResult(0),
                             [&](OpOperand &operand) {
                               Operation *userOp = operand.getOwner();
                               return !fallbackIf->isAncestor(userOp);
                             });
}

// Check if previous L0C could be reuse
bool couldReuse(Value ccfInVal) {
  // Check the user of previous L0C is only one
  if (!ccfInVal.hasOneUse())
    return false;

  // check the user of previous L0C is mmadL1
  if (auto mmadOp = ccfInVal.getDefiningOp<hivm::MmadL1Op>()) {
    return true;
  }
  if (auto mmadMxOp = ccfInVal.getDefiningOp<hivm::MmadMxL1Op>()) {
    return true;
  }

  // check the user of previous L0C is output from mmadL1 in CCF. A CCF that may
  // not have run is still reusable, because the init condition derived from
  // getOrCreateCounterPrevious clears L0C on the paths where it did not.
  if (Operation *defOp = ccfInVal.getDefiningOp()) {
    if (!isa<scf::ForOp, scf::IfOp>(defOp))
      return false;
    return isCCFOpResultInL0C(defOp, ccfInVal);
  }

  return false;
}

struct BrcBiasInfo {
  MatmulBiasMode brcBiasMode = MatmulBiasMode::NoBias;
  Value perChannelValue;
  hivm::VAddOp addOp;
  Value biasBrcSrc;
  Value biasVbrcResult;
};

// Match PostPerChannelAddWithSplitK: ccfOutVal has one use that is VAddOp
// with a perChannel VBrcOp source. Covers:
//   vbrc(x,[0]) + vadd, empty() + vadd, vbrc(0) + vadd
static std::optional<BrcBiasInfo>
matchPostPerChannelAddWithSplitK(Value ccfOutVal, Operation *hookOp) {
  if (!ccfOutVal.hasOneUse())
    return std::nullopt;
  auto addOp = dyn_cast<hivm::VAddOp>(*ccfOutVal.getUsers().begin());
  if (!addOp)
    return std::nullopt;
  for (Value src : addOp.getSrc()) {
    if (auto brcOp = src.getDefiningOp<hivm::VBrcOp>()) {
      if (isSatisfiedBrcForPerChannel(brcOp, hookOp)) {
        BrcBiasInfo result;
        result.perChannelValue = getBiasInputForPerChannelAdd(src);
        result.addOp = addOp;
        result.biasBrcSrc = brcOp.getSrc();
        result.biasVbrcResult = src;
        result.brcBiasMode = MatmulBiasMode::PostPerChannelAddWithSplitK;
        return result;
      }
    }
  }
  return std::nullopt;
}

// Get the Bias mode — rules tried in priority order, first match wins:
//   1. PerChannelAdd:          ccfInVal = Vbrc(x, [0])
//   2. PostPerChannel(vbrc0):  ccfInVal = Vbrc(0) with perChannel vadd
//   3. ZeroInitNoAccumulation: ccfInVal = Vbrc(0) without vadd
//   4. ReuseL0C:               ccfInVal is reusable L0C from previous mmad
//                              TODO: mayNotExecWithIf should be removed
//   5. PostPerChannel(empty):  ccfInVal traces to empty() with perChannel vadd
//   6. NoBias:                 ccfInVal traces to empty() without vadd
//   7. ElementwiseAdd:         fallback x[n, m]
BrcBiasInfo getBrcBiasMode(CCFInfo ccfinfo) {
  // refer to getMatmulLikeBiasMode
  Value ccfInVal = ccfinfo.inVal;
  Value ccfOutVal = ccfinfo.outVal;
  if (auto brcOp = ccfInVal.getDefiningOp<hivm::VBrcOp>()) {
    if (isSatisfiedBrcForPerChannel(brcOp)) {
      BrcBiasInfo info;
      info.perChannelValue = getBiasInputForPerChannelAdd(ccfInVal);
      info.brcBiasMode = MatmulBiasMode::PerChannelAdd;
      return info;
    }
    if (isConstZero(brcOp.getSrc())) {
      auto postPerChannel =
          matchPostPerChannelAddWithSplitK(ccfOutVal, ccfinfo.insertPointOp);
      if (postPerChannel)
        return *postPerChannel;
      BrcBiasInfo info;
      info.brcBiasMode = MatmulBiasMode::ZeroInitNoAccumulation;
      return info;
    }
  }

  if (couldReuse(ccfInVal) && (!ccfinfo.mayNotExecWithIf)) {
    BrcBiasInfo info;
    info.brcBiasMode = MatmulBiasMode::ReuseL0C;
    return info;
  }

  auto emptyOps =
      traceDefOps<tensor::EmptyOp>(ccfInVal,
                                   /*isSingleChain=*/false,
                                   /*traceMode=*/TraceResultMode::StrictSame);
  if (!emptyOps.empty()) {
    auto postPerChannel =
        matchPostPerChannelAddWithSplitK(ccfOutVal, ccfinfo.insertPointOp);
    if (postPerChannel)
      return *postPerChannel;
    BrcBiasInfo info;
    info.brcBiasMode = MatmulBiasMode::NoBias;
    return info;
  }

  BrcBiasInfo info;
  info.brcBiasMode = MatmulBiasMode::ElementwiseAdd;
  return info;
}

// Add counter and if block in the tail for the case that mmad is probably not
// executed
struct NormalizeCtx {
  LocalMatmulLikeOpInterface op;
  Value ccfInVal;
  Value ccfOutVal;
  Operation *insertPointOp = nullptr;
  BlockArgument blockArg;
  BrcBiasInfo biasInfo;
  bool mayNotExec = false;
  Value counterBuf;
  Value counterPrevious;
  Value newInit;
  LocalMatmulLikeOpInterface tmpNewMmad;
  Value initCondition;
  bool isUsingCounter = false;
};

class BiasModeStrategy {
public:
  virtual ~BiasModeStrategy() = default;
  virtual LogicalResult handle(PatternRewriter &rewriter,
                               NormalizeCtx &ctx) = 0;
};

static void setupCCFInitCondition(PatternRewriter &rewriter,
                                  NormalizeCtx &ctx) {
  ctx.initCondition = updateInitCondition(rewriter, ctx.op, ctx.counterBuf);
  ctx.isUsingCounter = true;
  if (!isa<scf::IfOp>(ctx.op->getParentOp())) {
    ctx.tmpNewMmad.getOperation()->setAttr(hivm::RemainInL0CAttr::name,
                                           rewriter.getUnitAttr());
    ctx.op->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
  }
}

class ReuseL0CStrategy : public BiasModeStrategy {
public:
  LogicalResult handle(PatternRewriter &rewriter, NormalizeCtx &ctx) override {
    if (ctx.insertPointOp != ctx.op) {
      ctx.op->setAttr(kNormalizedInL0C, rewriter.getUnitAttr());
      LDBG("ReuseL0C in for-loop or if: no need to decompose matmul");
      setRemainInL0CAttr(rewriter, ctx.ccfInVal);

      if (matchPattern(ctx.counterPrevious, m_Zero())) {
        // counterPrevious is compile-time false → L0C dirty, reuse directly.
        // Skip counter creation (updateInitCondition) since initCondition is
        // statically false; no need to track execution count. The next CCF's
        // counterPrevIn will also trace to a dirty source (matmul/normalized
        // ForOp), so it short-circuits to false regardless of this counter.
        if (!isa<scf::IfOp>(ctx.op->getParentOp())) {
          ctx.tmpNewMmad.getOperation()->setAttr(hivm::RemainInL0CAttr::name,
                                                 rewriter.getUnitAttr());
          ctx.op->setAttr(hivm::RemainInL0CAttr::name, rewriter.getUnitAttr());
        }
        ctx.op.setInitCondition(ctx.counterPrevious);
      } else {
        setupCCFInitCondition(rewriter, ctx);
        rewriter.setInsertionPoint(ctx.op);
        ctx.initCondition = rewriter.create<arith::AndIOp>(
            ctx.op->getLoc(), ctx.counterPrevious, ctx.initCondition);
        ctx.op.setInitCondition(ctx.initCondition);
        if (ctx.mayNotExec) {
          ctx.insertPointOp->setAttr(kMayNotExec, rewriter.getUnitAttr());
          ctx.op->setAttr(kDeferredTailFallback, rewriter.getUnitAttr());
        }
      }
      setNormalizedInL0CWithIndex(rewriter, *ctx.insertPointOp, ctx.ccfOutVal);
      return success();
    }
    LDBG("ReuseL0C bare mmad: initCondition = counterPrevious");
    ctx.op->setAttr(kNormalizedInL0C, rewriter.getUnitAttr());
    setRemainInL0CAttr(rewriter, ctx.ccfInVal);
    ctx.op.setInitCondition(ctx.counterPrevious);
    return success();
  }
};

class DecomposeStrategyBase : public BiasModeStrategy {
public:
  virtual ~DecomposeStrategyBase() = default;
  LogicalResult handle(PatternRewriter &rewriter, NormalizeCtx &ctx) override {
    if (ctx.insertPointOp != ctx.op) {
      setupCCFInitCondition(rewriter, ctx);
      LDBG("initCondition using counter");
    } else {
      rewriter.setInsertionPoint(ctx.insertPointOp);
      ctx.initCondition =
          rewriter.create<arith::ConstantIntOp>(ctx.op->getLoc(), 1, 1);
      ctx.mayNotExec = false;
      LDBG("initCondition always true");
    }
    ctx.tmpNewMmad.setInitCondition(ctx.initCondition);

    if (failed(normalizeInitArg(rewriter, ctx)))
      return failure();

    mergeBias(rewriter, ctx);

    ctx.tmpNewMmad.getOperation()->setAttr(kNormalizedInL0C,
                                           rewriter.getUnitAttr());
    setNormalizedInL0CWithIndex(rewriter, *ctx.insertPointOp, ctx.ccfOutVal);
    if (ctx.mayNotExec)
      ctx.insertPointOp->setAttr(kMayNotExec, rewriter.getUnitAttr());

    handleTailFallback(rewriter, ctx);

    finalize(rewriter, ctx);
    return success();
  }

protected:
  virtual LogicalResult normalizeInitArg(PatternRewriter &rewriter,
                                         NormalizeCtx &ctx) {
    if (ctx.insertPointOp != ctx.op && ctx.blockArg) {
      auto forOp = dyn_cast<scf::ForOp>(ctx.blockArg.getOwner()->getParentOp());
      if (forOp) {
        auto blockArgIdx = ctx.blockArg.getArgNumber() - 1;
        forOp.getInitArgsMutable()[blockArgIdx].assign(ctx.newInit);
      } else {
        return rewriter.notifyMatchFailure(
            ctx.op, "expected the outermost init arg to be a block argument "
                    "of a for op");
      }
    } else {
      ctx.tmpNewMmad.setMatmulC(ctx.newInit);
    }
    return success();
  }

  virtual void mergeBias(PatternRewriter &rewriter, NormalizeCtx &ctx) {}

  virtual void handleTailFallback(PatternRewriter &rewriter,
                                  NormalizeCtx &ctx) = 0;

  static void finalize(PatternRewriter &rewriter, NormalizeCtx &ctx) {
    Operation *lastOperandDef = ctx.initCondition.getDefiningOp();
    for (auto operand : ctx.tmpNewMmad->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (defOp->getBlock() != lastOperandDef->getBlock())
        continue;
      if (lastOperandDef->isBeforeInBlock(defOp))
        lastOperandDef = defOp;
    }
    rewriter.setInsertionPointAfter(lastOperandDef);
    auto newMmad = cloneLocalMatmulLikeOp(rewriter, ctx.tmpNewMmad);
    rewriter.eraseOp(ctx.tmpNewMmad);
    rewriter.replaceOp(ctx.op, newMmad);
  }
};

class NoBiasStrategy : public DecomposeStrategyBase {
  LogicalResult normalizeInitArg(PatternRewriter &, NormalizeCtx &) override {
    return success();
  }
  void handleTailFallback(PatternRewriter &rewriter,
                          NormalizeCtx &ctx) override {
    if (ctx.mayNotExec && !isa<scf::IfOp>(ctx.insertPointOp))
      ctx.tmpNewMmad.getOperation()->setAttr(kDeferredTailFallback,
                                             rewriter.getUnitAttr());
  }
};

class ZeroInitStrategy : public DecomposeStrategyBase {
  void handleTailFallback(PatternRewriter &rewriter,
                          NormalizeCtx &ctx) override {
    if (!ctx.isUsingCounter) {
      LDBG("decompose matmul with zero init no accumulation");
      return;
    }
    // if-only ZeroInit: the else block already yields the zero bias
    // (vbrc(0)), which is exactly the fallback value AddIfPattern would
    // create. No deferred tail fallback IfOp needed.
    if (ctx.mayNotExec && !isa<scf::IfOp>(ctx.insertPointOp))
      ctx.tmpNewMmad.getOperation()->setAttr(kDeferredTailFallback,
                                             rewriter.getUnitAttr());
  }
};

class ElementwiseAddStrategy : public DecomposeStrategyBase {
  void handleTailFallback(PatternRewriter &rewriter,
                          NormalizeCtx &ctx) override {
    if (ctx.mayNotExec) {
      addTailFallback(rewriter, *ctx.insertPointOp, ctx.tmpNewMmad,
                      ctx.counterBuf, ctx.ccfInVal, ctx.ccfOutVal,
                      /*isAdd=*/true);
    } else {
      rewriter.setInsertionPointAfter(ctx.insertPointOp);
      Location loc = ctx.insertPointOp->getLoc();
      auto addOp = createVadd(rewriter, loc,
                              ctx.insertPointOp->getResults()[0].getType(),
                              ctx.insertPointOp->getResults()[0], ctx.ccfInVal);
      mlir::DominanceInfo domInfo(ctx.op->getParentOp());
      for (auto &use : llvm::make_early_inc_range(ctx.ccfOutVal.getUses())) {
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
  }
};

class PerChannelAddStrategy : public DecomposeStrategyBase {
  void mergeBias(PatternRewriter &rewriter, NormalizeCtx &ctx) override {
    ctx.tmpNewMmad.setPerChannelBias(ctx.biasInfo.perChannelValue);
    if (ctx.biasInfo.addOp) {
      rewriter.replaceAllUsesWith(ctx.biasInfo.addOp->getResults()[0],
                                  ctx.insertPointOp->getResults()[0]);
    }
    ctx.tmpNewMmad->setAttr(kNormalizedInitOrBias, rewriter.getUnitAttr());
  }
  void handleTailFallback(PatternRewriter &rewriter,
                          NormalizeCtx &ctx) override {
    if (ctx.mayNotExec && !isa<scf::IfOp>(ctx.insertPointOp))
      addTailFallback(rewriter, *ctx.insertPointOp, ctx.tmpNewMmad,
                      ctx.counterBuf, ctx.ccfInVal, ctx.ccfOutVal);
    LDBG("decompose matmul with other cases");
  }
};

class PostPerChannelAddStrategy : public DecomposeStrategyBase {
  void mergeBias(PatternRewriter &rewriter,
                 NormalizeCtx &ctx) override {
    ctx.tmpNewMmad.setPerChannelBias(
        ctx.biasInfo.perChannelValue);
    if (ctx.biasInfo.addOp) {
      rewriter.replaceAllUsesWith(ctx.biasInfo.addOp->getResults()[0],
                                  ctx.ccfOutVal);
    }
    ctx.tmpNewMmad->setAttr(kNormalizedInitOrBias, rewriter.getUnitAttr());
  }
  void handleTailFallback(PatternRewriter &rewriter,
                          NormalizeCtx &ctx) override {
    if (!ctx.mayNotExec)
      return;
    // Insert fallback IfOp after the original vbrc (not after the CCF) so
    // that the vbrc and its src def-chain (vcast/expand_shape) dominate the
    // fallback then-block where the vbrc is cloned.
    addTailFallback(rewriter, *ctx.biasInfo.biasVbrcResult.getDefiningOp(),
                    ctx.tmpNewMmad, ctx.counterBuf,
                    ctx.biasInfo.biasVbrcResult, ctx.ccfOutVal);
    LDBG("decompose matmul with post per channel add");
  }
};


struct NormalizeMmadCCFPattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    Operation *mmadOp = op.getOperation();
    // TODO: need to be reverted when Affinity GMM supported
    auto moduleOp = mmadOp->getParentOfType<ModuleOp>();
    bool isDisableHfusionVectorize = false;
    if (moduleOp) {
      isDisableHfusionVectorize =
          moduleOp->hasAttr("hfusion.disableHfusionVectorize");
    }

    auto scopeOp = mmadOp->getParentOfType<scope::ScopeOp>();
    if ((scopeOp && !scopeOp->hasAttr(hivm::MatmulLimitedInCubeAttr::name)) ||
        isDisableHfusionVectorize) {
      LDBG("Affinity pattern already applied");
      return rewriter.notifyMatchFailure(mmadOp,
                                         "Affinity pattern already applied");
    }

    if (mmadOp->hasAttr(kNormalizedInL0C)) {
      LDBG("Pattern already applied");
      return rewriter.notifyMatchFailure(mmadOp, "Pattern already applied");
    }

    if (!matchPattern(op.getMatmulInitCondition(), m_Zero())) {
      LDBG("Init condition is not zero");
      return rewriter.notifyMatchFailure(mmadOp, "Init condition is not zero");
    }
    // Check if in CCF
    if (!mmadOp->getParentOfType<scf::ForOp>() &&
        !mmadOp->getParentOfType<scf::IfOp>()) {
      LDBG("Not in CCF");
    } else {
      LDBG("In CCF");
    }

    // get Mmad Pattern: isSingleUseChain blockOp
    auto ccfInfo = getResFromSingleUseChain(op);
    // If upper user enssure matmul limited in cube, we need to set mayNotExec
    // to false to avoid adding if condition.
    if (auto parantop = mmadOp->getParentOp()) {
      if (parantop->hasAttr(hivm::MatmulLimitedInCubeAttr::name)) {
        ccfInfo.mayNotExec = false;
        ccfInfo.mayNotExecWithIf = false;
      }
    }

    // Erase the annotation.mark {matmul_at_least_once} after consuming it.
    if (auto markOp = utils::getAnnotateOpWithAttr(ccfInfo.outVal,
                                                   "matmul_at_least_once"))
      rewriter.eraseOp(*markOp);

    // get bias Info
    BrcBiasInfo biasInfo = getBrcBiasMode(ccfInfo);
    LDBG("BiasMode:" << biasInfo.brcBiasMode);
    LDBG("skipOptimize:" << ccfInfo.isFailure);
    LDBG("mayNotExec:" << ccfInfo.mayNotExec);

    // create counter buffer
    NormalizeCtx ctx;
    ctx.op = op;
    ctx.ccfInVal = ccfInfo.inVal;
    ctx.ccfOutVal = ccfInfo.outVal;
    ctx.insertPointOp = ccfInfo.insertPointOp;
    ctx.blockArg = ccfInfo.blockArg;
    ctx.biasInfo = biasInfo;
    ctx.mayNotExec = ccfInfo.mayNotExec;

    if (ccfInfo.insertPointOp != mmadOp) {
      ctx.counterBuf = initCounter(rewriter, *ctx.insertPointOp);
      if (isa<scf::ForOp, scf::IfOp>(ctx.insertPointOp)) {
        ctx.counterBuf.getDefiningOp()->setAttr(
            kNormalizeMatmulCounterAttr,
            rewriter.getI32IntegerAttr(
                getResultIndex(*ctx.insertPointOp, ctx.ccfOutVal)));
      }
    }
    ctx.counterPrevious =
        getOrCreateCounterPrevious(rewriter, ctx.ccfInVal, op->getLoc());

    // create new mmad op
    ctx.newInit = mlir::utils::createEmptyOp(
        rewriter, ctx.insertPointOp->getLoc(), ctx.ccfInVal);
    rewriter.setInsertionPointAfter(op);
    ctx.tmpNewMmad = cloneLocalMatmulLikeOp(rewriter, op);

    static ReuseL0CStrategy sReuseL0C;
    static NoBiasStrategy sNoBias;
    static ZeroInitStrategy sZeroInit;
    static ElementwiseAddStrategy sElemAdd;
    static PerChannelAddStrategy sPerChannel;
    static PostPerChannelAddStrategy sPostPerChannel;

    BiasModeStrategy *strategy = nullptr;
    switch (biasInfo.brcBiasMode) {
    case MatmulBiasMode::ReuseL0C:
      strategy = &sReuseL0C;
      break;
    case MatmulBiasMode::NoBias:
      strategy = &sNoBias;
      break;
    case MatmulBiasMode::ZeroInitNoAccumulation:
      strategy = &sZeroInit;
      break;
    case MatmulBiasMode::ElementwiseAdd:
      strategy = &sElemAdd;
      break;
    case MatmulBiasMode::PerChannelAdd:
      strategy = &sPerChannel;
      break;
    case MatmulBiasMode::PostPerChannelAddWithSplitK:
      strategy = &sPostPerChannel;
      break;
    default:
      return rewriter.notifyMatchFailure(op, "unknown bias mode");
    }
    return strategy->handle(rewriter, ctx);
  }
};

// Emits the tail fallback that NormalizeMmadCCFPattern deferred: when a CCF may
// not execute, its L0C result has to be zeroed before anyone reads it. Running
// as a separate pattern means the CCF's consumer is already visible, so the
// guard can be skipped whenever the consumer clears L0C by itself.
struct ReuseL0CAddIfPattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    Operation *mmadOp = op.getOperation();
    if (!mmadOp->hasAttr(kDeferredTailFallback))
      return failure();

    auto ccfInfo = getResFromSingleUseChain(op);
    Value ccfInVal = ccfInfo.inVal;
    Value ccfOutVal = ccfInfo.outVal;
    Operation *insertPointOp = ccfInfo.insertPointOp;
    Location loc = op.getLoc();

    // Drop the marker up front. Everything below only decides whether a guard
    // is needed, and no path may report failure once ops have been created.
    rewriter.modifyOpInPlace(
        mmadOp, [&]() { mmadOp->removeAttr(kDeferredTailFallback); });

    if (!needsTailFallback(ccfOutVal))
      return success();
    if (!isa<scf::ForOp, scf::IfOp>(insertPointOp))
      return success();
    Value counterBuf = findCounterBufFor(
        insertPointOp, getResultIndex(*insertPointOp, ccfOutVal));
    if (!counterBuf)
      return success();

    // L0C provably holds real data, so there is nothing to clear.
    Value counterPrevious = getOrCreateCounterPrevious(rewriter, ccfInVal, loc);
    if (matchPattern(counterPrevious, m_Zero()))
      return success();

    rewriter.setInsertionPointAfter(insertPointOp);
    Value postCount =
        rewriter.create<memref::LoadOp>(loc, counterBuf, ValueRange{});
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value neverRan = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, postCount, zeroI32);
    Value ifCond = neverRan;
    if (!matchPattern(counterPrevious, m_One()))
      ifCond = rewriter.create<arith::AndIOp>(loc, counterPrevious, neverRan);

    auto fallbackIf = rewriter.create<scf::IfOp>(
        loc, TypeRange{ccfOutVal.getType()}, ifCond, /*withElseRegion=*/true);

    // Then block: the CCF never ran, so materialize an explicit zero.
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(fallbackIf.thenBlock());
      auto shapedType = cast<ShapedType>(ccfOutVal.getType());
      Type elementType = shapedType.getElementType();
      Value zeroVal = rewriter.create<arith::ConstantOp>(
          loc, cast<TypedAttr>(rewriter.getZeroAttr(elementType)));
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, shapedType.getShape(), elementType);
      auto vbrcZero = rewriter.create<hivm::VBrcOp>(
          loc, TypeRange{ccfOutVal.getType()}, zeroVal, emptyTensor,
          rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
      rewriter.create<scf::YieldOp>(loc, ValueRange{vbrcZero.getResult()[0]});
    }

    // Else block: the accumulated result is valid.
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(fallbackIf.elseBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{ccfOutVal});
    }

    rewriter.replaceUsesWithIf(ccfOutVal, fallbackIf.getResult(0),
                               [&](OpOperand &operand) {
                                 Operation *userOp = operand.getOwner();
                                 return !fallbackIf->isAncestor(userOp);
                               });
    return success();
  }
};

struct DecomposeMatmulWithBiasPattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    Operation *mmadOp = op.getOperation();
    if (mmadOp->hasAttr(kNormalizedInitOrBias) ||
        mmadOp->hasAttr(kNormalizedInL0C)) {
      LDBG("Pattern already applied");
      return rewriter.notifyMatchFailure(mmadOp, "Pattern already applied");
    }

    // PostPerChannel / MMInit folding leaves a per-channel bias operand; do not
    // further decompose as elementwise add on the loop-carried C.
    if (!isa<MmadMxL1Op>(mmadOp)) {
      if (auto mmad = dyn_cast<MmadL1Op>(mmadOp)) {
        if (mmad.getPerChannelBias()) {
          return rewriter.notifyMatchFailure(mmadOp,
                                             "already has per-channel bias");
        }
      } else if (auto batchMmad = dyn_cast<BatchMmadL1Op>(mmadOp)) {
        if (batchMmad.getPerChannelBias()) {
          return rewriter.notifyMatchFailure(mmadOp,
                                             "already has per-channel bias");
        }
      }
    }

    MatmulBiasMode biasMode = op.getMatmulBiasMode();
    if (biasMode == MatmulBiasMode::NoBias) {
      LDBG("no bias");
      return rewriter.notifyMatchFailure(mmadOp, "no bias");
    }
    if (op.shouldDecomposeBiasByElementAdd() && op.isInitConstant(true)) {
      LDBG("no need to decompose matmul with elemwise add since init is true");
      return failure();
    }
    if (op.shouldDecomposeBiasByElementAdd() && op.isInitConstant(false)) {
      LDBG("decompose matmul with elemwise add");
      return decomposeMatmulWithElementwiseAdd(rewriter, op);
    }
    if (biasMode == MatmulBiasMode::PerChannelAdd) {
      LDBG("decompose matmul with per channel add");
      return decomposeMatmulWithPerChannelAdd(rewriter, op);
    }
    if (biasMode == MatmulBiasMode::PostPerChannelAddWithSplitK) {
      LDBG("decompose matmul with post per channel add with split k add");
      return decomposeMatmulWithPostPerChannelAddWithSplitKAdd(rewriter, op);
    }
    if (biasMode == MatmulBiasMode::MMInitPerChannelAddWithSplitK) {
      LDBG("decompose matmul with mm init per channel add with split k add");
      return decomposeMatmulWithMMInitPerChannelAddWithSplitK(rewriter, op);
    }

    Value mmadResult = mmadOp->getResult(0);
    if (scope::utils::isInCubeScope(mmadOp) && hasDebugUse(mmadResult)) {
      return failure();
    }

    if (op.shouldDecomposeBiasByElementAdd() &&
        !op.isInitConstant(std::nullopt)) {
      LDBG("decompose matmul with elemwise add and non-const init");
      return decomposeMatmulWithConditionalElementwiseAdd(rewriter, op);
    }

    return failure();
  }
};

/// The a_transpose/b_transpose flags transpose the two innermost axes of the
/// operand while loading it, so only a vtranspose with that exact permutation
/// can be folded away. hivm.hir.vtranspose merely requires *some* two axes to
/// be swapped, so permutations such as [2, 1, 0] on a rank-3 operand are legal
/// but not representable by the flag.
bool isInnermostAxisSwap(ArrayRef<int64_t> permutation) {
  const int64_t rank = static_cast<int64_t>(permutation.size());
  if (rank < 2)
    return false;
  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    if (permutation[dim] != dim)
      return false;
  }
  return permutation[rank - 2] == rank - 1 && permutation[rank - 1] == rank - 2;
}

/// True if `src` has the shape `operand` would have before its two innermost
/// axes were swapped. Folding walks through view-like ops to reach the
/// vtranspose but then feeds its source straight to the mmad, so this rejects
/// the cases where an intervening view was silently dropped.
bool isSwappedInnermostShape(Value src, Value operand) {
  auto srcType = dyn_cast<ShapedType>(src.getType());
  auto operandType = dyn_cast<ShapedType>(operand.getType());
  if (!srcType || !operandType || !srcType.hasRank() || !operandType.hasRank())
    return false;
  const int64_t rank = operandType.getRank();
  if (srcType.getRank() != rank || rank < 2)
    return false;
  SmallVector<int64_t> expected(operandType.getShape());
  std::swap(expected[rank - 2], expected[rank - 1]);
  return succeeded(verifyCompatibleShape(srcType.getShape(), expected));
}

/// Look through ViewLike ops for a foldable VTransposeOp; stops at
/// ConvertLayoutOp, which is handled by FoldFractalVtransposePattern.
hivm::VTransposeOp lookThroughViewLikes(Value v) {
  Value cur = v;
  while (true) {
    if (auto vtrans = cur.getDefiningOp<hivm::VTransposeOp>())
      return isInnermostAxisSwap(vtrans.getPermutation()) ? vtrans
                                                          : hivm::VTransposeOp();
    if (cur.getDefiningOp<ConvertLayoutOp>())
      return hivm::VTransposeOp();
    auto viewOp = cur.getDefiningOp<ViewLikeOpInterface>();
    if (!viewOp)
      return hivm::VTransposeOp();
    cur = viewOp->getOperand(0);
  }
}

/// Absorb a vtranspose feeding A or B into the matching transpose flag, so the
/// transpose happens while loading the operand instead of as a separate op.
struct FoldVtransposePattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
public:
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult
  matchAndRewriteInterface(LocalMatmulLikeOpInterface op,
                           PatternRewriter &rewriter) const {
    auto aVtrans = lookThroughViewLikes(op.getMatmulA());
    auto bVtrans = lookThroughViewLikes(op.getMatmulB());
    if (aVtrans && !isSwappedInnermostShape(aVtrans.getSrc(), op.getMatmulA()))
      aVtrans = hivm::VTransposeOp();
    if (bVtrans && !isSwappedInnermostShape(bVtrans.getSrc(), op.getMatmulB()))
      bVtrans = hivm::VTransposeOp();
    if (!aVtrans && !bVtrans)
      return rewriter.notifyMatchFailure(op, "no foldable vtranspose found");

    auto newOp = rewriter.clone(*op.getOperation());
    auto newMmad = cast<LocalMatmulLikeOpInterface>(newOp);
    bool setVtransposeFolded = false;

    auto foldSide = [&](hivm::VTransposeOp vtrans, bool isA) {
      Value src = vtrans.getSrc();
      if (auto convert = src.getDefiningOp<ConvertLayoutOp>())
        if (convert.getSrcLayout().getDataLayout() ==
                hivm::DataLayout::Fractal &&
            convert.getDstLayout().isNDLayout())
          setVtransposeFolded = true;
      if (isA) {
        newMmad.setMatmulA(src);
        newMmad.setMatmulATranspose(rewriter.getUnitAttr());
      } else {
        newMmad.setMatmulB(src);
        newMmad.setMatmulBTranspose(rewriter.getUnitAttr());
      }
    };

    if (aVtrans)
      foldSide(aVtrans, /*isA=*/true);
    if (bVtrans)
      foldSide(bVtrans, /*isA=*/false);
    if (setVtransposeFolded)
      newMmad->setAttr(kFractalVtransposeFolded, rewriter.getUnitAttr());

    rewriter.replaceOp(op, newOp);
    return success();
  }

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteInterface(op, rewriter);
  }
};

/// Return a ConvertLayoutOp with swapped innermost output_shape dims.
Value createSwappedConvertLayout(PatternRewriter &rewriter,
                                 ConvertLayoutOp oldConvert, Value newSource) {
  auto mixedShape = oldConvert.getMixedOutputShape();
  std::swap(mixedShape[mixedShape.size() - 2],
            mixedShape[mixedShape.size() - 1]);

  auto oldResultType = cast<RankedTensorType>(oldConvert.getResult().getType());
  SmallVector<int64_t> newShape(oldResultType.getShape());
  std::swap(newShape[newShape.size() - 2], newShape[newShape.size() - 1]);
  auto newResultType =
      RankedTensorType::get(newShape, oldResultType.getElementType());

  rewriter.setInsertionPoint(oldConvert);
  return rewriter
      .create<ConvertLayoutOp>(oldConvert.getLoc(), newResultType, newSource,
                               oldConvert.getSrcLayout(),
                               oldConvert.getDstLayout(), mixedShape)
      .getResult();
}

/// Absorb a vtranspose sitting behind a Fractal->ND convert_layout by swapping
/// the innermost dims of the convert's output shape instead.
std::optional<Value> absorbFractalVtranspose(PatternRewriter &rewriter,
                                             ConvertLayoutOp oldConvert) {
  if (oldConvert.getSrcLayout().getDataLayout() != hivm::DataLayout::Fractal ||
      !oldConvert.getDstLayout().isNDLayout())
    return std::nullopt;
  if (!isa<RankedTensorType>(oldConvert.getResult().getType()))
    return std::nullopt;

  auto vtrans = oldConvert.getSource().getDefiningOp<hivm::VTransposeOp>();
  if (!vtrans || !isInnermostAxisSwap(vtrans.getPermutation()))
    return std::nullopt;
  // A second transpose behind this one cancels it out; the single flag cannot
  // express that, so leave such chains to the canonicalizer.
  if (vtrans.getSrc().getDefiningOp<hivm::VTransposeOp>())
    return std::nullopt;

  return createSwappedConvertLayout(rewriter, oldConvert, vtrans.getSrc());
}

/// Absorb vtranspose ops behind a Fractal->ND convert_layout into the mmad
/// transpose flags, and set kFractalVtransposeFolded so extractRealMKN treats
/// those flags as real 2D transposes.
struct FoldFractalVtransposePattern
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
public:
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult
  matchAndRewriteInterface(LocalMatmulLikeOpInterface op,
                           PatternRewriter &rewriter) const {
    std::optional<Value> newA, newB;
    if (auto convert = op.getMatmulA().getDefiningOp<ConvertLayoutOp>())
      newA = absorbFractalVtranspose(rewriter, convert);
    if (auto convert = op.getMatmulB().getDefiningOp<ConvertLayoutOp>())
      newB = absorbFractalVtranspose(rewriter, convert);
    if (!newA && !newB)
      return rewriter.notifyMatchFailure(
          op, "no Fractal->ND convert_layout with vtranspose on A or B");

    rewriter.modifyOpInPlace(op, [&]() {
      if (newA) {
        op.setMatmulA(*newA);
        if (!op.isMatmulATransposed())
          op.setMatmulATranspose(rewriter.getUnitAttr());
      }
      if (newB) {
        op.setMatmulB(*newB);
        if (!op.isMatmulBTransposed())
          op.setMatmulBTranspose(rewriter.getUnitAttr());
      }
      op->setAttr(kFractalVtransposeFolded, rewriter.getUnitAttr());
    });

    return success();
  }

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteInterface(op, rewriter);
  }
};

void populateFoldVtransposePattern(RewritePatternSet &patterns) {
  patterns.add<FoldVtransposePattern, FoldFractalVtransposePattern>(
      patterns.getContext());
}

void populateSetRealMKNPattern(RewritePatternSet &patterns) {
  patterns.add<SetRealMKNPattern>(patterns.getContext());
}

void populateNormalizeMatmulPattern(RewritePatternSet &patterns) {
  patterns.add<NormalizeMmadCCFPattern, DecomposeMatmulWithBiasPattern>(
      patterns.getContext());
}

void populateAddIfPattern(RewritePatternSet &patterns) {
  patterns.add<ReuseL0CAddIfPattern>(patterns.getContext());
}

LogicalResult runRegBasedNormalizeMatmul(func::FuncOp funcOp,
                                         MLIRContext *context) {
  // Must run before SetRealMKN: folding a vtranspose into a_transpose/
  // b_transpose changes which operand dims m, k and n are read from.
  {
    RewritePatternSet patterns(context);
    populateFoldVtransposePattern(patterns);
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      return failure();
  }
  {
    RewritePatternSet patterns(context);
    populateSetRealMKNPattern(patterns);
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      return failure();
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
      return failure();
  }
  // Block 3: AddIfPattern — all CCFs are normalized (kNormalizedInL0C set),
  // so isCCFOpResultInL0C will succeed. No timing issue.
  {
    RewritePatternSet patterns(context);
    populateAddIfPattern(patterns);
    GreedyRewriteConfig config = GreedyRewriteConfig();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      return failure();
  }
  return success();
}

} // namespace regbase

void NormalizeMatmulPass::runOnOperation() {
  Operation *rootOp = getOperation();

  // Resolve architecture from the nearest ModuleOp that carries hacc.target
  // (tests may wrap funcs in a nested module without the attr).
  auto resolveIsRegBased = [](Operation *from) -> bool {
    for (Operation *p = from; p; p = p->getParentOp()) {
      if (auto moduleOp = dyn_cast<ModuleOp>(p)) {
        if (hacc::utils::getTargetDevice(moduleOp).has_value())
          return hacc::utils::isRegBasedArch(moduleOp);
      }
    }
    if (auto moduleOp = dyn_cast<ModuleOp>(from))
      return hacc::utils::isRegBasedArch(moduleOp);
    if (auto moduleOp = from->getParentOfType<ModuleOp>())
      return hacc::utils::isRegBasedArch(moduleOp);
    return false;
  };

  auto runOnFunc = [&](func::FuncOp funcOp) {
    if (resolveIsRegBased(funcOp)) {
      if (failed(regbase::runRegBasedNormalizeMatmul(funcOp, &getContext())))
        signalPassFailure();
    } else {
      membase::runMemBasedNormalizeMatmul(funcOp, &getContext());
    }
  };

  if (auto funcOp = dyn_cast<func::FuncOp>(rootOp)) {
    runOnFunc(funcOp);
    return;
  }

  // Module-level (and nested modules): visit every func.
  rootOp->walk([&](func::FuncOp funcOp) { runOnFunc(funcOp); });
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeMatmulPass() {
  return std::make_unique<NormalizeMatmulPass>();
}
