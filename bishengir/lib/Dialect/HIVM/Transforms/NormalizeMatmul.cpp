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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEMATMUL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-normalize-matmul"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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
void tryOptimizePad(Operation *maybeLoadOp, Value mmadSource,
                    PatternRewriter &rewriter) {
  bool padKOnly =
      utils::getAnnotateOpWithAttr(mmadSource, kDotPadOnlyK).has_value();
  if (!padKOnly)
    return;

  auto loadOp = dyn_cast_if_present<hivm::LoadOp>(maybeLoadOp);
  if (!loadOp)
    return;

  if (!isConstZero(loadOp.getPadValue()))
    return;

  LDBG("removing pad for load op: " << *loadOp);
  rewriter.modifyOpInPlace(loadOp, [&loadOp]() {
    loadOp.setPadModeAttr(hivm::PadModeAttr());
    loadOp.getPadValueMutable().clear();
    loadOp.getLeftPaddingNumMutable().clear();
    loadOp.getRightPaddingNumMutable().clear();
    loadOp.setInitOutBuffer(false);
    loadOp.getInitConditionMutable().clear();
  });
}

struct NormalizeMatmulPass
    : public impl::NormalizeMatmulBase<NormalizeMatmulPass> {
  using Base::Base;
  void runOnOperation() override;
};

// If value is from memref, we get shape from memref.subview or memref.alloc.
// If value is from tensor, we get shape from value directly.
FailureOr<SmallVector<Value>>
getRealShapeFromMemrefOrTensor(Value val, Location loc,
                               PatternRewriter &rewriter) {
  FailureOr<memref::AllocOp> status = getMemRefAlloc(val);
  if (failed(status)) {
    SmallVector<Value> tensorMixedSizes;
    for (OpFoldResult size : tensor::getMixedSizes(rewriter, loc, val)) {
      tensorMixedSizes.push_back(
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, size));
    }
    return tensorMixedSizes;
  }
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
  tryOptimizePad(writerOp, val, rewriter);
  auto subview = dyn_cast<memref::SubViewOp>(*candidateSubViews.begin());
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
      assert(op.isInitConstant(false));
      LDBG("decompose matmul with elemwise add");
      return decomposeMatmulWithElementwiseAdd<T>(rewriter, op);
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

void NormalizeMatmulPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
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

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeMatmulPass() {
  return std::make_unique<NormalizeMatmulPass>();
}
