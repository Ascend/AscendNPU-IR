//===- Utils.cpp - Implementation of VFFusion Utils -----------------------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "vf-fusion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {

// Checks whether operation is part of tensor or memref reshaping operations.
bool isReshapeOp(Operation *const op) {
  return isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp, tensor::ReshapeOp>(
      op);
}

bool isInitialOp(Operation *const op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (utils::isAllocLikeOp(op))
    return true;
  if (reshape_utils::isInitOp(op))
    return true;
  return false;
}

bool isSafeToExcludeOps(Operation *const op) {
  return isInitialOp(op) && op->getNumOperands() == 0;
}

bool isCubeFunc(func::FuncOp funcOp) {
  auto fusionKind = mlir::hfusion::tryGetFusionKind(funcOp);
  if (fusionKind.has_value() &&
      (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
       fusionKind.value() == mlir::hfusion::FusionKind::SingleCube)) {
    return true;
  }
  std::optional<mlir::hivm::TFuncCoreType> funcCoreType =
      mlir::hivm::queryFuncCoreType(funcOp);
  return (funcCoreType.has_value() &&
          funcCoreType.value() != hivm::TFuncCoreType::AIV);
}

bool isVsstbPatternTransposeOp(Operation *op) {
  auto transpose = dyn_cast<linalg::TransposeOp>(op);
  if (!transpose) {
    return false;
  }

  auto inputType = dyn_cast<ShapedType>(transpose.getInput().getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }

  auto elemType = inputType.getElementType();
  if (!(elemType.isBF16() || elemType.isF16() || elemType.isF32() ||
        isa<Float8E4M3FNType>(elemType) || isa<Float8E5M2Type>(elemType))) {
    return false;
  }

  auto perm = transpose.getPermutation();
  int64_t rank = static_cast<int64_t>(perm.size());
  // Rule 0: Should be 3-dim transpose
  if (rank != 3)
    return false;

  // Rule 1: Should not be inner axis transpose
  if (perm[rank - 1] != rank - 1)
    return false;

  // Rule 2: Last axis should fit in exactly 32 bytes
  ArrayRef<int64_t> shape = inputType.getShape();
  // Calculate element width in bytes
  uint64_t elemByteWidth =
      llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(),
                       utils::INTR_BITS_PER_BYTE);
  int64_t lastDim = shape[rank - 1];
  return lastDim * static_cast<int64_t>(elemByteWidth) == 32;
}

static bool isSyncBarrierOp(Operation *op) {
  return isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
             hivm::CreateSyncBlockLockOp, hivm::SyncBlockLockOp,
             hivm::SyncBlockUnlockOp>(op);
}

static bool hasSyncBetween(Block::iterator from, Block::iterator to) {
  auto callback = [](Operation *inner) {
    return isSyncBarrierOp(inner) ? WalkResult::interrupt()
                                  : WalkResult::advance();
  };
  auto check = [callback](Operation &o) {
    return o.walk(callback).wasInterrupted();
  };
  return llvm::any_of(llvm::make_range(std::next(from), to), check);
}

// Returns true iff ALL users of op are ultimately in the vsstb transpose fusion
// chain with no sync barrier on any hop, and at least one vsstb consumer
// exists. "All users" is enforced universally and propagated recursively
// through generic ops, so any non-participating user at any depth blocks the
// entire chain.
bool userCanFuseIntoVsstbPatternTransposeOp(Operation *op) {
  auto users = op->getUsers();
  if (users.empty())
    return false;

  // Single pass: verify same block and find the last user in program order.
  Operation *lastUser = nullptr;
  for (Operation *user : users) {
    if (user->getBlock() != op->getBlock())
      return false;
    if (!lastUser || lastUser->isBeforeInBlock(user))
      lastUser = user;
  }

  if (hasSyncBetween(op->getIterator(), lastUser->getIterator()))
    return false;

  return llvm::all_of(users, [](Operation *user) {
    return isVsstbPatternTransposeOp(user) ||
           (isa<linalg::GenericOp>(user) &&
            userCanFuseIntoVsstbPatternTransposeOp(user));
  });
}

bool isExpandShapeOpCanFuseIntoVsstbPatternTranspose(Operation *op) {
  auto expandShape = dyn_cast<tensor::ExpandShapeOp>(op);
  if (!expandShape) {
    return false;
  }
  // FIXME: expand_shape with one-dim src will cause error when tile after
  // fusing into vsstb pattern transpose, see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/1100
  auto srcType = dyn_cast<TensorType>(expandShape.getSrc().getType());
  auto resType = dyn_cast<TensorType>(expandShape.getResult().getType());
  if (srcType.getShape().size() != 2 || resType.getShape().size() != 3) {
    return false;
  }

  return userCanFuseIntoVsstbPatternTransposeOp(op);
}

//===----------------------------------------------------------------------===//
// PreVectorizationFusion eliminability check
//
// Mirrors the two reshape-elimination paths inside PreVectorizationFusion so
// that VFFusion can safely admit reshape ops that would be folded away by
// the earlier pass (or by a later canonicalization/auto-vectorize run that
// re-applies the same patterns).
//===----------------------------------------------------------------------===//

// CollapseShapeOp: the collapse feeds into a linalg::BroadcastOp.
// PreVectorizationFusion::generalizeBroadcastOp (triggered when a broadcast's
// input is from CollapseShapeOp) creates an inverse expand_shape on the
// collapse; MLIR's ExpandShapeOp::fold() then cancels
// expand_shape(collapse_shape(x)) -> x, leaving the collapse dead.
static bool isCollapseShapeEliminable(Operation *op) {
  auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op);
  if (!collapseOp)
    return false;
  return llvm::any_of(collapseOp->getUsers(), [](Operation *user) {
    return isa<linalg::BroadcastOp>(user);
  });
}

// ExpandShapeOp: the expand only inserts unit dims (each reassociation group
// has at most one non-unit dim — isPureUnitExpand) and is consumed by a named
// linalg/hfusion op. At VFFusion time the consumer is still a named op
// (linalg.mul / elemwise_* / hfusion.cast / bitcast / ...); it generalizes to
// an IDENTITY-map generic that ExpandShapeToImplicitBrcInGenericPattern does
// NOT fold (no constant-0 broadcast axis).
//
// Admit rule: admit only when the consumer has a constant operand. The
// constant broadcasts, so the expand's unit dim is removable by later passes
// (flatten / vectorization). Without a constant, every operand carries the
// unit dim and the expand survives in the VF -> reject. This also rejects
// hfusion.cast/bitcast (single input, no constant operand).
//
// all_of (not any_of): the expand is folded away only if EVERY user is
// admissible; a single non-admissible user (e.g. func.return, or a named op
// without a constant operand) keeps the expand alive.
//
// linalg::TransposeOp is excluded because PreVectorizationFusion never
// generalizes it (HFusionGeneralizationPatterns skips TransposeOp), so the
// reshape folding cannot fire on it.
//
// Note: the inverse reshape pair (expand source from CollapseShapeOp) is
// always folded by preProcess() -> applyPatternsGreedily ->
// ExpandShapeOp::fold() before the fusion phase, so it is not handled here.
static bool isExpandShapeEliminable(Operation *op) {
  auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op);
  if (!expandOp)
    return false;

  auto resType = dyn_cast<RankedTensorType>(expandOp.getResult().getType());
  if (!resType)
    return false;

  ArrayRef<int64_t> resShape = resType.getShape();
  auto reassoc = expandOp.getReassociationIndices();

  // isPureUnitExpand: each reassociation group has at most one non-unit dim.
  bool isPureUnitExpand = true;
  for (const auto &group : reassoc) {
    unsigned numExtentDims = 0;
    for (int64_t d : group)
      if (resShape[d] != 1)
        ++numExtentDims;
    if (numExtentDims > 1) {
      isPureUnitExpand = false;
      break;
    }
  }
  if (!isPureUnitExpand)
    return false;

  // The expand is folded away only if EVERY user is admissible (so the expand
  // goes dead). all_of, not any_of: a single non-admissible user keeps it
  // alive.
  if (expandOp->getUsers().empty())
    return false;
  return llvm::all_of(expandOp->getUsers(), [&](Operation *user) {
    if (isa<linalg::TransposeOp>(user))
      return false;
    auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
    if (!linalgOp)
      return false;
    // At VFFusion time the consumer is a named op (linalg.mul / elemwise_* /
    // hfusion.cast / bitcast / ...); it generalizes to an identity-map generic
    // that ExpandShapeToImplicitBrcInGenericPattern does NOT fold. Admit only
    // when the op has a constant operand: the constant broadcasts, so the
    // expand's unit dim is removable by later passes (flatten / vectorization).
    // Without a constant, every operand carries the unit dim and the expand
    // survives -> reject. This also rejects hfusion.cast/bitcast (single input,
    // no constant operand).
    for (Value input : linalgOp.getDpsInputs()) {
      if (input.getDefiningOp() == expandOp)
        continue;
      if (Operation *defOp = input.getDefiningOp())
        if (defOp->hasTrait<OpTrait::ConstantLike>())
          return true;
    }
    return false;
  });
}

bool isReshapeEliminableByPreVectorizationFusion(Operation *op) {
  return isCollapseShapeEliminable(op) || isExpandShapeEliminable(op);
}

//===----------------------------------------------------------------------===//
// Each function returns true if the op should NOT participate in fusion.
// Add new skip categories here — no other sites need to change.
//===----------------------------------------------------------------------===//
template <typename... ReduceTypes>
std::optional<size_t> getReductionDim(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    std::optional<size_t> reductionDim = std::nullopt;
    auto iteratorTypes = genericOp.getIteratorTypesArray();

    for (size_t i = 0; i < iteratorTypes.size(); ++i) {
      if (iteratorTypes[i] == utils::IteratorType::reduction) {
        if (reductionDim.has_value())
          return std::nullopt;
        reductionDim = i;
      }
    }

    if (!reductionDim.has_value())
      return std::nullopt;

    Block &body = genericOp.getRegion().front();
    if (!llvm::isa<ReduceTypes...>(body.front()))
      return std::nullopt;

    return reductionDim;
  }

  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    SmallVector<unsigned> reductionDims;
    reduceOp.getReductionDims(reductionDims);
    if (reductionDims.size() != 1)
      return std::nullopt;

    Block &body = reduceOp.getRegion().front();
    for (Operation &bodyOp : body) {
      if (llvm::isa<ReduceTypes...>(bodyOp))
        return static_cast<size_t>(reductionDims[0]);
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static bool shouldSkipSumReduction(Operation *op,
                                   const VFFusionKindOption &option) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInputs() == 0)
    return false;

  std::optional<size_t> dimOpt =
      getReductionDim<arith::AddFOp, arith::AddIOp>(op);
  if (!dimOpt)
    return false;
  size_t dim = *dimOpt;

  auto inputType = dyn_cast<ShapedType>(linalgOp.getDpsInputs()[0].getType());
  if (!inputType)
    return false;

  Type elemType = inputType.getElementType();
  if (!llvm::isa<Float32Type, Float16Type, BFloat16Type>(elemType))
    return false;

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                          << "] shouldSkipFusion: sum-reduce dim=" << dim
                          << " rank=" << inputType.getRank()
                          << " enableRA=" << option.enableRA
                          << " enableAR=" << option.enableAR << "\n");

  // 1D reduction is not handled by RA/AR.
  if (inputType.getRank() == 1) {
    LLVM_DEBUG(llvm::dbgs() << " -> allow fusion (1D reduce)\n");
    return false;
  }

  // Only handle 2D reductions for RA/AR.
  if (inputType.getRank() == 2) {
    // RA: reduce along dim 0.
    if (dim == 0 && option.enableRA) {
      if (option.enableNewTreeReducePolicy) {
        // Only structurally supported reductions beyond the size cutoff take
        // the regular fusion path. Keep reductions rejected by the selector
        // on the established isolated 2-D RA path.
        return !hfusion::isSupportedTreeReductionCandidate(op) ||
               hfusion::shouldUseTreeReduction(op);
      }
      LLVM_DEBUG(llvm::dbgs()
                 << " -> skip fusion (2D RA)\n");
      return true;
    }

    // AR: reduce along dim 1.
    if (dim == 1 && option.enableAR) {
      LLVM_DEBUG(llvm::dbgs() << " -> skip fusion (2D AR)\n");
      return true;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << " -> allow fusion\n");
  return false;
}

bool shouldSkipFusion(Operation *op, const VFFusionKindOption &option) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<linalg::GenericOp, linalg::ReduceOp>(
          [&](auto /*typedOp*/) { return shouldSkipSumReduction(op, option); })
      .Default([](Operation *) { return false; });
}

bool isComputeOp(Operation *op) { return !isa<annotation::MarkOp>(op); }

} // namespace mlir::analysis
