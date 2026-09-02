//===- Utils.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UTILS_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UTILS_H

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/UnionFind.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
namespace mlir {
namespace analysis {

enum class FusionMode { AllOp, MaxParallel, UBAwareOp };

struct VFFusionKindOption {
  VFFusionKindOption(const bool enableOutlineCF, const bool enableOutlineMemref,
                     const bool enableOutlineArith,
                     const bool enableOutlineCube, int64_t ubBudgetBytes = 0,
                     int64_t ubAlignBytes = 0, const bool enableRA = false,
                     const bool enableAR = false, int64_t maxVFParams = -1,
                     const bool enableVFStackLimit = false,
                     const bool enableCastOpt = false,
                     const bool enableNewTreeReducePolicy = false,
                      const bool enablePredicateSinkAcrossSync = true,
                      const bool isMixCV = false)
      : enableOutlineCF(enableOutlineCF),
        enableOutlineMemref(enableOutlineMemref),
        enableOutlineArith(enableOutlineArith),
        enableOutlineCube(enableOutlineCube), ubBudgetBytes(ubBudgetBytes),
        ubAlignBytes(ubAlignBytes), enableRA(enableRA), enableAR(enableAR),
        maxVFParams(maxVFParams), enableVFStackLimit(enableVFStackLimit),
        enableCastOpt(enableCastOpt),
        enableNewTreeReducePolicy(enableNewTreeReducePolicy),
        enablePredicateSinkAcrossSync(enablePredicateSinkAcrossSync),
        isMixCV(isMixCV) {};

  VFFusionKindOption(const VFFusionKindOption &option) = default;

  VFFusionKindOption &operator=(const VFFusionKindOption &other) = delete;

  const bool enableOutlineCF;
  const bool enableOutlineMemref;
  const bool enableOutlineArith;
  const bool enableOutlineCube;
  const int64_t ubBudgetBytes;
  const int64_t ubAlignBytes;
  const bool enableRA;
  const bool enableAR;
  const int64_t maxVFParams;
  const bool enableVFStackLimit;
  const bool enableCastOpt;
  const bool enableNewTreeReducePolicy;
  const bool enablePredicateSinkAcrossSync;
  const bool isMixCV;
};

bool isReshapeOp(Operation *op);

// alloc-like op, constant-like, and EmptyOp
bool isInitialOp(Operation *op);

// is operation safe to not be outlined.
bool isSafeToExcludeOps(Operation *op);

bool isOutlineableOp(const VFFusionKindOption &option, Operation *op);

bool isCubeFunc(func::FuncOp funcOp);

bool isVsstbPatternTransposeOp(Operation *op);

bool userCanFuseIntoVsstbPatternTransposeOp(Operation *op);

bool isExpandShapeOpCanFuseIntoVsstbPatternTranspose(Operation *op);

/// Checks whether a reshape operation (tensor::CollapseShapeOp or
/// tensor::ExpandShapeOp) would be eliminated by PreVectorizationFusion.
///
/// CollapseShapeOp: eliminated when its result feeds into a
///   linalg::BroadcastOp — generalizeBroadcastOp creates an inverse
///   expand_shape, then MLIR fold cancels the pair.
///
/// ExpandShapeOp: eliminated when consumed by a linalg::LinalgOp (not
///   TransposeOp) where the expand only inserts unit dims (each
///   reassociation group has at most one non-unit dim) —
///   PreVectorizationFusion generalizes named ops to linalg.generic,
///   then the upstream reshape folding in
///   populateElementwiseOpsFusionPatterns projects out the unit dims
///   and folds the expand into the indexing map.
///
/// Note: inverse reshape pairs (expand source from CollapseShapeOp) are
/// not checked here — they are always folded by preProcess() ->
/// applyPatternsGreedily -> ExpandShapeOp::fold() before the fusion
/// phase, so they never reach areReshapesValidIfFused.
bool isReshapeEliminableByPreVectorizationFusion(Operation *op);

bool shouldSkipFusion(Operation *op, const VFFusionKindOption &option);

bool isComputeOp(Operation *op);

} // namespace analysis
} // namespace mlir
#endif
