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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include <string>
namespace mlir {
namespace analysis {

enum class FusionMode { AllOp, NMostOp, MaxParallel, UBAwareOp };

struct VFFusionKindOption {
  VFFusionKindOption(const bool enableOutlineCF, const bool enableOutlineMemref,
                     const bool enableOutlineArith,
                     const bool enableOutlineCube, int64_t ubBudgetBytes = 0,
                     int64_t ubAlignBytes = 0, const bool enableRA = false,
                     const bool enableAR = false)
      : enableOutlineCF(enableOutlineCF),
        enableOutlineMemref(enableOutlineMemref),
        enableOutlineArith(enableOutlineArith),
        enableOutlineCube(enableOutlineCube), ubBudgetBytes(ubBudgetBytes),
        ubAlignBytes(ubAlignBytes), enableRA(enableRA), enableAR(enableAR) {};

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

bool shouldSkipFusion(Operation *op, const VFFusionKindOption &option);

} // namespace analysis
} // namespace mlir
#endif