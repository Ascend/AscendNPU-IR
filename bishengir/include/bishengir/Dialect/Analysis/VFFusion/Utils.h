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
namespace mlir {
namespace analysis {

enum class FusionMode { AllOp, NMostOp, MaxParallel, UBAwareOp };

struct VFFusionKindOption {
  VFFusionKindOption(const bool enableOutlineCF, const bool enableOutlineMemref,
                     const bool enableOutlineArith,
                     const bool enableOutlineCube,
                     const bool enableReshapeTiling, int64_t ubBudgetBytes = 0,
                     int64_t ubAlignBytes = 0)
      : enableOutlineCF(enableOutlineCF),
        enableOutlineMemref(enableOutlineMemref),
        enableOutlineArith(enableOutlineArith),
        enableOutlineCube(enableOutlineCube),
        enableReshapeTiling(enableReshapeTiling), ubBudgetBytes(ubBudgetBytes),
        ubAlignBytes(ubAlignBytes){};

  VFFusionKindOption(const VFFusionKindOption &option) = default;

  VFFusionKindOption &operator=(const VFFusionKindOption &other) = delete;

  const bool enableOutlineCF;
  const bool enableOutlineMemref;
  const bool enableOutlineArith;
  const bool enableOutlineCube;
  const bool enableReshapeTiling;
  const int64_t ubBudgetBytes;
  const int64_t ubAlignBytes;
};

bool isReshapeOp(Operation *op);

// alloc-like op, constant-like, and EmptyOp
bool isInitialOp(Operation *op);

// is operation safe to not be outlined.
bool isSafeToExcludeOps(Operation *op);

bool isOutlineableOp(const VFFusionKindOption &option, Operation *op);

bool isCubeFunc(func::FuncOp funcOp);

} // namespace analysis
} // namespace mlir
#endif