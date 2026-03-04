//===- Utils.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UTILS_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UTILS_H

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Utils/UnionFind.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
namespace mlir {
namespace analysis {

enum class FusionMode { AllOp, NMostOp };

struct VFFusionKindOption {
  VFFusionKindOption(const bool enableOutlineCF, const bool enableOutlineMemref,
                     const bool enableOutlineArith,
                     const bool enableOutlineCube)
      : enableOutlineCF(enableOutlineCF),
        enableOutlineMemref(enableOutlineMemref),
        enableOutlineArith(enableOutlineArith),
        enableOutlineCube(enableOutlineCube){};

  VFFusionKindOption(const VFFusionKindOption &option) = default;

  VFFusionKindOption &operator=(const VFFusionKindOption &other) = delete;

  const bool enableOutlineCF;
  const bool enableOutlineMemref;
  const bool enableOutlineArith;
  const bool enableOutlineCube;
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