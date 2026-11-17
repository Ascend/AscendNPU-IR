//===- SplitKernelUtils.h - MIX split helpers -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <utility>

namespace mlir {
namespace hivm {
class TCoreRatioAttr;

/// Verify that a mixed function or split core half has no required work when
/// its core_ratio reserves zero instances of that core.
LogicalResult verifyCoreRatio(func::FuncOp func);

bool isSingleCoreRatio(TCoreRatioAttr ratio);

/// Rename the original function to AIC and clone an AIV function. Validate
/// symbol uses and preserve a private MIX declaration for host callers.
/// Return the AIC/AIV pair; each caller must prune the two bodies.
FailureOr<std::pair<func::FuncOp, func::FuncOp>>
createSplitMixFunctions(func::FuncOp func);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H
