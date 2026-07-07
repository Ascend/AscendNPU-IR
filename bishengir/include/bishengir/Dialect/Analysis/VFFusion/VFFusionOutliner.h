//===- VFFusionOutliner.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_OUTLINER_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_OUTLINER_H

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionBlock.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace analysis {

class VFFusionOutliner {
protected:
  IRMapping mapFromCloning;

public:
  virtual FailureOr<func::FuncOp>
  outline(func::FuncOp funcOp, VFFusionBlock &fusedBlock, OpBuilder &builder);
  virtual LogicalResult createInvoke(func::FuncOp fusedFunction,
                                     VFFusionBlock &fusedBlock,
                                     OpBuilder &builder);
  VFFusionOutliner() = default;
  virtual ~VFFusionOutliner() = default;
};

} // namespace analysis
} // namespace mlir
#endif
