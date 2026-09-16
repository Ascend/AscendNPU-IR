//===- TorchToHFusion.h - Main pass entry for Torch to HFusion ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_TORCHTOHFUSION_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_TORCHTOHFUSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_CONVERTTORCHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

/// Creates a pass to convert torch dialect ops to linalg/hfusion ops
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToHFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToHFusionPass(const ConvertTorchToHFusionOptions &options);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_ATENTONAMEDOP_H
