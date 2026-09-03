//===- HFusionToHIVMPass.h - HFusion to HIVM Conversion Pass ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert HFusion dialect to HIVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTHFUSIONTOHIVM
#include "bishengir/Conversion/Passes.h.inc"

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createHFusionToHIVMConversionPass();

std::unique_ptr<Pass>
createHFusionToHIVMConversionPass(const ConvertHFusionToHIVMOptions &option);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H
