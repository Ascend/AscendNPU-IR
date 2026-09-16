//===- Passes.h - HACC pipeline entry points --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all HACC pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_HACC_PIPELINES_PASSES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hacc {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the pipeline to lower HACC dialect to LLVM.
void buildLowerHACCToLLVMPipeline(OpPassManager &pm,
                                  std::string tmpDeviceBinName);

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_PIPELINES_PASSES_H
