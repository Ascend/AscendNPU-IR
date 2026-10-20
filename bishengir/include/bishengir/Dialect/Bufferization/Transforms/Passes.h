//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace bishengir {
namespace bufferization {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"

/// Create a pass to normalize bufferization.to_tensor ops: isolate the
/// memory-space info and expose memref-level writes to tensor-level
/// analysis.
std::unique_ptr<mlir::Pass> createNormalizeToTensorOpPass();

/// Create a pass that runs One-Shot analysis and inserts tensor copies to
/// resolve bufferization conflicts, without performing the bufferization.
std::unique_ptr<mlir::Pass> createTensorCopyInsertionPass();
std::unique_ptr<mlir::Pass> createTensorCopyInsertionPass(
    const mlir::bufferization::OneShotBufferizationOptions &options);

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace bishengir

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
