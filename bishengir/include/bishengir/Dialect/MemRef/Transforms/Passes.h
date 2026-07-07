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
#ifndef BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"

namespace memref {
std::unique_ptr<Pass> createFoldAllocReshapePass();
std::unique_ptr<Pass> createDeadStoreEliminationPass();
std::unique_ptr<Pass> createRemoveRedundantCopyPass();

/// Create a pass to bind buffer according to annotation.
std::unique_ptr<Pass> createBindBufferPass();


#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
