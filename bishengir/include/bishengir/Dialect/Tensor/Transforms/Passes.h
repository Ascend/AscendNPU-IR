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
#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
namespace tensor {

/// Create a pass to canonicalize tensor reshape.
std::unique_ptr<Pass> createCanonicalizeTensorReshapePass();

/// Create a pass to narrow liveness of tensor ops
std::unique_ptr<Pass> createNarrowTensorOpPass();

/// Create a pass to propagate reshape
std::unique_ptr<Pass>
createPropagateReshapePass(const PropagateReshapeOptions &options = {});

/// Create a pass to fold tensor empty.
std::unique_ptr<Pass> createFoldTensorEmptyPass();

/// Create a pass to normalize tensor ops.
std::unique_ptr<Pass> createNormalizeTensorOpsPass();

/// Create a pass to trickle tensor::concatOp down.
std::unique_ptr<Pass> createTrickleConcatDownPass();

/// Create a pass to bubble tensor::padOp up.
std::unique_ptr<Pass> createBubblePadUpPass();

/// Create a pass to normalize tensor ops with unaligned last dimension
std::unique_ptr<Pass> createNormalizeLastDimUnalignedTensorOpPass();

/// Create a pass to bubble up extract slice
std::unique_ptr<Pass>
createBubbleUpExtractSlicePass(const BubbleUpExtractSliceOptions &options = {});

/// Create a pass to merge consecutive insert extract slice
std::unique_ptr<Pass> createMergeConsecutiveInsertExtractSlicePass();

/// Decompose tensor::ConcatOp using upstream patterns
std::unique_ptr<Pass> createDecomposeTensorConcatPass();

/// Create a pass to optimize dps ops that are inserted and yielded.
std::unique_ptr<Pass> createOptimizeDpsOpWithYieldedInsertSlicePass();

/// Create a pass to clone dst out tensor empty.
std::unique_ptr<Pass> createReplicateOutEmptyTensorPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H
