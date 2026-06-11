//===- BubbleUpUtils.h ----------------------------------------------------===//
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
//============================================================================//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_BUBBLEUPUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_BUBBLEUPUTILS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::hivm::detail {

extern const llvm::StringLiteral kBubbleUpPropagateUp;
extern const llvm::StringLiteral kBubbleUpPropagateDown;

struct ExtractSliceTilingInfo {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  RankedTensorType resultType;
};

struct BufferizationPropagationState {
  memref::AllocOp allocOp;
  memref::AllocOp newAllocOp;
  TensorType sourceTensorType;
  ExtractSliceTilingInfo sliceInfo;
  SmallVector<Operation *, 4> pathOps;
  SmallVector<UnrealizedConversionCastOp, 4> upPropagators;
  SmallVector<UnrealizedConversionCastOp, 4> downPropagators;
  llvm::SmallDenseMap<Value, Value, 4> oldToNew;
};

struct TilingDimInfo {
  int64_t tilingDim = -1;
  OpFoldResult offset = nullptr;
  OpFoldResult size = nullptr;
};

LogicalResult
checkBufferizationBubbleUpPath(bufferization::ToTensorOp toTensorOp);

MemRefType getSlicedMemRefType(MemRefType oldType,
                               RankedTensorType slicedTensorType);

MemRefType getSlicedMemRefType(MemRefType oldType, int64_t tilingDim);

void resolveUpLinksForOldValue(Value oldValue, Value newValue,
                               PatternRewriter &rewriter);

void clearBubblePropagatorAttrs(Operation *op, PatternRewriter &rewriter);

void cleanupResolvedBufferizationPropagators(func::FuncOp funcOp);

UnrealizedConversionCastOp
createBubblePropagatorDown(Value oldValue, Value newValue, OpFoldResult offset,
                           OpFoldResult size, int64_t tilingDim,
                           PatternRewriter &rewriter);

UnrealizedConversionCastOp
createBubblePropagatorUpLink(Value oldValue, Type slicedType,
                             OpFoldResult offset, OpFoldResult size,
                             int64_t tilingDim, PatternRewriter &rewriter);

FailureOr<memref::AllocOp> createSlicedAllocLike(MemRefType slicedMemRefType,
                                                 memref::AllocOp allocOp,
                                                 PatternRewriter &rewriter);

void cleanupDeadBufferizationPropagators(func::FuncOp funcOp,
                                         PatternRewriter &rewriter);

void cleanupBufferizationPropagators(func::FuncOp funcOp,
                                     bufferization::ToTensorOp toTensorOp,
                                     BufferizationPropagationState &state,
                                     PatternRewriter &rewriter);

void markTiledTightlyCoupledAllocIfNeeded(RewriterBase &rewriter,
                                          Value memrefValue);

TilingDimInfo getTilingDimInfo(UnrealizedConversionCastOp propagateOp);

} // namespace mlir::hivm::detail

#endif
