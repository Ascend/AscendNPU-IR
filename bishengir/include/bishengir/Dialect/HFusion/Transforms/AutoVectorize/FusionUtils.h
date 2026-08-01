//===- FusionUtils.h - Op classification helpers for AV2 -----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUTILS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {

class Operation;
class Block;

namespace hfusion {

// Create a transform::SequenceOp anchored after `func`.
transform::SequenceOp buildTransformSequenceOp(OpBuilder &builder,
                                               func::FuncOp func);

// Whether `op` resides (directly or indirectly) in `block`.
bool isOpInBlock(Operation *op, Block *block);

// Whether `op` is a LinalgOp that operates on memref operands/results.
bool isMemrefLinalgOp(Operation *op);

// Whether `op` has a rank-reducing indexing map.
bool hasRankReducingIndexingMap(Operation *op);

// Whether `op` is a non-vectorizable barrier (sync, copy, region, etc.).
bool isNonVectorizableOp(Operation *op);

// Whether `op` is eligible for fusion (LinalgOp, interleave/deinterleave).
bool isFusableOp(Operation *op);

// Traverse downstream (user) IR graph to find fusable ops reachable from `op`.
void findDownstreamFusableOpOf(
    Operation *op, Block *block,
    llvm::DenseSet<Operation *> &downstreamFusableOps,
    llvm::DenseSet<Operation *> &visitedOps);

// Traverse upstream (def-use) IR graph to find fusable ops that `op` depends
// on.
void findUpstreamFusableOpOf(Operation *op, Block *block,
                             llvm::DenseSet<Operation *> &upstreamFusableOps,
                             llvm::DenseSet<Operation *> &visitedOps);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUTILS_H
