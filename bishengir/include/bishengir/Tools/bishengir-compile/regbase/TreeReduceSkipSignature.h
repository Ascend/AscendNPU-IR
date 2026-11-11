//===- TreeReduceSkipSignature.h - TreeReduce skip signature ----*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// Recognition of the kernels that regress under the new TreeReduce policy, so
// the caller can give them the same treatment as --enable-tree-reduce=false.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_TREEREDUCESKIPSIGNATURE_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_TREEREDUCESKIPSIGNATURE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

namespace bishengir {
namespace regbase {

/// Module-wide, shape-independent IR signature read at runRegBasePipeline
/// entry.  Every member is a plain op count so the match below can use exact
/// equality only: no shape extents and no greater/less comparisons, which
/// keeps the affected set as narrow as possible.
///
/// `scf.if` is deliberately NOT part of the signature: it is the one counter
/// that is not stable across shapes of the same kernel (chunk_kda_bwd wy_dqkg
/// yields 9 for one shape and 8 for another), so folding it into an equality
/// test would silently miss shapes.
struct NewTreeReduceCloseSignature {
  unsigned numSumReduceDim0 = 0; // f32 single-dim sum reduce over dim 0 (RA)
  unsigned numSumReduceOtherDim = 0; // ... over a non-zero dim (AR)
  unsigned numOtherReduce = 0;       // any reduce not of the two forms above
  unsigned numMatmul = 0;
  unsigned numTranspose = 0;
  unsigned numBroadcast = 0;
  unsigned numFill = 0;
  unsigned numScfFor = 0;
  unsigned numExpandShape = 0;
  unsigned numCollapseShape = 0;
  unsigned numFuncs = 0;

  bool operator==(const NewTreeReduceCloseSignature &o) const {
    return numSumReduceDim0 == o.numSumReduceDim0 &&
           numSumReduceOtherDim == o.numSumReduceOtherDim &&
           numOtherReduce == o.numOtherReduce && numMatmul == o.numMatmul &&
           numTranspose == o.numTranspose && numBroadcast == o.numBroadcast &&
           numFill == o.numFill && numScfFor == o.numScfFor &&
           numExpandShape == o.numExpandShape &&
           numCollapseShape == o.numCollapseShape && numFuncs == o.numFuncs;
  }
};

/// True when `reduceOp` is a canonical f32 single-dimension sum reduction.
inline bool isF32SingleDimSumReduce(mlir::linalg::ReduceOp reduceOp) {
  if (reduceOp.getDimensions().size() != 1 || reduceOp.getInputs().size() != 1)
    return false;
  auto inputType =
      mlir::dyn_cast<mlir::ShapedType>(reduceOp.getInputs().front().getType());
  if (!inputType || !inputType.getElementType().isF32())
    return false;
  bool isSum = false;
  reduceOp.walk([&](mlir::arith::AddFOp) {
    isSum = true;
    return mlir::WalkResult::interrupt();
  });
  return isSum;
}

inline NewTreeReduceCloseSignature
computeNewTreeReduceCloseSignature(mlir::ModuleOp module) {
  NewTreeReduceCloseSignature sig;
  module.walk([&](mlir::Operation *op) {
    if (auto reduceOp = mlir::dyn_cast<mlir::linalg::ReduceOp>(op)) {
      if (!isF32SingleDimSumReduce(reduceOp))
        ++sig.numOtherReduce;
      else if (reduceOp.getDimensions().front() == 0)
        ++sig.numSumReduceDim0;
      else
        ++sig.numSumReduceOtherDim;
      return;
    }
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      ++sig.numMatmul;
    else if (mlir::isa<mlir::linalg::TransposeOp>(op))
      ++sig.numTranspose;
    else if (mlir::isa<mlir::linalg::BroadcastOp>(op))
      ++sig.numBroadcast;
    else if (mlir::isa<mlir::linalg::FillOp>(op))
      ++sig.numFill;
    else if (mlir::isa<mlir::scf::ForOp>(op))
      ++sig.numScfFor;
    else if (mlir::isa<mlir::tensor::ExpandShapeOp>(op))
      ++sig.numExpandShape;
    else if (mlir::isa<mlir::tensor::CollapseShapeOp>(op))
      ++sig.numCollapseShape;
    else if (mlir::isa<mlir::func::FuncOp>(op))
      ++sig.numFuncs;
  });
  return sig;
}

// TODO: replace this signature table with a cost model. The new TreeReduce
// policy regresses these kernels, and until the cost model can tell them apart
// the only narrow lever available here is an exact IR signature match.
inline bool isSpecialCaseCloseNewTreeReduce(mlir::ModuleOp mod) {
  NewTreeReduceCloseSignature sig = computeNewTreeReduceCloseSignature(mod);

  // Signatures measured at this exact program point with the counting rules
  // implemented above. Field order:
  //   {RA, AR, otherReduce, matmul, transpose, broadcast, fill, scf.for,
  //    expandShape, collapseShape, func}
  // Only kernels confirmed to regress under the new TreeReduce policy are
  // listed; causal_conv1d_fwd_kernel is intentionally absent because it does
  // not regress, and its signature {0, 8, 0, 0, 0, 17, 21, 0, 0, 0, 1} differs
  // from every entry here, so it stays on the default path.
  //
  // chunk_gla_bwd_kernel_inter is also intentionally absent: its skip is not
  // wanted, so its signature {3, 0, 0, 2, 2, 3, 13, 1, 0, 0, 2} is not listed
  // and it stays on the default path as well.
  static const NewTreeReduceCloseSignature kSignatures[] = {
      // causal_conv1d_bwd_kernel
      {14, 8, 0, 0, 0, 16, 37, 0, 0, 0, 1},
      // chunk_gla_bwd_kernel_inter
      {3, 0, 0, 2, 2, 3, 13, 1, 0, 0, 2},
      // chunk_kda_bwd_kernel_wy_dqkg_fused needs two entries: fill and func
      // counts are not stable across its shapes (16/1 vs 15/3).
      {2, 2, 0, 9, 3, 9, 16, 0, 1, 1, 1},
      {2, 2, 0, 9, 3, 9, 15, 0, 1, 1, 3},
      // prepare_wy_repr_bwd_kernel (identical across all 4 measured shapes)
      {1, 5, 0, 9, 6, 10, 13, 3, 1, 1, 1},
  };
  return llvm::is_contained(kSignatures, sig);
}

} // namespace regbase
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_TREEREDUCESKIPSIGNATURE_H
