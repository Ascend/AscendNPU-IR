//===- RootReachability.h - dataflow root reachability for OTP -*- C++ -*-===//
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
// One-shot dataflow answer to the question the placement pass asks per
// candidate: "which nz2nd fixpipes reach this value through layout-agnostic
// single-use transparent ops?" Two projections:
//
//   - localRoots:  same-block paths only; this is what the decision reads.
//   - globalRoots: paths that may cross control flow (the framework joins
//     root sets into block arguments / region results, so scf.for
//     iter_args reach a fixed point). Computed for future policy use.
//
// The framework joins whole lattices across region edges, so the local
// projection is re-gated on the consumption side (stepsInBlock,
// localRootsIn).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_ROOT_REACHABILITY_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_ROOT_REACHABILITY_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace hivm {
namespace otp {

/// Set of nz2nd fixpipe ops whose result reaches a value through
/// layout-agnostic single-use transparent ops. Bottom is both sets empty,
/// join is set union.
struct RootSet {
  SmallPtrSet<Operation *, 2> localRoots;
  SmallPtrSet<Operation *, 2> globalRoots;

  static RootSet fromRoot(Operation *root) {
    RootSet set;
    set.localRoots.insert(root);
    set.globalRoots.insert(root);
    return set;
  }

  static RootSet join(const RootSet &lhs, const RootSet &rhs) {
    RootSet result = lhs;
    result.localRoots.insert(rhs.localRoots.begin(), rhs.localRoots.end());
    result.globalRoots.insert(rhs.globalRoots.begin(), rhs.globalRoots.end());
    return result;
  }

  bool operator==(const RootSet &rhs) const {
    return localRoots == rhs.localRoots && globalRoots == rhs.globalRoots;
  }

  void print(raw_ostream &os) const {
    // Stream and return: some interleaveComma versions stream the result.
    auto printRoot = [&os](Operation *root) {
      os << root->getName();
      return root->getName();
    };
    os << "local";
    llvm::interleaveComma(localRoots, os, printRoot);
    os << " | global";
    llvm::interleaveComma(globalRoots, os, printRoot);
  }
};

/// Sparse forward dataflow analysis computing RootSet for every SSA value.
/// An nz2nd fixpipe with a tensor result injects itself; a transparent
/// single-use chain op joins the roots of its shaped DPS inputs (globalRoots
/// unconditionally, localRoots only along same-block steps); everything else
/// blocks both projections.
class RootReachabilityAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                                     dataflow::Lattice<RootSet>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<RootSet> *> operands,
                 ArrayRef<dataflow::Lattice<RootSet> *> results) override;

  /// Values that cannot be analyzed (function arguments, region entry block
  /// arguments) carry no roots.
  void setToEntryState(dataflow::Lattice<RootSet> *lattice) override;
};

/// The roots a same-block backward walk would find from `value` in `block`.
/// Empty unless `value` is defined in `block` by a steppable op.
SmallPtrSet<Operation *, 2> localRootsIn(const DataFlowSolver &solver,
                                         Value value, Block *block);

} // namespace otp
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_ROOT_REACHABILITY_H
