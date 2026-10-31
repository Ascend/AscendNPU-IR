//===- RootReachability.cpp - dataflow root reachability for OTP ---------===//
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

#include "bishengir/Dialect/HIVM/Transforms/OptimizeTransposePlacement/RootReachability.h"
#include "bishengir/Dialect/HIVM/Transforms/OptimizeTransposePlacement/Utils.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {
namespace otp {

using RootLattice = dataflow::Lattice<RootSet>;

/// Per-step condition: `value` continues a same-block chain when it is
/// defined in `block` by an nz2nd fixpipe with a tensor result (a root) or
/// by a transparent op whose result has a single real use. Re-applied
/// wherever localRoots is consumed, so the local projection stays the
/// same-block relation even where the framework joined cross-block
/// information into the lattice.
static bool stepsInBlock(Value value, Block *block) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block)
    return false;
  if (auto fixpipe = dyn_cast<FixpipeOp>(def))
    return fixpipe.getDmaMode() == FixpipeDMAMode::NZ2ND &&
           fixpipe.getResultTensor();
  return isTransparentChainOp(def) && hasSingleRealUse(def->getResult(0));
}

LogicalResult
RootReachabilityAnalysis::visitOperation(Operation *op,
                                         ArrayRef<const RootLattice *> operands,
                                         ArrayRef<RootLattice *> results) {
  // An nz2nd fixpipe with a tensor result is an absorption root. No
  // single-use gate here: reaching a multi-user fixpipe is reported, and
  // rejecting it is a later policy decision.
  if (auto fixpipe = dyn_cast<FixpipeOp>(op)) {
    if (fixpipe.getDmaMode() == FixpipeDMAMode::NZ2ND &&
        fixpipe.getResultTensor()) {
      for (auto [lattice, result] : llvm::zip(results, op->getResults()))
        if (result == fixpipe.getResultTensor())
          propagateIfChanged(lattice, lattice->join(RootSet::fromRoot(op)));
    }
    return success();
  }

  // Roots flow only through layout-agnostic transparent ops whose result
  // has a single real use. isTransparentChainOp also requires exactly one
  // result.
  if (!isTransparentChainOp(op) || !hasSingleRealUse(op->getResult(0)))
    return success();

  // Only DPS inputs carry roots. Inits are never walked.
  auto dps = cast<DestinationStyleOpInterface>(op);
  llvm::SmallPtrSet<Value, 4> dpsInputs;
  for (Value input : dps.getDpsInputs())
    dpsInputs.insert(input);

  RootSet reached;
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!dpsInputs.contains(operand) || !isa<ShapedType>(operand.getType()))
      continue;
    const RootSet &input = operands[index]->getValue();
    reached.globalRoots.insert(input.globalRoots.begin(),
                               input.globalRoots.end());
    // Local roots flow in only along a same-block step.
    if (stepsInBlock(operand, op->getBlock()))
      reached.localRoots.insert(input.localRoots.begin(),
                                input.localRoots.end());
  }
  propagateIfChanged(results[0], results[0]->join(reached));
  return success();
}

void RootReachabilityAnalysis::setToEntryState(RootLattice *lattice) {
  // The empty set: values we cannot analyze carry no roots.
  propagateIfChanged(lattice, lattice->join(RootSet()));
}

SmallPtrSet<Operation *, 2> localRootsIn(const DataFlowSolver &solver,
                                         Value value, Block *block) {
  SmallPtrSet<Operation *, 2> none;
  if (!stepsInBlock(value, block))
    return none;
  const RootLattice *lattice = solver.lookupState<RootLattice, Value>(value);
  if (!lattice)
    return none;
  return lattice->getValue().localRoots;
}

} // namespace otp
} // namespace hivm
} // namespace mlir
