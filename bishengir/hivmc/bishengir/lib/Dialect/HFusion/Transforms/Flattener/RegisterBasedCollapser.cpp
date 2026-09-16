//===- RegisterBasedCollapser.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Flattener/Flattener.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "flattener-collapser"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {
void Flattener::adjustIndirectLoadOp(hfusion::IndirectLoadOp indirectLoadOp,
                                     mlir::OpBuilder &builder) {
  indirectLoadOp.getResult().setType(indirectLoadOp.getDst().getType());
}

template <typename OpTy>
void Flattener::adjustMemrefAccessOp(
    OpTy op, Value memref, OperandRange &indices, OpBuilder &builder,
    llvm::function_ref<void(SmallVector<Value> &)> prepareOperands) {

  auto collapseGroups = getCollapseGroup(memref);
  SmallVector<OpFoldResult> oldMixedSize = getFlattenMixedSizes(memref);

  builder.setInsertionPoint(op);
  Location loc = op.getLoc();

  // Define getDimSize lambda function.
  auto getDimSize = [&](int idx) -> Value {
    return getValueOrCreateConstantIndexOp(builder, loc, oldMixedSize[idx]);
  };

  // Compute new indices using the utility function.
  auto newIndices = hfusion::computeExtractCollapsedIndices(
      collapseGroups, indices, getDimSize, builder, loc);

  // Prepare the new operands.
  SmallVector<Value> newOperands;
  newOperands.reserve(newIndices.size() + 2);
  newOperands.push_back(memref);
  newOperands.append(newIndices);

  // Allow caller-specific operand preparation
  prepareOperands(newOperands);

  op->setOperands(newOperands);

  LLVM_DEBUG(llvm::dbgs() << "Ok " << op->getName() << " done\n";);
  LLVM_DEBUG(llvm::dbgs() << *op->template getParentOfType<func::FuncOp>(););
}

void Flattener::adjustMemrefLoadOp(memref::LoadOp loadOp,
                                   OpBuilder &builder) {
  LDBG("[adjustMemrefLoadOp] " << loadOp);
  auto indices = loadOp.getIndices();
  adjustMemrefAccessOp(loadOp, loadOp.getMemRef(), indices,
                       builder, [&](auto newOperands) {
                         updatePreviousType(loadOp.getResult());
                       });
}

void Flattener::adjustMemrefStoreOp(memref::StoreOp storeOp,
                                    OpBuilder &builder) {
  LDBG("[adjustMemrefStoreOp] " << storeOp);
  auto indices = storeOp.getIndices();
  adjustMemrefAccessOp(storeOp, storeOp.getMemRef(), indices,
                       builder, [&](SmallVector<Value> &newOperands) {
                         // For store: [value, memref, indices...]
                         newOperands.insert(newOperands.begin(),
                                            storeOp.getValue());
                       });
}


} // namespace detail
} // namespace hfusion
} // namespace mlir
