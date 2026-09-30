//===- LinearScanFusion.cpp - Linear-scan VF fusion ----------------------===//
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

#include "bishengir/Dialect/Analysis/Partition/HIVM/VFFusionHIVM.h"
#include "bishengir/Dialect/Analysis/Partition/Partition.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionOutliner.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vf-fusion-hivm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis::partition::hivm {

Groups sameBlockRelation(GroupRef group) {
  DenseMap<Block *, Group> blockToOps;

  for (auto *op : group)
    op->walk<WalkOrder::PreOrder>([&blockToOps](Operation *op) {
      blockToOps[op->getBlock()].push_back(op);
    });

  Groups out;
  for (auto &[block, ops] : blockToOps)
    out.emplace_back(std::move(ops));
  return out;
}

Groups filterFuncOpRelation(GroupRef group) {
  Groups out{{}};

  for (auto *op : group)
    if (!isa<func::FuncOp>(op))
      out.back().push_back(op);

  if (out.back().empty())
    return {};
  return out;
}

Groups linearScanVectorizableOpsRelation(GroupRef group) {
  Groups out{{}};

  for (auto *op : group)
    if (isa<mlir::hivm::VectorizableOpInterface>(op))
      out.back().push_back(op);
    else if (!out.back().empty())
      out.emplace_back(Group{});

  if (out.back().empty())
    out.pop_back();
  return out;
}

LogicalResult LinearScanFusion::rewriteImpl(Group &&group) {
  VFFusionBlock fusedBlock{};

  LDBG("rewriting group: size=" << group.size());
  for (auto &op : group) {
    LDBG("  op: " << op->getName());
  }
  handoverGroupForOutlining(std::move(group));
  return success();
}

} // namespace mlir::analysis::partition::hivm
