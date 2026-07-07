//===- Utils.cpp - Utilities to support the HIVM dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HIVM/Utils/MemrefAliasAnalysisState.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include <cassert>
#include <cstdint>
#include <sys/param.h>

#define DEBUG_TYPE "hivm-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {

namespace utils {
MemrefAliasAnalysisState::MemrefAliasAnalysisState(Operation *op) {
  op->walk([&](Operation *op) {
    for (Value v : op->getResults())
      if (isa<MemRefType>(v.getType()))
        createAliasInfoEntry(v);
    for (Region &r : op->getRegions())
      for (Block &b : r.getBlocks())
        for (auto bbArg : b.getArguments())
          if (isa<MemRefType>(bbArg.getType()))
            createAliasInfoEntry(bbArg);
  });
}

void MemrefAliasAnalysisState::unionAliasSets(Value v1, Value v2) {
  if (!isa<MemRefType>(v1.getType()) || !isa<MemRefType>(v2.getType()))
    return;
  aliasInfo.unionSets(v1, v2);
}

void MemrefAliasAnalysisState::createAliasInfoEntry(Value v) {
  aliasInfo.insert(v);
}

bool MemrefAliasAnalysisState::isAlias(Value v1, Value v2) {
  return aliasInfo.isEquivalent(v1, v2);
}

void MemrefAliasAnalysisState::printAlias() {
  LDBG("=== Memref Alias Analysis State ===");

  for (auto it = aliasInfo.begin(), e = aliasInfo.end(); it != e; ++it) {
    if (!(*it)->isLeader())
      continue;

    Value leader = (*it)->getData();
    LDBG("Alias Set (Leader: " << leader << "):");
    for (auto memberIt = aliasInfo.member_begin(**it);
         memberIt != aliasInfo.member_end(); ++memberIt) {
      Value val = *memberIt;
      LDBG("  - " << val);
    }
    LDBG("-----------------------------------");
  }
}
} // namespace utils
} // namespace mlir
