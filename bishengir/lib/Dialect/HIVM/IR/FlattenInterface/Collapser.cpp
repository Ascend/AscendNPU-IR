//===- Collapser.cpp - Collapser utilities for FlattenInterface -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#define DEBUG_TYPE "flatten-collapser"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

namespace mlir::hivm {
namespace detail {
// Helper function to collapse a type if it's a MemRefType
Type collapseTypeIfMemRef(Type type,
                          ArrayRef<ReassociationIndices> reassociation) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memref::CollapseShapeOp::computeCollapsedType(memrefType,
                                                         reassociation);
  }
  return type;
}

// Helper function to collapse a collection of types
static SmallVector<KindTypePair>
collapseTypes(ArrayRef<KindTypePair> types,
              ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<KindTypePair> collapsedTypes;
  collapsedTypes.reserve(types.size());
  for (const KindTypePair &kindTypePair : types) {
    collapsedTypes.emplace_back(
        kindTypePair.first,
        collapseTypeIfMemRef(kindTypePair.second, reassociation));
  }
  return collapsedTypes;
}

FlattenResult
collapseOperandsUniformly(FlattenResult &payload,
                          ArrayRef<ReassociationIndices> reassociation) {
  LDBG(to_string(reassociation));
  FlattenResult res = payload;
  res.reassociation = {llvm::to_vector(reassociation)};
  res.operandTypes = collapseTypes(payload.operandTypes, reassociation);
  return res;
}
} // namespace detail
} // namespace mlir::hivm