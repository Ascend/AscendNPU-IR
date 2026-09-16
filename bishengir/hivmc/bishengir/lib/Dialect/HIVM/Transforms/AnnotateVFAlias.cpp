//===----------------------- AnnotateVFAlias.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_ANNOTATEVFALIAS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using AliasGroup = DenseSet<Operation *>;
namespace {

static constexpr llvm::StringRef kNoAliasScope = "llvm.noalias_scopes";

struct AnnotateVFAliasPass
    : public impl::AnnotateVFAliasBase<AnnotateVFAliasPass> {
  using AnnotateVFAliasBase<AnnotateVFAliasPass>::AnnotateVFAliasBase;

public:
  void runOnOperation() override;
};
} // namespace

static Value traceMemref(VectorTransferOpInterface transferOp) {
  auto src = transferOp.getSource();
  if (isa<BlockArgument>(src)) {
    return src;
  }
  Value baseMemref = src;
  while (auto defOp = baseMemref.getDefiningOp()) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      baseMemref = subviewOp.getSource();
    } else {
      llvm::dbgs() << "unrecognized intermediate Op!\n";
      break;
    }
  }
  return baseMemref;
}

static void removeDuplicateAliasScopes(SmallVector<AliasGroup> &aliasScopes) {
  SmallVector<AliasGroup> uniqueScopes;
  for (auto &aliasScope : aliasScopes) {
    bool isUnique = true;
    for (auto &existingScope : uniqueScopes) {
      if (existingScope == aliasScope) {
        isUnique = false;
        break;
      }
    }
    if (isUnique) {
      uniqueScopes.push_back(aliasScope);
    }
  }
  aliasScopes = std::move(uniqueScopes);
}

static SmallVector<AliasGroup>
initGroups(SmallVector<Operation *, 4> &memReadsAndWrites) {
  SmallVector<AliasGroup> aliasGroups;
  for (auto *transferOp : memReadsAndWrites) {
    AliasGroup group;
    group.insert(transferOp);
    aliasGroups.push_back(group);
  }
  return aliasGroups;
}

static void populateAliasGroups(SmallVector<Operation *, 4> &memReadsAndWrites,
                                SmallVector<AliasGroup> &aliasGroups,
                                DenseMap<Operation *, Value> &baseMemrefs) {
  for (auto *transferOp : memReadsAndWrites) {
    auto srcTransferOp = cast<VectorTransferOpInterface>(transferOp);
    Value src = srcTransferOp.getSource();
    auto curBaseMemref = baseMemrefs[transferOp];
    for (AliasGroup &group : aliasGroups) {
      if (group.contains(transferOp)) {
        continue;
      }
      bool aliasWithAll = true;
      for (Operation *targetOp : group) {
        auto targetTransferOp = cast<VectorTransferOpInterface>(targetOp);
        Value target = targetTransferOp.getSource();
        auto targetBaseMemref = baseMemrefs[targetOp];
        // transfer Op with same source value
        if (target == src &&
            vector::isDisjointTransferSet(targetTransferOp, srcTransferOp)) {
          aliasWithAll = false;
          break;
        }
        // Check if both points to the same buffer (before subview)
        // TODO: array section alias anaylsis
        if (curBaseMemref != targetBaseMemref) {
          aliasWithAll = false;
          break;
        }
      }
      // src transfer op alias with any other transfer ops in this group
      if (aliasWithAll) {
        group.insert(transferOp);
      }
    }
  }
}

static SmallVector<Attribute> getOrCreateAliasScopes(Operation *op) {
  if (!op->hasAttr(LLVM::AliasScopeAttr::name)) {
    return SmallVector<Attribute>();
  }
  Attribute attr = op->getAttr(LLVM::AliasScopeAttr::name);
  ArrayAttr aliasScopeArrayAttr = mlir::dyn_cast<ArrayAttr>(attr);
  assert(aliasScopeArrayAttr);
  SmallVector<Attribute> aliasScopes(aliasScopeArrayAttr.getValue());
  return aliasScopes;
}

static void setAliasScopes(Operation *op, SmallVector<Attribute> &aliasScopes) {
  OpBuilder b(op->getContext());
  auto updatedAliasScopeArrayAttr = b.getArrayAttr(aliasScopes);
  Attribute newAttr = dyn_cast<Attribute>(updatedAliasScopeArrayAttr);
  op->setAttr(LLVM::AliasScopeAttr::name, newAttr);
}

static SmallVector<Attribute> getOrCreateNoAliasScopes(Operation *op) {
  if (!op->hasAttr(kNoAliasScope)) {
    return SmallVector<Attribute>();
  }
  Attribute attr = op->getAttr(kNoAliasScope);
  ArrayAttr aliasScopeArrayAttr = mlir::dyn_cast<ArrayAttr>(attr);
  assert(aliasScopeArrayAttr);
  SmallVector<Attribute> aliasScopes(aliasScopeArrayAttr.getValue());
  return aliasScopes;
}

static void setAliasNoScopes(Operation *op,
                             SmallVector<Attribute> &aliasScopes) {
  OpBuilder b(op->getContext());
  auto updatedAliasScopeArrayAttr = b.getArrayAttr(aliasScopes);
  Attribute newAttr = dyn_cast<Attribute>(updatedAliasScopeArrayAttr);
  op->setAttr(kNoAliasScope, newAttr);
}

static void
populateAliasScopes(SmallVector<AliasGroup> &aliasGroups,
                    DenseMap<LLVM::AliasScopeAttr, AliasGroup> &aliasScopeMap,
                    LLVM::AliasScopeDomainAttr aliasScopeDomainAttr) {
  for (AliasGroup &group : aliasGroups) {
    // skip alias group with only 1 / 0 element
    // since it does not alias with any other op
    if (group.size() <= 1) {
      continue;
    }
    // create a llvm alias_scope attr
    auto aliasScopeAttr = LLVM::AliasScopeAttr::get(aliasScopeDomainAttr);
    for (Operation *op : group) {
      auto aliasScopes = getOrCreateAliasScopes(op);
      aliasScopes.push_back(dyn_cast<Attribute>(aliasScopeAttr));
      setAliasScopes(op, aliasScopes);
    }
    aliasScopeMap.try_emplace(aliasScopeAttr, group);
  }
}

static void populateNoAliasScopes(
    OpBuilder &b, SmallVector<AliasGroup> &aliasGroups,
    SmallVector<Operation *, 4> memReadsAndWrites,
    DenseMap<LLVM::AliasScopeAttr, AliasGroup> aliasScopeMap) {
  for (Operation *op : memReadsAndWrites) {
    bool hasAliasInfo = false;
    for (auto &[scopeAttr, group] : aliasScopeMap) {
      if (!group.contains(op)) {
        hasAliasInfo = true;
        auto noAliasScopes = getOrCreateNoAliasScopes(op);
        noAliasScopes.push_back(dyn_cast<Attribute>(scopeAttr));
        setAliasNoScopes(op, noAliasScopes);
      }
    }
    if (hasAliasInfo) {
      op->setAttr(hivm::HasAliaScopesAttr::name, b.getUnitAttr());
    }
  }
}

void AnnotateVFAliasPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;
  MLIRContext *ctx = funcOp->getContext();
  // record each memory read/write operation, trace its memrefVal
  SmallVector<Operation *, 4> memReadsAndWrites;
  DenseMap<Operation *, Value> baseMemrefs;
  funcOp->walk([&](Operation *op) {
    if (isa<VectorTransferOpInterface>(op)) {
      memReadsAndWrites.push_back(op);
      Value baseMemref = traceMemref(cast<VectorTransferOpInterface>(op));
      baseMemrefs.insert(std::make_pair(op, baseMemref));
    }
    // TODO: Scalar read/write
  });

  auto aliasGroups = initGroups(memReadsAndWrites);

  populateAliasGroups(memReadsAndWrites, aliasGroups, baseMemrefs);

  removeDuplicateAliasScopes(aliasGroups);

  OpBuilder b(ctx);
  auto aliasScopeDomainAttr = LLVM::AliasScopeDomainAttr::get(
      ctx, b.getStringAttr(funcOp.getSymName()));

  DenseMap<LLVM::AliasScopeAttr, AliasGroup> aliasScopeMap;

  // populate alias_scopes
  populateAliasScopes(aliasGroups, aliasScopeMap, aliasScopeDomainAttr);

  // populate noalias_scopes
  populateNoAliasScopes(b, aliasGroups, memReadsAndWrites, aliasScopeMap);
}

std::unique_ptr<Pass> hivm::createAnnotateVFAliasPass() {
  return std::make_unique<AnnotateVFAliasPass>();
}
