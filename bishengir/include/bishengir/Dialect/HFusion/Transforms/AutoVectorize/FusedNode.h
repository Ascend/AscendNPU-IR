//===- FusedNode.h - Fusion group for AutoVectorizeV2 planning ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSEDNODE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSEDNODE_H

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include <memory>
#include <string>

namespace mlir {

class Block;
class Operation;

namespace hfusion {

class PlanContext;
struct FusableOpInfo;

/// Per-call-site guard subsets, preserved exactly to keep behavioral
/// equivalence during refactoring. The three flags differ because the old
/// code was inconsistent; the gaps are intentionally not fixed here.
///
/// TODO: Unify the three flags after verifying no regression:
/// - isMemref:       investigate whether Producer context also needs it.
/// - reverse vsstb:  enable in Producer/FallbackLeaf (pure bugfix).
/// - all-ops cap:    enable in Sibling/FallbackLeaf (pure bugfix).
///
/// planFuseSiblingForLeafNodes:
/// isMemref, reverse vsstb, leaf cap.
/// findBestFusedNodeForProducer:
/// no isMemref, no reverse vsstb, all-ops cap.
/// planFuseProducerIntoFusedNode fallback:
/// isMemref, no reverse vsstb, leaf cap.
enum class AcceptContext {
  Sibling,
  Producer,
  FallbackLeaf,
};

class FusedNode : public std::enable_shared_from_this<FusedNode> {
  PlanContext &ctx;
  std::string loopLabel;
  SetVector<Operation *> leaf;
  SetVector<Operation *> producers;

public:
  FusedNode(PlanContext &ctx, const std::string &loopLabel)
      : ctx(ctx), loopLabel(loopLabel) {}
  const std::string &getLabel() const { return loopLabel; }

  size_t leafCnt() const { return leaf.size(); }
  size_t size() const { return leaf.size() + producers.size(); }

  bool containsLeaf(Operation *op) const { return leaf.contains(op); }
  bool containsProducer(Operation *op) const { return producers.contains(op); }
  bool contains(Operation *op) const {
    return containsLeaf(op) || containsProducer(op);
  }

  auto ops() const { return llvm::concat<Operation *const>(leaf, producers); }
  auto leafOps() const { return llvm::make_range(leaf.begin(), leaf.end()); }
  auto producerOps() const {
    return llvm::make_range(producers.begin(), producers.end());
  }

  void addProducer(Operation *op);
  void addLeaf(Operation *op);

  bool canAccept(Operation *candidate,
                 AcceptContext actx = AcceptContext::Sibling) const;

  bool hasVsstb() const {
    return llvm::any_of(leaf, analysis::isVsstbPatternTransposeOp);
  }
  bool isMemref() const;

  // Estimate the tile size for `fusedOp` within this fused node.
  void estimateTileSizeForOp(Operation *fusedOp,
                             SmallVectorImpl<int64_t> &tileSize,
                             SmallVectorImpl<int64_t> &tileInterchange) const;

  // Whether `producer` can be legally fused into this node (tile-local +
  // reduction-domain checks).
  bool canFuseProducer(Operation *producer) const;

private:
  // Whether `producer` would become tile-local if fused into this node.
  bool wouldBecomeTileLocal(Operation *producer,
                            SmallVectorImpl<int64_t> &estimatedTileSize) const;

  // Whether any reduction consumer of `producer` within this node needs the
  // full (untiled) producer domain.
  bool
  reductionConsumerNeedsFullDomain(Operation *producer,
                                   ArrayRef<int64_t> estimatedTileSize) const;

  bool conflictsWith(const FusableOpInfo &otherInfo) const;
  bool hasCommonAxis(Operation *op) const;
  bool canFitStack(Operation *candidate) const;
  bool hasRankReducing() const;

  void consolidateToTail() const;
  void updateConflicts(Operation *newOp) const;

  bool useAsDpsInits(Operation *op) const;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSEDNODE_H
