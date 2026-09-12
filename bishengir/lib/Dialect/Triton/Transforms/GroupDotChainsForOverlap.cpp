//===- GroupDotChainsForOverlap.cpp - Direct dot-chain grouping -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass finds direct producer-to-consumer dot chains with unique C or
// A/B dependency edges.  It assigns group IDs to eligible chains so later
// tiling and lowering passes can preserve the chain and apply a fused
// accumulator schedule instead of treating each dot independently.
//
// C-chain grouping is currently enabled.  A/B-chain grouping is constructed
// for analysis but remains disabled until its lowering path is profitable.

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <optional>

using namespace mlir;

namespace bishengir {
#define GEN_PASS_DEF_GROUPDOTCHAINSFOROVERLAP
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace triton {
namespace {

static constexpr llvm::StringLiteral kCGroupedAttr =
    "bishengir.dot.c_grouped_for_overlap";
static constexpr llvm::StringLiteral kABGroupedAttr =
    "bishengir.dot.ab_grouped_for_overlap";
static constexpr llvm::StringLiteral kGroupIdAttr = "bishengir.dot.group_id";

enum class EdgeKind { C, AB };

struct Node {
  mlir::triton::DotOp op;
  SmallVector<Node *, 4> cPreds;
  SmallVector<Node *, 4> cSuccs;
  SmallVector<Node *, 4> abPreds;
  SmallVector<Node *, 4> abSuccs;
};

static bool is16x16f32Tensor(Type ty) {
  auto rt = dyn_cast<RankedTensorType>(ty);
  return rt && rt.getRank() == 2 && rt.getDimSize(0) == 16 &&
         rt.getDimSize(1) == 16 && rt.getElementType().isF32();
}

static bool isLegalForGrouping(mlir::triton::DotOp dot) {
  return dot && is16x16f32Tensor(dot.getResult().getType()) &&
         is16x16f32Tensor(dot.getA().getType()) &&
         is16x16f32Tensor(dot.getB().getType()) && dot->getBlock();
}

static mlir::triton::DotOp getDirectDotProducer(Value v) {
  return v.getDefiningOp<mlir::triton::DotOp>();
}

static DenseMap<Operation *, Node *>
buildNodes(mlir::triton::FuncOp func,
           SmallVectorImpl<std::unique_ptr<Node>> &storage) {
  DenseMap<Operation *, Node *> nodes;
  func.walk([&](mlir::triton::DotOp dot) {
    if (!isLegalForGrouping(dot))
      return;
    storage.push_back(std::make_unique<Node>());
    storage.back()->op = dot;
    nodes[dot.getOperation()] = storage.back().get();
  });
  return nodes;
}

static void buildEdges(DenseMap<Operation *, Node *> &nodes) {
  for (auto &[op, node] : nodes) {
    auto dot = cast<mlir::triton::DotOp>(op);
    struct PairInfo {
      bool c = false;
      bool ab = false;
      unsigned abCount = 0;
      Node *producer = nullptr;
    };
    DenseMap<Node *, PairInfo> pairs;

    auto record = [&](Value operand, EdgeKind kind) {
      auto producerDot = getDirectDotProducer(operand);
      if (!producerDot)
        return;
      auto it = nodes.find(producerDot.getOperation());
      if (it == nodes.end())
        return;
      Node *producer = it->second;
      auto &info = pairs[producer];
      info.producer = producer;
      if (kind == EdgeKind::C)
        info.c = true;
      else {
        info.ab = true;
        ++info.abCount;
      }
    };

    record(dot.getC(), EdgeKind::C);
    record(dot.getA(), EdgeKind::AB);
    record(dot.getB(), EdgeKind::AB);

    for (auto &[producer, info] : pairs) {
      if (info.c && info.ab)
        continue;
      if (info.ab && info.abCount > 1)
        continue;
      if (info.c) {
        node->cPreds.push_back(producer);
        producer->cSuccs.push_back(node);
      } else if (info.ab) {
        node->abPreds.push_back(producer);
        producer->abSuccs.push_back(node);
      }
    }
  }
}

static SmallVector<Node *> collectChain(Node *seed, EdgeKind kind,
                                        DenseSet<Node *> &visited) {
  SmallVector<Node *> chain;
  Node *cur = seed;
  while (cur && !visited.count(cur)) {
    chain.push_back(cur);
    visited.insert(cur);
    auto &succs = kind == EdgeKind::C ? cur->cSuccs : cur->abSuccs;
    if (succs.size() != 1)
      break;
    Node *next = succs.front();
    auto &preds = kind == EdgeKind::C ? next->cPreds : next->abPreds;
    if (preds.size() != 1 || visited.count(next))
      break;
    cur = next;
  }
  return chain;
}

static SmallVector<SmallVector<Node *>>
buildGroups(DenseMap<Operation *, Node *> &nodes, EdgeKind kind) {
  SmallVector<SmallVector<Node *>> groups;
  DenseSet<Node *> visited;
  SmallVector<Node *> roots;
  for (auto &[op, node] : nodes) {
    if ((kind == EdgeKind::C ? node->cPreds.empty() : node->abPreds.empty()))
      roots.push_back(node);
  }
  llvm::sort(roots, [](Node *lhs, Node *rhs) {
    return lhs->op->isBeforeInBlock(rhs->op);
  });

  for (Node *root : roots) {
    if (visited.count(root))
      continue;
    if (!isLegalForGrouping(root->op))
      continue;
    if (kind == EdgeKind::C ? !root->cSuccs.size() : !root->abSuccs.size())
      continue;
    auto chain = collectChain(root, kind, visited);
    if (chain.size() >= 2)
      groups.push_back(std::move(chain));
  }
  return groups;
}

static void markGroups(ArrayRef<SmallVector<Node *>> groups, StringRef attr,
                       unsigned groupIdOffset) {
  if (groups.empty())
    return;
  Builder b(groups.front().front()->op->getContext());
  for (auto [idx, group] : llvm::enumerate(groups)) {
    auto id = b.getI64IntegerAttr(groupIdOffset + idx);
    for (Node *node : group) {
      node->op->setAttr(attr, b.getUnitAttr());
      node->op->setAttr(kGroupIdAttr, id);
    }
  }
}

} // namespace

struct GroupDotChainsForOverlapPass
    : public impl::GroupDotChainsForOverlapBase<GroupDotChainsForOverlapPass> {
  using impl::GroupDotChainsForOverlapBase<
      GroupDotChainsForOverlapPass>::GroupDotChainsForOverlapBase;

  void runOnOperation() override {
    auto func = getOperation();
    SmallVector<std::unique_ptr<Node>> storage;
    auto nodes = buildNodes(func, storage);
    buildEdges(nodes);

    auto cGroups = buildGroups(nodes, EdgeKind::C);
    DenseSet<Operation *> selectedOps;
    for (auto &group : cGroups)
      for (Node *node : group)
        selectedOps.insert(node->op.getOperation());

    auto abGroups = buildGroups(nodes, EdgeKind::AB);
    abGroups.erase(llvm::remove_if(abGroups, [&](ArrayRef<Node *> group) {
                     return llvm::any_of(group, [&](Node *node) {
                       return selectedOps.count(node->op.getOperation());
                     });
                   }),
                   abGroups.end());

    markGroups(cGroups, kCGroupedAttr, 0);
    // Disable ab chain for now. Performance decreases with it on.
    if (false)
      markGroups(abGroups, kABGroupedAttr, cGroups.size());
  }
};

std::unique_ptr<mlir::Pass> createGroupDotChainsForOverlapPass() {
  return std::make_unique<GroupDotChainsForOverlapPass>();
}

} // namespace triton
} // namespace bishengir
