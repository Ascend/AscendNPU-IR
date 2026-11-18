//===- OptimizeTransposePlacement.cpp - sink vtranspose onto a fixpipe ----===//
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
// Sink a rank-2 hivm.hir.vtranspose permutation [1, 0] backwards across a
// straight-line chain of single-use layout-agnostic elementwise ops, so the
// transpose becomes the direct consumer of an nz2nd fixpipe.
// hivm-inline-fixpipe then folds that pair into one NZ2DN fixpipe.
//
// One-shot: the root-reachability dataflow solver runs once on the
// unchanged IR, every vtranspose candidate is analyzed against it, and the
// accepted plans apply in program order. A sibling of a chain op is either
// a matching single-use vtranspose [1, 0] (cancelled, its unary prefix
// replayed at the swapped shape) or a freely recreatable source
// (tensor.empty, scalar vbrc); anything else declines the candidate —
// keeping one transposed layout per chain value is what makes the rewrite
// safe. A plan is re-analyzed once when an earlier apply consumed one of
// its ops (a cancellable sibling that was itself a candidate). A scoped
// sweep erases the ops the applies orphaned.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/OptimizeTransposePlacement/RootReachability.h"
#include "bishengir/Dialect/HIVM/Transforms/OptimizeTransposePlacement/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "hivm-optimize-transpose-placement"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZETRANSPOSEPLACEMENT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::otp;

namespace {

/// One side operand of a chain op and what the rewrite needs to rebuild it
struct Sibling {
  Value value;
  /// The chain op that consumes `value`.
  Operation *user = nullptr;
  /// The matching single-use vtranspose [1, 0] to cancel.
  /// `cancelled == nullptr` means a freely recreatable source
  /// i.e (tensor.empty or scalar vbrc).
  VTransposeOp cancelled = nullptr;
  /// Unary agnostic ops between `cancelled` and `value`, producer to
  /// consumer. Replayed on `cancelled`'s source at the swapped shape.
  SmallVector<Operation *> replay;
};

struct SinkPlan {
  FixpipeOp root;
  /// Producer to consumer. Excludes the fixpipe and the candidate
  /// transpose; chain[0] consumes the root result.
  SmallVector<Operation *> chain;
  SmallVector<Sibling> siblings;
};

/// Whether the transpose result reaches at least one real user through
/// forward transparent single-use ops.
static bool hasRealLeaf(Value value, Block *block) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (isIgnorableUser(user) || !seen.insert(user).second)
        continue;
      if (user->getBlock() == block && isTransparentChainOp(user) &&
          hasSingleRealUse(user->getResult(0))) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      return true;
    }
  }
  return false;
}

/// Walk sibling back to another vtranspose[1, 0] to cancel them out.
static FailureOr<std::pair<VTransposeOp, SmallVector<Operation *>>>
matchCancellable(Value value, Block *block) {
  SmallVector<Operation *> reverseReplay;
  Value cur = value;
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (true) {
    Operation *def = cur.getDefiningOp();
    if (!def || def->getBlock() != block || !seen.insert(def).second)
      return failure();
    if (auto transpose = dyn_cast<VTransposeOp>(def)) {
      if (!isSwapTranspose(transpose) ||
          !hasSingleRealUse(transpose.getResult()[0]))
        return failure();
      SmallVector<Operation *> replay(llvm::reverse(reverseReplay));
      return std::make_pair(transpose, std::move(replay));
    }
    if (!isTransparentChainOp(def) || !hasSingleRealUse(def->getResult(0)))
      return failure();
    FailureOr<Value> shaped = getOnlyShapedInput(def);
    if (failed(shaped))
      return failure();
    reverseReplay.push_back(def);
    cur = *shaped;
  }
}

/// At this point solver already ran the RootReachabilityAnalysis.
/// 'reason' is used purely for debugging purposes.
static std::optional<SinkPlan> analyze(VTransposeOp transpose,
                                       const DataFlowSolver &solver,
                                       StringRef *reason = nullptr) {
  auto reject = [&](StringRef why) -> std::optional<SinkPlan> {
    if (reason)
      *reason = why;
    LDBG("skip: " << why);
    return std::nullopt;
  };
  if (!isSwapTranspose(transpose))
    return reject("not a rank-2 [1, 0] transpose");

  Block *block = transpose->getBlock();
  if (!hasRealLeaf(transpose.getResult()[0], block))
    return reject("transpose result has no real user");

  SinkPlan plan;
  Value cur = transpose.getSrc();
  SmallVector<Operation *> reverseChain;
  // The chain goes UP only, backward to one nz2nd fixpipe root.
  while (Operation *def = cur.getDefiningOp()) {
    if (def->getBlock() != block)
      return reject("chain crosses a block boundary");

    if (auto fixpipe = dyn_cast<FixpipeOp>(def)) {
      if (!fixpipe.getResultTensor() ||
          fixpipe.getDmaMode() != FixpipeDMAMode::NZ2ND ||
          !hasSingleRealUse(fixpipe.getResultTensor()))
        return reject("fixpipe is not a single-user nz2nd root");
      plan.root = fixpipe;
      break;
    }

    if (isa<LoadOp>(def))
      return reject("producer is hivm.hir.load");
    if (!isTransparentChainOp(def) || !hasSingleRealUse(def->getResult(0)))
      return reject("producer is not a single-use layout-agnostic op");
    if (failed(swap2D(def->getResult(0).getType())))
      return reject("shape is not static rank 2");

    auto dps = cast<DestinationStyleOpInterface>(def);
    for (Value init : dps.getDpsInits())
      if (failed(swap2D(init.getType())))
        return reject("init shape is not static rank 2");

    // Two roots (a diamond, or two fixpipes) currently not supported.
    SmallVector<Value> inputs = dps.getDpsInputs();
    Value chainSrc;
    for (Value input : inputs) {
      if (!isa<RankedTensorType>(input.getType()))
        continue;
      if (localRootsIn(solver, input, block).empty())
        continue;
      if (chainSrc)
        return reject("more than one shaped input reaches a root");
      chainSrc = input;
    }
    if (!chainSrc)
      return reject("no shaped input reaches a root");

    for (Value input : inputs) {
      if (input == chainSrc || !isa<RankedTensorType>(input.getType()))
        continue;
      if (!isRank2Static(input.getType()))
        return reject("sibling shape is not static rank 2");
      Sibling sibling;
      sibling.value = input;
      sibling.user = def;
      if (auto matched = matchCancellable(input, block); succeeded(matched)) {
        sibling.cancelled = matched->first;
        sibling.replay = std::move(matched->second);
        for (Operation *op : sibling.replay)
          if (failed(swap2D(op->getResult(0).getType())))
            return reject("replay shape is not static rank 2");
      } else if (!isFreelyTransposableSource(input)) {
        // Rebuilding this operand would need a fresh vtranspose.
        // This is not supported yet. (Cost of fresh vtranspose unknown)
        return reject("sibling is neither cancellable nor free");
      }
      plan.siblings.push_back(std::move(sibling));
    }
    cur = chainSrc;
    reverseChain.push_back(def);
  }

  if (!plan.root)
    return reject("no nz2nd fixpipe root");
  // Already direct consumer. InlineFixpipe folds already.
  if (reverseChain.empty())
    return reject("already the direct user of the fixpipe");
  plan.chain.assign(llvm::reverse(reverseChain).begin(),
                    llvm::reverse(reverseChain).end());
  if (reason)
    *reason = "ok";
  return plan;
}

/// Write one Mermaid flowchart for a candidate: solid edges along the
/// root-to-anchor chain, dashed edges for siblings (they are rebuilt, not
/// moved). A rejected candidate is a single node with the reject cause.
/// DEBUGGING PURPOSES ONLY.
static void dumpMermaid(VTransposeOp anchor, const SinkPlan *plan,
                        StringRef reason, int serial, StringRef dumpDir) {
  std::string path = (Twine(dumpDir) + "/otp-" + Twine(serial) + ".mmd").str();
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    LDBG("mermaid dump failed for " << path << ": " << ec.message());
    return;
  }
  Operation *scope = anchor->getParentOfType<func::FuncOp>();
  if (!scope)
    scope = anchor->getParentOfType<ModuleOp>();
  AsmState state(scope);

  DenseMap<Operation *, unsigned> ids;
  auto node = [&](Operation *op) {
    return ids.try_emplace(op, ids.size()).first->second;
  };
  auto emitNode = [&](Operation *op) {
    std::string text;
    llvm::raw_string_ostream ss(text);
    ss << op->getName().getStringRef();
    for (Value result : op->getResults()) {
      ss << "\\n";
      result.printAsOperand(ss, state);
    }
    os << "  n" << node(op) << "[\"" << ss.str() << "\"]\n";
  };
  auto emitEdge = [&](Operation *from, Operation *to, bool dashed) {
    os << "  n" << node(from) << (dashed ? " -.-> " : " --> ") << 'n'
       << node(to) << "\n";
  };

  os << "flowchart LR\n";
  emitNode(anchor);
  if (!plan) {
    os << "  %% rejected: " << reason << "\n";
    return;
  }
  emitNode(plan->root);
  Operation *prev = plan->root;
  for (Operation *link : plan->chain) {
    emitNode(link);
    emitEdge(prev, link, /*dashed=*/false);
    prev = link;
  }
  emitEdge(prev, anchor, /*dashed=*/false);
  for (const Sibling &sibling : plan->siblings) {
    if (sibling.cancelled) {
      emitNode(sibling.cancelled);
      Operation *tail = sibling.cancelled;
      for (Operation *op : sibling.replay) {
        emitNode(op);
        emitEdge(tail, op, /*dashed=*/true);
        tail = op;
      }
      emitEdge(tail, sibling.user, /*dashed=*/true);
    } else if (Operation *def = sibling.value.getDefiningOp()) {
      emitNode(def);
      emitEdge(def, sibling.user, /*dashed=*/true);
    }
  }
}

/// cloneWithSwappedLayout requires a replacement for every shaped DPS
/// input; check upfront so a violating plan is skipped before any rewrite.
static bool hasUnmappedShapedInput(Operation *op, Value chainSrc,
                                   const SmallPtrSetImpl<Value> &mapped) {
  auto dps = cast<DestinationStyleOpInterface>(op);
  for (Value input : dps.getDpsInputs()) {
    if (input == chainSrc || !isa<ShapedType>(input.getType()))
      continue;
    if (!mapped.contains(input))
      return true;
  }
  return false;
}

static LogicalResult validatePlan(SinkPlan &plan) {
  for (auto [index, link] : llvm::enumerate(plan.chain)) {
    Value chainSrc = index == 0 ? Value(plan.root.getResultTensor())
                                : plan.chain[index - 1]->getResult(0);
    SmallPtrSet<Value, 4> mapped;
    for (const Sibling &sibling : plan.siblings)
      if (sibling.user == link)
        mapped.insert(sibling.value);
    if (hasUnmappedShapedInput(link, chainSrc, mapped))
      return failure();
  }
  for (const Sibling &sibling : plan.siblings) {
    if (!sibling.cancelled)
      continue;
    // Replay clones remap only the single shaped input.
    for (Operation *op : sibling.replay)
      if (failed(getOnlyShapedInput(op)))
        return failure();
  }
  return success();
}

/// Applies one analyzed plan already proven safe; fails without touching
/// the IR when validatePlan declines.
/// `consumed` records every op this apply erases so the driver can tell when
/// a not-yet-applied plan references ops an earlier apply already erased.
static LogicalResult applyPlan(IRRewriter &rewriter, VTransposeOp transpose,
                               SinkPlan &plan,
                               SmallPtrSetImpl<Operation *> &consumed) {
  if (failed(validatePlan(plan))) {
    LDBG("skip: plan leaves a shaped input without replacement");
    return failure();
  }
  DenseMap<Value, Value> materialized;
  DenseMap<Operation *, Value> rewritten;
  auto materializeSibling = [&](Sibling &sibling) -> Value {
    if (auto found = materialized.find(sibling.value);
        found != materialized.end())
      return found->second;
    Value replaced;
    if (sibling.cancelled) {
      replaced = sibling.cancelled.getSrc();
      DenseMap<Value, Value> noSiblings;
      for (Operation *op : sibling.replay) {
        Value replaySrc = *getOnlyShapedInput(op);
        replaced = cloneWithSwappedLayout(rewriter, op, replaySrc, replaced,
                                          noSiblings);
        rewritten[op] = replaced;
      }
    } else {
      RankedTensorType swapped = *swap2D(sibling.value.getType());
      if (auto vbrc = sibling.value.getDefiningOp<VBrcOp>()) {
        Value init =
            createSwappedEmpty(rewriter, vbrc.getLoc(), vbrc.getDst(), swapped);
        auto rebuilt = rewriter.create<VBrcOp>(
            vbrc.getLoc(), TypeRange{swapped}, vbrc.getSrc(), init);
        replaced = rebuilt.getResult()[0];
      } else {
        replaced = createSwappedEmpty(rewriter, sibling.value.getLoc(),
                                      sibling.value, swapped);
      }
    }
    materialized[sibling.value] = replaced;
    return replaced;
  };

  rewriter.setInsertionPoint(transpose);
  RankedTensorType movedType = *swap2D(plan.root.getResultTensor().getType());
  Value movedInit = createSwappedEmpty(rewriter, transpose.getLoc(),
                                       plan.root.getDst(), movedType);
  Value movedSrc = plan.root.getResultTensor();
  // need to handle tempBuffer correctly. cast can change elem type
  Value carriedTemp;
  if (Value temp = transpose.getTempBuffer()) {
    auto memTy = dyn_cast<MemRefType>(temp.getType());
    if (memTy && memTy.getElementType() == movedType.getElementType() &&
        memTy.getShape() == movedType.getShape())
      carriedTemp = temp;
  }
  VTransposeOp moved;
  if (carriedTemp)
    moved = rewriter.create<VTransposeOp>(
        transpose.getLoc(), TypeRange{movedType}, movedSrc, movedInit,
        carriedTemp, transpose.getPermutationAttr());
  else
    moved = rewriter.create<VTransposeOp>(
        transpose.getLoc(), TypeRange{movedType}, movedSrc, movedInit,
        transpose.getPermutationAttr(), transpose.getDisableAlign());
  if (carriedTemp && transpose.getDisableAlign())
    moved.setDisableAlign(true);
  Value cur = moved.getResult()[0];
  rewriter.setInsertionPointAfter(moved);

  for (auto [index, link] : llvm::enumerate(plan.chain)) {
    DenseMap<Value, Value> siblingValues;
    for (Sibling &sibling : plan.siblings)
      if (sibling.user == link)
        siblingValues[sibling.value] = materializeSibling(sibling);
    Value chainSrc = index == 0 ? Value(plan.root.getResultTensor())
                                : plan.chain[index - 1]->getResult(0);
    cur = cloneWithSwappedLayout(rewriter, link, chainSrc, cur, siblingValues);
    rewritten[link] = cur;
    rewriter.setInsertionPointAfter(cur.getDefiningOp());
  }

  if (auto dps = dyn_cast<DestinationStyleOpInterface>(cur.getDefiningOp())) {
    for (Value init : dps.getDpsInits())
      overlayBufferAttrs(init, transpose.getDst());
  }

  rewriter.replaceAllUsesWith(transpose.getResult()[0], cur);
  eraseIfDead(rewriter, transpose);
  for (Operation *link : llvm::reverse(plan.chain))
    eraseIfDead(rewriter, link, rewritten.lookup(link));
  for (Sibling &sibling : plan.siblings) {
    if (!sibling.cancelled)
      continue;
    Value siblingRepl = materialized.lookup(sibling.value);
    for (Operation *op : llvm::reverse(sibling.replay))
      eraseIfDead(rewriter, op, rewritten.lookup(op));
    eraseIfDead(rewriter, sibling.cancelled, siblingRepl);
  }
  LDBG("sank vtranspose onto nz2nd fixpipe");

  // All of these were erased above: single-real-use by analysis.
  consumed.insert(transpose);
  for (Operation *link : plan.chain)
    consumed.insert(link);
  for (Sibling &sibling : plan.siblings) {
    if (!sibling.cancelled)
      continue;
    for (Operation *op : sibling.replay)
      consumed.insert(op);
    consumed.insert(sibling.cancelled);
  }
  return success();
}

/// Whether an earlier apply erased an op this plan recorded. Only a
/// cancellable sibling can be shared across plans; chain and root ops are
/// single-use per candidate.
static bool touchesConsumed(const SinkPlan &plan,
                            const SmallPtrSetImpl<Operation *> &consumed) {
  if (consumed.contains(plan.root))
    return true;
  for (Operation *link : plan.chain)
    if (consumed.contains(link))
      return true;
  for (const Sibling &sibling : plan.siblings) {
    if (!sibling.cancelled)
      continue;
    if (consumed.contains(sibling.cancelled))
      return true;
    for (Operation *op : sibling.replay)
      if (consumed.contains(op))
        return true;
  }
  return false;
}

/// Run the root-reachability solver on `top`. DeadCodeAnalysis and
/// SparseConstantPropagation are required, or region bodies (scf.for)
/// silently stay dead and roots go missing.
static LogicalResult runSolver(DataFlowSolver &solver, Operation *top) {
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<RootReachabilityAnalysis>();
  return solver.initializeAndRun(top);
}

struct OptimizeTransposePlacementPass
    : public impl::OptimizeTransposePlacementBase<
          OptimizeTransposePlacementPass> {
  void runOnOperation() override {
    // Ops already dead before any apply: the final sweep skips them, so the
    // pass cleans up only what it orphaned (no blanket DCE).
    SmallPtrSet<Operation *, 16> preExistingDead;
    getOperation()->walk([&](Operation *op) {
      if (!isIgnorableUser(op) && isOpTriviallyDead(op))
        preExistingDead.insert(op);
    });

    // Analyze every candidate on the unchanged IR first, then apply
    DataFlowSolver solver;
    if (failed(runSolver(solver, getOperation())))
      return;
    SmallVector<std::pair<VTransposeOp, std::optional<SinkPlan>>> plans;
    if (!dumpDir.empty())
      (void)llvm::sys::fs::create_directories(dumpDir);
    int serial = 0;
    getOperation()->walk([&](VTransposeOp transpose) {
      if (!isRegBaseScope(transpose))
        return;
      StringRef reason;
      plans.emplace_back(transpose, analyze(transpose, solver, &reason));
      if (!dumpDir.empty())
        dumpMermaid(transpose,
                    plans.back().second ? &*plans.back().second : nullptr,
                    reason, serial++, dumpDir);
    });

    IRRewriter rewriter(&getContext());
    SmallPtrSet<Operation *, 8> consumed;
    bool applied = false;
    for (auto &[anchor, plan] : plans) {
      if (!plan)
        continue;
      // A candidate cancelled as a sibling by an earlier apply is gone:
      // nothing left to place.
      if (consumed.contains(anchor))
        continue;
      if (touchesConsumed(*plan, consumed)) {
        // Plan went stale: re-analyze once on the current IR.
        DataFlowSolver fresh;
        if (failed(runSolver(fresh, getOperation())))
          continue;
        std::optional<SinkPlan> reanalyzed = analyze(anchor, fresh);
        if (!reanalyzed) {
          LDBG("overlap revalidation declined");
          continue;
        }
        applied |=
            succeeded(applyPlan(rewriter, anchor, *reanalyzed, consumed));
      } else {
        applied |= succeeded(applyPlan(rewriter, anchor, *plan, consumed));
      }
    }
    if (applied)
      sweepDeadOps(getOperation(), preExistingDead);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createOptimizeTransposePlacementPass() {
  return std::make_unique<OptimizeTransposePlacementPass>();
}
