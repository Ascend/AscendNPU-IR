//===- InlineLoadCopy.cpp ----- inline copied load ------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_INLINELOADCOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-load-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

static inline bool isViewLikeOp(Operation *op) {
  return isa<ViewLikeOpInterface>(op);
}

// Collects all Values reachable from `root` by walking through view-like ops
// (subview, reshape, cast-like memref views, etc...).
static void collectViewClosure(Value root, DenseSet<Value> &closure) {
  // BFS
  SmallVector<Value, 8> worklist;
  auto push = [&](Value v) {
    if (closure.insert(v).second) {
      worklist.push_back(v);
    }
  };
  push(root);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;

    // go up
    if (isViewLikeOp(defOp)) {
      for (Value operand : defOp->getOperands()) {
        if (isa_and_present<BaseMemRefType>(operand.getType()))
          push(operand);
      }
      // go down
      for (Value res : defOp->getResults()) {
        if (res != v && isa_and_present<BaseMemRefType>(res.getType()))
          push(res);
      }
    }
  }
}

static bool writeBeforeCopy(Operation *writeOp, Operation *loadOp,
                            Operation *copyOp) {
  if (writeOp == loadOp) {
    return false;
  }
  if (writeOp->getBlock() == copyOp->getBlock()) {
    LDBG("write in same block \n");
    return writeOp->isBeforeInBlock(copyOp);
  }

  // Different blocks: be conservative. assume can reach
  // TODO: implement inter-block reachability
  LDBG("found write in different block, assume reachability \n");
  return true;
}

static bool readAfterLoad(Operation *readOp, Operation *loadOp,
                          Operation *copyOp) {
  if (readOp == copyOp)
    return false;

  if (readOp->getBlock() == loadOp->getBlock()) {
    LDBG("read in same block \n");
    return loadOp->isBeforeInBlock(readOp);
  }

  // Different blocks: be conservative. assume can reach
  // TODO: implement inter-block reachability
  return true;
}

static bool CopyBeforeLoad(Operation *loadOp, Operation *copyOp) {
  if (loadOp->getBlock() == copyOp->getBlock()) {
    return copyOp->isBeforeInBlock(loadOp);
  }

  // Different blocks, be conservative
  LDBG("Copy and load in different blocks, do not optimize \n");
  return true;
}
// we know order is always load then copy, need only check order on one side
static bool happensAfterLoadBeforeCopy(Operation *candidate, Operation *loadOp,
                                       Operation *copyOp) {
  if (candidate == loadOp || candidate == copyOp)
    return false;

  Block *bc = candidate->getBlock();
  Block *bl = loadOp->getBlock();
  Block *bp = copyOp->getBlock();

  if (bc == bl) {
    return loadOp->isBeforeInBlock(candidate);
  }

  if (bc == bp) {
    return candidate->isBeforeInBlock(copyOp);
  }

  // Different blocks: be conservative. assume can reach
  // TODO: implement inter-block reachability
  return true;
}
// For a general scenario : load A -> B
//                          copy B -> C
// we need check constraints differently on B and A
enum class BufferRole {
  CopySrc, // Middle buffer B
  LoadSrc  // Source buffer A
};

// Returns true if any intervening op performs a dangerous
// memory effect on the memory in memref "src" based on its role.
static bool hasInterveningMemoryEffect(Value src, hivm::LoadOp matchedLoad,
                                       hivm::CopyOp copyOp, BufferRole role) {

  // Build an alias set consisting of all view-like projections that are
  // reachable from the source buffer or its original allocated memory
  DenseSet<Value> aliasSet;

  if (auto maybeAllocSrc =
          utils::tracebackMemRefToAlloc(src).value_or(nullptr)) {
    collectViewClosure(maybeAllocSrc.getResult(), aliasSet);
  }
  collectViewClosure(src, aliasSet);

  SmallVector<Value, 16> toVisit(aliasSet.begin(), aliasSet.end());
  DenseSet<Operation *> visitedOps;

  auto enqueueViewOutputs = [&](Operation *op) {
    for (Value res : op->getResults()) {
      if (isa<BaseMemRefType>(res.getType()) &&
          aliasSet.insert(res).second)
        toVisit.push_back(res);
    }
  };

  while (!toVisit.empty()) {
    Value v = toVisit.pop_back_val();

    for (Operation *user : v.getUsers()) {
      if (user == matchedLoad)
        continue;
      if (!visitedOps.insert(user).second)
        continue;

      // view-like -> trace further
      if (isViewLikeOp(user)) {
        enqueueViewOutputs(user);
        for (Value opnd : user->getOperands()) {
          if (isa<BaseMemRefType>(opnd.getType()) &&
              aliasSet.insert(opnd).second)
            toVisit.push_back(opnd);
        }
        continue;
      }

      // use memory effect interface to check if writes/reads
      if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance, 8> effects;
        memEffect.getEffects(effects);

        for (const auto &eff : effects) {
          bool isWrite = isa<MemoryEffects::Write>(eff.getEffect());
          bool isRead = isa<MemoryEffects::Read>(eff.getEffect());

          if (!isWrite && !isRead)
            continue;

          bool aliases = true;
          if (Value effected = eff.getValue()) {
            aliases = aliasSet.contains(effected);
          } else {
            LDBG("effect on unknown value, do not optimize\n");
            return true;
          }

          if (!aliases)
            continue;

          if (isWrite) {
            LDBG("aliasSet Lookup found a write (could be fine)\n");
            if (role == BufferRole::CopySrc) {
              if (writeBeforeCopy(user, matchedLoad, copyOp)) {
                LDBG("position of write dangerous, do not optimize \n");
                return true;
              }
            } else if (role == BufferRole::LoadSrc) {
              if (happensAfterLoadBeforeCopy(user, matchedLoad, copyOp)) {
                LDBG("write to load source A occurs between load and copy; do "
                     "not optimize\n");
                return true;
              }
            }
          } else if (isRead && role == BufferRole::CopySrc) {
            // For LoadSrc, we don't care about intervening reads, only for CopySrc
            LDBG("aliasSet Lookup found a read (could be fine) \n");
            if (readAfterLoad(user, matchedLoad, copyOp)) {
              LDBG("position of read dangerous, do not optimize \n");
              return true;
            }
          }
        }
        enqueueViewOutputs(user);
      } else {
        // Unknown op kind without memory effects interface
      }
    }
  }
  return false;
}

struct LoadCopyInlinePattern : public OpRewritePattern<hivm::CopyOp> {
  using OpRewritePattern<hivm::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (!copyOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(copyOp,
                                         " op should have buffer semantics.");
    }
    auto src = copyOp.getSrc();
    hivm::LoadOp matchedLoad = nullptr;
    for (Operation *user : src.getUsers()) {
      auto load = dyn_cast<hivm::LoadOp>(user);
      if (!load)
        continue;
      // Ensure the load writes into exactly this buffer.
      if (load.getDst() != src)
        continue;

      matchedLoad = load;
      break;
    }

    if (!matchedLoad) {
      return rewriter.notifyMatchFailure(
          copyOp, "no LoadOp found that writes into copy src");
    }

    if (CopyBeforeLoad(matchedLoad, copyOp)) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "copy and load are not in order");
    }

    bool memoryEffected =
        hasInterveningMemoryEffect(src, matchedLoad, copyOp, BufferRole::CopySrc);
    if (memoryEffected) {
      return rewriter.notifyMatchFailure(
          copyOp, "Cannot optimize, middle buffer is used");
    }
    
    Value loadSrcA = matchedLoad.getSrc();
    bool loadSrcEffected =
        hasInterveningMemoryEffect(loadSrcA, matchedLoad, copyOp, BufferRole::LoadSrc);
    if (loadSrcEffected) {
      return rewriter.notifyMatchFailure(
          copyOp, "Cannot optimize, source buffer is used");
    }

    rewriter.replaceOpWithNewOp<hivm::LoadOp>(
        copyOp, TypeRange{}, matchedLoad.getSrc(), copyOp.getDst(),
        matchedLoad.getPadModeAttr(), matchedLoad.getPadValue(),
        matchedLoad.getLeftPaddingNum(), matchedLoad.getRightPaddingNum());
    rewriter.eraseOp(matchedLoad);
    return success();
  }
};

struct InlineLoadCopyPass
    : public impl::InlineLoadCopyBase<InlineLoadCopyPass> {
public:
  void runOnOperation() override;
};
} // namespace

void InlineLoadCopyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LoadCopyInlinePattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createInlineLoadCopyPass() {
  return std::make_unique<InlineLoadCopyPass>();
}
