//===-------------------- SplitMixedIfConditionals.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass splits if conditionals for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_SPLITMIXEDIFCONDITIONALS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "split-mixed-if-conditionals"

using hivm::detail::queryCoreTypeHelper;

namespace {
struct SplitMixedIfConditionalsPass
    : public impl::SplitMixedIfConditionalsBase<SplitMixedIfConditionalsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // anonymous namespace

// Check if an operation is a region operation (if, for, etc.)
static bool isRegionOperation(Operation *op) { return op->getNumRegions() > 0; }

static std::optional<TCoreType> getRegionOperationCoreType(Operation *op) {
  if (!isRegionOperation(op)) {
    return std::nullopt;
  }

  bool hasC = false, hasV = false;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      auto [blockHasC, blockHasV] = analyzeCoreTypes(&block);
      hasC = hasC || blockHasC;
      hasV = hasV || blockHasV;
    }
  }

  if (hasC && hasV) {
    return std::nullopt;
  }

  if (hasC) {
    return TCoreType::CUBE;
  } else if (hasV) {
    return TCoreType::VECTOR;
  }

  return std::nullopt;
}

struct OperationGroup {
  SmallVector<Operation *> coreOps;
  SmallVector<Operation *> nonCoreOps;
  SmallVector<Operation *> allOps;
  std::optional<TCoreType> coreType;

  bool hasCoreOps() const { return !coreOps.empty(); }
  bool isEmpty() const { return coreOps.empty() && nonCoreOps.empty(); }

  SmallVector<Value> getAllResults() const {
    SmallVector<Value> results;

    for (Operation *op : allOps) {
      for (Value res : op->getResults()) {
        results.push_back(res);
      }
    }
    return results;
  }

  SmallVector<Value> getExternallyUsedResults() const {
    DenseSet<Operation *> groupOps(allOps.begin(), allOps.end());
    SmallVector<Value> results;
    for (Operation *op : allOps) {
      for (Value res : op->getResults()) {
        if (llvm::any_of(res.getUsers(), [&](Operation *user) {
              return !groupOps.contains(user);
            })) {
          results.push_back(res);
        }
      }
    }
    return results;
  }
};

struct CloneContext {
  OpBuilder &builder;
  Location loc;
  std::shared_ptr<OperationGroup> group;
  ArrayRef<Value> externallyUsed;
  DenseMap<Value, Value> &valueMap;
  Block *originalThenBlock;
  Block *originalElseBlock;
};

struct SplitIfContext {
  std::shared_ptr<OperationGroup> group;
  Value cond;
  Location loc;
  bool isThenBlock;
  DenseMap<Value, Value> &valueMap;
  PatternRewriter &rewriter;
  Block *originalThenBlock;
  Block *originalElseBlock;
};

struct BlockProcessContext {
  Block *block;
  bool isThenBlock;
  Value cond;
  Location loc;
  ArrayRef<Value> thenYields;
  ArrayRef<Value> elseYields;
  PatternRewriter &rewriter;
  Block *originalThenBlock;
  Block *originalElseBlock;
};

struct GroupResultsContext {
  SmallVector<Value> &externallyUsed;
  SmallVector<Value> &newResults;
  ArrayRef<Value> thenYields;
  ArrayRef<Value> elseYields;
  DenseMap<Value, Value> &valueMap;
  DenseMap<Value, Value> &yieldMap;
};

struct FinalIfContext {
  PatternRewriter &rewriter;
  Location loc;
  Value cond;
  Value thenResult;
  Value elseResult;
};

// Check if a region has mixed core operations
static bool regionOpMixedCoreTypes(Operation *op) {
  if (isa<scf::IfOp>(op)) {
    return false;
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      auto [hasC, hasV] = analyzeCoreTypes(&block);
      if (hasC && hasV)
        return true;

      for (Operation &inner : block) {
        if (isRegionOperation(&inner) && regionOpMixedCoreTypes(&inner))
          return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Operation Grouping
//===----------------------------------------------------------------------===//

// Group operations in a block by core type
static SmallVector<std::shared_ptr<OperationGroup>>
groupOperations(Block *block) {
  SmallVector<std::shared_ptr<OperationGroup>> groups;
  auto currentGroup = std::make_shared<OperationGroup>();

  for (Operation &op : block->getOperations()) {
    if (isa<scf::YieldOp>(&op)) {
      continue;
    }

    auto directCoreType = queryCoreTypeHelper(&op);
    auto coreType = directCoreType;

    if (!coreType.has_value() && isRegionOperation(&op)) {
      coreType = getRegionOperationCoreType(&op);
    }

    if (coreType.has_value()) {
      // Core-type operation
      if (currentGroup->hasCoreOps() && currentGroup->coreType != coreType) {
        // Different core type - start new group
        if (!currentGroup->isEmpty()) {
          groups.push_back(currentGroup);
          currentGroup = std::make_shared<OperationGroup>();
        }
      }

      if (!currentGroup->hasCoreOps()) {
        currentGroup->coreType = coreType;
      }

      if (directCoreType.has_value()) {
        currentGroup->coreOps.push_back(&op);
      } else {
        currentGroup->nonCoreOps.push_back(&op);
      }
      currentGroup->allOps.push_back(&op);
    } else {
      // Non-core operation
      if (isRegionOperation(&op)) {
        // Region opearation - if we have a group, flush it first
        if (!currentGroup->isEmpty()) {
          groups.push_back(currentGroup);
          currentGroup = std::make_shared<OperationGroup>();
        }

        // Create separate group for region op
        auto regionGroup = std::make_shared<OperationGroup>();
        regionGroup->nonCoreOps.push_back(&op);
        regionGroup->allOps.push_back(&op);
        groups.push_back(regionGroup);
      } else {
        // Regular non-core op - add to current group
        currentGroup->nonCoreOps.push_back(&op);
        currentGroup->allOps.push_back(&op);
      }
    }
  }

  if (!currentGroup->isEmpty()) {
    groups.push_back(currentGroup);
  }

  return groups;
}

// Clone operaions and yield results
static void cloneAndYield(const CloneContext &ctx) {
  DenseMap<Value, Value> localValueMap = ctx.valueMap;

  for (Operation *op : ctx.group->allOps) {
    IRMapping mapping;
    for (auto [oldVal, newVal] : localValueMap) {
      mapping.map(oldVal, newVal);
    }

    for (Value operand : op->getOperands()) {
      if (mapping.lookupOrNull(operand)) {
        Value mapped = localValueMap.count(operand)
                           ? localValueMap.lookup(operand)
                           : operand;
        mapping.map(operand, mapped);
      }
    }

    Operation *clone = ctx.builder.clone(*op, mapping);

    for (auto [oldR, newR] : llvm::zip(op->getResults(), clone->getResults())) {
      localValueMap[oldR] = newR;
    }
  }

  // Yield all results
  SmallVector<Value> yields;
  for (Value v : ctx.externallyUsed) {
    yields.push_back(localValueMap.count(v) ? localValueMap.lookup(v) : v);
  }

  ctx.builder.create<scf::YieldOp>(ctx.loc, yields);
}

// Check if all operands of an operation are accounted for
static bool tallyOperands(Operation *op,
                          const DenseMap<Value, Value> &localValueMap,
                          Block *originalThenBlock, Block *originalElseBlock) {
  for (Value operand : op->getOperands()) {
    if (localValueMap.count(operand))
      continue; // found internally

    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;

    bool inOriginalScope =
        (defOp->getBlock() == originalThenBlock) ||
        (originalElseBlock && defOp->getBlock() == originalElseBlock);
    if (!inOriginalScope)
      continue; // found externally

    return false; // located internally but not in current split
  }
  return true;
}

// Using DFS to track Operation dependencies
static SmallVector<Operation *>
trackDependencies(Value value, const DenseMap<Value, Value> &localValueMap,
                  Block *originalThenBlock, Block *originalElseBlock,
                  DenseSet<Operation *> &visited) {
  SmallVector<Operation *> dependencies;
  if (localValueMap.count(value))
    return dependencies; // Already available

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return dependencies;

  bool inOriginalScope =
      (defOp->getBlock() == originalThenBlock) ||
      (originalElseBlock && defOp->getBlock() == originalElseBlock);
  if (!inOriginalScope)
    return dependencies;

  if (visited.contains(defOp))
    return dependencies;
  visited.insert(defOp);

  for (Value operand : defOp->getOperands()) {
    auto subDeps = trackDependencies(operand, localValueMap, originalThenBlock,
                                     originalElseBlock, visited);
    dependencies.append(subDeps);
  }

  dependencies.push_back(defOp);
  return dependencies;
}

// Cloning only the needed ops for inactive branch
static void cloneInactiveAndYield(const CloneContext &ctx) {
  DenseMap<Value, Value> localValueMap = ctx.valueMap;
  DenseSet<Value> extUsedSet(ctx.externallyUsed.begin(),
                             ctx.externallyUsed.end());

  for (Operation *op : ctx.group->coreOps) {
    for (Value res : op->getResults()) {
      if (extUsedSet.contains(res)) {
        Value dummy = mlir::utils::createEmptyOp(ctx.builder, ctx.loc, res);
        localValueMap[res] = dummy;
      }
    }
  }

  for (Operation *op : ctx.group->nonCoreOps) {
    if (isRegionOperation(op)) {
      for (Value res : op->getResults()) {
        if (extUsedSet.contains(res)) {
          Value dummy;
          if (auto tensorTy = dyn_cast<RankedTensorType>(res.getType())) {
            dummy = mlir::utils::createEmptyOp(ctx.builder, ctx.loc, res);
          }
          if (dummy) {
            localValueMap[res] = dummy;
          }
        }
      }
    }
  }

  DenseSet<Operation *> visited;
  SmallVector<Operation *> neededOps;

  for (Value v : ctx.externallyUsed) {
    if (!localValueMap.count(v)) {
      Operation *defOp = v.getDefiningOp();
      if (defOp && tallyOperands(defOp, localValueMap, ctx.originalThenBlock,
                                 ctx.originalElseBlock)) {
        if (!visited.contains(defOp)) {
          visited.insert(defOp);
          neededOps.push_back(defOp);
        }
      } else {
        auto deps = trackDependencies(v, localValueMap, ctx.originalThenBlock,
                                      ctx.originalElseBlock, visited);
        neededOps.append(deps);
      }
    }
  }

  for (Operation *op : neededOps) {
    if (op->getNumResults() == 0 || !localValueMap.count(op->getResult(0))) {
      IRMapping mapping;
      for (auto [oldVal, newVal] : localValueMap) {
        mapping.map(oldVal, newVal);
      }

      for (Value operand : op->getOperands()) {
        if (!mapping.lookupOrNull(operand)) {
          Value mapped = localValueMap.count(operand)
                             ? localValueMap.lookup(operand)
                             : operand;
          mapping.map(operand, mapped);
        }
      }

      Operation *clone = ctx.builder.clone(*op, mapping);

      for (auto [oldR, newR] :
           llvm::zip(op->getResults(), clone->getResults())) {
        localValueMap[oldR] = newR;
      }
    }
  }

  // Yield all results
  SmallVector<Value> yields;

  for (Value v : ctx.externallyUsed) {
    yields.push_back(localValueMap.count(v) ? localValueMap.lookup(v) : v);
  }

  ctx.builder.create<scf::YieldOp>(ctx.loc, yields);
}

//===----------------------------------------------------------------------===//
// Create Split IF for a group
//===----------------------------------------------------------------------===//

static scf::IfOp createSplitIfForGroup(const SplitIfContext &ctx) {
  // Collect all results from group
  SmallVector<Value> externallyUsed = ctx.group->getExternallyUsedResults();

  if (ctx.group->isEmpty()) {
    return nullptr;
  }

  SmallVector<Type> resultTypes;
  for (Value v : externallyUsed) {
    resultTypes.push_back(v.getType());
  }

  auto createBranchLogic =
      [&](bool activeForThen) -> std::function<void(OpBuilder &, Location)> {
    return [&, activeForThen](OpBuilder &b, Location loc) {
      CloneContext cloneCtx{b,
                            loc,
                            ctx.group,
                            externallyUsed,
                            ctx.valueMap,
                            ctx.originalThenBlock,
                            ctx.originalElseBlock};
      if ((ctx.isThenBlock && activeForThen) ||
          (!ctx.isThenBlock && !activeForThen)) {
        cloneAndYield(cloneCtx);
      } else {
        cloneInactiveAndYield(cloneCtx);
      }
    };
  };

  return ctx.rewriter.create<scf::IfOp>(
      ctx.loc, ctx.cond, createBranchLogic(true), createBranchLogic(false));
}

//===----------------------------------------------------------------------===//
// Create IF-Else for Final Result
//===----------------------------------------------------------------------===//

static scf::IfOp createFinalIf(const FinalIfContext &ctx) {
  return ctx.rewriter.create<scf::IfOp>(
      ctx.loc, ctx.cond,
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, ctx.thenResult);
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, ctx.elseResult);
      });
}

static scf::IfOp createFinalIfNoElse(PatternRewriter &rewriter, Location loc,
                                     Value cond, Value thenResult) {
  return rewriter.create<scf::IfOp>(loc, cond, [&](OpBuilder &b, Location loc) {
    b.create<scf::YieldOp>(loc, thenResult);
  });
}

//===----------------------------------------------------------------------===//
// Process IF-Else Blocks for Yield Value Mapping
//===----------------------------------------------------------------------===//
static void processGroupResults(const GroupResultsContext &ctx) {
  for (auto [oldV, newV] : llvm::zip(ctx.externallyUsed, ctx.newResults)) {
    if (!llvm::is_contained(ctx.thenYields, oldV) &&
        !llvm::is_contained(ctx.elseYields, oldV)) {
      oldV.replaceAllUsesWith(newV);
    }
    ctx.valueMap[oldV] = newV;

    if (llvm::is_contained(ctx.thenYields, oldV) ||
        llvm::is_contained(ctx.elseYields, oldV)) {
      ctx.yieldMap[oldV] = newV;
    }
  }
}

static DenseMap<Value, Value> processBlock(const BlockProcessContext &ctx) {
  DenseMap<Value, Value> valueMap;
  DenseMap<Value, Value> yieldMap;

  // Copy block arguments to valueMap
  for (auto arg : ctx.block->getArguments()) {
    valueMap[arg] = arg;
  }

  // Group Operations
  SmallVector<std::shared_ptr<OperationGroup>> groups =
      groupOperations(ctx.block);

  for (auto group : groups) {
    SplitIfContext splitCtx{group,
                            ctx.cond,
                            ctx.loc,
                            ctx.isThenBlock,
                            valueMap,
                            ctx.rewriter,
                            ctx.originalThenBlock,
                            ctx.originalElseBlock};
    auto newIf = createSplitIfForGroup(splitCtx);
    if (newIf) {
      SmallVector<Value> externallyUsed = group->getExternallyUsedResults();
      SmallVector<Value> newResults(newIf.getResults().begin(),
                                    newIf.getResults().end());
      // Map all results
      GroupResultsContext groupCtx{externallyUsed, newResults, ctx.thenYields,
                                   ctx.elseYields, valueMap,   yieldMap};
      processGroupResults(groupCtx);
      ctx.rewriter.setInsertionPointAfter(newIf);
    }
  }

  return yieldMap;
}

//===----------------------------------------------------------------------===//
// SplitMixedIfConditionalsPass
//===----------------------------------------------------------------------===//

struct SplitMixedIfConditionalsPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!needsSplit(ifOp)) {
      return failure();
    }

    for (Block *block : {ifOp.thenBlock(), ifOp.elseBlock()}) {
      if (!block)
        continue;
      for (Operation &op : *block) {
        if (isRegionOperation(&op) && !isa<scf::IfOp>(&op) &&
            regionOpMixedCoreTypes(&op)) {
          return failure();
        }
      }
    }

    Location loc = ifOp.getLoc();
    Value cond = ifOp.getCondition();
    bool hasElse = !ifOp.getElseRegion().empty();

    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    SmallVector<Value> thenYields(thenYield.getOperands().begin(),
                                  thenYield.getOperands().end());

    SmallVector<Value> elseYields;
    if (hasElse) {
      auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
      elseYields.assign(elseYield.getOperands().begin(),
                        elseYield.getOperands().end());
    }

    Block *originalThenBlock = ifOp.thenBlock();
    Block *originalElseBlock = hasElse ? ifOp.elseBlock() : nullptr;

    // Processing IF branch
    BlockProcessContext thenCtx{
        originalThenBlock, true,       cond,     loc,
        thenYields,        elseYields, rewriter, originalThenBlock,
        originalElseBlock};
    auto thenMap = processBlock(thenCtx);

    // Processing ELSE branch
    DenseMap<Value, Value> elseMap;
    if (hasElse) {
      BlockProcessContext elseCtx{
          originalElseBlock, false,      cond,     loc,
          thenYields,        elseYields, rewriter, originalThenBlock,
          originalElseBlock};
      elseMap = processBlock(elseCtx);
    }

    // Create Final Results
    SmallVector<Value> results;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Value thenV = thenYields[i];
      Value elseV = hasElse ? elseYields[i] : Value();

      Value thenResult = thenMap.count(thenV) ? thenMap.lookup(thenV) : thenV;
      Value elseResult =
          hasElse ? (elseMap.count(elseV) ? elseMap.lookup(elseV) : elseV)
                  : Value();

      // Create combined If
      if (hasElse) {
        FinalIfContext finalCtx{rewriter, loc, cond, thenResult, elseResult};
        auto finalIf = createFinalIf(finalCtx);
        results.push_back(finalIf.getResult(0));
      } else {
        auto finalIf = createFinalIfNoElse(rewriter, loc, cond, thenResult);
        results.push_back(finalIf.getResult(0));
      }
    }

    rewriter.replaceOp(ifOp, results);
    return success();
  }
};

void SplitMixedIfConditionalsPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  patterns.insert<SplitMixedIfConditionalsPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSplitMixedIfConditionalsPass() {
  return std::make_unique<SplitMixedIfConditionalsPass>();
}
