//===- Util.cpp - Common utilities for CV Pipelining V2 -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"

namespace mlir {
namespace hivm {

bool isCoreOp(Operation *op) {
  if (!op || !op->getDialect())
    return false;
  return op->getDialect()->getNamespace() ==
             HIVMDialect::getDialectNamespace() &&
         isa<DestinationStyleOpInterface>(op);
}

std::optional<TCoreType> queryConcreteCoreType(Operation *op) {
  auto ct = detail::queryCoreTypeHelper(op);
  if (ct && (*ct == TCoreType::CUBE || *ct == TCoreType::VECTOR))
    return *ct;
  return std::nullopt;
}

std::pair<Value, Value> getIfYieldedValues(scf::IfOp ifOp, unsigned resIdx) {
  auto thenYield = ifOp.thenYield();
  Value thenVal = (resIdx < thenYield.getNumOperands())
                      ? thenYield.getOperand(resIdx)
                      : Value();
  Value elseVal;
  if (ifOp.elseBlock()) {
    auto elseYield = ifOp.elseYield();
    if (resIdx < elseYield.getNumOperands())
      elseVal = elseYield.getOperand(resIdx);
  }
  return {thenVal, elseVal};
}

Value getOrCreateElseYieldValue(OpBuilder &builder, Location loc, Value thenVal,
                                Value elseVal) {
  if (elseVal)
    return elseVal;
  Type valType = thenVal.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valType))
    return builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                           tensorType.getElementType());
  if (valType.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  if (auto intType = dyn_cast<IntegerType>(valType))
    return builder.create<arith::ConstantIntOp>(loc, 0, intType.getWidth());
  if (auto fltType = dyn_cast<FloatType>(valType))
    return builder.create<arith::ConstantFloatOp>(
        loc, fltType, APFloat(fltType.getFloatSemantics(), 0));
  return thenVal;
}

ScopeOp recreateScope(ScopeOp srcScope, ValueRange returnOperands,
                      Operation *insertBefore) {
  Block &block = srcScope.getRegion().front();
  if (block.empty() || !isa<scope::ReturnOp>(block.back())) {
    OpBuilder::atBlockEnd(&block).create<scope::ReturnOp>(srcScope.getLoc(),
                                                          returnOperands);
  } else {
    block.getTerminator()->setOperands(returnOperands);
  }
  OpBuilder builder(insertBefore ? insertBefore : srcScope.getOperation());
  auto newScope =
      builder.create<ScopeOp>(srcScope.getLoc(), TypeRange(returnOperands),
                              ValueRange(), srcScope->getAttrs());
  newScope.getRegion().takeBody(srcScope.getRegion());
  return newScope;
}

// NOTE: Moving individual ops across scopes is expensive; prefer merging scopes
// whenever possible.
void moveOpToScope(Operation *op, ScopeOp target, bool insertAtEnd) {
  if (!op || !target)
    return;

  if (op->getParentOfType<scf::IfOp>())
    llvm_unreachable("TODO: support moving op within scf.if");

  auto origScope = op->getParentOfType<ScopeOp>();
  if (!origScope || origScope == target)
    return;

  SmallPtrSet<Operation *, 16> opsToMoveSet = {op};
  if (isCoreOp(op)) {
    SmallVector<Operation *> worklist = {op};
    while (!worklist.empty()) {
      for (Value opnd : worklist.pop_back_val()->getOperands()) {
        Operation *defOp = opnd.getDefiningOp();
        if (!defOp || !origScope->isProperAncestor(defOp))
          continue;
        if (defOp->getParentOfType<scf::IfOp>())
          llvm_unreachable("TODO: support moving op within scf.if");
        if (!isCoreOp(defOp) && opsToMoveSet.insert(defOp).second)
          worklist.push_back(defOp);
      }
    }
  }

  Block &origBlock = origScope.getRegion().front();
  Block &targetBlock = target.getRegion().front();
  auto origReturn = cast<scope::ReturnOp>(origBlock.getTerminator());
  auto targetReturn = cast<scope::ReturnOp>(targetBlock.getTerminator());

  SmallVector<Operation *> orderedOpsToMove;
  for (Operation &currOp : origBlock) {
    if (opsToMoveSet.contains(&currOp))
      orderedOpsToMove.push_back(&currOp);
  }

  Operation *insertPt =
      insertAtEnd ? targetReturn.getOperation() : &targetBlock.front();

  IRMapping sharedPureMapping;
  SmallPtrSet<Operation *, 16> actuallyMovedOps;
  OpBuilder builder(insertPt);
  for (Operation *mOp : orderedOpsToMove) {
    bool hasRemainingUseInOrig =
        mOp != op && isMemoryEffectFree(mOp) &&
        llvm::any_of(mOp->getUsers(), [&](Operation *user) {
          return user != origReturn && origScope->isProperAncestor(user) &&
                 !opsToMoveSet.contains(user);
        });
    if (hasRemainingUseInOrig) {
      builder.clone(*mOp, sharedPureMapping);
      continue;
    }
    mOp->moveBefore(insertPt);
    for (OpOperand &opnd : mOp->getOpOperands()) {
      if (Value mapped = sharedPureMapping.lookupOrNull(opnd.get()))
        opnd.set(mapped);
    }
    actuallyMovedOps.insert(mOp);
  }

  for (Operation *mOp : actuallyMovedOps) {
    for (OpOperand &opnd : mOp->getOpOperands()) {
      if (auto res = dyn_cast<OpResult>(opnd.get());
          res && res.getOwner() == target.getOperation())
        opnd.set(targetReturn.getOperand(res.getResultNumber()));
    }
  }

  SmallVector<Value> newReturnOperands;
  SmallVector<std::pair<unsigned, unsigned>> keptResultMapping;
  DenseMap<Value, unsigned> valToNewResultIdx;
  bool scopeResultsChanged = false;

  for (auto [idx, retVal] : llvm::enumerate(origReturn.getOperands())) {
    if (Operation *retDef = retVal.getDefiningOp();
        retDef && actuallyMovedOps.contains(retDef)) {
      origScope.getResult(idx).replaceAllUsesWith(retVal);
      scopeResultsChanged = true;
      continue;
    }
    unsigned newIdx = newReturnOperands.size();
    newReturnOperands.push_back(retVal);
    keptResultMapping.emplace_back(idx, newIdx);
    valToNewResultIdx.try_emplace(retVal, newIdx);
  }

  SmallVector<std::pair<Value, unsigned>> depToNewResultIdx;
  for (Operation *mOp : orderedOpsToMove) {
    if (!actuallyMovedOps.contains(mOp))
      continue;
    for (Value opnd : mOp->getOperands()) {
      Operation *defOp = opnd.getDefiningOp();
      if (!defOp || !origScope->isProperAncestor(defOp))
        continue;
      auto [it, inserted] =
          valToNewResultIdx.try_emplace(opnd, newReturnOperands.size());
      if (inserted) {
        newReturnOperands.push_back(opnd);
        scopeResultsChanged = true;
      }
      depToNewResultIdx.emplace_back(opnd, it->second);
    }
  }

  ScopeOp activeOrigScope = origScope;
  if (scopeResultsChanged) {
    auto newOrigScope = recreateScope(origScope, newReturnOperands);
    for (auto [oldIdx, newIdx] : keptResultMapping) {
      origScope.getResult(oldIdx).replaceAllUsesWith(
          newOrigScope.getResult(newIdx));
    }
    origScope.erase();
    activeOrigScope = newOrigScope;
  }

  for (auto [depVal, newIdx] : depToNewResultIdx) {
    Value scopeRes = activeOrigScope.getResult(newIdx);
    depVal.replaceUsesWithIf(scopeRes, [&](OpOperand &use) {
      return target->isProperAncestor(use.getOwner());
    });
  }
}

ScopeOp mergeScopes(ScopeOp scope1, ScopeOp scope2, Operation *at) {
  if (!scope1 || !scope2)
    return scope1 ? scope1 : scope2;
  if (scope1 == scope2)
    return scope1;

  auto coreAttr1 =
      scope1->getAttrOfType<TCoreTypeAttr>(kPipelinedLoopCoreTypeAttrName);
  auto coreAttr2 =
      scope2->getAttrOfType<TCoreTypeAttr>(kPipelinedLoopCoreTypeAttrName);
  assert(coreAttr1 && coreAttr2 && "expected scopes to have core type attr");
  TCoreType ct1 = coreAttr1.getTcoretype();
  TCoreType ct2 = coreAttr2.getTcoretype();
  assert(!((ct1 == TCoreType::CUBE && ct2 == TCoreType::VECTOR) ||
           (ct1 == TCoreType::VECTOR && ct2 == TCoreType::CUBE)) &&
         "cannot merge CUBE and VECTOR scopes");
  TCoreType mergedCoreType = ct1;
  if (ct1 == TCoreType::CUBE || ct2 == TCoreType::CUBE)
    mergedCoreType = TCoreType::CUBE;
  else if (ct1 == TCoreType::VECTOR || ct2 == TCoreType::VECTOR)
    mergedCoreType = TCoreType::VECTOR;

  DominanceInfo domInfo(scope1->getParentOp());
  ScopeOp firstScope = scope1;
  ScopeOp secondScope = scope2;
  if (domInfo.properlyDominates(scope2.getOperation(), scope1.getOperation()))
    std::swap(firstScope, secondScope);

  Block &firstBlock = firstScope.getRegion().front();
  Block &secondBlock = secondScope.getRegion().front();
  auto firstReturn = cast<scope::ReturnOp>(firstBlock.getTerminator());
  auto secondReturn = cast<scope::ReturnOp>(secondBlock.getTerminator());

  // Redirect uses of firstScope's results inside secondScope to firstReturn's
  // operands.
  for (auto [retVal, res] :
       llvm::zip(firstReturn.getOperands(), firstScope.getResults())) {
    res.replaceUsesWithIf(retVal, [&](OpOperand &use) {
      return secondScope->isProperAncestor(use.getOwner());
    });
  }

  auto pruneUnusedResults = [](ScopeOp scope, scope::ReturnOp retOp) {
    SmallVector<Value> keptReturns, keptResults;
    for (auto [retVal, res] :
         llvm::zip(retOp.getOperands(), scope.getResults())) {
      if (!res.use_empty()) {
        keptReturns.push_back(retVal);
        keptResults.push_back(res);
      }
    }
    retOp->setOperands(keptReturns);
    return keptResults;
  };

  SmallVector<Value> keptFirstScopeResults =
      pruneUnusedResults(firstScope, firstReturn);
  SmallVector<Value> keptSecondScopeResults =
      pruneUnusedResults(secondScope, secondReturn);

  // Index scf.if operations in firstBlock by their pipeline.if.cond ID.
  DenseMap<int64_t, scf::IfOp> firstIfById;
  for (auto ifOp : firstBlock.getOps<scf::IfOp>()) {
    if (auto idAttr = ifOp->getAttrOfType<IntegerAttr>(kPipelineIfCondAttrName))
      firstIfById[idAttr.getInt()] = ifOp;
  }

  SmallVector<scf::IfOp> secondIfs(secondBlock.getOps<scf::IfOp>());
  for (scf::IfOp if2 : secondIfs) {
    auto idAttr = if2->getAttrOfType<IntegerAttr>(kPipelineIfCondAttrName);
    if (!idAttr)
      continue;
    scf::IfOp if1 = firstIfById.lookup(idAttr.getInt());
    if (!if1)
      continue;

    // Move prerequisite ops preceding if2 in secondBlock before if1 if they do
    // not depend on if1 or later ops in firstBlock.
    for (Operation &op : llvm::make_early_inc_range(secondBlock)) {
      if (&op == if2.getOperation())
        break;
      bool dependsOnIf1OrLater =
          llvm::any_of(op.getOperands(), [&](Value opnd) {
            Operation *def = opnd.getDefiningOp();
            return def && def->getBlock() == &firstBlock &&
                   !def->isBeforeInBlock(if1);
          });
      if (!dependsOnIf1OrLater)
        op.moveBefore(if1);
    }

    // Redirect uses of if1's results inside if2's branches to if1's branch
    // yield operands.
    for (auto [idx, res] : llvm::enumerate(if1.getResults())) {
      auto [thenVal, elseVal] = getIfYieldedValues(if1, idx);
      if (thenVal) {
        res.replaceUsesWithIf(thenVal, [&](OpOperand &use) {
          return if2.getThenRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });
      }
      if (elseVal && if2.elseBlock()) {
        res.replaceUsesWithIf(elseVal, [&](OpOperand &use) {
          return if2.getElseRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });
      }
    }

    SmallVector<Value> mergedThenYieldVals;
    SmallVector<Value> mergedElseYieldVals;
    SmallVector<Value> keptIfResults;

    for (scf::IfOp ifOp : {if1, if2}) {
      for (auto [idx, res] : llvm::enumerate(ifOp.getResults())) {
        if (res.use_empty())
          continue;
        auto [tVal, eVal] = getIfYieldedValues(ifOp, idx);
        keptIfResults.push_back(res);
        mergedThenYieldVals.push_back(tVal);
        mergedElseYieldVals.push_back(eVal);
      }
    }

    bool needElse = (if1.elseBlock() != nullptr) ||
                    (if2.elseBlock() != nullptr) || !keptIfResults.empty();

    OpBuilder ifBuilder(if1);
    auto mergedIf = ifBuilder.create<scf::IfOp>(
        if1.getLoc(), TypeRange(ValueRange(keptIfResults)), if1.getCondition(),
        needElse);
    mergedIf->setAttrs(if1->getAttrs());

    if (!mergedIf.thenBlock()->empty())
      mergedIf.thenBlock()->getTerminator()->erase();
    if (mergedIf.elseBlock() && !mergedIf.elseBlock()->empty())
      mergedIf.elseBlock()->getTerminator()->erase();

    auto moveBranchOps = [](Block *srcBlock, Block *dstBlock) {
      if (!srcBlock)
        return;
      for (Operation &op : llvm::make_early_inc_range(*srcBlock)) {
        if (!isa<scf::YieldOp>(&op))
          op.moveBefore(dstBlock, dstBlock->end());
      }
    };

    moveBranchOps(if1.thenBlock(), mergedIf.thenBlock());
    moveBranchOps(if2.thenBlock(), mergedIf.thenBlock());
    ifBuilder.setInsertionPointToEnd(mergedIf.thenBlock());
    ifBuilder.create<scf::YieldOp>(if1.thenYield().getLoc(),
                                   mergedThenYieldVals);

    if (needElse) {
      moveBranchOps(if1.elseBlock(), mergedIf.elseBlock());
      moveBranchOps(if2.elseBlock(), mergedIf.elseBlock());
      ifBuilder.setInsertionPointToEnd(mergedIf.elseBlock());
      Location elseLoc = if1.elseBlock()   ? if1.elseYield().getLoc()
                         : if2.elseBlock() ? if2.elseYield().getLoc()
                                           : if1.getLoc();
      SmallVector<Value> finalElseYieldVals;
      for (auto [thenVal, elseVal] :
           llvm::zip(mergedThenYieldVals, mergedElseYieldVals)) {
        finalElseYieldVals.push_back(
            getOrCreateElseYieldValue(ifBuilder, elseLoc, thenVal, elseVal));
      }
      ifBuilder.create<scf::YieldOp>(elseLoc, finalElseYieldVals);
    }

    for (auto [oldRes, newRes] :
         llvm::zip(keptIfResults, mergedIf.getResults()))
      oldRes.replaceAllUsesWith(newRes);

    if1.erase();
    if2.erase();
  }

  // Move all remaining non-terminator ops from secondBlock into firstBlock
  // before firstReturn.
  for (Operation &op : llvm::make_early_inc_range(secondBlock)) {
    if (&op != secondReturn.getOperation())
      op.moveBefore(firstReturn);
  }

  SmallVector<Value> mergedReturnOperands(firstReturn.getOperands());
  llvm::append_range(mergedReturnOperands, secondReturn.getOperands());
  SmallVector<Value> oldScopeResultsToReplace(keptFirstScopeResults);
  llvm::append_range(oldScopeResultsToReplace, keptSecondScopeResults);

  auto mergedScope = recreateScope(firstScope, mergedReturnOperands, at);
  mergedScope->setAttr(
      kPipelinedLoopCoreTypeAttrName,
      TCoreTypeAttr::get(mergedScope.getContext(), mergedCoreType));

  for (auto [oldRes, newRes] :
       llvm::zip(oldScopeResultsToReplace, mergedScope.getResults()))
    oldRes.replaceAllUsesWith(newRes);

  firstScope.erase();
  secondScope.erase();
  return mergedScope;
}

namespace {

struct ScopeDepNode;

struct ScopeDepEdge {
  const ScopeDepNode *target = nullptr;
  std::string label;
  bool isLoopCarried = false;
};

struct ScopeDepNode {
  unsigned id = 0;
  ScopeOp scopeOp;
  TCoreType coreType = TCoreType::CUBE_OR_VECTOR;
  SmallVector<const ScopeDepNode *> children;
  SmallVector<ScopeDepEdge> edges;

  void addEdge(const ScopeDepNode *dst, StringRef edgeLabel, bool loopCarried) {
    for (ScopeDepEdge &existing : edges) {
      if (existing.target == dst && existing.isLoopCarried == loopCarried) {
        if (!edgeLabel.empty() &&
            StringRef(existing.label).find(edgeLabel) == StringRef::npos) {
          if (!existing.label.empty())
            existing.label += ", ";
          existing.label += edgeLabel.str();
        }
        return;
      }
    }
    children.push_back(dst);
    edges.push_back({dst, edgeLabel.str(), loopCarried});
  }
};

struct DOTScopeDepGraphInfo {
  scf::ForOp forOp;
  std::string funcName;
  SmallVector<std::unique_ptr<ScopeDepNode>> nodes;
  SmallVector<const ScopeDepNode *> nodePtrs;

  explicit DOTScopeDepGraphInfo(scf::ForOp loop) : forOp(loop) {
    if (auto func = loop->getParentOfType<func::FuncOp>())
      funcName = func.getName().str();
    else
      funcName = "loop";

    DenseMap<Operation *, ScopeDepNode *> scopeToNode;
    for (auto [idx, scope] :
         llvm::enumerate(loop.getBody()->getOps<ScopeOp>())) {
      auto node = std::make_unique<ScopeDepNode>();
      node->id = idx;
      node->scopeOp = scope;
      if (auto attr = scope->getAttrOfType<TCoreTypeAttr>(
              kPipelinedLoopCoreTypeAttrName))
        node->coreType = attr.getTcoretype();
      scopeToNode[scope.getOperation()] = node.get();
      nodePtrs.push_back(node.get());
      nodes.push_back(std::move(node));
    }

    auto findEnclosingScopeNode = [&](Operation *op) -> ScopeDepNode * {
      while (op && op != loop.getOperation()) {
        if (auto *n = scopeToNode.lookup(op))
          return n;
        op = op->getParentOp();
      }
      return nullptr;
    };

    // Edges:
    // 1. `scope result` -> `user scope`
    // 2. `scope result` -> `scf.yield` -> `corresponding iter_arg user scope`
    auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (const auto &srcNode : nodes) {
      for (auto [resIdx, res] :
           llvm::enumerate(srcNode->scopeOp.getResults())) {
        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          if (user == yieldOp.getOperation()) {
            unsigned argIdx = use.getOperandNumber();
            if (argIdx >= loop.getNumRegionIterArgs())
              continue;
            std::string label =
                llvm::formatv("LCD(res#{0}->arg#{1})", resIdx, argIdx).str();
            for (OpOperand &argUse : loop.getRegionIterArg(argIdx).getUses()) {
              if (ScopeDepNode *dstNode =
                      findEnclosingScopeNode(argUse.getOwner()))
                srcNode->addEdge(dstNode, label, /*loopCarried=*/true);
            }
          } else if (ScopeDepNode *dstNode = findEnclosingScopeNode(user);
                     dstNode && dstNode != srcNode.get()) {
            srcNode->addEdge(dstNode, llvm::formatv("res#{0}", resIdx).str(),
                             /*loopCarried=*/false);
          }
        }
      }
    }
  }
};

} // namespace

} // namespace hivm
} // namespace mlir

namespace llvm {

using mlir::hivm::DOTScopeDepGraphInfo;
using mlir::hivm::ScopeDepEdge;
using mlir::hivm::ScopeDepNode;

template <> struct GraphTraits<DOTScopeDepGraphInfo *> {
  using NodeRef = const ScopeDepNode *;
  using ChildIteratorType =
      SmallVectorImpl<const ScopeDepNode *>::const_iterator;
  using nodes_iterator = SmallVectorImpl<const ScopeDepNode *>::const_iterator;

  static NodeRef getEntryNode(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.empty() ? nullptr : G->nodePtrs.front();
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return N->children.begin();
  }
  static ChildIteratorType child_end(NodeRef N) { return N->children.end(); }

  static nodes_iterator nodes_begin(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.begin();
  }
  static nodes_iterator nodes_end(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.end();
  }
  static size_t size(DOTScopeDepGraphInfo *G) { return G->nodePtrs.size(); }
};

template <>
struct DOTGraphTraits<DOTScopeDepGraphInfo *> : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(DOTScopeDepGraphInfo *G) {
    return "Scope Dependence Graph for '" + G->funcName + "'";
  }

  static std::string getSimpleNodeLabel(const ScopeDepNode *Node,
                                        DOTScopeDepGraphInfo *) {
    return formatv("Scope {0} ({1})", Node->id,
                   mlir::hivm::stringifyTCoreType(Node->coreType))
        .str();
  }

  static void printScopeBlock(raw_string_ostream &OS,
                              const ScopeDepNode &Node) {
    OS << formatv("Scope {0} ({1}):\n", Node.id,
                  mlir::hivm::stringifyTCoreType(Node.coreType));
    mlir::scope::ScopeOp scopeOp = Node.scopeOp;
    for (mlir::Operation &op : scopeOp.getRegion().front())
      OS << op << "\n";
  }

  static void noOpHandleComment(std::string &, unsigned &, unsigned) {}

  std::string getNodeLabel(const ScopeDepNode *Node, DOTScopeDepGraphInfo *G) {
    if (isSimple())
      return getSimpleNodeLabel(Node, G);
    return CompleteNodeLabelString(
        Node,
        function_ref<void(raw_string_ostream &, const ScopeDepNode &)>(
            printScopeBlock),
        function_ref<void(std::string &, unsigned &, unsigned)>(
            noOpHandleComment));
  }

  static std::string getNodeAttributes(const ScopeDepNode *Node,
                                       DOTScopeDepGraphInfo *) {
    StringRef fillColor = "#e6e6e6";
    if (Node->coreType == mlir::hivm::TCoreType::CUBE)
      fillColor = "#ffcccc";
    else if (Node->coreType == mlir::hivm::TCoreType::VECTOR)
      fillColor = "#cce5ff";
    return ("style=filled, fillcolor=\"" + fillColor +
            "\", fontname=\"Courier\"")
        .str();
  }

  static std::string
  getEdgeAttributes(const ScopeDepNode *Node,
                    GraphTraits<DOTScopeDepGraphInfo *>::ChildIteratorType EI,
                    DOTScopeDepGraphInfo *) {
    size_t idx = std::distance(Node->children.begin(), EI);
    const ScopeDepEdge &edge = Node->edges[idx];
    std::string attrs =
        formatv("tooltip=\"Scope {0} -> Scope {1}\"", Node->id, edge.target->id)
            .str();
    if (!edge.label.empty())
      attrs += formatv(" label=\"{0}\"", DOT::EscapeString(edge.label)).str();
    if (edge.isLoopCarried) {
      attrs += " style=dashed color=red penwidth=2";
      if (Node->id >= edge.target->id)
        attrs += " constraint=false";
    }
    return attrs;
  }
};

} // namespace llvm

namespace mlir {
namespace hivm {

void writeScopeDependenceGraphToDotFile(scf::ForOp forOp, StringRef filename,
                                        bool shortNames) {
  if (!forOp)
    return;
  DOTScopeDepGraphInfo graphInfo(forOp);
  std::string outFilename = filename.str();
  if (outFilename.empty()) {
    Operation *rootOp = forOp->getParentOfType<func::FuncOp>();
    if (!rootOp)
      rootOp = forOp->getParentOp();
    SmallVector<scf::ForOp> scopedLoops;
    if (rootOp) {
      rootOp->walk([&](scf::ForOp loop) {
        if (!loop.getBody()->getOps<ScopeOp>().empty())
          scopedLoops.push_back(loop);
      });
    }
    if (scopedLoops.size() > 1) {
      auto it = llvm::find(scopedLoops, forOp);
      unsigned idx = (it != scopedLoops.end())
                         ? std::distance(scopedLoops.begin(), it)
                         : 0;
      outFilename =
          llvm::formatv("scope-dep.{0}.{1}.dot", graphInfo.funcName, idx).str();
    } else {
      outFilename = "scope-dep." + graphInfo.funcName + ".dot";
    }
  }
  llvm::errs() << "Writing '" << outFilename << "'...";

  std::error_code ec;
  llvm::raw_fd_ostream file(outFilename, ec, llvm::sys::fs::OF_Text);
  if (!ec)
    llvm::WriteGraph(file, &graphInfo, shortNames);
  else
    llvm::errs() << "  error opening file for writing!";
  llvm::errs() << "\n";
}

} // namespace hivm
} // namespace mlir
