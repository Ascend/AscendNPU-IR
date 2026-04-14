//===----------------------- MergeVecScopePass.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/MemrefAliasAnalysisState.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#define DEBUG_TYPE "hfusion-merge-vf"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using mlir::utils::debugger::getPrettyOpName;

namespace {
#define GEN_PASS_DEF_MERGEVECSCOPE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {
void printEffects(ArrayRef<MemoryEffects::EffectInstance> effects) {
  for (const auto &effect : effects) {
    auto *effectType = effect.getEffect();
    std::string effectName = "Unknown";

    if (llvm::isa<MemoryEffects::Read>(effectType)) {
      effectName = "Read";
    } else if (llvm::isa<MemoryEffects::Write>(effectType)) {
      effectName = "Write";
    } else if (llvm::isa<MemoryEffects::Allocate>(effectType)) {
      effectName = "Allocate";
    } else if (llvm::isa<MemoryEffects::Free>(effectType)) {
      effectName = "Free";
    }

    Value value = effect.getValue();
    if (value) {
      LDBG("  Effect: " << effectName << " on Value: " << value);
    } else {
      LDBG("  Effect: " << effectName << " on Resource");
    }
  }
}

bool hasMemDependency(ArrayRef<MemoryEffects::EffectInstance> effects1,
                      ArrayRef<MemoryEffects::EffectInstance> effects2,
                      llvm::function_ref<bool(Value, Value)> isAlias) {

  for (const auto &eff1 : effects1) {
    for (const auto &eff2 : effects2) {

      Value v1 = eff1.getValue();
      Value v2 = eff2.getValue();

      if (!v1 || !v2 || !isa<MemRefType>(v1.getType()) ||
          !isa<MemRefType>(v2.getType()))
        continue;

      // RAW, WAR, WAW
      bool oneIsWrite = isa<MemoryEffects::Write>(eff1.getEffect()) ||
                        isa<MemoryEffects::Write>(eff2.getEffect());

      if (oneIsWrite) {
        if (isAlias(v1, v2)) {
          return true;
        }
      }
    }
  }
  return false;
}

void collectAllEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    for (Value arg : callOp.getArgOperands()) {
      if (isa<MemRefType>(arg.getType())) {
        auto addEffect = [&](auto effectType, Value v) {
          if (auto res = llvm::dyn_cast<OpResult>(v)) {
            effects.emplace_back(effectType, res, 0, true);
          } else if (auto bArg = llvm::dyn_cast<BlockArgument>(v)) {
            effects.emplace_back(effectType, bArg, 0, true);
          }
        };

        addEffect(MemoryEffects::Write::get(), arg);
      }
    }
    return;
  }

  if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
    interface.getEffects(effects);
  }

  if (op->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      for (Operation &innerOp : region.getOps()) {
        collectAllEffects(&innerOp, effects);
      }
    }
  }
}

void getEffectsOrConservative(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectAllEffects(op, effects);

  // only consider memref
  llvm::erase_if(effects, [](const MemoryEffects::EffectInstance &it) {
    Value value = it.getValue();
    if (!value)
      return true;
    if (!llvm::isa<MemRefType>(value.getType()))
      return true;
    return false;
  });
}

bool analyzeMemrefDepdencies(Operation *op1, Operation *op2,
                             llvm::function_ref<bool(Value, Value)> isAlias) {
  SmallVector<MemoryEffects::EffectInstance> effects1;
  getEffectsOrConservative(op1, effects1);
  
  if (effects1.empty()) {
    return false;
  }
  
  SmallVector<MemoryEffects::EffectInstance> effects2;
  getEffectsOrConservative(op2, effects2);
  
  if (effects2.empty()) {
    return false;
  }
  
  bool hasDep = hasMemDependency(effects1, effects2, isAlias);
  LLVM_DEBUG(if(hasDep){
    LDBG("analyze memref dependency, op1: " << op1->getName());
    printEffects(effects1);
    LDBG("analyze memref dependency, op2: " << op2->getName());
    printEffects(effects2);
  });

  return hasDep;
}

using DependMap = DenseMap<Operation *, DenseSet<Operation *>>;

DependMap computeDependencyClosure(const DependMap &directDeps) {
  DependMap closure;
  llvm::DenseSet<mlir::Operation *> processing;

  std::function<void(mlir::Operation *)> resolve =
      [&](mlir::Operation *current) {
        if (closure.count(current))
          return;

        if (processing.count(current)) {
          llvm::errs() << "Circular dependency detected at: " << *current
                       << "\n";
          return;
        }

        processing.insert(current);
        llvm::DenseSet<mlir::Operation *> tmpSet;

        auto it = directDeps.find(current);
        if (it != directDeps.end()) {
          for (mlir::Operation *dep : it->second) {
            tmpSet.insert(dep);

            resolve(dep);
            auto depIt = closure.find(dep);
            if (depIt != closure.end()) {
              tmpSet.insert(depIt->second.begin(), depIt->second.end());
            }
          }
        }
        closure[current] = std::move(tmpSet);

        processing.erase(current);
      };

  for (auto &entry : directDeps) {
    resolve(entry.first);
  }
  return closure;
}
} // namespace

struct MergeVecScopePass : public impl::MergeVecScopeBase<MergeVecScopePass> {
  using MergeVecScopeBase::MergeVecScopeBase;

  void runOnOperation() override;

private:
  bool tryMerge(func::FuncOp callGraphRoot, func::FuncOp vf1, func::FuncOp vf2,
                func::FuncOp *insertedPos);

  void mergeNoBetween(func::FuncOp root, func::FuncOp vf1, func::FuncOp vf2,
                      func::CallOp call1, func::CallOp call2,
                      func::FuncOp *insertedPos);

  void mergeNoMemory(
      func::FuncOp root, func::FuncOp vf1, func::FuncOp vf2,
      const llvm::MapVector<Operation *, std::pair<bool, bool>> &betweenInfo,
      func::CallOp call1, func::CallOp call2, func::FuncOp *insertedPos);

  void patchCalls(func::FuncOp root, func::CallOp call1, func::CallOp call2,
                  func::FuncOp mergedFunc);

  void GetMemOps(const SmallVector<Operation *> &betweenOps,
                 SmallVector<Operation *> &memOps);

  // Try to move part of ops before vf1, part of ops after vf2
  bool SplitMove(const SmallVector<Operation *> &memOps,
                 SmallVector<Operation *> &canBeBeforeVf1,
                 SmallVector<Operation *> &canBeAfterVf2, func::CallOp &call1,
                 func::CallOp &call2);

  bool isDep(Operation *op1, Operation *op2);

  SmallVector<func::FuncOp> vfs;
  DependMap allDepsClosure;
  SmallVector<Operation *> toErase;
};
} // namespace

SmallVector<func::CallOp> findCallOpsForFunction(func::FuncOp root,
                                                 func::FuncOp vf) {
  SmallVector<func::CallOp> callOps;

  StringRef targetFuncName = vf.getName();

  root.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == targetFuncName) {
      callOps.push_back(callOp);
    }
  });

  return callOps;
}

void printScoreMatrix(const llvm::SmallVector<llvm::SmallVector<int>> &matrix) {
  llvm::dbgs() << "Score Matrix (" << matrix.size() << "x"
               << (matrix.empty() ? 0 : matrix[0].size()) << "):\n";
  llvm::dbgs() << "          ";
  for (size_t i = 0; i < matrix.size(); ++i) {
    llvm::dbgs() << i << ", ";
  }
  llvm::dbgs() << "\n";
  for (size_t i = 0; i < matrix.size(); ++i) {
    llvm::dbgs() << "  Row " << i << ": [";
    for (size_t j = 0; j < matrix[i].size(); ++j) {
      llvm::dbgs() << matrix[i][j];
      if (j < matrix[i].size() - 1) {
        llvm::dbgs() << ", ";
      }
    }
    llvm::dbgs() << "]\n";
  }
}

void MergeVecScopePass::runOnOperation() {
  if (mergeLevel == 0) {
    return;
  }

  ModuleOp mod = getOperation();
  // get vector functions
  mod.walk([&](func::FuncOp f) {
    if (hivm::isVF(f))
      vfs.push_back(f);
  });

  unsigned int numberVfBefore = static_cast<unsigned int>(vfs.size());

  // get call function/root
  func::FuncOp root = nullptr;
  mod.walk([&](func::FuncOp f) {
    if (!hivm::isVF(f))
      root = f;
  });

  // loop throught the root function and update vfs function order by the calls
  // the first function call the VF should be first

  if (!root)
    return;

  SmallVector<func::FuncOp> callOrder;
  DenseSet<StringRef> seenCallees;
  root.walk([&](func::CallOp call) {
    StringRef callee = call.getCallee();
    if (seenCallees.insert(callee).second) {
      if (auto func = mod.lookupSymbol<func::FuncOp>(callee)) {
        if (hivm::isVF(func)) {
          callOrder.push_back(func);
        }
      }
    }
  });

  DenseSet<func::FuncOp> calledSet(callOrder.begin(), callOrder.end());
  SmallVector<func::FuncOp> newVfs = callOrder;
  for (func::FuncOp vf : vfs) {
    if (!calledSet.contains(vf)) {
      newVfs.push_back(vf);
    }
  }
  vfs = newVfs;
  LLVM_DEBUG(size_t n = vfs.size(); if (n > 0) {
    llvm::dbgs() << "VFs(" << n << " entries) = [";
    for (size_t i = 0; i < n - 1; ++i) {
      llvm::dbgs() << vfs[i].getName() << ", ";
    }
    llvm::dbgs() << vfs[n - 1].getName() << "]\n";
  });
  // Before greedy merging VFs, analyze to find the VFs without dependency.
  // First merge those VFs.
  DenseMap<func::FuncOp, SmallVector<func::CallOp>> func2callMap;
  for (size_t i = 0; i < vfs.size(); ++i) {
    func2callMap[vfs[i]] = findCallOpsForFunction(root, vfs[i]);
  }

  SmallVector<SmallVector<int>> useScoreMat(vfs.size(),
                                            SmallVector<int>(vfs.size(), 0));

  // compute memref alias
  utils::MemrefAliasAnalysisState state(root);

  SmallVector<Operation *> allOps;

  root.walk([&](Operation *op) {
    allOps.push_back(op);
    if (auto aliasPairs = hivm::getOperationAliasInfo(op);
        !aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        state.unionAliasSets(aliasPair.first, aliasPair.second);
      }
    }
  });

  state.printAlias();

  using DependMap = DenseMap<Operation *, DenseSet<Operation *>>;
  DependMap ssaDeps;
  // SSA dependency, only consider ops in same block
  for (Operation *op : allOps) {
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (op->getBlock() == defOp->getBlock()) {
          ssaDeps[op].insert(defOp);
        }
      }
    }

    if (op->getNumRegions() > 0) {
      op->walk([&](Operation *innerOp) {
        for (Value operand : innerOp->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (!op->isAncestor(defOp) && defOp->getBlock() == op->getBlock()) {
              ssaDeps[op].insert(defOp);
            }
          }
        }
      });
    }
  }

  // memref Read Write dependency
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();

  auto isAlias = [&state](Value v1, Value v2) { return state.isAlias(v1, v2); };
  DependMap memDeps;
  for (auto op1 : allOps) {
    for (auto op2 : allOps) {
      if (op1->getBlock() != op2->getBlock()) {
        continue;
      }

      if (!domInfo.properlyDominates(op1, op2)) {
        continue;
      }

      if (analyzeMemrefDepdencies(op1, op2, isAlias)) {
        memDeps[op2].insert(op1);
      }
    }
  }

  DependMap combinedDeps = ssaDeps;
  for (auto &entry : memDeps) {
    combinedDeps[entry.first].insert(entry.second.begin(), entry.second.end());
  }

  LDBG("=== Combined Direct Dependencies (SSA + Mem) ===");
  LLVM_DEBUG(for (auto &entry : combinedDeps) {
    Operation *op = entry.first;
    auto &deps = entry.second;
    LDBG("Op: " << getPrettyOpName(op));

    for (Operation *depOp : deps) {
      LDBG("  -> Depends on: " << getPrettyOpName(depOp) << " (" << depOp
                               << ")");
    }
  });
  LDBG("================================================");

  // Compute dependencies closure
  allDepsClosure = computeDependencyClosure(combinedDeps);

  LDBG("=== Combined Dependencies Closure ===");
  LLVM_DEBUG(for (auto &entry : combinedDeps) {
    Operation *op = entry.first;
    auto &deps = entry.second;
    LDBG("Op: " << *op);

    for (Operation *depOp : deps) {
      LDBG("  -> Depends on: " << getPrettyOpName(depOp) << " (" << depOp
                               << ")");
    }
  });
  LDBG("================================================");

  // Assume each vf is used only once, there won't be situations like:
  // b = vf1(a); d = vf2(c); e = vf1(d)
  // So there is no chance that vf1 depend on vf2
  for (size_t i = 0; i < vfs.size(); ++i) {
    func::FuncOp vf1 = vfs[i];
    auto vf1_callers = func2callMap[vf1];
    for (size_t j = i + 1; j < vfs.size(); ++j) {
      func::FuncOp vf2 = vfs[j];
      auto vf2_callers = func2callMap[vf2];
      // check if callers of vf1 and callers of vf2 have dependency
      bool hasDependency = false;
      for (size_t i1 = 0; i1 < vf1_callers.size(); ++i1) {
        for (size_t j2 = 0; j2 < vf2_callers.size(); ++j2) {
          hasDependency |= isDep(vf2_callers[j2], vf1_callers[i1]);
        }
      }
      if (hasDependency) {
        // Mark forward usage, i.e. i's result is used by j.
        useScoreMat[i][j] = 1;
      }
    }
  }
  LLVM_DEBUG(printScoreMatrix(useScoreMat););
  SmallVector<int> useScoreTotal(vfs.size(), 0);
  for (size_t i = 0; i < vfs.size(); ++i) {
    useScoreTotal[i] =
        std::accumulate(useScoreMat[i].begin(), useScoreMat[i].end(), 0);
  }
  // Adjust the VF merging order
  SmallVector<int> useIncrIdx(vfs.size());
  std::iota(useIncrIdx.begin(), useIncrIdx.end(), 0);
  std::stable_partition(useIncrIdx.begin(), useIncrIdx.end(),
                        [&useScoreTotal, &useIncrIdx](int bValue) {
                          auto it = std::find(useIncrIdx.begin(),
                                              useIncrIdx.end(), bValue);
                          size_t index = static_cast<size_t>(std::distance(useIncrIdx.begin(), it));
                          return useScoreTotal[index] == 0;
                        });
  LLVM_DEBUG(size_t n = useIncrIdx.size(); if (n > 0) {
    llvm::dbgs() << "useIncrIdx(" << n << " entries) = [";
    for (size_t i = 0; i < n - 1; ++i) {
      llvm::dbgs() << useIncrIdx[i] << ", ";
    }
    llvm::dbgs() << useIncrIdx[n - 1] << "]\n";
  });
  SmallVector<Operation *> startVFs;
  DenseMap<Operation *, DenseSet<Operation *>> relatedVFIdxs;
  for (size_t i = 0; i < vfs.size(); ++i) {
    auto idx = useIncrIdx[i];
    if (mergeLevel != 1 || useScoreTotal[idx] == 0) {
      startVFs.push_back(vfs[idx].getOperation());
    }
    DenseSet<Operation *> relatedVFs;
    for (size_t j = 0; j < vfs.size(); ++j) {
      if (useScoreMat[i][j] > 0) {
        relatedVFs.insert(vfs[j].getOperation());
      }
    }
    relatedVFIdxs[vfs[i].getOperation()] = relatedVFs;
  }

  // try to merge vfs
  size_t scoreIdx = 0;
  SmallVector<func::FuncOp> toRemoveVFs;

  DenseMap<Operation *, int> numVFInMergedVF;
  for (auto vf : vfs) {
    numVFInMergedVF[vf.getOperation()] = 1;
  }

  while (scoreIdx < startVFs.size()) {
    // clang-format off
    LLVM_DEBUG(
      llvm::dbgs() << "idx = " << scoreIdx << "/" << startVFs.size() << "\n";
      size_t n = startVFs.size();
      if (n > 0) {
        llvm::dbgs() << "startVFs(" << n << " entries) = [";
        for (size_t i = 0; i < n - 1; ++i) {
          llvm::dbgs() << cast<func::FuncOp>(startVFs[i]).getName() << ", ";
        }
        llvm::dbgs() << cast<func::FuncOp>(startVFs[n - 1]).getName() << "]\n";
      }
      //
      n = vfs.size();
      if (n > 0) {
        llvm::dbgs() << "vfs(" << n << " entries) = [";
        for (size_t i = 0; i < n - 1; ++i) {
          llvm::dbgs() << vfs[i].getName() << ", ";
        }
        llvm::dbgs() << vfs[n - 1].getName() << "]\n";
      }
      //
      llvm::dbgs() << "relatedVFIdxs:\n";
      for (const auto &entry : relatedVFIdxs) {
        llvm::dbgs() << "Key: " << cast<func::FuncOp>(entry.first).getName() << "\n";
        llvm::dbgs() << "  Operations: {";
        bool first = true;
        for (mlir::Operation *op : entry.second) {
          if (!first) llvm::dbgs() << ", ";
          llvm::dbgs() << cast<func::FuncOp>(op).getName();
          first = false;
        }
        llvm::dbgs() << "}\n";
      }
    );
    // clang-format on
    auto vf1Op = startVFs[scoreIdx];
    auto vf1 = cast<func::FuncOp>(vf1Op);
    auto vf1Ptr = std::find(vfs.begin(), vfs.end(), vf1);
    if (vf1Ptr == vfs.end()) {
      scoreIdx++;
      continue;
    }

    size_t vf1Idx = static_cast<size_t>(std::distance(vfs.begin(), vf1Ptr));
    if (vf1Idx == vfs.size() - 1) {
      scoreIdx++;
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "vf1Idx = " << vf1Idx << "\n";);
    // Note: this assign will create a new object
    auto vf1Obj = vfs[vf1Idx];
    auto vf2Obj = vfs[vf1Idx + 1];
    //
    if (mergeLevel == 1) {
      // Merge VFs only without dependency/usage
      if (relatedVFIdxs[vf1Obj.getOperation()].contains(
              vf2Obj.getOperation()) ||
          relatedVFIdxs[vf2Obj.getOperation()].contains(
              vf1Obj.getOperation())) {
        scoreIdx++;
        continue;
      }
    }
    // Such & operator extract the iterator. Later in tryMerge function,
    // we insert new created merged VF at this iterator.
    if (numVFInMergedVF[vf1Obj.getOperation()] +
            numVFInMergedVF[vf2Obj.getOperation()] >
        mergeVFNumLimit) {
      // clang-format off
      LLVM_DEBUG(llvm::dbgs()
        << "numVF of " << vf1Obj.getOperation() << " = "
        << numVFInMergedVF[vf1Obj.getOperation()] << "\n"
        << "numVF of " << vf2Obj.getOperation() << " = "
        << numVFInMergedVF[vf2Obj.getOperation()] << "\n"
        << "Sum of vf1 and vf2 is larger than " << mergeVFNumLimit
        << ", thus skip this merge\n";
      );
      // clang-format on
      scoreIdx++;
      continue;
    }

    auto insertedPos = &vfs[vf1Idx];
    bool mergeSuccess = tryMerge(root, vf1Obj, vf2Obj, insertedPos);
    if (mergeSuccess) {
      auto &mergedVfRef = vfs[vf1Idx];
      auto &vf1Ref = vfs[vf1Idx + 1];
      auto &vf2Ref = vfs[vf1Idx + 2];

      // update dependency

      numVFInMergedVF[mergedVfRef.getOperation()] =
          numVFInMergedVF[vf1Ref.getOperation()] +
          numVFInMergedVF[vf2Ref.getOperation()];
      // Note: you must erase vf2 before erase vf1
      vfs.erase(&vf2Ref);
      vfs.erase(&vf1Ref);
      // Here we cannot directly erase vf1 and vf2 object because that would
      // also remove the element in startVFs. So we collect them.
      // After the loop, we erase all of them.
      toRemoveVFs.push_back(vf1Obj);
      toRemoveVFs.push_back(vf2Obj);
      // clang-format off
      LLVM_DEBUG(llvm::dbgs()
        << "mergedVfRef = " << &mergedVfRef << ", "
        << mergedVfRef.getOperation() << ", "
        << mergedVfRef.getName() << ", "
        << "numVFInMergedVF = " << numVFInMergedVF[mergedVfRef.getOperation()]
        << "\n";
      );
      // clang-format on
      // We still need to merge the new created merged VF with its right VF
      startVFs.push_back(mergedVfRef.getOperation());
      //
      auto mergedRelatedVFIdxs = relatedVFIdxs[vf1Obj];
      mergedRelatedVFIdxs.insert(relatedVFIdxs[vf2Obj].begin(),
                                 relatedVFIdxs[vf2Obj].end());
      relatedVFIdxs[mergedVfRef.getOperation()] = mergedRelatedVFIdxs;
      relatedVFIdxs.erase(vf2Obj);
      relatedVFIdxs.erase(vf1Obj);
    }
    scoreIdx++;
  }

  LLVM_DEBUG(size_t n = toRemoveVFs.size(); if (n > 0) {
    llvm::dbgs() << "toRemoveVFs(" << n << " entries) = [";
    for (size_t i = 0; i < n - 1; ++i) {
      llvm::dbgs() << toRemoveVFs[i].getName() << ", ";
    }
    llvm::dbgs() << toRemoveVFs[n - 1].getName() << "]\n";
  });
  for (auto func : toRemoveVFs) {
    func.erase();
  }

  SmallVector<func::FuncOp> vfsAfter;
  mod.walk([&](func::FuncOp f) {
    if (hivm::isVF(f))
      vfsAfter.push_back(f);
  });
  unsigned int numberVfafter = vfsAfter.size();
  // debug
  // mod->dump();

  LLVM_DEBUG(llvm::dbgs() << "Number of VFs before: " << numberVfBefore
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Number of VFs after: " << numberVfafter << "\n");
}

bool MergeVecScopePass::isDep(Operation *op1, Operation *op2) {
  if (op1 == op2)
    return false;
  auto it = allDepsClosure.find(op1);
  if (it == allDepsClosure.end()) {
    return false;
  }
  return it->second.contains(op2);
}

bool MergeVecScopePass::tryMerge(func::FuncOp root, func::FuncOp vf1,
                                 func::FuncOp vf2, func::FuncOp *insertedPos) {
  SmallVector<Operation *> betweenOps;
  LLVM_DEBUG(llvm::dbgs() << "[tryMerge] " << vf1.getName() << " <-> "
                          << vf2.getName() << "]\n";);
  func::CallOp call1 = nullptr, call2 = nullptr;
  bool seenVf1 = false;

  root.walk([&](Operation *op) {
    if (auto c = dyn_cast<func::CallOp>(op)) {
      if (c.getCallee() == vf1.getName()) {
        call1 = c;
        seenVf1 = true;
        Operation *next = c->getNextNode();
        while (next) {
          if (auto c2 = dyn_cast<func::CallOp>(next)) {
            if (c2.getCallee() == vf2.getName()) {
              call2 = c2;
              return WalkResult::interrupt();
            }
          }
          next = next->getNextNode();
        }
      }
    }
    return WalkResult::advance();
  });

  if (!call1 || !call2)
    return false;

  if (call1->getParentRegion() != call2->getParentRegion()) {
    // in different control regions (e.g., different scf.if/scf.for blocks)
    return false;
  }

  Operation *between = call1->getNextNode();
  while (between && between != call2) {
    betweenOps.push_back(between);
    between = between->getNextNode();
  }

  for (Operation *op : betweenOps) {
    if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
      // temporarily regard sync_block instr as memory op
      // so that vf1 and vf2 would not be merged
      LLVM_DEBUG(llvm::dbgs() << "SyncBlock|Load op prevents fusion\n";);
      return false;
    }
  }

  SmallVector<Operation *> memOps;
  GetMemOps(betweenOps, memOps);

  bool moveAfterVf2 = true;
  bool moveBeforeVf1 = true;

  // check if any memory op result feeds into vf2
  for (Operation *memOp : memOps) {
    if (isDep(call2, memOp)) {
      moveAfterVf2 = false;
      break;
    }
  }

  // if any memory op uses vf1's result
  for (Operation *memOp : memOps) {
    if (isDep(memOp, call1)) {
      moveBeforeVf1 = false;
      break;
    }
  }

  // move operations
  if (moveAfterVf2) {
    LLVM_DEBUG(llvm::dbgs() << "all move after vf2\n";);
    Operation *insertionPoint = call2;
    for (Operation *op : memOps) {
      op->moveAfter(insertionPoint);
      insertionPoint = op;
    }
  } else if (moveBeforeVf1) {
    LLVM_DEBUG(llvm::dbgs() << "all move before vf1\n";);
    Operation *insertionPoint = call1;
    for (Operation *op : memOps) {
      op->moveBefore(insertionPoint);
    }
  } else {

    SmallVector<Operation *> canBeBeforeVf1;
    SmallVector<Operation *> canBeAfterVf2;

    if (!SplitMove(memOps, canBeBeforeVf1, canBeAfterVf2, call1, call2)) {
      return false;
    }

    LLVM_DEBUG(for (auto op : canBeBeforeVf1) {
      llvm::dbgs() << "canBeBefore: " << *op << "\n";
    });
    LLVM_DEBUG(for (auto op : canBeAfterVf2) {
      llvm::dbgs() << "canBeAfter: " << *op << "\n";
    });

    Operation *insertionPoint = call1;
    for (Operation *op : canBeBeforeVf1) {
      op->moveBefore(insertionPoint);
    }

    insertionPoint = call2;
    for (Operation *op : canBeAfterVf2) {
      op->moveAfter(insertionPoint);
      insertionPoint = op;
    }
  }

  betweenOps.clear();
  Operation *current = call1->getNextNode();
  while (current && current != call2) {
    betweenOps.push_back(current);
    current = current->getNextNode();
  }

  // when megeLevel==1, there should be no betweenOps here
  if (mergeLevel == 1 && !betweenOps.empty()) {
    return false;
  }

  // gain info about the code inbetween
  llvm::MapVector<Operation *, std::pair<bool, bool>> betweenInfo;
  SmallVector<Value> betweenVals;

  for (Operation *op : betweenOps) {
    bool usedAfter = false;
    bool feedsVF2 = false;

    SmallVector<Value> opUsedResults;
    for (Value result : op->getResults()) {
      bool resultUsedAfter =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return user->getBlock() == call2->getBlock() &&
                   call2->isBeforeInBlock(user);
          });

      if (resultUsedAfter) {
        usedAfter = true;
        opUsedResults.push_back(result);
      }
    }

    feedsVF2 = llvm::any_of(op->getResults(), [&](Value result) {
      return llvm::is_contained(call2.getArgOperands(), result);
    });

    betweenInfo[op] = {usedAfter, feedsVF2};
    if (usedAfter) {
      betweenVals.append(opUsedResults.begin(), opUsedResults.end());
    }
  }

  // debug stuff
  // for (auto betweenOp : betweenOps) {
  //   betweenOp->dump();
  //   llvm::dbgs() << "***\n";
  // }
  LLVM_DEBUG(size_t n = betweenOps.size(); if (n > 0) {
    llvm::dbgs() << "betweenOps(" << n << " entries) = \n";
    for (size_t i = 0; i < n; ++i) {
      llvm::dbgs() << "[" << i << "]: " << *betweenOps[i] << "\n";
    }
  });
  // do the diff merge types
  if (betweenOps.empty()) {
    // llvm::dbgs() << "here  empty function "
    //                 "=========================================================="
    //                 "======================";
    mergeNoBetween(root, vf1, vf2, call1, call2, insertedPos);

    return true;
  } else {
    // llvm::dbgs() << "here no memory function "
    //                 "=========================================================="
    //                 "======================";
    mergeNoMemory(root, vf1, vf2, betweenInfo, call1, call2, insertedPos);
    return true;
  }
  return false;
}

void MergeVecScopePass::patchCalls(func::FuncOp root, func::CallOp call1,
                                   func::CallOp call2,
                                   func::FuncOp mergedFunc) {
  OpBuilder builder(call1);

  SmallVector<Value> newArgs;

  newArgs.append(call1.getArgOperands().begin(), call1.getArgOperands().end());

  DenseSet<Value> call1Results(call1.getResults().begin(),
                               call1.getResults().end());
  for (Value arg : call2.getArgOperands()) {
    if (!call1Results.contains(arg)) {
      newArgs.push_back(arg);
    }
  }

  auto newCall =
      builder.create<func::CallOp>(call1.getLoc(), mergedFunc, newArgs);
  newCall->setAttr("ptc_simdvf", builder.getUnitAttr());
  for (auto attr : call1->getAttrs()) {
    if (attr.getName() == "callee" ||
        attr.getName() == "result_segment_sizes") {
      continue;
    }
    newCall->setAttr(attr.getName(), attr.getValue());
  }

  unsigned numCall1Results = call1.getNumResults();
  for (unsigned i = 0; i < numCall1Results; ++i) {
    call1.getResult(i).replaceAllUsesWith(newCall.getResult(i));
  }

  for (unsigned i = 0; i < call2.getNumResults(); ++i) {
    call2.getResult(i).replaceAllUsesWith(
        newCall.getResult(numCall1Results + i));
  }

  // update Dependency info
  auto &newDeps = allDepsClosure[newCall];

  if (allDepsClosure.count(call1))
    newDeps.insert(allDepsClosure[call1].begin(), allDepsClosure[call1].end());
  if (allDepsClosure.count(call2))
    newDeps.insert(allDepsClosure[call2].begin(), allDepsClosure[call2].end());

  for (auto &entry : allDepsClosure) {
    auto &depsSet = entry.second;
    if (depsSet.contains(call1) || depsSet.contains(call2)) {
      depsSet.erase(call1);
      depsSet.erase(call2);
      depsSet.insert(newCall);
    }
  }

  call1.erase();
  call2.erase();
}

void MergeVecScopePass::GetMemOps(const SmallVector<Operation *> &betweenOps,
                                  SmallVector<Operation *> &memOps) {
  if (mergeLevel == 1) {
    // all betweenOps should be moved
    memOps = betweenOps;
    return;
  }

  DenseSet<Operation *> memOpsSet;

  // get direct memory operations
  // TODO: add Flag to not move DMA ops
  for (Operation *op : betweenOps) {
    if (isa<hivm::LoadOp, hivm::StoreOp, hivm::CopyOp, hivm::IndirectLoadOp,
            hivm::IndirectStoreOp, hfusion::LoadOp, hfusion::StoreOp,
            hfusion::IndirectLoadOp, hfusion::IndirectStoreOp, memref::LoadOp,
            memref::StoreOp, memref::CopyOp, memref::SubViewOp, memref::AllocOp,
            memref::ExpandShapeOp, memref::CollapseShapeOp, tensor::EmptyOp,
            tensor::ExtractSliceOp, tensor::DimOp, annotation::MarkOp>(op)) {
      if (memOpsSet.insert(op).second) {
        memOps.push_back(op);
      }
    }
  }

  SmallVector<Operation *> worklist = memOps;
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();

    for (Value operand : current->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (llvm::is_contained(betweenOps, defOp) &&
            !memOpsSet.contains(defOp)) {
          memOpsSet.insert(defOp);
          memOps.push_back(defOp);
          worklist.push_back(defOp);
        }
      }
    }

    for (Value result : current->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (llvm::is_contained(betweenOps, user) && !memOpsSet.contains(user)) {
          memOpsSet.insert(user);
          memOps.push_back(user);
          worklist.push_back(user);
        }
      }
    }
  }

  llvm::sort(memOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
}

bool MergeVecScopePass::SplitMove(const SmallVector<Operation *> &memOps,
                                  SmallVector<Operation *> &canBeBeforeVf1,
                                  SmallVector<Operation *> &canBeAfterVf2,
                                  func::CallOp &call1, func::CallOp &call2) {
  auto vf2Depend = [&call2](Operation *op) {
    for (Value result : op->getResults()) {
      if (llvm::is_contained(call2.getArgOperands(), result)) {
        return true;
      }
    }
    return false;
  };

  if (mergeLevel == 1) {
    // Assume that vf1 and vf2 have no data flow dependency
    // Then an op cannot be depend on vf1 and be depended on by vf2 at the same
    // time.

    for (Operation *op : memOps) {
      bool isDependent = isDep(op, call1);

      if (isDependent) {
        if (isDep(call2, op)) {
          return false;
        }
        canBeAfterVf2.push_back(op);
      } else {
        canBeBeforeVf1.push_back(op);
      }
    }
    return true;
  }

  bool foundDependency = false;

  for (Operation *op : memOps) {
    if (foundDependency) {
      canBeAfterVf2.push_back(op);
    } else {
      bool usesCall1 = false;
      for (Value operand : op->getOperands()) {
        if (operand.getDefiningOp() == call1) {
          usesCall1 = true;
          break;
        }
      }
      if (usesCall1) {
        foundDependency = true;
        canBeAfterVf2.push_back(op);
      } else {
        canBeBeforeVf1.push_back(op);
      }
    }
  }

  for (Operation *op : canBeAfterVf2) {
    if (vf2Depend(op)) {
      return false;
    }
  }
  return true;
}

void MergeVecScopePass::mergeNoBetween(func::FuncOp root, func::FuncOp vf1,
                                       func::FuncOp vf2, func::CallOp call1,
                                       func::CallOp call2,
                                       func::FuncOp *insertedPos) {
  OpBuilder builder(root.getContext());
  ModuleOp module = root->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(module);

  LLVM_DEBUG(llvm::dbgs() << "[mergeNoBetween] vf1 = " << vf1.getName()
                          << ", vf2 = " << vf2.getName() << "\n";);

  std::string originalName = vf1.getName().str();
  size_t pos = originalName.rfind("vf_");
  std::string baseName = originalName;
  if (pos != std::string::npos) {
    baseName = originalName.substr(0, pos);
  }
  std::string newSuffix = "merged_vf";
  baseName = baseName + newSuffix;
  std::string newName = baseName + "_0";
  unsigned counter = 0;
  while (symbolTable.lookup(newName)) {
    newName = baseName + "_" + std::to_string(counter++);
  }

  auto getFuncCoreType = [](func::FuncOp f) -> hivm::TFuncCoreType {
    auto maybeFuncCoreType = hivm::queryFuncCoreType(f);
    if (!maybeFuncCoreType.has_value()) {
      llvm_unreachable(
          (f.getName() + " has no func_core_type\n").str().c_str());
    }
    return maybeFuncCoreType.value();
  };
  auto vf1FuncCoreType = getFuncCoreType(vf1);
  auto vf2FuncCoreType = getFuncCoreType(vf2);
  if (vf1FuncCoreType != vf2FuncCoreType) {
    llvm_unreachable((vf1.getName() + " must have the same func_core_type as " +
                      vf2.getName() + "\n")
                         .str()
                         .c_str());
  }

  SmallVector<Type> newArgTypes;
  DenseSet<Value> call1Results(call1.getResults().begin(),
                               call1.getResults().end());

  for (auto arg : vf1.getArguments())
    newArgTypes.push_back(arg.getType());

  // vf2 arguments that don't come from vf1 results
  DenseMap<BlockArgument, Value> vf2ArgReplacements;
  for (auto arg : vf2.getArguments()) {
    Value operand = call2.getOperand(arg.getArgNumber());
    if (call1Results.contains(operand)) {
      // mark argument for replacement with vf1 result
      vf2ArgReplacements[arg] = operand;
    } else {
      newArgTypes.push_back(arg.getType());
    }
  }

  SmallVector<Type> newResultTypes;
  newResultTypes.append(vf1.getFunctionType().getResults().begin(),
                        vf1.getFunctionType().getResults().end());
  newResultTypes.append(vf2.getFunctionType().getResults().begin(),
                        vf2.getFunctionType().getResults().end());

  builder.setInsertionPointToStart(module.getBody());
  auto mergedFunc = builder.create<func::FuncOp>(
      root.getLoc(), newName,
      builder.getFunctionType(newArgTypes, newResultTypes));
  mergedFunc->setAttr("hivm.vector_function", builder.getUnitAttr());
  mergedFunc->setAttr(
      hivm::TFuncCoreTypeAttr::name,
      hivm::TFuncCoreTypeAttr::get(root.getContext(), vf1FuncCoreType));
  symbolTable.insert(mergedFunc);

  Block *mergedBlock = mergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(mergedBlock);

  // clone vf1 operations
  IRMapping vf1Mapper;
  for (auto [origArg, newArg] :
       llvm::zip(vf1.getArguments(),
                 mergedFunc.getArguments().take_front(vf1.getNumArguments()))) {
    vf1Mapper.map(origArg, newArg);
  }

  for (Operation &op : vf1.getBody().front().without_terminator()) {
    builder.clone(op, vf1Mapper);
  }

  // clone vf2 operations with argument remapping
  IRMapping vf2Mapper;
  unsigned mergedArgIdx = vf1.getNumArguments();

  for (BlockArgument arg : vf2.getArguments()) {
    if (vf2ArgReplacements.count(arg)) {
      // map to corresponding vf1 result
      Value vf1Result = vf2ArgReplacements[arg];
      auto resultIt = llvm::find(call1.getResults(), vf1Result);
      auto resultIdx = std::distance(call1.result_begin(), resultIt);
      Value mappedResult = vf1Mapper.lookup(
          vf1.getBody().front().getTerminator()->getOperand(resultIdx));
      vf2Mapper.map(arg, mappedResult);
    } else {
      // map to new merged function argument
      vf2Mapper.map(arg, mergedFunc.getArgument(mergedArgIdx++));
    }
  }
  for (Operation &op : vf2.getBody().front().without_terminator()) {
    Operation *cloned = builder.clone(op, vf2Mapper);
    for (unsigned i = 0; i < cloned->getNumOperands(); ++i) {
      Value operand = cloned->getOperand(i);
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getParentBlock() == &vf2.getBody().front()) {
          cloned->setOperand(i, vf2Mapper.lookup(blockArg));
        }
      }
    }
  }

  SmallVector<Value> mergedReturns;
  auto vf1Return = cast<func::ReturnOp>(vf1.getBody().front().getTerminator());
  auto vf2Return = cast<func::ReturnOp>(vf2.getBody().front().getTerminator());

  for (Value ret : vf1Return.getOperands())
    mergedReturns.push_back(vf1Mapper.lookup(ret));

  for (Value ret : vf2Return.getOperands())
    mergedReturns.push_back(vf2Mapper.lookup(ret));

  builder.create<func::ReturnOp>(root.getLoc(), mergedReturns);

  // update call sites
  patchCalls(root, call1, call2, mergedFunc);

  // verify the merged function (debugging stuff)
  if (failed(mergedFunc.verify())) {
    mergedFunc.emitError() << "Merged function verification failed";
    signalPassFailure();
  }
  vfs.insert(insertedPos, mergedFunc);
}

void MergeVecScopePass::mergeNoMemory(
    func::FuncOp root, func::FuncOp vf1, func::FuncOp vf2,
    const llvm::MapVector<Operation *, std::pair<bool, bool>> &betweenInfo,
    func::CallOp call1, func::CallOp call2, func::FuncOp *insertedPos) {
  OpBuilder builder(root.getContext());
  ModuleOp module = root->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(module);

  LLVM_DEBUG(llvm::dbgs() << "[mergeNoMemory] vf1 = " << vf1.getName()
                          << ", vf2 = " << vf2.getName() << "\n";);

  std::string originalName = vf1.getName().str();
  size_t pos = originalName.rfind("vf_");
  std::string baseName = originalName;
  if (pos != std::string::npos) {
    baseName = originalName.substr(0, pos);
  }
  std::string newSuffix = "_merged_vf";
  baseName = baseName + newSuffix;
  std::string newName = baseName + "_0";
  unsigned counter = 0;
  while (symbolTable.lookup(newName)) {
    newName = baseName + "_" + std::to_string(counter++);
  }

  SmallVector<Type> newArgTypes;
  DenseSet<Value> call1Results(call1.getResults().begin(),
                               call1.getResults().end());

  for (auto arg : vf1.getArguments())
    newArgTypes.push_back(arg.getType());

  // vf2 arguments not provided by vf1 or intermediate ops
  SmallVector<Value> remainingVf2Operands;
  for (unsigned i = 0; i < vf2.getNumArguments(); i++) {
    Value operand = call2.getOperand(i);
    if (call1Results.contains(operand) ||
        (operand.getDefiningOp() &&
         betweenInfo.count(operand.getDefiningOp()))) {
      continue;
    }
    remainingVf2Operands.push_back(operand);
    newArgTypes.push_back(vf2.getArgument(i).getType());
  }

  // external inputs for intermediate ops add it to args
  DenseSet<Value> seen;
  SmallVector<Value> externalInputsVec;
  for (auto &opInfo : betweenInfo) {
    Operation *op = opInfo.first;
    for (Value operand : op->getOperands()) {
      if (seen.contains(operand))
        continue;
      Operation *defOp = operand.getDefiningOp();
      // if defined by intermediate or vf1 continue
      if ((defOp && betweenInfo.count(defOp)) ||
          call1Results.contains(operand)) {
        continue;
      }
      seen.insert(operand);
      externalInputsVec.push_back(operand);
      newArgTypes.push_back(operand.getType());
    }
  }

  SmallVector<Type> newResultTypes;
  auto vf1Return = cast<func::ReturnOp>(vf1.getBody().front().getTerminator());
  for (Value ret : vf1Return.getOperands())
    newResultTypes.push_back(ret.getType());
  auto vf2Return = cast<func::ReturnOp>(vf2.getBody().front().getTerminator());
  for (Value ret : vf2Return.getOperands())
    newResultTypes.push_back(ret.getType());
  for (auto &opInfo : betweenInfo) {
    if (opInfo.second.first) { // usedAfter
      for (Value res : opInfo.first->getResults())
        newResultTypes.push_back(res.getType());
    }
  }

  builder.setInsertionPointToStart(module.getBody());
  auto mergedFunc = builder.create<func::FuncOp>(
      root.getLoc(), newName,
      builder.getFunctionType(newArgTypes, newResultTypes));
  mergedFunc->setAttr("hivm.vector_function", builder.getUnitAttr());
  symbolTable.insert(mergedFunc);
  Block *mergedBlock = mergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(mergedBlock);

  // map vf1 args and clone body
  IRMapping vf1Mapper;
  for (unsigned i = 0; i < vf1.getNumArguments(); i++)
    vf1Mapper.map(vf1.getArgument(i), mergedBlock->getArgument(i));
  for (Operation &op : vf1.getBody().front().without_terminator())
    builder.clone(op, vf1Mapper);

  // map call1 results to cloned vf1 outputs
  DenseMap<Value, Value> call1ResultMap;
  for (unsigned i = 0; i < call1.getNumResults(); i++) {
    Value origRet = vf1Return.getOperand(i);
    call1ResultMap[call1.getResult(i)] = vf1Mapper.lookup(origRet);
  }

  // map external inputs and clone intermediate ops
  IRMapping betweenMapper;
  unsigned externalStartIdx =
      vf1.getNumArguments() + remainingVf2Operands.size();
  for (unsigned i = 0; i < externalInputsVec.size(); i++)
    betweenMapper.map(externalInputsVec[i],
                      mergedBlock->getArgument(externalStartIdx + i));
  for (auto &it : call1ResultMap)
    betweenMapper.map(it.first, it.second);
  for (auto &opInfo : betweenInfo) {
    Operation *op = opInfo.first;
    Operation *cloned = builder.clone(*op, betweenMapper);
    for (unsigned i = 0; i < op->getNumResults(); i++)
      betweenMapper.map(op->getResult(i), cloned->getResult(i));
  }

  // map vf2 args and clone body
  IRMapping vf2Mapper;
  unsigned remainingIdx = 0;
  for (unsigned i = 0; i < vf2.getNumArguments(); i++) {
    Value operand = call2.getOperand(i);
    if (call1ResultMap.count(operand)) {
      vf2Mapper.map(vf2.getArgument(i), call1ResultMap[operand]);
    } else if (operand.getDefiningOp() &&
               betweenInfo.count(operand.getDefiningOp())) {
      vf2Mapper.map(vf2.getArgument(i), betweenMapper.lookup(operand));
    } else {
      vf2Mapper.map(
          vf2.getArgument(i),
          mergedBlock->getArgument(vf1.getNumArguments() + remainingIdx++));
    }
  }
  for (Operation &op : vf2.getBody().front().without_terminator())
    builder.clone(op, vf2Mapper);

  SmallVector<Value> mergedReturns;
  for (Value ret : vf1Return.getOperands())
    mergedReturns.push_back(vf1Mapper.lookup(ret));
  for (Value ret : vf2Return.getOperands())
    mergedReturns.push_back(vf2Mapper.lookup(ret));
  for (auto &opInfo : betweenInfo) {
    if (opInfo.second.first) { // usedAfter
      for (Value res : opInfo.first->getResults())
        mergedReturns.push_back(betweenMapper.lookup(res));
    }
  }
  builder.create<func::ReturnOp>(root.getLoc(), mergedReturns);

  OpBuilder rootBuilder(call1);

  SmallVector<Value> newCallOperands;
  // vf1 operands
  newCallOperands.append(call1.getOperands().begin(),
                         call1.getOperands().end());
  // vf2 operands not fed by vf1/intermediates
  for (Value operand : remainingVf2Operands)
    newCallOperands.push_back(operand);
  // External inputs for intermediates
  for (Value operand : externalInputsVec)
    newCallOperands.push_back(operand);

  auto newCall = rootBuilder.create<func::CallOp>(call1.getLoc(), mergedFunc,
                                                  newCallOperands);

  for (auto attr : call1->getAttrs()) {
    if (attr.getName() == "callee" ||
        attr.getName() == "result_segment_sizes") {
      continue;
    }
    newCall->setAttr(attr.getName(), attr.getValue());
  }

  unsigned resultIdx = 0;
  for (Value res : call1.getResults())
    res.replaceAllUsesWith(newCall.getResult(resultIdx++));
  for (Value res : call2.getResults())
    res.replaceAllUsesWith(newCall.getResult(resultIdx++));
  for (auto &opInfo : betweenInfo) {
    if (opInfo.second.first) {
      Operation *op = opInfo.first;
      for (Value res : op->getResults()) {
        res.replaceAllUsesWith(newCall.getResult(resultIdx++));
      }
    }
  }

  // when merge-level = 1 won't goto here
  call2.erase();
  call1.erase();
  for (auto it = betweenInfo.rbegin(); it != betweenInfo.rend(); ++it) {
    it->first->erase();
  }

  // (debug stuff)
  if (failed(mergedFunc.verify())) {
    mergedFunc.emitError() << "Merged function verification failed";
    signalPassFailure();
  }

  vfs.insert(insertedPos, mergedFunc);
}

std::unique_ptr<Pass>
hfusion::createMergeVecScopePass(const MergeVecScopeOptions &options) {
  return std::make_unique<MergeVecScopePass>(options);
}
