//===------- SyncSolverTester.cpp ---- Graph Sync Solver ------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTester.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTest.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "hivm-gss-sync-tester"

using namespace mlir;
using namespace hivm::syncsolver;

// Random test IR generator: recursively builds scopes, loops, conditions and RW
// ops. Used by the tester to create synthetic cases exercising the solver.
void SyncTester::generateRandTest(
    Scope *scopeOp, const std::vector<int> &pointerOps,
    const llvm::SmallVector<hivm::PIPE> &pipesVec, int &remOpNum, int depth,
    Scope *counterScope, std::optional<hivm::TCoreType> scopeCoreType) {
  auto blockBeginPlaceHolderOp =
      std::make_unique<PlaceHolder>(nullptr, scopeOp);
  blockBeginPlaceHolderOp->scopeBegin = scopeOp;
  scopeOp->body.push_back(std::move(blockBeginPlaceHolderOp));
  bool empty = true;
  while (remOpNum > 0) {
    if (!scopeCoreType.has_value() && depth + 1 < max_depth &&
        pointerOps.size() >= read_write_vals_max_num * 2 &&
        remOpNum >= loop_preload_max_num &&
        isTrueWithProbability(preload_loop_prob_a, preload_loop_prob_b)) {

      auto loopOp = std::make_unique<Loop>(nullptr, scopeOp);
      loopOp->haveCounter = true;
      loopOp->isCVPreloadingLoop = true;
      loopOp->loopPreloadNum = loop_preload_max_num;
      auto loopBody = std::make_unique<Scope>();
      loopBody->parentOp = loopOp.get();

      int coreTypeParity = getRand() % 2;
      int skipEvenOddItrs =
          syncMode == SyncMode::TEST_INTRA_CORE_MODE ? (getRand() % 2) : -1;

      std::vector<int> pointerOps0(pointerOps.begin(),
                                   pointerOps.begin() + pointerOps.size() / 2);
      std::vector<int> pointerOps1(pointerOps.begin() + pointerOps.size() / 2,
                                   pointerOps.end());
      int remOpNumDivLoopPreloadNum = remOpNum / loop_preload_max_num;
      for (int i = 0; i < loop_preload_max_num; i++) {
        if (skipEvenOddItrs != -1) {
          if ((i + skipEvenOddItrs) % 2 == 0) {
            continue;
          }
        }

        hivm::TCoreType coreType = (coreTypeParity + i) % 2
                                       ? hivm::TCoreType::CUBE
                                       : hivm::TCoreType::VECTOR;

        std::vector<int> curPointerOps;
        if (i == 0) {
          curPointerOps = pointerOps0;
        } else if (i == 3) {
          curPointerOps = pointerOps1;
        } else {
          curPointerOps = pointerOps;
        }

        auto preloadScope = std::make_unique<Scope>();
        preloadScope->parentOp = loopBody.get();
        preloadScope->preloadNum = loop_preload_max_num - i - 1;
        preloadScope->maxPreloadNum = loop_preload_max_num;
        preloadScope->haveCounter = true;

        int curRemOpNum = remOpNumDivLoopPreloadNum;
        int ogCurRemOpNum = curRemOpNum;
        auto preloadScopeBody = std::make_unique<Scope>();
        preloadScopeBody->parentOp = preloadScope.get();
        generateRandTest(preloadScopeBody.get(), curPointerOps, pipesVec,
                         curRemOpNum, depth + 1, loopOp.get(), coreType);
        remOpNum -= ogCurRemOpNum - curRemOpNum;

        preloadScope->body.push_back(std::move(preloadScopeBody));
        loopBody->body.push_back(std::move(preloadScope));
      }

      loopOp->body.push_back(std::move(loopBody));
      auto beforePlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      beforePlaceHolderOp->beforeOp = loopOp.get();
      auto afterPlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      afterPlaceHolderOp->afterOp = loopOp.get();
      scopeOp->body.push_back(std::move(beforePlaceHolderOp));
      scopeOp->body.push_back(std::move(loopOp));
      scopeOp->body.push_back(std::move(afterPlaceHolderOp));
      empty = false;
    } else if (depth < max_depth &&
               isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
               isTrueWithProbability(scope_for_loop_prob_a,
                                     scope_for_loop_prob_b)) {
      auto loopOp = std::make_unique<Loop>(nullptr, scopeOp);
      loopOp->isParallel =
          isTrueWithProbability(parallel_loop_prob_a, parallel_loop_prob_b);
      loopOp->haveCounter = true;
      auto loopBlock = std::make_unique<Scope>();
      loopBlock->parentOp = loopOp.get();
      generateRandTest(loopBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1, loopOp.get(), scopeCoreType);
      loopOp->body.push_back(std::move(loopBlock));
      auto beforePlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      beforePlaceHolderOp->beforeOp = loopOp.get();
      auto afterPlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      afterPlaceHolderOp->afterOp = loopOp.get();
      scopeOp->body.push_back(std::move(beforePlaceHolderOp));
      scopeOp->body.push_back(std::move(loopOp));
      scopeOp->body.push_back(std::move(afterPlaceHolderOp));
      empty = false;
    } else if (depth < max_depth &&
               isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
               isTrueWithProbability(scope_while_loop_prob_a,
                                     scope_while_loop_prob_b)) {
      auto loopOp = std::make_unique<Loop>(nullptr, scopeOp);
      loopOp->haveCounter = true;
      auto beforeBlock = std::make_unique<Scope>();
      beforeBlock->parentOp = loopOp.get();
      generateRandTest(beforeBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1, loopOp.get(), scopeCoreType);
      auto afterBlock = std::make_unique<Scope>();
      afterBlock->parentOp = loopOp.get();
      generateRandTest(afterBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1, loopOp.get(), scopeCoreType);
      loopOp->body.push_back(std::move(beforeBlock));
      loopOp->body.push_back(std::move(afterBlock));
      auto beforePlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      beforePlaceHolderOp->beforeOp = loopOp.get();
      auto afterPlaceHolderOp =
          std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
      afterPlaceHolderOp->afterOp = loopOp.get();
      scopeOp->body.push_back(std::move(beforePlaceHolderOp));
      scopeOp->body.push_back(std::move(loopOp));
      scopeOp->body.push_back(std::move(afterPlaceHolderOp));
      empty = false;
    } else if (depth < max_depth &&
               isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
               isTrueWithProbability(scope_cond_prob_a, scope_cond_prob_b)) {
      auto conditionOp =
          std::make_unique<Condition>(nullptr, scopeOp, nullptr, nullptr);
      conditionOp->isUnlikely =
          isTrueWithProbability(unlikely_cond_prob_a, unlikely_cond_prob_b);
      auto trueBlock = std::make_unique<Scope>();
      trueBlock->parentOp = conditionOp.get();
      generateRandTest(trueBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1, counterScope, scopeCoreType);
      auto falseBlock = std::make_unique<Scope>();
      falseBlock->parentOp = conditionOp.get();
      generateRandTest(falseBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1, counterScope, scopeCoreType);
      conditionOp->setTrueScope(std::move(trueBlock));
      conditionOp->setFalseScope(std::move(falseBlock));
      scopeOp->body.push_back(std::move(conditionOp));
      empty = false;
    } else {
      hivm::PIPE pipeRead =
          pipesVec[getRand() % static_cast<int>(pipesVec.size())];
      // hivm::PIPE pipeWrite = pipesVec[getRand() % pipesVec.size()];
      hivm::PIPE pipeWrite = pipeRead;

      auto coreType = hivm::TCoreType::CUBE_OR_VECTOR;
      if (syncMode == SyncMode::TEST_CROSS_CORE_MODE) {
        if (scopeCoreType.has_value()) {
          coreType = scopeCoreType.value();
        } else {
          coreType =
              (getRand() % 2) ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
        }
      }

      int readValsNum = 1;
      int writeValsNum = 1;
      if (counterScope != nullptr && enableMultiBuffer) {
        readValsNum = getRand(1, read_write_vals_max_num);
        writeValsNum = getRand(1, read_write_vals_max_num);
      }
      SmallVector<SmallVector<int64_t>> readVals(1);
      SmallVector<SmallVector<int64_t>> writeVals(1);
      for (auto i : getNDifferentRandNums(readValsNum, pointerOps.size())) {
        readVals.back().push_back(pointerOps[i]);
      }
      for (auto i : getNDifferentRandNums(writeValsNum, pointerOps.size())) {
        writeVals.back().push_back(pointerOps[i]);
      }
      llvm::SmallVector<MemInfo> readMemInfo;
      llvm::SmallVector<MemInfo> writeMemInfo;
      for (auto &val : readVals) {
        readMemInfo.push_back(MemInfo::getMemInfo(counterScope, val));
      }
      for (auto &val : writeVals) {
        writeMemInfo.push_back(MemInfo::getMemInfo(counterScope, val));
      }

      auto rwOp =
          std::make_unique<RWOperation>(nullptr, scopeOp, coreType, pipeRead,
                                        pipeWrite, readMemInfo, writeMemInfo);
      assert(rwOp != nullptr);
      scopeOp->body.push_back(std::move(rwOp));
      empty = false;
      remOpNum--;
    }
    if (!empty && (scopeOp->getDepth() > 3) &&
        isTrueWithProbability(scope_out_prob_a, scope_out_prob_b)) {
      break;
    }
  }
  auto blockEndPlaceHolderOp = std::make_unique<PlaceHolder>(nullptr, scopeOp);
  blockEndPlaceHolderOp->scopeEnd = scopeOp;
  scopeOp->body.push_back(std::move(blockEndPlaceHolderOp));
}

std::unique_ptr<OperationBase> SyncTester::getGeneratedRandomTest() {
  std::vector<int> pointerOps(numPointers);
  std::iota(pointerOps.begin(), pointerOps.end(), 0);

  const llvm::SmallVector<hivm::PIPE> pipesVec = {
      hivm::PIPE::PIPE_MTE1, hivm::PIPE::PIPE_MTE2, hivm::PIPE::PIPE_MTE3};

  auto funcIr = std::make_unique<Function>(nullptr);

  int numFunctionBlocks = 1;
  if (isTrueWithProbability(multi_function_blocks_prob_a,
                            multi_function_blocks_prob_b)) {
    numFunctionBlocks = 1 + (getRand() % function_blocks_max_num);
  }

  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = funcIr.get();
  auto *parScopeOp = scopeOp.get();
  funcIr->body.push_back(std::move(scopeOp));

  for (int i = 0; i < numFunctionBlocks; i++) {
    auto functionBlockOp = std::make_unique<FunctionBlock>();
    functionBlockOp->parentOp = parScopeOp;
    int remOpNum = numOperations / numFunctionBlocks;
    if (i + 1 >= numFunctionBlocks) {
      remOpNum += numOperations % numFunctionBlocks;
    }
    generateRandTest(functionBlockOp.get(), pointerOps, pipesVec, remOpNum, 0);
    parScopeOp->body.push_back(std::move(functionBlockOp));
  }

  return funcIr;
}

// Walk generated IR and populate per-pipeline queues. The produced queues are
// consumed by runSimulation to emulate runtime ordering and check conflicts.
void SyncTester::fillPipelines(OperationBase *op, Scope *counterScope) {
  assert(op != nullptr);

  if (isa<SyncOp>(op) && op->getDepth() <= 3) {
    llvm::dbgs() << op->getDepth() << ' ' << op->str(0, false) << '\n';
    assert(false && "unexpected sync operation outside of function-block");
  }

  if (auto *setWaitOp = dyn_cast<SetWaitOp>(op)) {
    assert(!setWaitOp->checkFirstIter && !setWaitOp->checkLastIter);
  }

  if (isa<FunctionBlock>(op) &&
      isTrueWithProbability(skip_function_block_prob_a,
                            skip_function_block_prob_b)) {
    return;
  }

  int loopIdx = 0;
  if (counterScope != nullptr) {
    loopIdx = countersMap[counterScope];
  }

  if (auto *loopOp = dyn_cast<Loop>(op)) {
    if (loopOp->isCVPreloadingLoop) {
      assert(loopOp->haveCounter);
      assert(!loopOp->isParallel);
      assert(loopOp->loopPreloadNum.has_value());

      int numIter = loopOp->loopPreloadNum.value() * 2 + 3;
      auto &loopCounter = countersMap[loopOp];
      for (int i = 0; i < numIter; i++) {
        for (auto &outerScope : loopOp->body) {
          for (auto &innerOp : cast<Scope>(outerScope.get())->body) {
            if (auto innerScope = dyn_cast<Scope>(innerOp.get())) {
              if (innerScope->preloadNum.has_value()) {
                if (i == 0) {
                  countersMap[innerScope] = 0;
                }
                assert(innerScope->haveCounter);
                // loopPreloadNum-scopePreloadNum <= i
                // < (numIter-scopePreloadNum)
                int lb = loopOp->loopPreloadNum.value() -
                         innerScope->preloadNum.value() - 1;
                int ub = numIter - innerScope->preloadNum.value();
                if (!(lb <= i && i < ub)) {
                  continue;
                }
              }
            }
            fillPipelines(innerOp.get(), loopOp);
          }
        }
        loopCounter++;
      }
      return;
    }
  }

  if (auto *loopOp = dyn_cast<Loop>(op)) {
    assert(loopOp->haveCounter);
    if (isTrueWithProbability(dead_loop_prob_a, dead_loop_prob_b)) {
      return;
    }

    int numIter = 1;
    if (!loopOp->isParallel) {
      numIter = enableMultiBuffer ? loop_unrolling_num : 1;
    }

    auto &loopCounter = countersMap[loopOp];
    for (int i = 0; i < numIter; i++) {
      for (auto &op : loopOp->body) {
        assert(isa<Scope>(op));
        fillPipelines(op.get(), loopOp);
      }
      loopCounter++;
    }
    // if (loopOp->body.size() > 1) {
    //   fillPipelines(loopOp->body.front().get(), loopCnt + 1,
    //                 loopIdx * loop_unrolling_num + 0);
    // }
    return;
  }

  if (auto *condOp = dyn_cast<Condition>(op)) {
    if (isTrueWithProbability(true_branch_prob_a, true_branch_prob_b)) {
      fillPipelines(condOp->getTrueScope(), counterScope);
    } else if (condOp->hasFalseScope()) {
      fillPipelines(condOp->getFalseScope(), counterScope);
    }
    return;
  }

  if (auto *scopeOp = dyn_cast<Scope>(op)) {
    auto newCounterScope = scopeOp->haveCounter ? scopeOp : counterScope;
    for (auto &op : scopeOp->body) {
      fillPipelines(op.get(), newCounterScope);
    }
    if (scopeOp->haveCounter) {
      countersMap[scopeOp]++;
    }
    return;
  }

  if (auto *setFlagOp = dyn_cast<SetFlagOp>(op)) {
    assert(!setFlagOp->eventIds.empty());
    CorePipeInfo corePipe(setFlagOp->coreType, setFlagOp->pipeSrc);
    if (!setFlagOp->allAtOnce) {
      pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, loopIdx}});
    } else {
      for (size_t i = 0; i < setFlagOp->eventIds.size(); i++) {
        pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, i}});
      }
    }
    return;
  }

  if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(op)) {
    assert(!waitFlagOp->eventIds.empty());
    CorePipeInfo corePipe(waitFlagOp->coreType, waitFlagOp->pipeDst);
    if (!waitFlagOp->allAtOnce) {
      pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, loopIdx}});
    } else {
      for (size_t i = 0; i < waitFlagOp->eventIds.size(); i++) {
        pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, i}});
      }
    }
    return;
  }

  if (auto *barrierOp = dyn_cast<BarrierOp>(op)) {
    assert(syncMode == SyncMode::TEST_INTRA_CORE_MODE);
    CorePipeInfo corePipe(hivm::TCoreType::CUBE_OR_VECTOR, barrierOp->pipe);
    pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, loopIdx}});
    return;
  }

  if (auto *rwOp = dyn_cast<RWOperation>(op)) {
    CorePipeInfo corePipe(rwOp->coreType, rwOp->pipeRead);
    pipelineQue[corePipe].push_back({{idx++, getRand()}, {op, loopIdx}});
    return;
  }
}

// Simulate execution of pipeline queues, detect memory and synchronization
// violations. Returns success when no conflicts occur for the run.
llvm::LogicalResult SyncTester::runSimulation(int runId, bool debugPrint) {

  auto compairPipelines = [&](const CorePipeInfo &corePipe1,
                              const CorePipeInfo &corePipe2) {
    auto &pipeQue1 = pipelineQue[corePipe1];
    auto &pipeQue2 = pipelineQue[corePipe2];
    assert(!pipeQue1.empty() || !pipeQue2.empty());
    if (pipeQue1.empty() || pipeQue2.empty()) {
      return pipeQue1.empty();
    }
    auto &[idx1, _op1] = pipeQue1.front();
    auto &[idx2, _op2] = pipeQue2.front();
    auto &op1 = _op1.first;
    auto &op2 = _op2.first;
    if (op1->opType == op2->opType) {
      if (idx1.second != idx2.second) {
        return idx1.second < idx2.second;
      }
      return idx1.first < idx2.first;
    }
    // rwOp < (setFlagOp < barrierOp < syncOp)
    return (isa<RWOperation>(op1) && isa<SyncOp>(op2)) ||
           (isa<SetFlagOp>(op1) && isa<SyncOp>(op2) && !isa<SetFlagOp>(op2)) ||
           (isa<BarrierOp>(op1) && isa<SyncOp>(op2) &&
            !isa<SetFlagOp, BarrierOp>(op2));
  };

  std::set<CorePipeInfo, decltype(compairPipelines)> mainQue(compairPipelines);
  llvm::DenseMap<hivm::TCoreType,
                 llvm::DenseMap<int, llvm::DenseSet<RWOperation *>>>
      ongoingWrites, ongoingReads;
  llvm::DenseMap<CorePipeInfo, std::vector<std::pair<RWOperation *, int>>>
      runningOps;
  llvm::DenseMap<std::tuple<hivm::TCoreType, hivm::PIPE, hivm::PIPE>,
                 std::multiset<int64_t>>
      triggeredSetFlagOps;
  std::set<int> allIndexes;
  auto corePipeAll =
      CorePipeInfo(hivm::TCoreType::CUBE_OR_VECTOR, hivm::PIPE::PIPE_ALL);
  auto &pipeAllQue = pipelineQue[corePipeAll];

  for (auto &[corePipe, pipeQue] : pipelineQue) {
    if (corePipe != corePipeAll) {
      mainQue.insert(corePipe);
    }
  }
  for (int i = 0; i < idx; i++) {
    allIndexes.insert(i);
  }

  auto getTriggeredGroup = [this](SetWaitOp *syncOp) {
    if (syncMode == SyncMode::TEST_INTRA_CORE_MODE) {
      return std::make_tuple(TCoreType::CUBE_OR_VECTOR, syncOp->pipeSrc,
                             syncOp->pipeDst);
    } else {
      if (isa<SetFlagOp>(syncOp)) {
        return std::make_tuple(getOppositeCoreType(syncOp->coreType),
                               hivm::PIPE::PIPE_UNASSIGNED,
                               hivm::PIPE::PIPE_UNASSIGNED);
      } else {
        return std::make_tuple(syncOp->coreType, hivm::PIPE::PIPE_UNASSIGNED,
                               hivm::PIPE::PIPE_UNASSIGNED);
      }
    }
  };

  auto refreshPipeQue = [&](const CorePipeInfo &corePipe) {
    if (corePipe == corePipeAll) {
      return;
    }
    mainQue.erase(corePipe);
    auto &pipeQue = pipelineQue[corePipe];
    while (!pipeQue.empty()) {
      auto [op, loopIdx] = pipeQue.front().second;
      if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(op)) {
        assert(CorePipeInfo(waitFlagOp->coreType, waitFlagOp->pipeDst) ==
               corePipe);
        auto &triggeredOps = triggeredSetFlagOps[getTriggeredGroup(waitFlagOp)];
        assert(!waitFlagOp->eventIds.empty());
        auto eventId = waitFlagOp->eventIds[loopIdx % static_cast<int>(
            waitFlagOp->eventIds.size())];
        auto it = triggeredOps.find(eventId);
        if (it != triggeredOps.end()) {
          assert((*it) == eventId);
          triggeredOps.erase(it);
          allIndexes.erase(pipeQue.front().first.first);
          pipeQue.pop_front();
          LLVM_DEBUG({
            if (debugPrint) {
              llvm::dbgs() << "wait-flag triggered: "
                           << waitFlagOp->str(0, false) << '\n';
            }
          });
          continue;
        }
      }
      break;
    }
    if (!pipeQue.empty()) {
      mainQue.insert(corePipe);
    }
  };

  auto checkPipeAll = [&]() {
    if (pipeAllQue.empty()) {
      return false;
    }
    if (!allIndexes.empty() &&
        *allIndexes.begin() < pipeAllQue.front().first.first) {
      return false;
    }
    return true;
  };

  auto getCurCorePipe = [&]() -> CorePipeInfo {
    if (checkPipeAll()) {
      return corePipeAll;
    }
    std::optional<CorePipeInfo> retCorePipe;
    for (auto corePipe : mainQue) {
      if (pipeAllQue.empty() || (pipelineQue[corePipe].front().first.first <
                                 pipeAllQue.front().first.first)) {
        retCorePipe = corePipe;
        break;
      }
    }
    assert(retCorePipe.has_value());
    mainQue.erase(retCorePipe.value());
    return retCorePipe.value();
  };

  [[maybe_unused]] auto printMainQue = [&]() {
    for (auto &[triggerGroup, eventIds] : triggeredSetFlagOps) {
      auto [coreType, setPipe, waitPipe] = triggerGroup;
      if (!eventIds.empty()) {
        llvm::dbgs() << coreType << ' ' << setPipe << ' ' << waitPipe << ' ';
        for (auto e : eventIds) {
          llvm::dbgs() << e << ' ';
        }
        llvm::dbgs() << '\n';
      }
    }
    for (auto corePipe : mainQue) {
      int szLimit = 100;
      llvm::dbgs() << corePipe.coreType << ' ' << corePipe.pipe << ": ";
      for (auto e : pipelineQue[corePipe]) {
        if (!szLimit--) {
          break;
        }
        llvm::dbgs() << e.second.second << ' ' << e.second.first->str(0, false)
                     << ' ';
      }
      llvm::dbgs() << '\n';
    }
    for (auto corePipe : {corePipeAll}) {
      int szLimit = 100;
      llvm::dbgs() << corePipe.coreType << ' ' << corePipe.pipe << ": ";
      for (auto e : pipelineQue[corePipe]) {
        if (!szLimit--) {
          break;
        }
        llvm::dbgs() << e.second.second << ' ' << e.second.first->str(0, false)
                     << ' ';
      }
      llvm::dbgs() << '\n';
    }
  };

  [[maybe_unused]] auto decomposeIndex = [](int index) {
    std::vector<int> ret;
    int x = index;
    while (true) {
      ret.push_back(x % loop_unrolling_num);
      x /= loop_unrolling_num;
      if (!x) {
        break;
      }
    }
    reverse(ret.begin(), ret.end());
    return ret;
  };

  auto checkMemoryConflict = [&](RWOperation *rwOp, int loopIndex) {
    auto curCoreType = rwOp->coreType;
    auto oppCoreType = getOppositeCoreType(curCoreType);
    for (auto &memInfo : rwOp->readMemInfo) {
      auto &addrs = memInfo.pointerLikeInfo->addresses;
      auto index = loopIndex % static_cast<int>(addrs.size());
      auto ptrVal = addrs[index];
      auto ongoingWriteOps = ongoingWrites[oppCoreType][ptrVal];
      if (!ongoingWriteOps.empty()) {
        LLVM_DEBUG({
          if (debugPrint) {
            llvm::dbgs() << "read while write memory conflict: "
                         << "curLoopIdx(" << loopIndex << ") idx(" << index
                         << ") ptr(" << ptrVal << ")\n"
                         << rwOp->str(0, false) << '\n';
            for (auto op : ongoingWriteOps) {
              llvm::dbgs() << op->str(0, false) << '\n';
            }
          }
        });
        return llvm::failure();
      }
    }
    for (auto &memInfo : rwOp->writeMemInfo) {
      auto &addrs = memInfo.pointerLikeInfo->addresses;
      auto index = loopIndex % static_cast<int>(addrs.size());
      auto ptrVal = addrs[index];
      auto ongoingReadOps = ongoingReads[oppCoreType][ptrVal];
      auto ongoingWriteOps = ongoingWrites[oppCoreType][ptrVal];
      if (!ongoingReadOps.empty()) {
        LLVM_DEBUG({
          if (debugPrint) {
            llvm::dbgs() << "write while read memory conflict: "
                         << "curLoopIdx(" << loopIndex << ") idx(" << index
                         << ") ptr(" << ptrVal << ")\n"
                         << rwOp->str(0, false) << '\n';
            for (auto op : ongoingReadOps) {
              llvm::dbgs() << op->str(0, false) << '\n';
            }
          }
        });
        return llvm::failure();
      }
      if (!ongoingWriteOps.empty()) {
        LLVM_DEBUG({
          if (debugPrint) {
            llvm::dbgs() << "write while write memory conflict: "
                         << "curLoopIdx(" << loopIndex << ") idx(" << index
                         << ") ptr(" << ptrVal << ")\n"
                         << rwOp->str(0, false) << '\n';
            for (auto op : ongoingWriteOps) {
              llvm::dbgs() << op->str(0, false) << '\n';
            }
          }
        });
        return llvm::failure();
      }
    }
    return llvm::success();
  };

  auto handleRWOperation = [&](RWOperation *rwOp, int loopIndex) {
    for (auto &memInfo : rwOp->readMemInfo) {
      auto &addrs = memInfo.pointerLikeInfo->addresses;
      auto index = loopIndex % static_cast<int>(addrs.size());
      auto ptrVal = addrs[index];
      ongoingReads[rwOp->coreType][ptrVal].insert(rwOp);
    }
    for (auto &memInfo : rwOp->writeMemInfo) {
      auto &addrs = memInfo.pointerLikeInfo->addresses;
      auto index = loopIndex % static_cast<int>(addrs.size());
      auto ptrVal = addrs[index];
      ongoingWrites[rwOp->coreType][ptrVal].insert(rwOp);
    }
    CorePipeInfo corePipe(rwOp->coreType, rwOp->pipeRead);
    assert(rwOp->pipeRead == rwOp->pipeWrite);
    runningOps[corePipe].emplace_back(rwOp, loopIndex);
  };

  auto handleSetFlagOp = [&](SetFlagOp *setFlagOp, int loopIndex) {
    CorePipeInfo corePipeSrc(setFlagOp->coreType, setFlagOp->pipeSrc);
    CorePipeInfo corePipeDst(getOppositeCoreType(setFlagOp->coreType),
                             setFlagOp->pipeDst);
    for (auto [rwOp, loopIdx] : runningOps[corePipeSrc]) {
      for (auto &memInfo : rwOp->readMemInfo) {
        auto &addrs = memInfo.pointerLikeInfo->addresses;
        auto index = loopIdx % static_cast<int>(addrs.size());
        auto ptrVal = addrs[index];
        ongoingReads[setFlagOp->coreType][ptrVal].erase(rwOp);
      }
      for (auto &memInfo : rwOp->writeMemInfo) {
        auto &addrs = memInfo.pointerLikeInfo->addresses;
        auto index = loopIdx % static_cast<int>(addrs.size());
        auto ptrVal = addrs[index];
        ongoingWrites[setFlagOp->coreType][ptrVal].erase(rwOp);
      }
    }
    assert(!setFlagOp->eventIds.empty());
    triggeredSetFlagOps[getTriggeredGroup(setFlagOp)].insert(
        setFlagOp->eventIds[loopIndex %
                            static_cast<int>(setFlagOp->eventIds.size())]);
    refreshPipeQue(corePipeDst);
  };

  auto clearPipeline = [&](const CorePipeInfo &corePipe) {
    for (auto [rwOp, loopIdx] : runningOps[corePipe]) {
      for (auto &memInfo : rwOp->readMemInfo) {
        auto &addrs = memInfo.pointerLikeInfo->addresses;
        auto index = loopIdx % static_cast<int>(addrs.size());
        auto ptrVal = addrs[index];
        ongoingReads[rwOp->coreType][ptrVal].erase(rwOp);
      }
    }
    for (auto [rwOp, loopIdx] : runningOps[corePipe]) {
      for (auto &memInfo : rwOp->writeMemInfo) {
        auto &addrs = memInfo.pointerLikeInfo->addresses;
        auto index = loopIdx % static_cast<int>(addrs.size());
        auto ptrVal = addrs[index];
        ongoingWrites[rwOp->coreType][ptrVal].erase(rwOp);
      }
    }
  };

  auto handleBarrierOp = [&](BarrierOp *barrierOp) {
    if (barrierOp->pipe == hivm::PIPE::PIPE_ALL) {
      for (auto &[corePipe, rwOps] : runningOps) {
        clearPipeline(corePipe);
      }
    } else {
      clearPipeline(
          CorePipeInfo(hivm::TCoreType::CUBE_OR_VECTOR, barrierOp->pipe));
    }
  };

  while (!mainQue.empty()) {

    LLVM_DEBUG({
      if (debugPrint) {
        printMainQue();
      }
    });

    auto curPipe = getCurCorePipe();
    auto &pipeQue = pipelineQue[curPipe];
    assert(!pipeQue.empty());
    auto [curOp, curLoopIdx] = pipeQue.front().second;
    allIndexes.erase(pipeQue.front().first.first);
    pipeQue.pop_front();

    LLVM_DEBUG({
      if (debugPrint) {
        llvm::dbgs() << "[loopIdx: " << curLoopIdx << " - ";
        llvm::interleaveComma(decomposeIndex(curLoopIdx), llvm::dbgs());
        llvm::dbgs() << "] " << curOp->str(0, false) << "\n\n";
      }
    });

    if (auto *rwOp = dyn_cast<RWOperation>(curOp)) {
      if (llvm::failed(checkMemoryConflict(rwOp, curLoopIdx))) {
        return llvm::failure();
      }
      handleRWOperation(rwOp, curLoopIdx);
    } else if (auto *setFlagOp = dyn_cast<SetFlagOp>(curOp)) {
      handleSetFlagOp(setFlagOp, curLoopIdx);
    } else if (auto *barrierOp = dyn_cast<BarrierOp>(curOp)) {
      handleBarrierOp(barrierOp);
    } else if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(curOp)) {
      LLVM_DEBUG({
        if (debugPrint) {
          llvm::dbgs() << "untriggered waitFlagOp: "
                       << waitFlagOp->str(0, false) << '\n';
        }
      });
      return failure();
    } else {
      assert(false && "unexpected op type");
    }

    refreshPipeQue(curPipe);
  }
#ifndef NDEBUG
  for (auto &[_, value] : pipelineQue) {
    assert(value.empty());
  }
#endif
  return success();
}

// High-level test runner: generate random test, run solver, generate result
// ops, and run multiple simulation runs to verify correctness.
llvm::LogicalResult SyncTester::test() {
  auto funcIr = getGeneratedRandomTest();
  LLVM_DEBUG(llvm::dbgs() << "before-tester:\n"
                          << funcIr->str(0, true) << '\n';);

  SyncSolverOptions options(syncMode, /*isMemBasedArch=*/false,
                            /*isRegBasedArch=*/false);
  options.enableCVPatterns = true;

  auto irTranslator =
      std::make_unique<IRTranslator>(std::move(funcIr), options);

  auto solver = std::make_unique<SolverTest>(std::move(irTranslator));

  DEBUG_WITH_TYPE("gss-print-unrolled-sync-ir", {
    for (auto &occ : solver->syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->syncIrIndex << ' ' << occ->startIndex << ' '
                   << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  solver->solve();

  CodeGenerator codeGen(std::move(solver));
  codeGen.generateFuncIrResultOps();

  llvm::outs() << codeGen.funcIr->str(0, true) << '\n';
  llvm::outs().flush();

  int simulatedRuns = 100;
  auto result = llvm::success();
  std::unique_ptr<std::mt19937> tmpRandGenerator = std::move(randGenerator);
  for (int i = 0; i < simulatedRuns; i++) {
    reset();
    randGenerator = std::make_unique<std::mt19937>(i);
    fillPipelines(codeGen.funcIr.get());
    if (llvm::failed(runSimulation(i))) {
      LLVM_DEBUG({
        reset();
        randGenerator = std::make_unique<std::mt19937>(i);
        fillPipelines(codeGen.funcIr.get());
        auto status = runSimulation(i, true);
        assert(llvm::failed(status));
        llvm::dbgs() << "runId: " << i << '\n';
      });
      result = llvm::failure();
      break;
    }
  }
  randGenerator = std::move(tmpRandGenerator);
  return result;
}

// If environment indicates tester mode, parse env vars and run SyncTester.
void SyncTester::runTestMode(const SmallVector<int64_t> &options) {
  if (options.size() != SyncTester::getOptionsNum()) {
    llvm::report_fatal_error(("Expected size of sync-tester options to be equal to " +
                              std::to_string(SyncTester::getOptionsNum()))
                              .c_str());
    return;
  }

  int64_t numRuns = options[0];
  int64_t initSeed = options[1];
  int64_t numOperations = options[2];
  int64_t numPointers = options[3];
  int64_t enableMultiBuffer = options[4];
  int64_t enableCrossCoreMode = options[5];

  llvm::outs() << "tester-options:"
               << " seed(" << initSeed << ")"
               << " num_ops(" << numOperations << ")"
               << " num_ptrs(" << numPointers << ")"
               << " multibuffer(" << enableMultiBuffer << ")"
               << " enableCrossCoreMode(" << enableCrossCoreMode << ")"
               << "\n";

  SyncTester tester(numOperations, numPointers, enableMultiBuffer,
                    enableCrossCoreMode, initSeed);

  llvm::LogicalResult result = llvm::success();
  for (int64_t runI = 0; runI < numRuns; ++runI) {
    auto status = tester.test();
    if (llvm::failed(status)) {
      result = llvm::failure();
      break;
    }
  }

  llvm::outs() << (llvm::succeeded(result) ? "succeeded" : "failed") << "\n";
}
