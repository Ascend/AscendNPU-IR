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
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <asm-generic/errno.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#define DEBUG_TYPE "hivm-graph-sync-solver-tester"

using namespace mlir;
using namespace hivm::syncsolver;

// Random test IR generator: recursively builds scopes, loops, conditions and RW
// ops. Used by the tester to create synthetic cases exercising the solver.
void SyncTester::generateRandTest(Scope *scopeOp,
                                  const std::vector<int> &pointerOps,
                                  const std::vector<hivm::PIPE> &pipesVec,
                                  int &remOpNum, int depth) {
  bool empty = true;
  while (remOpNum > 0) {

    if (depth < max_depth &&
        isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
        isTrueWithProbability(scope_for_loop_prob_a, scope_for_loop_prob_b)) {
      auto loopOp = std::make_unique<Loop>(nullptr, scopeOp);
      auto loopBlock = std::make_unique<Scope>();
      loopBlock->parentOp = loopOp.get();
      generateRandTest(loopBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1);
      loopOp->body.push_back(std::move(loopBlock));
      scopeOp->body.push_back(std::move(loopOp));
      empty = false;
    } else if (depth < max_depth &&
               isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
               isTrueWithProbability(scope_while_loop_prob_a,
                                     scope_while_loop_prob_b)) {
      auto loopOp = std::make_unique<Loop>(nullptr, scopeOp);
      auto beforeBlock = std::make_unique<Scope>();
      beforeBlock->parentOp = loopOp.get();
      generateRandTest(beforeBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1);
      auto afterBlock = std::make_unique<Scope>();
      afterBlock->parentOp = loopOp.get();
      generateRandTest(afterBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1);
      loopOp->body.push_back(std::move(beforeBlock));
      loopOp->body.push_back(std::move(afterBlock));
      scopeOp->body.push_back(std::move(loopOp));
      empty = false;
    } else if (depth < max_depth &&
               isTrueWithProbability(scope_in_prob_a, scope_in_prob_b) &&
               isTrueWithProbability(scope_cond_prob_a, scope_cond_prob_b)) {
      auto conditionOp =
          std::make_unique<Condition>(nullptr, scopeOp, nullptr, nullptr);
      auto trueBlock = std::make_unique<Scope>();
      trueBlock->parentOp = conditionOp.get();
      generateRandTest(trueBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1);
      auto falseBlock = std::make_unique<Scope>();
      falseBlock->parentOp = conditionOp.get();
      generateRandTest(falseBlock.get(), pointerOps, pipesVec, remOpNum,
                       depth + 1);
      conditionOp->setTrueScope(std::move(trueBlock));
      conditionOp->setFalseScope(std::move(falseBlock));
      scopeOp->body.push_back(std::move(conditionOp));
      empty = false;
    } else {
      hivm::PIPE pipeRead = pipesVec[getRand() % static_cast<int>(pipesVec.size())];
      // hivm::PIPE pipeWrite = pipesVec[getRand() % pipesVec.size()];
      hivm::PIPE pipeWrite = pipeRead;

      int readValsNum = 1;
      int writeValsNum = 1;
      if (enableMultiBuffer) {
        // readValsNum = writeValsNum = 2;
        readValsNum = (getRand() % read_write_vals_max_num) + 1;
        writeValsNum = (getRand() % read_write_vals_max_num) + 1;
      }
      SmallVector<SmallVector<int>> readVals(1);
      SmallVector<SmallVector<int>> writeVals(1);
      for (auto i : getNDifferentRandNums(readValsNum, pointerOps.size())) {
        readVals.back().push_back(pointerOps[i]);
      }
      for (auto i : getNDifferentRandNums(writeValsNum, pointerOps.size())) {
        writeVals.back().push_back(pointerOps[i]);
      }

      auto rwOp = std::make_unique<RWOperation>(
          nullptr, scopeOp, pipeRead, pipeWrite,
          hivm::TCoreType::CUBE_OR_VECTOR, SmallVector<Value>(),
          SmallVector<Value>());
      rwOp->testReadMemVals = readVals;
      rwOp->testWriteMemVals = writeVals;
      assert(rwOp != nullptr);
      scopeOp->body.push_back(std::move(rwOp));
      empty = false;
      remOpNum--;
    }

    if (!empty && (scopeOp->parentOp != nullptr) &&
        isTrueWithProbability(scope_out_prob_a, scope_out_prob_b)) {
      break;
    }
  }

  auto ghostOp = std::make_unique<Ghost>(nullptr, scopeOp, nullptr);
  scopeOp->body.push_back(std::move(ghostOp));
}

std::unique_ptr<OperationBase> SyncTester::getGeneratedRandomTest() {
  std::vector<int> pointerOps(numPointers);
  std::iota(pointerOps.begin(), pointerOps.end(), 0);

  std::vector<hivm::PIPE> pipesVec;
  for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
    if ((static_cast<unsigned>(usedPipesMask) >> i) & 1u) {
      pipesVec.push_back(static_cast<hivm::PIPE>(i));
    }
  }

  int remOpNum = numOperations;
  auto funcIr = std::make_unique<Function>(nullptr);
  auto scopeOp = std::make_unique<Scope>();
  generateRandTest(scopeOp.get(), pointerOps, pipesVec, remOpNum, 0);
  scopeOp->parentOp = funcIr.get();
  funcIr->body.push_back(std::move(scopeOp));
  return funcIr;
}

// Walk generated IR and populate per-pipeline queues. The produced queues are
// consumed by runSimulation to emulate runtime ordering and check conflicts.
void SyncTester::fillPipelines(const OperationBase *op, int loopCnt,
                               int loopIdx) {

  assert(op != nullptr);
  bool doubled = loopCnt > 0;
  bool allDeadLoops =
      isTrueWithProbability(all_dead_loops_prob_a, all_dead_loops_prob_b);

  if (auto *setWaitOp = dyn_cast<const SetWaitOp>(op)) {
    if (setWaitOp->checkFirstIter && (loopIdx % loop_unrolling_num != 0)) {
      return;
    }
    if (setWaitOp->checkLastIter && ((loopIdx + 1) % loop_unrolling_num != 0)) {
      return;
    }
  }

  if (auto *loopOp = dyn_cast<const Loop>(op)) {
    int numIter = (enableMultiBuffer || !doubled) ? loop_unrolling_num : 1;
    if (allDeadLoops ||
        isTrueWithProbability(dead_loop_prob_a, dead_loop_prob_b)) {
      numIter = 0;
    }
    for (int i = 0; i < numIter; i++) {
      for (auto &op : loopOp->body) {
        assert(isa<Scope>(op));
        fillPipelines(op.get(), loopCnt + 1, loopIdx * loop_unrolling_num + i);
      }
    }
    // if (loopOp->body.size() > 1) {
    //   fillPipelines(loopOp->body.front().get(), loopCnt + 1,
    //                 loopIdx * loop_unrolling_num + 0);
    // }
    return;
  }

  if (auto *condOp = dyn_cast<const Condition>(op)) {
    if (isTrueWithProbability(true_branch_prob_a, true_branch_prob_b)) {
      fillPipelines(condOp->getTrueScope(), loopCnt, loopIdx);
    } else if (condOp->hasFalseScope()) {
      fillPipelines(condOp->getFalseScope(), loopCnt, loopIdx);
    }
    return;
  }

  if (auto *scopeOp = dyn_cast<const Scope>(op)) {
    for (auto &op : scopeOp->body) {
      fillPipelines(op.get(), loopCnt, loopIdx);
    }
    return;
  }

  if (auto *setOp = dyn_cast<const SetFlagOp>(op)) {
    assert(!setOp->eventIds.empty());
    auto pipeline = setOp->pipeSrc;
    if (!setOp->allAtOnce) {
      pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, loopIdx}});
    } else {
      for (size_t i = 0; i < setOp->eventIds.size(); i++) {
        pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, i}});
      }
    }
    return;
  }

  if (auto *waitOp = dyn_cast<const WaitFlagOp>(op)) {
    assert(!waitOp->eventIds.empty());
    auto pipeline = waitOp->pipeDst;
    if (!waitOp->allAtOnce) {
      pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, loopIdx}});
    } else {
      for (size_t i = 0; i < waitOp->eventIds.size(); i++) {
        pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, i}});
      }
    }
    return;
  }

  if (auto *barrierOp = dyn_cast<const BarrierOp>(op)) {
    auto pipeline = barrierOp->pipe;
    pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, loopIdx}});
    return;
  }

  if (auto *rwOp = dyn_cast<const RWOperation>(op)) {
    auto pipeline = rwOp->pipeRead;
    pipelineQue[pipeline].push_back({{idx++, getRand()}, {op, loopIdx}});
    return;
  }
}

// Simulate execution of pipeline queues, detect memory and synchronization
// violations. Returns success when no conflicts occur for the run.
llvm::LogicalResult SyncTester::runSimulation(int runId, bool debugPrint) {

  auto compairPipelines = [&](const hivm::PIPE &pipe1,
                              const hivm::PIPE &pipe2) {
    auto &pipeQue1 = pipelineQue[pipe1];
    auto &pipeQue2 = pipelineQue[pipe2];
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

  std::set<hivm::PIPE, decltype(compairPipelines)> mainQue(compairPipelines);
  llvm::DenseMap<int, llvm::DenseSet<const RWOperation *>> ongoingWrites,
      ongoingReads;
  llvm::DenseMap<hivm::PIPE, std::vector<std::pair<const RWOperation *, int>>>
      runningOps;
  llvm::DenseMap<hivm::PIPE,
                 llvm::DenseMap<hivm::PIPE, std::multiset<hivm::EVENT>>>
      triggeredSetFlagOps;
  std::set<int> allIndexes;
  auto &pipeAllQue = pipelineQue[hivm::PIPE::PIPE_ALL];

  for (auto &[pipeId, pipeQue] : pipelineQue) {
    if (pipeId != hivm::PIPE::PIPE_ALL) {
      mainQue.insert(pipeId);
    }
  }
  for (int i = 0; i < idx; i++) {
    allIndexes.insert(i);
  }

  auto refreshPipeQue = [&](const hivm::PIPE &pipe) {
    if (pipe == hivm::PIPE::PIPE_ALL) {
      return;
    }
    mainQue.erase(pipe);
    auto &pipeQue = pipelineQue[pipe];
    while (!pipeQue.empty()) {
      auto [op, loopIdx] = pipeQue.front().second;
      if (auto *waitOp = dyn_cast<const WaitFlagOp>(op)) {
        assert(waitOp->pipeDst == pipe);
        auto &triggeredOps =
            triggeredSetFlagOps[waitOp->pipeSrc][waitOp->pipeDst];
        assert(!waitOp->eventIds.empty());
        auto it = triggeredOps.find(
            waitOp->eventIds[loopIdx % waitOp->eventIds.size()]);
        if (it != triggeredOps.end()) {
          triggeredOps.erase(it);
          allIndexes.erase(pipeQue.front().first.first);
          pipeQue.pop_front();
          continue;
        }
      }
      break;
    }
    if (!pipeQue.empty()) {
      mainQue.insert(pipe);
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

  auto getCurPipe = [&]() {
    if (checkPipeAll()) {
      return hivm::PIPE::PIPE_ALL;
    }
    auto retPipe = hivm::PIPE::PIPE_UNASSIGNED;
    for (auto pipe : mainQue) {
      if (pipeAllQue.empty() || (pipelineQue[pipe].front().first.first <
                                 pipeAllQue.front().first.first)) {
        retPipe = pipe;
        break;
      }
    }
    assert(retPipe != hivm::PIPE::PIPE_UNASSIGNED);
    mainQue.erase(retPipe);
    return retPipe;
  };

  [[maybe_unused]] auto printMainQue = [&]() {
    for (auto pipe : mainQue) {
      int szLimit = 100;
      llvm::dbgs() << stringifyPIPE(pipe).str() << ": ";
      for (auto e : pipelineQue[pipe]) {
        if (!szLimit--) {
          break;
        }
        llvm::dbgs() << e.second.first->str(0, false) << ' ';
      }
      llvm::dbgs() << '\n';
    }
    for (auto pipe : {hivm::PIPE::PIPE_ALL}) {
      int szLimit = 100;
      llvm::dbgs() << stringifyPIPE(pipe).str() << ": ";
      for (auto e : pipelineQue[pipe]) {
        if (!szLimit--) {
          break;
        }
        llvm::dbgs() << e.second.first->str(0, false) << ' ';
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

  auto checkMemoryConflict = [&](const RWOperation *rwOp, int loopIndex) {
    for (const auto &readPtr : rwOp->testReadMemVals) {
      auto index = loopIndex % static_cast<int>(readPtr.size());
      auto ptrVal = readPtr[index];
      auto ongoingWriteOps = ongoingWrites[ptrVal];
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
    for (const auto &writePtr : rwOp->testWriteMemVals) {
      auto index = loopIndex % static_cast<int>(writePtr.size());
      auto ptrVal = writePtr[index];
      auto ongoingReadOps = ongoingReads[ptrVal];
      auto ongoingWriteOps = ongoingWrites[ptrVal];
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

  auto handleRWOperation = [&](const RWOperation *rwOp, int loopIndex) {
    for (const auto &readPtr : rwOp->testReadMemVals) {
      auto index = loopIndex % static_cast<int>(readPtr.size());
      auto ptrVal = readPtr[index];
      ongoingReads[ptrVal].insert(rwOp);
    }
    for (const auto &writePtr : rwOp->testWriteMemVals) {
      auto index = loopIndex % static_cast<int>(writePtr.size());
      auto ptrVal = writePtr[index];
      ongoingWrites[ptrVal].insert(rwOp);
    }
    auto rwPipe = rwOp->pipeRead;
    assert(rwOp->pipeRead == rwOp->pipeWrite);
    runningOps[rwPipe].emplace_back(rwOp, loopIndex);
  };

  auto handleSetFlagOp = [&](const SetFlagOp *setFlagOp, int loopIndex) {
    for (auto [rwOp, loopIdx] : runningOps[setFlagOp->pipeSrc]) {
      for (auto readPtr : rwOp->testReadMemVals) {
        auto index = loopIdx % static_cast<int>(readPtr.size());
        auto ptrVal = readPtr[index];
        ongoingReads[ptrVal].erase(rwOp);
      }
      for (auto writePtr : rwOp->testWriteMemVals) {
        auto index = loopIdx % static_cast<int>(writePtr.size());
        auto ptrVal = writePtr[index];
        ongoingWrites[ptrVal].erase(rwOp);
      }
    }
    assert(!setFlagOp->eventIds.empty());
    triggeredSetFlagOps[setFlagOp->pipeSrc][setFlagOp->pipeDst].insert(
        setFlagOp->eventIds[loopIndex % setFlagOp->eventIds.size()]);
    refreshPipeQue(setFlagOp->pipeDst);
  };

  auto clearPipeline = [&](const hivm::PIPE &pipe) {
    for (auto [rwOp, loopIdx] : runningOps[pipe]) {
      for (auto readPtr : rwOp->testReadMemVals) {
        auto index = loopIdx % static_cast<int>(readPtr.size());
        auto ptrVal = readPtr[index];
        ongoingReads[ptrVal].erase(rwOp);
      }
    }
    for (auto [rwOp, loopIdx] : runningOps[pipe]) {
      for (auto writePtr : rwOp->testWriteMemVals) {
        auto index = loopIdx % static_cast<int>(writePtr.size());
        auto ptrVal = writePtr[index];
        ongoingWrites[ptrVal].erase(rwOp);
      }
    }
  };

  auto handleBarrierOp = [&](const BarrierOp *barrierOp) {
    if (barrierOp->pipe == hivm::PIPE::PIPE_ALL) {
      for (auto &[pipe, rwOps] : runningOps) {
        clearPipeline(pipe);
      }
    } else {
      clearPipeline(barrierOp->pipe);
    }
  };

  while (!mainQue.empty()) {

    LLVM_DEBUG({
      if (debugPrint) {
        printMainQue();
      }
    });

    auto curPipe = getCurPipe();
    auto &pipeQue = pipelineQue[curPipe];
    assert(!pipeQue.empty());
    auto [curOp, curLoopIdx] = pipeQue.front().second;
    allIndexes.erase(pipeQue.front().first.first);
    pipeQue.pop_front();

    LLVM_DEBUG({
      if (debugPrint) {
        llvm::dbgs() << "[loopIdx: ";
        llvm::interleaveComma(decomposeIndex(curLoopIdx), llvm::dbgs());
        llvm::dbgs() << "] " << curOp->str(0, false) << "\n\n";
      }
    });

    if (auto *rwOp = dyn_cast<const RWOperation>(curOp)) {
      if (llvm::failed(checkMemoryConflict(rwOp, curLoopIdx))) {
        return llvm::failure();
      }
      handleRWOperation(rwOp, curLoopIdx);
    } else if (auto *setFlagOp = dyn_cast<const SetFlagOp>(curOp)) {
      handleSetFlagOp(setFlagOp, curLoopIdx);
    } else if (auto *barrierOp = dyn_cast<const BarrierOp>(curOp)) {
      handleBarrierOp(barrierOp);
    } else if (auto *waitOp = dyn_cast<const WaitFlagOp>(curOp)) {
      LLVM_DEBUG({
        if (debugPrint) {
          llvm::dbgs() << "untriggered waitOp: " << waitOp->str(0, false)
                       << '\n';
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
  llvm::outs() << "generated test with: "
               << " seed(" << seed << ")"
               << " multibuffer(" << enableMultiBuffer << ")"
               << " num_ops(" << numOperations << ")"
               << " num_ptrs(" << numPointers << ") :\n";

  Solver solver(getGeneratedRandomTest());
  LLVM_DEBUG(llvm::dbgs() << solver.funcIr->str(0, true) << '\n';);

  LLVM_DEBUG({
    for (auto &occ : solver.syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->syncIrIndex << ' ' << occ->startIndex << ' '
                   << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  solver.solveTest();
  solver.generateFuncIrResultOps();
  llvm::outs() << solver.funcIr->str(0, true) << '\n';
  llvm::outs().flush();

  int simulatedRuns = 100;
  for (int i = 0; i < simulatedRuns; i++) {
    reset();
    randGenerator = std::make_unique<std::mt19937>(i);
    fillPipelines(solver.funcIr.get());
    if (llvm::failed(runSimulation(i))) {
      LLVM_DEBUG({
        reset();
        randGenerator = std::make_unique<std::mt19937>(i);
        fillPipelines(solver.funcIr.get());
        auto status = runSimulation(i, true);
        assert(llvm::failed(status));
        llvm::dbgs() << "runId: " << i << '\n';
      });
      return llvm::failure();
    }
  }
  return llvm::success();
}