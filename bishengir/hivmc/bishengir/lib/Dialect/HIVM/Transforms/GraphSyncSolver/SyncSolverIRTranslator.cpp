//===--------- IRTranslator.cpp ------- Graph Sync Solver -------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <climits>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "hivm-gss-ir-translator"

using namespace mlir;
using namespace hivm::syncsolver;

// Resolve a Value into the underlying pointer-like Values used for memory
// conflict analysis (handles block args, selects, scf::If, scf::For/While
// results etc.).
llvm::SmallVector<Value>
IRTranslator::collectPointerLikeOps(Value val, func::FuncOp funcOp) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (auto *iterArgOperand = forOp.getTiedLoopInit(blockArg)) {
        return collectPointerLikeOps(iterArgOperand->get(), funcOp);
      }
    }

    if (auto whileOp =
            dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner()->getParent() == &whileOp.getAfter()) {
        auto argNum = blockArg.getArgNumber();
        return collectPointerLikeOps(whileOp.getConditionOp().getArgs()[argNum],
                                     funcOp);
      } else {
        assert(blockArg.getOwner()->getParent() == &whileOp.getBefore());
        return collectPointerLikeOps(whileOp.getTiedLoopInit(blockArg)->get(),
                                     funcOp);
      }
    }

    if (syncMode == SyncMode::NORMAL_SYNC) {
      if (hacc::utils::isKernelArg(funcOp, blockArg.getArgNumber(),
                                   hacc::KernelArgType::kWorkspace)) {
        bool isSplittedMixKernel =
            funcOp->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name);
        if (isSplittedMixKernel) {
          return {};
        }
      }
    }

    return {val};
  }

  auto *op = val.getDefiningOp();
  assert(op != nullptr);

  if (syncMode == SyncMode::NORMAL_SYNC) {
    if (isa<hivm::PointerCastOp, tensor::EmptyOp, memref::AllocOp>(op)) {
      return {val};
    }
  }
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    if (isa<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
      return {val};
    }
    if (this->isRegBasedArch) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        auto allocOpResult = allocOp.getResult();
        if (auto spaceAttr = GetBufferSpaceAttr(allocOpResult)) {
          return {val};
        }
      }
    }
  }

  if (auto resultVal = dyn_cast<OpResult>(val)) {
    if (auto dsiOp =
            dyn_cast<DestinationStyleOpInterface>(resultVal.getDefiningOp())) {
      return collectPointerLikeOps(
          dsiOp.getDpsInitOperand(resultVal.getResultNumber())->get(), funcOp);
    }
  }

  if (auto aliasInfoVec = getOperationAliasInfo(op); !aliasInfoVec.empty()) {
    llvm::SmallVector<Value> collectedOps;
    for (auto aliasInfo : aliasInfoVec) {
      collectedOps.append(collectPointerLikeOps(aliasInfo.second, funcOp));
    }
    return collectedOps;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    llvm::SmallVector<Value> collectedOps;
    Operation::result_range resultVals = ifOp.getResults();
    auto it = std::find(resultVals.begin(), resultVals.end(), val);
    assert(it != resultVals.end());
    OpResult resultVal = *it;
    auto operandNum = resultVal.getResultNumber();
    // then
    auto thenYield = ifOp.thenYield();
    auto firstPath =
        collectPointerLikeOps(thenYield->getOperand(operandNum), funcOp);
    collectedOps.append(firstPath);
    // else
    auto elseYield = ifOp.elseYield();
    if (elseYield) {
      auto secondPath =
          collectPointerLikeOps(elseYield->getOperand(operandNum), funcOp);
      collectedOps.append(secondPath);
    }
    return collectedOps;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto resultNum = dyn_cast<OpResult>(val).getResultNumber();
    auto yieldedVal = forOp.getYieldedValues()[resultNum];
    return collectPointerLikeOps(yieldedVal, funcOp);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    auto resultNum = dyn_cast<OpResult>(val).getResultNumber();
    auto yieldedVal = whileOp.getYieldedValues()[resultNum];
    return collectPointerLikeOps(yieldedVal, funcOp);
  }

  return {};
}

// Collect pointer operands for a vector of Values (flattening aliases).
llvm::SmallVector<Value>
IRTranslator::getMemoryOps(const SmallVector<Value> &vals,
                           std::optional<func::FuncOp> funcOpOpt) {
  SmallVector<Value> collectedOps;
  auto curFuncOp = funcOpOpt.has_value() ? funcOpOpt.value() : this->funcOp;
  for (auto val : vals) {
    for (auto pointerOp : collectPointerLikeOps(val, curFuncOp)) {
      collectedOps.push_back(pointerOp);
    }
  }
  return collectedOps;
}

// Return read/write memory operands for a generic operation by consulting
// DestinationStyleOpInterface and ExtraBufferOpInterface.
std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
IRTranslator::getReadWriteMemoryOps(Operation *op) {
  assert(op != nullptr);
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  if (auto dsiOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    readMemVals = getMemoryOps(dsiOp.getDpsInputs());
    writeMemVals = getMemoryOps(dsiOp.getDpsInits());
  }
  if (auto extraBufferOp = dyn_cast<ExtraBufferOpInterface>(op)) {
    auto extraWriteMemVals = getMemoryOps(extraBufferOp.getExtraBuffers());
    llvm::append_range(writeMemVals, extraWriteMemVals);
  }
  return std::make_pair(readMemVals, writeMemVals);
}

// Wrap memref/affine load/store into RWOperation nodes when appropriate.
template <typename OP>
std::unique_ptr<OperationBase>
IRTranslator::getLoadStoreOp(OP loadStoreOp, OperationBase *parentOp) {
  auto op = loadStoreOp.getOperation();
  auto pipe = hivm::PIPE::PIPE_S;
  auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
  if (syncMode == SyncMode::NORMAL_SYNC) {
    auto memorySpaceAttr = GetBufferSpaceAttr(loadStoreOp.getMemRef());
    if (!memorySpaceAttr.has_value()) {
      return nullptr;
    }
  }
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    auto coreType = hivm::getCoreType(op);
    assert(llvm::succeeded(coreType));
    assert(coreType.value() != hivm::TCoreType::CUBE_OR_VECTOR);
    coreTypeVal = coreType.value();
  }
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  if constexpr (std::is_same_v<OP, memref::LoadOp> ||
                std::is_same_v<OP, affine::AffineLoadOp>) {
    readMemVals = getMemoryOps({loadStoreOp.getMemRef()});
  } else {
    static_assert(std::is_same_v<OP, memref::StoreOp> ||
                  std::is_same_v<OP, affine::AffineStoreOp>);
    writeMemVals = getMemoryOps({loadStoreOp.getMemRef()});
  }
  auto rwOp = std::make_unique<RWOperation>(op, parentOp, coreTypeVal, pipe,
                                            pipe, readMemVals, writeMemVals);
  return rwOp;
}

// Decompose specific MmadL1 ops into a small inline sequence in the IR for
// easier sync handling.
std::unique_ptr<OperationBase>
IRTranslator::getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op,
                                  OperationBase *parentOp) {

  auto outerScopeOp = std::make_unique<Scope>();
  outerScopeOp->parentOp = parentOp;
  outerScopeOp->op = mmadl1Op;

  auto mmadl1LoopOp =
      std::make_unique<MmadL1LoopOp>(mmadl1Op, outerScopeOp.get());
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = mmadl1LoopOp.get();
  auto coreType = TCoreType::CUBE_OR_VECTOR;
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    coreType = TCoreType::CUBE;
  }
  auto loadL0aOp = std::make_unique<LoadL0AOp>(
      nullptr, scopeOp.get(), coreType, hivm::PIPE::PIPE_MTE1,
      hivm::PIPE::PIPE_MTE1, getMemoryOps({mmadl1Op.getA()}),
      SmallVector<Value>());
  scopeOp->body.push_back(std::move(loadL0aOp));

  auto loadL0bOp = std::make_unique<LoadL0BOp>(
      nullptr, scopeOp.get(), coreType, hivm::PIPE::PIPE_MTE1,
      hivm::PIPE::PIPE_MTE1, getMemoryOps({mmadl1Op.getB()}),
      SmallVector<Value>());
  scopeOp->body.push_back(std::move(loadL0bOp));

  if (auto bias = mmadl1Op.getPerChannelBias()) {
    auto loadBiasOp = std::make_unique<LoadBiasOp>(
        nullptr, scopeOp.get(), coreType, hivm::PIPE::PIPE_MTE1,
        hivm::PIPE::PIPE_MTE1, getMemoryOps({mmadl1Op.getPerChannelBias()}),
        SmallVector<Value>());
    scopeOp->body.push_back(std::move(loadBiasOp));
  }

  auto mmadl0Op = std::make_unique<MmadL0Operation>(
      mmadl1Op, scopeOp.get(), coreType, hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_M,
      SmallVector<Value>(), getMemoryOps({mmadl1Op.getC()}));
  mmadl0Op->hasUnitFlagFeat = true;
  unitFlagFeaturedOps.insert(mmadl0Op.get());
  mmadl1LoopOp->mmadL0Op = mmadl0Op.get();
  scopeOp->body.push_back(std::move(mmadl0Op));
  mmadl1LoopOp->body.push_back(std::move(scopeOp));

  auto beforePlaceHolderOp =
      std::make_unique<PlaceHolder>(nullptr, mmadl1LoopOp->parentOp);
  beforePlaceHolderOp->beforeOp = mmadl1LoopOp.get();
  auto afterPlaceHolderOp =
      std::make_unique<PlaceHolder>(nullptr, mmadl1LoopOp->parentOp);
  afterPlaceHolderOp->afterOp = mmadl1LoopOp.get();
  outerScopeOp->body.push_back(std::move(beforePlaceHolderOp));
  outerScopeOp->body.push_back(std::move(mmadl1LoopOp));
  outerScopeOp->body.push_back(std::move(afterPlaceHolderOp));
  return outerScopeOp;
}

std::optional<hivm::PIPE>
IRTranslator::getInferredPipe(Operation *op, TCoreType coreType,
                              const llvm::SmallVector<Value> &writeMemInfo) {
  if (!isa<hivm::CopyOp, hivm::VBrcOp>(op) ||
      coreType == TCoreType::CUBE_OR_VECTOR || writeMemInfo.empty()) {
    return {};
  }
  std::optional<hivm::PIPE> pipe;
  for (auto &memInfoVal : writeMemInfo) {
    auto addressSpaceOpt = GetBufferSpaceAttr(memInfoVal);
    if (!addressSpaceOpt.has_value()) {
      return {};
    }
    auto addressSpace = addressSpaceOpt.value().getAddressSpace();
    std::optional<hivm::PIPE> curPipe;
    if (isa<hivm::CopyOp>(op) && addressSpace == AddressSpace::L1 &&
        coreType == TCoreType::VECTOR) {
      curPipe = PIPE::PIPE_MTE3;
    }
    if (isa<hivm::VBrcOp>(op) && addressSpace == AddressSpace::L1 &&
        coreType == TCoreType::VECTOR) {
      curPipe = PIPE::PIPE_MTE2;
    }
    if (curPipe.has_value()) {
      if (pipe.has_value() && curPipe != pipe.value()) {
        return {};
      }
      pipe = curPipe;
    }
  }
  return pipe;
}

std::unique_ptr<OperationBase>
IRTranslator::getPipeInterfaceOp(hivm::OpPipeInterface op,
                                 OperationBase *parentOp) {
  if (decomposeMmadl1Op && (syncMode == SyncMode::NORMAL_SYNC)) {
    if (auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(op.getOperation())) {
      return getDecomposedMmadl1(mmadl1Op, parentOp);
    }
  }
  auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    auto coreType = hivm::getCoreType(op.getOperation());
    assert(llvm::succeeded(coreType));
    assert(coreType.value() != hivm::TCoreType::CUBE_OR_VECTOR);
    coreTypeVal = coreType.value();
  }
  auto [readMemOps, writeMemOps] = getReadWriteMemoryOps(op.getOperation());
  std::optional<hivm::PIPE> pipe;
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    if (isa<hivm::CopyOp, hivm::VBrcOp>(op)) {
      if (auto pipeOpt = getInferredPipe(op, coreTypeVal, writeMemOps)) {
        pipe = pipeOpt.value();
      } else {
        pipe = PIPE::PIPE_S;
      }
    }
  }
  hivm::PIPE pipeRead, pipeWrite;
  if (pipe.has_value()) {
    pipeRead = pipe.value();
    pipeWrite = pipe.value();
  } else {
    pipeRead = op.isSinglePipeOp() ? op.getPipe() : op.getInPipe();
    pipeWrite = op.isSinglePipeOp() ? op.getPipe() : op.getOutPipe();
  }
  assert(pipeRead != hivm::PIPE::PIPE_UNASSIGNED &&
         pipeWrite != hivm::PIPE::PIPE_UNASSIGNED);
  auto rwOp = std::make_unique<RWOperation>(op.getOperation(), parentOp,
                                            coreTypeVal, pipeRead, pipeWrite,
                                            readMemOps, writeMemOps);
  if (isa<UnitFlagEnabledInterface>(op.getOperation())) {
    rwOp->hasUnitFlagFeat = true;
    unitFlagFeaturedOps.insert(rwOp.get());
  }
  return rwOp;
}

std::unique_ptr<OperationBase>
IRTranslator::getTensorExtractOp(tensor::ExtractOp extractOp,
                                 OperationBase *parentOp) {
  auto pipeRead = hivm::PIPE::PIPE_S;
  auto pipeWrite = hivm::PIPE::PIPE_S;
  auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
  if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    auto coreType = hivm::getCoreType(extractOp.getOperation());
    assert(llvm::succeeded(coreType));
    if (coreType.value() == hivm::TCoreType::CUBE_OR_VECTOR) {
      return nullptr;
    }
    coreTypeVal = coreType.value();
  }
  auto readMemOps = getMemoryOps({extractOp.getTensor()});
  auto rwOp = std::make_unique<RWOperation>(
      extractOp.getOperation(), parentOp, coreTypeVal, pipeRead, pipeWrite,
      readMemOps, llvm::SmallVector<Value>());
  return rwOp;
}

std::unique_ptr<OperationBase>
IRTranslator::getCallOp(func::CallOp callOp, OperationBase *parentOp) {
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  SymbolTable symtab(module);
  auto calledFuncOp = symtab.lookup<func::FuncOp>(callOp.getCallee());
  if (!calledFuncOp->hasAttr(hivm::VectorFunctionAttr::name)) {
    return nullptr;
  }
  SmallVector<Value> readMemVals, writeMemVals;
  auto handleRWValue = [&](Value val, bool isRead, bool isWrite) {
    for (auto &rwVal : getMemoryOps({val}, calledFuncOp)) {
      if (auto blockArg = dyn_cast<BlockArgument>(rwVal)) {
        auto callArg = callOp->getOperand(blockArg.getArgNumber());
        if (isRead) {
          readMemVals.push_back(callArg);
        }
        if (isWrite) {
          writeMemVals.push_back(callArg);
        }
      }
    }
  };
  calledFuncOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      handleRWValue(transferReadOp.getSource(), true, false);
    }
    if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      handleRWValue(transferWriteOp.getVector(), true, false);
      handleRWValue(transferWriteOp.getSource(), false, true);
    }
  });
  readMemVals = getMemoryOps(readMemVals);
  writeMemVals = getMemoryOps(writeMemVals);
  auto coreTypeVal = syncMode == SyncMode::NORMAL_SYNC
                         ? hivm::TCoreType::CUBE_OR_VECTOR
                         : hivm::TCoreType::VECTOR;
  auto rwOp = std::make_unique<RWOperation>(
      callOp.getOperation(), parentOp, coreTypeVal, hivm::PIPE::PIPE_V,
      hivm::PIPE::PIPE_V, readMemVals, writeMemVals);
  return rwOp;
}

bool IRTranslator::isUnlikelyCondition(Condition *condOp) {
  assert(condOp != nullptr);
  if (condOp->op != nullptr) {
    return condOp->op->hasAttrOfType<UnitAttr>(
        hivm::UnlikelyConditionAttr::name);
  }
  return false;
}

// Build a Scope tree (funcIr) from MLIR Region recursively.
std::unique_ptr<Scope> IRTranslator::funcIrBuilder(Region &region,
                                                   OperationBase *parentOp) {
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = parentOp;

  for (auto &block : region.getBlocks()) {
    auto blockBeginPlaceHolderOp =
        std::make_unique<PlaceHolder>(nullptr, scopeOp.get());
    blockBeginPlaceHolderOp->scopeBegin = scopeOp.get();
    blockBeginPlaceHolderOp->block = &block;
    scopeOp->body.push_back(std::move(blockBeginPlaceHolderOp));

    for (auto &op : block.getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto trueScope = funcIrBuilder(ifOp.getThenRegion(), nullptr);
        std::unique_ptr<Scope> falseScope;
        if (ifOp.elseBlock()) {
          falseScope = funcIrBuilder(ifOp.getElseRegion(), nullptr);
        }
        auto conditionOp = std::make_unique<Condition>(
            &op, scopeOp.get(), std::move(trueScope), std::move(falseScope));
        conditionOp->isUnlikely = isUnlikelyCondition(conditionOp.get());
        scopeOp->body.push_back(std::move(conditionOp));
        continue;
      }
      if (isa<LoopLikeOpInterface>(op)) {
        auto loopOp = std::make_unique<Loop>(&op, scopeOp.get());
        for (auto &region : op.getRegions()) {
          auto regionOp = funcIrBuilder(region, loopOp.get());
          loopOp->body.push_back(std::move(regionOp));
        }
        auto beforePlaceHolderOp =
            std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
        beforePlaceHolderOp->beforeOp = loopOp.get();
        auto afterPlaceHolderOp =
            std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
        afterPlaceHolderOp->afterOp = loopOp.get();
        scopeOp->body.push_back(std::move(beforePlaceHolderOp));
        scopeOp->body.push_back(std::move(loopOp));
        scopeOp->body.push_back(std::move(afterPlaceHolderOp));
        continue;
      }
      if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
        if (auto rwOp = getPipeInterfaceOp(pipeOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(storeOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(loadOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(storeOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(loadOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
        if (auto rwOp = getTensorExtractOp(extractOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (auto rwOp = getCallOp(callOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      }
    }

    auto blockEndPlaceHolderOp =
        std::make_unique<PlaceHolder>(nullptr, scopeOp.get());
    blockEndPlaceHolderOp->scopeEnd = scopeOp.get();
    blockEndPlaceHolderOp->block = &block;
    scopeOp->body.push_back(std::move(blockEndPlaceHolderOp));
  }

  return scopeOp;
}

// Various processing-order and sync IR builder helpers
// (generateProcessingOrders, syncIrBuilder).
void IRTranslator::generateProcessingOrders(Occurrence *occ, int l, int r,
                                            bool isUseless) {
  assert(llvm::isa_and_present<Scope>(occ->op));
  assert(r < occ->endIndex);
  int start = occ->syncIrIndex;
  int end = occ->syncIrEndIndex;
  assert(start != -1 && end != -1);
  for (int i = start; i < end; i++) {
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      ProcessingOrder order(syncIr[i].get(), start + 1, i - 1,
                            /*reverseOrder=*/true, isUseless);
      processingOrders.push_back(order);
    }
  }
  for (int i = r; i >= l; i--) {
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      ProcessingOrder order(syncIr[i].get(), start + 1, end - 1,
                            /*reverseOrder=*/false, isUseless);
      processingOrders.push_back(order);
    }
  }
}

void IRTranslator::generateProcessingOrders(int l, int r, bool isUseless) {
  for (int i = l; i <= r; i++) {
    if (llvm::isa_and_nonnull<Scope>(syncIr[i]->op)) {
      generateProcessingOrders(syncIr[i].get(), l, i - 1, isUseless);
      assert(syncIr[i]->syncIrIndex == i);
      assert(syncIr[i]->syncIrEndIndex != -1);
      i = syncIr[i]->syncIrEndIndex - 1;
      continue;
    }
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      assert(syncIr[i]->syncIrIndex == i);
      ProcessingOrder order(syncIr[i].get(), l, i - 1, /*reverseOrder=*/true,
                            isUseless);
      processingOrders.push_back(order);
    }
  }
}

void IRTranslator::generateProcessingOrders(int l1, int r1, int l2, int r2,
                                            bool isUseless) {
  assert(r1 < l2);
  for (int i = l2; i <= r2; i++) {
    if (llvm::isa_and_nonnull<Scope>(syncIr[i]->op)) {
      generateProcessingOrders(syncIr[i].get(), l1, r1, isUseless);
      assert(syncIr[i]->syncIrIndex == i);
      assert(syncIr[i]->syncIrEndIndex != -1);
      i = syncIr[i]->syncIrEndIndex - 1;
      continue;
    }
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      assert(syncIr[i]->syncIrIndex == i);
      ProcessingOrder order(syncIr[i].get(), l1, r1, /*reverseOrder=*/true,
                            isUseless);
      processingOrders.push_back(order);
    }
  }
}

// Build the linearized sync IR (syncIr) and record occurrence ranges for
// analysis.
void IRTranslator::syncIrBuilder(OperationBase *op, Occurrence *parentOcc,
                                 int depth, bool isUseless) {
  assert(op != nullptr);
  int startIndex = globalIndex++;
  auto occ = std::make_unique<Occurrence>(op, parentOcc, depth, startIndex, -1);
  occ->syncIrIndex = static_cast<int>(syncIr.size());
  if (auto *rwOp = dyn_cast<RWOperation>(op)) {
    occ->hasUnitFlagFeat = rwOp->hasUnitFlagFeat;
  }
  syncIr.push_back(std::move(occ));
  Occurrence *occPtr = syncIr.back().get();
  opAllOccurrences[op].push_back(occPtr);

  if (parentOcc != nullptr) {
    occChildrenMem[parentOcc].push_back(occPtr);
  }

  if (auto *scopeOp = dyn_cast<Scope>(op)) {

    bool unrollLoop = isa<Loop>(op);
    if (!unrollLoop) {
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
      }
    } else {
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
      }
      occPtr->loopSplitIndex = static_cast<int>(syncIr.size());
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, true);
      }
    }

    if (unrollLoop) {
      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               occPtr->loopSplitIndex - 1, isUseless);

      generateProcessingOrders(occPtr->loopSplitIndex,
                               static_cast<int>(syncIr.size()) - 1,
                               /*isUseless=*/true);

      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               occPtr->loopSplitIndex - 1,
                               occPtr->loopSplitIndex,
                               static_cast<int>(syncIr.size()) - 1, isUseless);

      ProcessingOrder orderSkipIter1(nullptr, occPtr->syncIrIndex + 1,
                                     occPtr->loopSplitIndex - 1,
                                     /*reverseOrder=*/true,
                                     /*isUseless=*/false,
                                     /*skip=*/true);

      ProcessingOrder orderSkipIter2(nullptr, occPtr->loopSplitIndex,
                                     static_cast<int>(syncIr.size()) - 1,
                                     /*reverseOrder=*/false,
                                     /*isUseless=*/false,
                                     /*skip=*/true);

      processingOrders.push_back(orderSkipIter1);
      processingOrders.push_back(orderSkipIter2);
    } else if (op->opType == OpType::SCOPE) {
      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               static_cast<int>(syncIr.size()) - 1, isUseless);
    }
  }

  int endIndex = globalIndex++;
  occPtr->endIndex = endIndex;
  occPtr->syncIrEndIndex = static_cast<int>(syncIr.size());
}
