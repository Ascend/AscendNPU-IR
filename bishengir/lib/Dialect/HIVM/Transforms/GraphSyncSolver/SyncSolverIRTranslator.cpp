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
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <climits>
#include <memory>
#include <optional>
#include <queue>
#include <utility>

#define DEBUG_TYPE "hivm-gss-ir-translator"

using namespace mlir;
using namespace hivm::syncsolver;

// Resolve a Value into the underlying pointer-like Values used for memory
// conflict analysis (handles block args, selects, scf::If, scf::For/While
// results etc.).
llvm::SmallVector<Value> IRTranslator::tracebackMemValsStep(Value val) {
  llvm::SmallVector<Value> collectedVals;
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (auto *initOperand = forOp.getTiedLoopInit(blockArg)) {
        collectedVals.push_back(initOperand->get());
      }
      if (auto *yieldedValue = forOp.getTiedLoopYieldedValue(blockArg)) {
        collectedVals.push_back(yieldedValue->get());
      }
    } else if (auto whileOp =
                   dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner()->getParent() == &whileOp.getBefore()) {
        if (auto *initOperand = whileOp.getTiedLoopInit(blockArg)) {
          collectedVals.push_back(initOperand->get());
        }
        if (auto *yieldedValueAfter =
                whileOp.getTiedLoopYieldedValue(blockArg)) {
          collectedVals.push_back(yieldedValueAfter->get());
        }
      } else {
        assert(blockArg.getOwner()->getParent() == &whileOp.getAfter());
        auto argNum = blockArg.getArgNumber();
        assert(whileOp.getConditionOp().getArgs().size() > argNum);
        auto yieldedValueBefore = whileOp.getConditionOp().getArgs()[argNum];
        collectedVals.push_back(yieldedValueBefore);
      }
    }
    if (blockArgAliases.contains(val)) {
      llvm::append_range(collectedVals, blockArgAliases[val]);
    }
    return collectedVals;
  }

  auto resultVal = dyn_cast<OpResult>(val);
  assert(resultVal != nullptr);
  if (!resultVal) {
    return collectedVals;
  }

  auto *defOp = resultVal.getDefiningOp();
  auto resultNum = resultVal.getResultNumber();
  assert(defOp != nullptr);

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    // then
    auto thenYield = ifOp.thenYield();
    auto yieldedValueThen = thenYield->getOperand(resultNum);
    collectedVals.push_back(yieldedValueThen);
    // else
    if (ifOp.elseBlock()) {
      auto elseYield = ifOp.elseYield();
      auto yieldedValueElse = elseYield->getOperand(resultNum);
      collectedVals.push_back(yieldedValueElse);
    }
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    assert(forOp.getYieldedValues().size() > resultNum);
    auto yieldedValue = forOp.getYieldedValues()[resultNum];
    collectedVals.push_back(yieldedValue);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
    assert(whileOp.getConditionOp().getArgs().size() > resultNum);
    auto yieldedValueBefore = whileOp.getConditionOp().getArgs()[resultNum];
    assert(whileOp.getYieldedValues().size() > resultNum);
    auto yieldedValueAfter = whileOp.getYieldedValues()[resultNum];
    collectedVals.push_back(yieldedValueBefore);
    collectedVals.push_back(yieldedValueAfter);
  } else if (auto scopeOp = dyn_cast<scope::ScopeOp>(defOp)) {
    Region &scopeRegion = scopeOp.getRegion();
    Block &scopeBlock = scopeRegion.front();
    auto returnOp = dyn_cast<scope::ReturnOp>(scopeBlock.getTerminator());
    assert(returnOp != nullptr);
    assert(returnOp->getOperands().size() > resultNum);
    auto returnedValue = returnOp->getOperand(resultNum);
    collectedVals.push_back(returnedValue);
  }

  if (auto aliasInfoVec = getOperationAliasInfo(defOp); !aliasInfoVec.empty()) {
    for (auto aliasInfo : aliasInfoVec) {
      if (aliasInfo.first == resultVal) {
        collectedVals.push_back(aliasInfo.second);
      }
    }
  } else if (auto dsiOp = dyn_cast<DestinationStyleOpInterface>(
                 resultVal.getDefiningOp())) {
    for (auto initOperand : dsiOp.getDpsInits()) {
      collectedVals.push_back(initOperand);
    }
  }

  return collectedVals;
}

llvm::SmallVector<Value> IRTranslator::tracebackMemVals(Value val,
                                                        func::FuncOp funcOp) {
  std::queue<Value> que;
  llvm::DenseSet<Value> visitedVals;
  llvm::SetVector<Value> collectedValsSet;
  que.push(val);
  visitedVals.insert(val);
  bool isSplittedMixKernel =
      funcOp->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name);

  while (!que.empty()) {
    auto curVal = que.front();
    que.pop();

    auto nextVals = tracebackMemValsStep(curVal);
    if (!nextVals.empty()) {
      for (auto nextVal : nextVals) {
        if (!visitedVals.contains(nextVal)) {
          que.push(nextVal);
          visitedVals.insert(nextVal);
        }
      }
      continue;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(curVal)) {
      if (options.isIntraCoreMode()) {
        if (hacc::utils::isKernelArg(funcOp, blockArg.getArgNumber(),
                                     hacc::KernelArgType::kWorkspace)) {
          if (isSplittedMixKernel) {
            continue;
          }
        }
      }
      collectedValsSet.insert(curVal);
      continue;
    }

    auto resultVal = dyn_cast<OpResult>(curVal);
    assert(resultVal != nullptr);
    if (!resultVal) {
      continue;
    }

    auto *defOp = resultVal.getDefiningOp();
    assert(defOp != nullptr);

    if (options.isIntraCoreMode()) {
      if (isa<hivm::PointerCastOp, tensor::EmptyOp, memref::AllocOp>(defOp)) {
        collectedValsSet.insert(resultVal);
        continue;
      }
    } else if (options.isCrossCoreMode()) {
      if (isa<bishengir::memref_ext::AllocWorkspaceOp>(defOp)) {
        collectedValsSet.insert(resultVal);
        continue;
      }
      if (options.isRegBasedArch) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(defOp)) {
          auto allocOpResult = allocOp.getResult();
          if (auto spaceAttr = GetBufferSpaceAttr(allocOpResult)) {
            collectedValsSet.insert(resultVal);
            continue;
          }
        }
      }
    }
  }

  return collectedValsSet.takeVector();
}

// Collect pointer operands for a vector of Values (flattening aliases).
llvm::SmallVector<Value>
IRTranslator::getMemoryOps(const SmallVector<Value> &vals,
                           std::optional<func::FuncOp> funcOpOpt) {
  llvm::SetVector<Value> collectedValsSet;
  auto curFuncOp = funcOpOpt.has_value() ? funcOpOpt.value() : this->funcOp;
  for (auto val : vals) {
    for (auto memVal : tracebackMemVals(val, curFuncOp)) {
      collectedValsSet.insert(memVal);
    }
  }
  return collectedValsSet.takeVector();
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
    llvm::SetVector<Value> extendedWriteMemVals(writeMemVals.begin(),
                                                writeMemVals.end());
    auto extraWriteMemVals = getMemoryOps(extraBufferOp.getExtraBuffers());
    extendedWriteMemVals.insert(extraWriteMemVals.begin(),
                                extraWriteMemVals.end());
    writeMemVals = extendedWriteMemVals.takeVector();
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
  if (options.isIntraCoreMode()) {
    auto memorySpaceAttr = GetBufferSpaceAttr(loadStoreOp.getMemRef());
    if (!memorySpaceAttr.has_value()) {
      return nullptr;
    }
  }
  if (options.isCrossCoreMode()) {
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
  if (options.isCrossCoreMode()) {
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

bool IRTranslator::isVectorOpResult(Value value) {
  if (auto resultVal = dyn_cast<OpResult>(value)) {
    if (auto op = dyn_cast<CoreTypeInterface>(resultVal.getDefiningOp())) {
      if (auto coreType = op.getCoreType()) {
        if (coreType.value() == TCoreType::VECTOR) {
          return true;
        }
      }
    }
  }
  return false;
}

std::optional<hivm::PIPE>
IRTranslator::getInferredPipe(Operation *op, TCoreType coreType,
                              const llvm::SmallVector<Value> &writeMemInfo) {
  if (!isa<hivm::CopyOp, hivm::VBrcOp, tensor::InsertSliceOp>(op) ||
      coreType == TCoreType::CUBE_OR_VECTOR) {
    return {};
  }
  if (coreType == TCoreType::VECTOR) {
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      if (isVectorOpResult(insertSliceOp.getDest())) {
        return PIPE::PIPE_V;
      }
    }
  }
  if (writeMemInfo.empty()) {
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
    if (isa<hivm::VBrcOp>(op) && (addressSpace == AddressSpace::L1)) {
      curPipe = PIPE::PIPE_MTE2;
    }
    if (isa<hivm::CopyOp, tensor::InsertSliceOp>(op) &&
        (coreType == TCoreType::VECTOR) && (addressSpace == AddressSpace::L1)) {
      curPipe = PIPE::PIPE_MTE3;
    }
    if (isa<hivm::VBrcOp, hivm::CopyOp, tensor::InsertSliceOp>(op) &&
        (coreType == TCoreType::VECTOR) && (addressSpace == AddressSpace::UB)) {
      curPipe = PIPE::PIPE_V;
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
IRTranslator::getDestinationStyleInterfaceOp(Operation *op,
                                             OperationBase *parentOp) {
  if (options.decomposeMmadl1Op) {
    if (auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(op)) {
      return getDecomposedMmadl1(mmadl1Op, parentOp);
    }
  }
  auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
  if (options.isCrossCoreMode()) {
    auto coreType = hivm::getCoreType(op);
    assert(llvm::succeeded(coreType));
    assert(coreType.value() != hivm::TCoreType::CUBE_OR_VECTOR);
    coreTypeVal = coreType.value();
  }
  auto [readMemOps, writeMemOps] = getReadWriteMemoryOps(op);
  std::optional<hivm::PIPE> pipe;
  if (options.isCrossCoreMode()) {
    if (isa<hivm::CopyOp, hivm::VBrcOp, tensor::InsertSliceOp,
            tensor::InsertOp>(op)) {
      if (auto pipeOpt = getInferredPipe(op, coreTypeVal, writeMemOps)) {
        pipe = pipeOpt.value();
      } else {
        pipe = PIPE::PIPE_S;
      }
    }
  }
  hivm::PIPE pipeRead = hivm::PIPE::PIPE_UNASSIGNED;
  hivm::PIPE pipeWrite = hivm::PIPE::PIPE_UNASSIGNED;
  if (pipe.has_value()) {
    pipeRead = pipe.value();
    pipeWrite = pipe.value();
  } else if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
    pipeRead = pipeOp.isSinglePipeOp() ? pipeOp.getPipe() : pipeOp.getInPipe();
    pipeWrite =
        pipeOp.isSinglePipeOp() ? pipeOp.getPipe() : pipeOp.getOutPipe();
  }
  assert(pipeRead != hivm::PIPE::PIPE_UNASSIGNED &&
         pipeWrite != hivm::PIPE::PIPE_UNASSIGNED);
  auto rwOp = std::make_unique<RWOperation>(op, parentOp, coreTypeVal, pipeRead,
                                            pipeWrite, readMemOps, writeMemOps);
  if (isa<UnitFlagEnabledInterface>(op)) {
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
  if (options.isCrossCoreMode()) {
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
  llvm::SetVector<Value> readMemVals, writeMemVals;
  auto handleRWValue = [&](Value val, hivm::MemoryEffect memoryEffect) {
    for (auto &rwVal : getMemoryOps({val}, calledFuncOp)) {
      if (auto blockArg = dyn_cast<BlockArgument>(rwVal)) {
        auto callArg = callOp->getOperand(blockArg.getArgNumber());
        if (memoryEffect == MemoryEffect::READ ||
            memoryEffect == MemoryEffect::READ_WRITE) {
          readMemVals.insert(callArg);
        }
        if (memoryEffect == MemoryEffect::WRITE ||
            memoryEffect == MemoryEffect::READ_WRITE) {
          writeMemVals.insert(callArg);
        }
      }
    }
  };

  // handle function arguments annotated with memory effect attributes
  for (auto [i, arg] : llvm::enumerate(calledFuncOp.getArguments())) {
    // get the attribute by name for i-th argument
    auto memEffectAttr =
        calledFuncOp.getArgAttr(i, hivm::MemoryEffectAttr::name);
    if (!memEffectAttr) {
      continue;
    }
    auto effect = cast<hivm::MemoryEffectAttr>(memEffectAttr).getEffect();
    auto callArg = callOp->getOperand(i);
    // logic based on the attribute value
    if (effect == hivm::MemoryEffect::READ) {
      handleRWValue(callArg, MemoryEffect::READ);
    } else if (effect == hivm::MemoryEffect::WRITE) {
      handleRWValue(callArg, MemoryEffect::WRITE);
    } else if (effect == hivm::MemoryEffect::READ_WRITE) {
      handleRWValue(callArg, MemoryEffect::READ_WRITE);
    }
  }

  calledFuncOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      handleRWValue(loadOp.getMemRef(), MemoryEffect::READ);
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      handleRWValue(storeOp.getMemRef(), MemoryEffect::WRITE);
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      handleRWValue(loadOp.getMemRef(), MemoryEffect::READ);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      handleRWValue(storeOp.getMemRef(), MemoryEffect::WRITE);
    } else if (auto tensorExtractOp = dyn_cast<tensor::ExtractOp>(op)) {
      handleRWValue(tensorExtractOp.getTensor(), MemoryEffect::READ);
    } else if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      handleRWValue(transferReadOp.getSource(), MemoryEffect::READ);
    } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      handleRWValue(transferWriteOp.getVector(), MemoryEffect::READ);
      handleRWValue(transferWriteOp.getSource(), MemoryEffect::WRITE);
    } else if (auto gatherOp = dyn_cast<vector::GatherOp>(op)) {
      handleRWValue(gatherOp.getBase(), MemoryEffect::READ);
    }
  });

  auto readMemValsVec = getMemoryOps(readMemVals.takeVector());
  auto writeMemValsVec = getMemoryOps(writeMemVals.takeVector());

  auto coreTypeVal = options.isIntraCoreMode() ? hivm::TCoreType::CUBE_OR_VECTOR
                                               : hivm::TCoreType::VECTOR;
  auto rwOp = std::make_unique<RWOperation>(
      callOp.getOperation(), parentOp, coreTypeVal, hivm::PIPE::PIPE_V,
      hivm::PIPE::PIPE_V, readMemValsVec, writeMemValsVec);
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

bool IRTranslator::isParallelLoop(Loop *loopOp) {
  assert(loopOp != nullptr);
  if (loopOp->op != nullptr) {
    return loopOp->op->hasAttrOfType<UnitAttr>(hivm::ParallelLoopAttr::name);
  }
  return false;
}

std::optional<int64_t> IRTranslator::getLoopMultibufferUnrollNum(Loop *loopOp) {
  assert(loopOp != nullptr);
  auto forOp = dyn_cast<scf::ForOp>(loopOp->op);
  if (!forOp) {
    return {};
  }
  if (auto intAttr =
          forOp->getAttrOfType<IntegerAttr>(kMultibufferUnrollAttrName)) {
    if (!scf::utils::isNormalized(forOp)) {
      // TODO: call normalize loop pass before plan memory, currently
      // CVPipelining ensure the loop is normalized
      forOp->emitOpError("multibuffer-enabled loop expected to be normalized");
      return {};
    }
    return intAttr.getInt();
  }
  return {};
}

void IRTranslator::updateBlockArgAliases(Block *block,
                                         OperandRange destOperands) {
  assert(block->getArguments().size() == destOperands.size());
  for (auto [destArg, destOperand] :
       llvm::zip(block->getArguments(), destOperands)) {
    blockArgAliases[destArg].push_back(destOperand);
  }
}

// Build a Scope tree (funcIr) from MLIR Region recursively.
std::unique_ptr<Scope> IRTranslator::funcIrBuilder(Region &region,
                                                   OperationBase *parentOp,
                                                   bool skipEmptyScopes) {
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = parentOp;

  if (!isa_and_present<Function>(parentOp) && region.getBlocks().size() > 1) {
    llvm_unreachable(
        "unsupported non-function region to have multiple blocks.");
  }

  for (auto &block : region.getBlocks()) {

    auto *parScope = scopeOp.get();
    if (isa_and_present<Function>(parentOp)) {
      auto blockOp = std::make_unique<FunctionBlock>();
      blockOp->parentOp = scopeOp.get();
      parScope = blockOp.get();
      scopeOp->body.push_back(std::move(blockOp));
    }

    auto blockBeginPlaceHolderOp =
        std::make_unique<PlaceHolder>(nullptr, parScope);
    blockBeginPlaceHolderOp->scopeBegin = parScope;
    blockBeginPlaceHolderOp->block = &block;
    parScope->body.push_back(std::move(blockBeginPlaceHolderOp));
    for (auto &op : block.getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto trueScope =
            funcIrBuilder(ifOp.getThenRegion(), nullptr, skipEmptyScopes);
        std::unique_ptr<Scope> falseScope;
        if (ifOp.elseBlock()) {
          falseScope =
              funcIrBuilder(ifOp.getElseRegion(), nullptr, skipEmptyScopes);
        }
        auto conditionOp = std::make_unique<Condition>(
            &op, parScope, std::move(trueScope), std::move(falseScope));
        conditionOp->isUnlikely = isUnlikelyCondition(conditionOp.get());
        if (!skipEmptyScopes || !isEmptyScope(conditionOp.get())) {
          parScope->body.push_back(std::move(conditionOp));
        }
        continue;
      }
      if (isa<LoopLikeOpInterface>(op)) {
        auto loopOp = std::make_unique<Loop>(&op, parScope);
        loopOp->isParallel = isParallelLoop(loopOp.get());
        loopOp->multibufferUnrollNum =
            getLoopMultibufferUnrollNum(loopOp.get());
        for (auto &region : op.getRegions()) {
          auto regionOp = funcIrBuilder(region, loopOp.get(), skipEmptyScopes);
          loopOp->body.push_back(std::move(regionOp));
        }
        auto beforePlaceHolderOp =
            std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
        beforePlaceHolderOp->beforeOp = loopOp.get();
        auto afterPlaceHolderOp =
            std::make_unique<PlaceHolder>(nullptr, loopOp->parentOp);
        afterPlaceHolderOp->afterOp = loopOp.get();
        if (!skipEmptyScopes || !isEmptyScope(loopOp.get())) {
          parScope->body.push_back(std::move(beforePlaceHolderOp));
          parScope->body.push_back(std::move(loopOp));
          parScope->body.push_back(std::move(afterPlaceHolderOp));
        }
        continue;
      }
      if (auto scopeScopeOp = dyn_cast<scope::ScopeOp>(op)) {
        auto curScopeOp =
            std::make_unique<Scope>(OpType::SCOPE, scopeScopeOp, scopeOp.get());
        for (auto &region : scopeScopeOp->getRegions()) {
          auto regionOp = funcIrBuilder(region, curScopeOp.get());
          curScopeOp->body.push_back(std::move(regionOp));
        }
        scopeOp->body.push_back(std::move(curScopeOp));
        continue;
      }
      if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
        updateBlockArgAliases(branchOp.getDest(), branchOp.getDestOperands());
        continue;
      }
      if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
        updateBlockArgAliases(condBranchOp.getTrueDest(),
                              condBranchOp.getTrueDestOperands());
        updateBlockArgAliases(condBranchOp.getFalseDest(),
                              condBranchOp.getFalseDestOperands());
        continue;
      }

      if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op)) {
        if (auto rwOp = getDestinationStyleInterfaceOp(dstOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(storeOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(loadOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(storeOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(loadOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
        if (auto rwOp = getTensorExtractOp(extractOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (auto rwOp = getCallOp(callOp, parScope)) {
          parScope->body.push_back(std::move(rwOp));
        }
      }
    }

    auto blockEndPlaceHolderOp =
        std::make_unique<PlaceHolder>(nullptr, parScope);
    blockEndPlaceHolderOp->scopeEnd = parScope;
    blockEndPlaceHolderOp->block = &block;
    parScope->body.push_back(std::move(blockEndPlaceHolderOp));
  }

  return scopeOp;
}

bool IRTranslator::skipLaterIterations(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (occ1->parentOcc != nullptr) {
    if (isa<Loop>(occ1->parentOcc->op)) {
      if (occ1->syncIrIndex < occ1->parentOcc->loopSplitIndex &&
          occ1->parentOcc->loopSplitIndex <= occ2->syncIrIndex) {
        return true;
      }
    }
  }
  if (occ2->parentOcc != nullptr) {
    if (isa<Loop>(occ2->parentOcc->op)) {
      if (occ1->syncIrIndex < occ2->parentOcc->loopSplitIndex &&
          occ2->parentOcc->loopSplitIndex <= occ2->syncIrIndex) {
        return true;
      }
    }
  }
  return false;
}

void IRTranslator::generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (skipLaterIterations(occ1, occ2)) {
    return;
  }
  if (isa<Scope>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, occ2->childOccs, isUseless);
  }
  if (isa<RWOperation>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders({occ1}, occ2->childOccs, isUseless);
  }
  if (isa<Scope>(occ1->op) && isa<RWOperation>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, {occ2}, isUseless);
  }
  if (auto *rwOp1 = dyn_cast<RWOperation>(occ1->op)) {
    if (auto *rwOp2 = dyn_cast<RWOperation>(occ2->op)) {
      generateProcessingOrders(rwOp1, rwOp2, occ1, occ2, isUseless);
    }
  }
}

void IRTranslator::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs, bool isUseless) {
  int64_t occsNum = static_cast<int64_t>(occs.size());
  for (int64_t i = 0; i < occsNum; i++) {
    for (int64_t j = i - 1; j >= 0; j--) {
      auto *occ1 = occs[j];
      auto *occ2 = occs[i];
      generateProcessingOrders(occ1, occ2, isUseless);
    }
  }
}

void IRTranslator::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs1,
    const llvm::SmallVector<Occurrence *> &occs2, bool isUseless) {
  for (auto *occ2 : occs2) {
    for (auto *occ1 : llvm::reverse(occs1)) {
      generateProcessingOrders(occ1, occ2, isUseless);
    }
  }
}

void IRTranslator::generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                            bool isUseless) {
  assert(scopeOp != nullptr && occ != nullptr);
  assert(occ->op == scopeOp);
  generateProcessingOrders(occ->childOccs, isUseless);
}

void IRTranslator::generateProcessingOrders(Loop *loopOp, Occurrence *occ,
                                            bool isUseless) {
  assert(loopOp != nullptr && occ != nullptr);
  assert(occ->op == loopOp);
  assert(occ->loopSplitIndex != -1);
  int64_t childNum = static_cast<int64_t>(occ->childOccs.size());
  assert(childNum % 2 == 0);
  assert(childNum == 2 || childNum == 4);
  SmallVector<Occurrence *> firstLoopIteration(
      occ->childOccs.begin(), occ->childOccs.begin() + childNum / 2);
  SmallVector<Occurrence *> secondLoopIteration(
      occ->childOccs.begin() + childNum / 2, occ->childOccs.end());
  generateProcessingOrders(firstLoopIteration, isUseless);
  generateProcessingOrders(secondLoopIteration, true);
  for (auto *occ2 : secondLoopIteration) {
    for (auto *occ1 : llvm::reverse(firstLoopIteration)) {
      generateProcessingOrders(occ1->childOccs, occ2->childOccs, isUseless);
    }
  }
}

void IRTranslator::generateProcessingOrders(RWOperation *rwOp1,
                                            RWOperation *rwOp2,
                                            Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  assert(rwOp1 != nullptr && occ1 != nullptr);
  assert(rwOp2 != nullptr && occ2 != nullptr);
  assert(occ1->op == rwOp1);
  assert(occ2->op == rwOp2);
  ProcessingOrder processingOrder(occ1, occ2, rwOp1, rwOp2, isUseless);
  processingOrders.push_back(processingOrder);
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
    parentOcc->childOccs.push_back(occPtr);
  }

  if (auto *loopOp = dyn_cast<Loop>(op)) {
    for (auto &op : loopOp->body) {
      syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
    }
    occPtr->loopSplitIndex = static_cast<int>(syncIr.size());
    for (auto &op : loopOp->body) {
      syncIrBuilder(op.get(), occPtr, depth + 1, true);
    }
    generateProcessingOrders(loopOp, occPtr, isUseless);
  } else if (auto *scopeOp = dyn_cast<Scope>(op)) {
    for (auto &op : scopeOp->body) {
      syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
    }
    generateProcessingOrders(scopeOp, occPtr, isUseless);
  }

  int endIndex = globalIndex++;
  occPtr->endIndex = endIndex;
  occPtr->syncIrEndIndex = static_cast<int>(syncIr.size());
}
