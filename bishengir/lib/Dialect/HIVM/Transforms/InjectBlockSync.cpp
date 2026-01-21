//===- InjectBlockSync.cpp ---- Inject Block Sync Pass --------------------===//
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
#include "bishengir/Dialect/HIVM/Transforms/InjectBlockSync.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MoveSyncState.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/RemoveRedundantSync.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include <type_traits>

#define DEBUG_TYPE "hivm-inject-block-sync"

namespace mlir {
#define GEN_PASS_DEF_INJECTBLOCKSYNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

/// This pass inject block sync
struct InjectBlockSyncPass
    : public impl::InjectBlockSyncBase<InjectBlockSyncPass> {
public:
  explicit InjectBlockSyncPass(const InjectBlockSyncOptions &options)
      : InjectBlockSyncBase(options) {}

  void runOnOperation() override;

private:
  std::optional<Value> getFFTSBaseAddrFromFunc(func::FuncOp funcOp) {
    auto funcParamSize = funcOp.getNumArguments();
    for (size_t i = 0; i < funcParamSize; i++) {
      if (hacc::utils::isKernelArg(funcOp, i,
                                   hacc::KernelArgType::kFFTSBaseAddr))
        return funcOp.getArgument(i);
    }
    return std::nullopt;
  }

  void insertSetFFTSBaseAddrOp(Value baseAddr) {
    auto funcOp = getOperation();
    OpBuilder opBuilder(funcOp);
    Block *firstBlock = &(funcOp.getBlocks().front());
    assert(firstBlock != nullptr);
    Operation *firstOperation = &(firstBlock->front());
    assert(firstOperation != nullptr);
    opBuilder.setInsertionPoint(firstOperation);
    opBuilder.create<hivm::SetFFTSBaseAddrOp>(firstOperation->getLoc(),
                                              baseAddr);
  }

  LogicalResult checkWorkSpaceValidity() {
    auto funcOp = getOperation();
    auto funcParamSize = funcOp.getNumArguments();
    for (size_t i = 0; i < funcParamSize; i++) {
      if (!hacc::utils::isKernelArg(funcOp, i,
                                    hacc::KernelArgType::kWorkspace)) {
        continue;
      }
      for (Operation *user : funcOp.getArgument(i).getUsers()) {
        auto allocWorkUser =
            dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(user);
        if (!allocWorkUser) {
          user->emitError(
              "All users of workspace arg must be AllocWorkspaceOp!");
          return failure();
        }
      }
    }
    return success();
  }
};

TCoreType InjectBlockSyncAnalysis::convertFuncCoreTypeToCoreType(
    TFuncCoreType funcCoreType) {
  if (funcCoreType == TFuncCoreType::AIC) {
    return TCoreType::CUBE;
  }
  if (funcCoreType == TFuncCoreType::AIV) {
    return TCoreType::VECTOR;
  }
  return TCoreType::CUBE_OR_VECTOR;
}

std::optional<::mlir::hivm::TCoreType>
InjectBlockSyncAnalysis::queryCoreType(Operation *op) {
  auto tCoreTypeAttr =
      op->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (tCoreTypeAttr) {
    return tCoreTypeAttr.getTcoretype();
  }
  auto module = op->getBlock()->getParent()->getParentOfType<ModuleOp>();
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    Operation *dstFunc = module.lookupSymbol(callOp.getCallee());
    auto funcCoreType = queryFuncCoreType(dstFunc);
    if (!funcCoreType.has_value()) {
      return std::nullopt;
    }
    return convertFuncCoreTypeToCoreType(funcCoreType.value());
  }
  return hivm::detail::queryCoreTypeHelper(op);
}

IntegerAttr InjectBlockSyncAnalysis::generateFlagId(OpBuilder opBuilder) {
  return opBuilder.getIntegerAttr(opBuilder.getI64Type(), 0x0f & flagIdCnt++);
}

SyncBlockOp InjectBlockSyncAnalysis::generateSyncBlockOp(OpBuilder opBuilder,
                                                         Location loc,
                                                         IntegerAttr flagId,
                                                         TCoreType coreType) {
  assert(coreType != TCoreType::CUBE_OR_VECTOR);
  auto syncCubeBlockMode = hivm::SyncBlockModeAttr::get(
      opBuilder.getContext(), hivm::SyncBlockMode::ALL_CUBE);
  auto syncVectorBlockMode = hivm::SyncBlockModeAttr::get(
      opBuilder.getContext(), hivm::SyncBlockMode::ALL_VECTOR);
  auto cubePipeAttr =
      hivm::PipeAttr::get(opBuilder.getContext(), hivm::PIPE::PIPE_FIX);
  auto vectorPipeAttr =
      hivm::PipeAttr::get(opBuilder.getContext(), hivm::PIPE::PIPE_MTE3);
  if (coreType == TCoreType::CUBE) {
    return opBuilder.create<hivm::SyncBlockOp>(loc, syncCubeBlockMode, flagId,
                                               Value{}, cubePipeAttr,
                                               hivm::PipeAttr{});
  }
  return opBuilder.create<hivm::SyncBlockOp>(loc, syncVectorBlockMode, flagId,
                                             Value{}, hivm::PipeAttr{},
                                             vectorPipeAttr);
}

template <typename OpType>
OpType InjectBlockSyncAnalysis::generateCVSyncOp(OpBuilder opBuilder,
                                                 Location loc,
                                                 TCoreType coreType, PIPE pipe,
                                                 IntegerAttr flagIdAttr) {
  auto coreTypeAttr =
      hivm::TCoreTypeAttr::get(opBuilder.getContext(), coreType);
  auto pipeAttr = hivm::PipeAttr::get(opBuilder.getContext(), pipe);
  auto mte2PipeAttr =
      hivm::PipeAttr::get(opBuilder.getContext(), hivm::PIPE::PIPE_MTE2);
  return opBuilder.create<OpType>(loc, coreTypeAttr, pipeAttr, mte2PipeAttr,
                                  flagIdAttr);
}

void InjectBlockSyncAnalysis::injectSyncBetweenOp(
    OpBuilder &opBuilder, Operation *op, TCoreType opCoreType,
    SetVector<TCoreType> &userOpCoreTypes) {
  if (userOpCoreTypes.empty()) {
    return;
  }
  if (opCoreType == TCoreType::CUBE_OR_VECTOR ||
      userOpCoreTypes.contains(TCoreType::CUBE_OR_VECTOR)) {
    func_.emitWarning("don't support inject block sync after/before "
                      "unrecognized cube/vector op");
    return;
  }

  auto flagIdForMode0 = generateFlagId(opBuilder);
  auto loc = op->getLoc();
  generateSyncBlockOp(opBuilder, loc, flagIdForMode0, opCoreType);
  if (userOpCoreTypes.size() > 1 || opCoreType != userOpCoreTypes.front()) {
    hivm::PIPE tpipe = opCoreType == TCoreType::CUBE ? hivm::PIPE::PIPE_FIX
                                                     : hivm::PIPE::PIPE_MTE3;
    auto userOpCoreType =
        opCoreType == TCoreType::CUBE ? TCoreType::VECTOR : TCoreType::CUBE;
    auto flagIdForMode2 = generateFlagId(opBuilder);
    generateCVSyncOp<SyncBlockSetOp>(opBuilder, loc, opCoreType, tpipe,
                                     flagIdForMode2);
    generateCVSyncOp<SyncBlockWaitOp>(opBuilder, loc, userOpCoreType, tpipe,
                                      flagIdForMode2);
  }
}

LogicalResult InjectBlockSyncAnalysis::injectShallowBlockSync(Operation *op) {
  func::FuncOp funcOp = func_;
  OpBuilder opBuilder(funcOp);
  opBuilder.setInsertionPointAfter(op);

  auto opCoreType = queryCoreType(op);
  if (!opCoreType.has_value()) {
    return op->emitError("Failed to query core type of this op");
  }
  SmallVector<Operation *, 8> userOps;
  SetVector<TCoreType> userOpCoreTypes;
  getOpUsers(op, userOps);
  for (Operation *userOp : userOps) {
    auto userOpCoreType = queryCoreType(userOp);
    if (!userOpCoreType.has_value()) {
      continue;
    }
    userOpCoreTypes.insert(userOpCoreType.value());
  }
  injectSyncBetweenOp(opBuilder, op, opCoreType.value(), userOpCoreTypes);
  return success();
}

void SyncBlockIRTranslator::SyncBlockBuild() {
  UpdateKernelArgMemInfo();
  Region &funcRegion = func_.getBody();
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion);
}

void SyncBlockIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto aliasPairs = getOperationAliasInfo(op);
    if (!aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      UpdateForOpInfo(forOp);
      std::unique_ptr<InstanceElement> &forEndElement = syncIR.back();
      assert(forEndElement->GetKind() == InstanceElement::KindTy::LOOP);
      auto *forPtr = dyn_cast<LoopInstanceElement>(forEndElement.get());
      assert(forPtr != nullptr);
      auto multibufferAttr =
          op->getAttrOfType<IntegerAttr>(kMultibufferUnrollAttrName);
      if (multibufferAttr) {
        forPtr->ignore_block_sync_move_out = true;
      }
      return WalkResult::skip();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      UpdateWhileOpInfo(whileOp);
      return WalkResult::skip();
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      UpdateIfOpInform(ifOp);
      return WalkResult::skip();
    } else if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(op)) {
      UpdateYieldOpInform(yieldOp);
    } else if (isa<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
      if (failed(UpdateAllocLikeOpMemInfo(op))) {
        return WalkResult::interrupt();
      }
    } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      UpdateTensorExtractOpInform(op, extractOp);
    } else if (auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      UpdateInitAndResAlias(dstStyleOp);
      UpdateDestinationStyleOpInform(op, dstStyleOp);
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      UpdateStoreOrLoadOpInfoBlockSync(loadOp);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateStoreOrLoadOpInfoBlockSync(storeOp);
    } else if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      UpdateStoreOrLoadOpInfoBlockSync(affineLoadOp);
    } else if (auto affineStoreOp = dyn_cast<affine::AffineStoreOp>(op)) {
      UpdateStoreOrLoadOpInfoBlockSync(affineStoreOp);
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("InjectSync Traverse IR Failed! ");
  }
}

void SyncBlockIRTranslator::UpdateInitAndResAlias(
    DestinationStyleOpInterface dstStyleOp) {
  for (auto [i, arg] : llvm::enumerate(dstStyleOp.getDpsInits())) {
    auto tensorType = dyn_cast_or_null<TensorType>(arg.getType());
    if (!tensorType) {
      continue;
    }
    UpdateAliasBufferInfo(dstStyleOp->getResult(i), arg);
  }
}

void SyncBlockIRTranslator::UpdateYieldOpInform(scf::YieldOp yieldOp) {
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOpOperands())) {
    UpdateAliasBufferInfo(yieldOp->getParentOp()->getResult(i), arg.get());
  }
}

void SyncBlockIRTranslator::UpdateDestinationStyleOpInform(
    Operation *op, DestinationStyleOpInterface dstStyleOp) {
  auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op);
  if (!pipeOp) {
    return;
  }
  if (!isa<hivm::FixpipeOp, hivm::StoreOp, hivm::LoadOp>(op)) {
    return;
  }
  hivm::PIPE pipe = pipeOp.getPipe();
  if (pipe == hivm::PIPE::PIPE_UNASSIGNED) {
    return;
  }
  SmallVector<const BaseMemInfo *> defVec;
  UpdateDefUseVec(dstStyleOp.getDpsInits(), defVec);
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(dstStyleOp.getDpsInputs(), useVec);
  assert(static_cast<unsigned int>(pipe) < getPipeNum());
  auto copPtr = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, dstStyleOp->getName());
  copPtr->elementOp = op;
  auto coreType = getCoreType(op);
  assert(succeeded(coreType));
  assert(coreType.value() != TCoreType::CUBE_OR_VECTOR);
  copPtr->compoundCoreType = coreType.value();
  syncIR.emplace_back(std::move(copPtr));
  index++;
}

void SyncBlockIRTranslator::UpdateTensorExtractOpInform(
    Operation *op, tensor::ExtractOp extractOp) {
  auto pipe = hivm::PIPE::PIPE_S;
  auto coreType = getCoreType(op);
  assert(succeeded(coreType));
  if (coreType.value() == TCoreType::CUBE_OR_VECTOR) {
    return;
  }
  auto coreTypeVal = coreType.value();
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec({extractOp.getTensor()}, useVec);
  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, extractOp->getName());
  compoundElement->elementOp = op;
  compoundElement->compoundCoreType = coreTypeVal;
  UpdateAliasBufferInfo(extractOp->getResult(0), extractOp.getTensor());
  syncIR.emplace_back(std::move(compoundElement));
  index++;
}

template <typename OP>
void SyncBlockIRTranslator::UpdateStoreOrLoadOpInfoBlockSync(OP op) {
  Value memRef = op.getMemRef();
  llvm::SmallVector<const BaseMemInfo *> memInfoVec;
  if (buffer2MemInfoMap.contains(memRef)) {
    for (auto &memInfo : buffer2MemInfoMap[memRef]) {
      memInfoVec.push_back(memInfo.get());
    }
  }
  if (memInfoVec.empty()) {
    return;
  }
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  if constexpr (std::is_same_v<OP, memref::LoadOp> ||
                std::is_same_v<OP, affine::AffineLoadOp>) {
    useVec = memInfoVec;
  } else {
    static_assert(std::is_same_v<OP, memref::StoreOp> ||
                  std::is_same_v<OP, affine::AffineStoreOp>);
    defVec = memInfoVec;
  }
  auto pipe = hivm::PIPE::PIPE_S;
  auto coreType = getCoreType(op);
  assert(succeeded(coreType));
  assert(coreType.value() != TCoreType::CUBE_OR_VECTOR);
  auto copPrt = std::make_unique<CompoundInstanceElement>(index, defVec, useVec,
                                                          pipe, op->getName());
  copPrt->elementOp = op.getOperation();
  copPrt->compoundCoreType = coreType.value();
  syncIR.emplace_back(std::move(copPrt));
  index++;
}

void InjectBlockSyncAnalysis::InjectAllBlockSync() {
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    OpBuilder opBuilder(func_);
    opBuilder.setInsertionPointAfter(op);
    auto loc = op->getLoc();
    if (isa<hivm::FixpipeOp>(op)) {
      auto flagIdForMode0 = generateFlagId(opBuilder);
      generateSyncBlockOp(opBuilder, loc, flagIdForMode0, TCoreType::CUBE);
      auto flagIdForMode2 = generateFlagId(opBuilder);
      generateCVSyncOp<SyncBlockSetOp>(opBuilder, loc, TCoreType::CUBE,
                                       hivm::PIPE::PIPE_FIX, flagIdForMode2);
      generateCVSyncOp<SyncBlockWaitOp>(opBuilder, loc, TCoreType::VECTOR,
                                        hivm::PIPE::PIPE_FIX, flagIdForMode2);
    }
    if (isa<hivm::StoreOp>(op)) {
      auto flagIdForMode0 = generateFlagId(opBuilder);
      generateSyncBlockOp(opBuilder, loc, flagIdForMode0, TCoreType::VECTOR);
      auto flagIdForMode2 = generateFlagId(opBuilder);
      generateCVSyncOp<SyncBlockSetOp>(opBuilder, loc, TCoreType::VECTOR,
                                       hivm::PIPE::PIPE_MTE3, flagIdForMode2);
      generateCVSyncOp<SyncBlockWaitOp>(opBuilder, loc, TCoreType::CUBE,
                                        hivm::PIPE::PIPE_MTE3, flagIdForMode2);
    }
  });
}

void InjectBlockSyncAnalysis::InjectBlockMixSync(bool assumeAliveLoops) {
  MemoryDependentAnalyzer memAnalyzer;
  SyncIRs syncIR;
  SyncOperations syncOperations;
  Buffer2MemInfoMap buffer2MemInfoMap;

  SyncBlockIRTranslator trans(syncIR, memAnalyzer, buffer2MemInfoMap, func_,
                              SyncAnalysisMode::BLOCKSYNC);
  trans.SyncBlockBuild();
  LLVM_DEBUG(llvm::dbgs() << "SyncBlockIRTranslator\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  SyncAnalyzer syncAnalyzer(syncIR, memAnalyzer, syncOperations, func_,
                            SyncAnalysisMode::BLOCKSYNC, false,
                            assumeAliveLoops);
  syncAnalyzer.Plan(false /*insertBarAllAtLast*/);
  LLVM_DEBUG(llvm::dbgs() << "SyncAnalyzer\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  MoveSyncState syncMove(syncIR, syncOperations);
  syncMove.StateOptimize();
  LLVM_DEBUG(llvm::dbgs() << "MoveSyncState\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  RemoveRedundantSync removeRedundantSync(syncIR, syncOperations,
                                          SyncAnalysisMode::BLOCKSYNC);
  removeRedundantSync.Plan();
  LLVM_DEBUG(llvm::dbgs() << "RemoveRedundantSync\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  SyncEventIdAllocation eventIdAllocation(syncIR, syncOperations);
  eventIdAllocation.Allocate();
  LLVM_DEBUG(llvm::dbgs() << "SyncEventIdAllocation\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  SyncCodegen syncCodegen(syncIR, func_, SyncAnalysisMode::BLOCKSYNC);
  syncCodegen.Build();
  LLVM_DEBUG(llvm::dbgs() << "SyncCodegen\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());
}

void InjectBlockSyncAnalysis::InjectBlockShallowSync() {
  func_.walk([&](Operation *op) {
    auto visitStatus = success();
    if (auto matmulOp = dyn_cast<hivm::MatmulOp>(op)) {
      visitStatus = injectShallowBlockSync(matmulOp);
    } else if (auto mixmatmulOp = dyn_cast<hivm::MixMatmulOp>(op)) {
      visitStatus = injectShallowBlockSync(mixmatmulOp);
    }

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      visitStatus = injectShallowBlockSync(callOp);
    }

    if (failed(visitStatus)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

void InjectBlockSyncPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (hacc::utils::isHost(funcOp)) {
    return;
  }
  auto funcCoreType = queryFuncCoreType(funcOp);
  if (!funcCoreType.has_value() ||
      (funcCoreType.value() != TFuncCoreType::MIX)) {
    return;
  }

  // get && set ffts base addr
  std::optional<Value> baseAddr = getFFTSBaseAddrFromFunc(funcOp);
  assert(baseAddr.has_value() &&
         "The mix kernel parameter must have a ffts_addr value");
  insertSetFFTSBaseAddrOp(baseAddr.value());
  // If module has attribute to disable auto inject block sync, skip injection.
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (moduleOp && moduleOp->hasAttr(hivm::DisableAutoInjectBlockSyncAttr::name)) {
    return;
  }
  if (this->disableAutoInjectBlockSync)
    return;

  InjectBlockSyncAnalysis injectBlockSyncAnalysis(funcOp);
  // TODO:
  //  refactor to implement block sync without distinguish
  //  between shallowcv and mix cv.
  auto fusionKind = mlir::hfusion::tryGetFusionKind(funcOp);
  if (this->blockAllSync) {
    injectBlockSyncAnalysis.InjectAllBlockSync();
  } else if (fusionKind.has_value() &&
             fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV) {
    injectBlockSyncAnalysis.InjectBlockShallowSync();
  } else {
    if (failed(checkWorkSpaceValidity())) {
      return signalPassFailure();
    }
    injectBlockSyncAnalysis.InjectBlockMixSync(assumeAliveLoops);
  }
}

std::unique_ptr<Pass>
mlir::hivm::createInjectBlockSyncPass(const InjectBlockSyncOptions &options) {
  return std::make_unique<InjectBlockSyncPass>(options);
}
