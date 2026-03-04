//===- InjectBlockSync.cpp ---- Inject Block Sync Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/Transforms/InjectBlockSync.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MoveSyncState.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/RemoveRedundantSync.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
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
    for (auto [i, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (!hacc::utils::isKernelArg(funcOp, i,
                                    hacc::KernelArgType::kWorkspace)) {
        continue;
      }
      auto argUsers = arg.getUsers();
      auto noneAllocWorkSpaceUserIter =
          llvm::find_if(argUsers, [](Operation *user) {
            return !isa<bishengir::memref_ext::AllocWorkspaceOp>(user);
          });
      if (noneAllocWorkSpaceUserIter != argUsers.end()) {
        return noneAllocWorkSpaceUserIter->emitError(
            "All users of workspace arg must be AllocWorkspaceOp!");
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
  auto pipeSAttr =
      hivm::PipeAttr::get(opBuilder.getContext(), hivm::PIPE::PIPE_S);
  return opBuilder.create<OpType>(loc, coreTypeAttr, pipeAttr, pipeSAttr,
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
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
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
    } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      UpdateAllocOpMeminfo(allocOp);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateCallOp(callOp);
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
    } else if (auto aliasPairs = getOperationAliasInfo(op);
               !aliasPairs.empty()) {
      for (auto aliasPair : aliasPairs) {
        UpdateAliasBufferInfo(aliasPair.first, aliasPair.second);
      }
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
  assert(yieldOp->getParentOp() != nullptr);
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOpOperands())) {
    UpdateAliasBufferInfo(yieldOp->getParentOp()->getResult(i), arg.get());
  }
}

std::optional<hivm::PIPE> SyncBlockIRTranslator::getInferredPipe(
    Operation *op, TCoreType coreType,
    const SmallVector<const BaseMemInfo *> &defVec) {
  if (!isa<hivm::CopyOp, hivm::VBrcOp>(op) ||
      coreType == TCoreType::CUBE_OR_VECTOR || defVec.empty()) {
    return {};
  }
  std::optional<hivm::PIPE> pipe;
  for (auto &memInfo : defVec) {
    auto addressSpace = memInfo->scope;
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

void SyncBlockIRTranslator::UpdateDestinationStyleOpInform(
    Operation *op, DestinationStyleOpInterface dstStyleOp) {

  auto coreType = getCoreType(op);
  if (failed(coreType) || coreType.value() == TCoreType::CUBE_OR_VECTOR) {
    return;
  }

  SmallVector<const BaseMemInfo *> defVec;
  UpdateDefUseVec(dstStyleOp.getDpsInits(), defVec);
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(dstStyleOp.getDpsInputs(), useVec);

  auto pipe = hivm::PIPE::PIPE_UNASSIGNED;
  if (isa<hivm::CopyOp, hivm::VBrcOp>(op)) {
    if (auto pipeOpt = getInferredPipe(op, coreType.value(), defVec)) {
      pipe = pipeOpt.value();
    } else {
      pipe = PIPE::PIPE_S;
    }
  } else if (auto mmadL1Op = dyn_cast<hivm::MmadL1Op>(op)) {
    pipe = mmadL1Op.getInPipe();
  } else if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
    if (pipeOp->hasTrait<OpTrait::SinglePipeOpTrait>()) {
      pipe = pipeOp.getPipe();
    }
  }
  if (pipe == hivm::PIPE::PIPE_UNASSIGNED) {
    return;
  }

  auto copPtr = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, dstStyleOp->getName());
  copPtr->elementOp = op;
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

void SyncBlockIRTranslator::UpdateAllocOpMeminfo(memref::AllocOp allocOp) {
  auto allocOpResult = allocOp.getResult();
  if (auto spaceAttr = GetBufferSpaceAttr(allocOpResult)) {
    auto space = spaceAttr.value().getAddressSpace();
    auto newMemInfo = std::make_unique<BaseMemInfo>(
        allocOpResult, allocOpResult, space, SmallVector<uint64_t>(1, 0), 0,
        std::nullopt);
    buffer2MemInfoMap[allocOpResult].emplace_back(std::move(newMemInfo));
  }
}

void InjectBlockSyncAnalysis::InjectAllBlockSync() {
  OpBuilder opBuilder(func_);
  auto insertBlockAll = [this, &opBuilder](Operation *op,
                                           bool insertBefore = false) {
    if (insertBefore) {
      opBuilder.setInsertionPoint(op);
    } else {
      opBuilder.setInsertionPointAfter(op);
    }

    auto flagId1 =
        opBuilder.getIntegerAttr(opBuilder.getI64Type(), blockAllFlagId1);
    auto flagId2 =
        opBuilder.getIntegerAttr(opBuilder.getI64Type(), blockAllFlagId2);
    auto loc = op->getLoc();
    auto pipeAllAttr =
        hivm::PipeAttr::get(opBuilder.getContext(), hivm::PIPE::PIPE_ALL);

    /*
    barrier-all(pipe_all)
    sync_block_wait(cube, pipe_mte2, flagId1)
    sync_block_set(cube, pipe_mte2, flagId2)
    sync_block_set(cube, pipe_mte2, flagId1)
    sync_block_wait(cube, pipe_mte2, flagId2)
    barrier-all(pipe_all)
    */
    opBuilder.create<hivm::PipeBarrierOp>(loc, pipeAllAttr);
    generateCVSyncOp<SyncBlockWaitOp>(opBuilder, loc, TCoreType::CUBE,
                                      hivm::PIPE::PIPE_MTE2, flagId1);
    generateCVSyncOp<SyncBlockSetOp>(opBuilder, loc, TCoreType::CUBE,
                                     hivm::PIPE::PIPE_MTE2, flagId2);
    generateCVSyncOp<SyncBlockSetOp>(opBuilder, loc, TCoreType::VECTOR,
                                     hivm::PIPE::PIPE_MTE2, flagId1);
    generateCVSyncOp<SyncBlockWaitOp>(opBuilder, loc, TCoreType::VECTOR,
                                      hivm::PIPE::PIPE_MTE2, flagId2);
    opBuilder.create<hivm::PipeBarrierOp>(loc, pipeAllAttr);
  };

  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    opBuilder.setInsertionPointAfter(op);
    if (isa<hivm::MmadL1Op, hivm::LoadOp>(op)) {
      // insert block-all before op.
      insertBlockAll(op, /*insertBefore=*/true);
    } else if (isa<hivm::FixpipeOp, hivm::StoreOp>(op)) {
      // insert block-all after op.
      insertBlockAll(op);
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
  if (funcOp->hasAttr(hivm::VectorFunctionAttr::name)) {
    return;
  }
  auto funcCoreType = queryFuncCoreType(funcOp);
  if (!funcCoreType.has_value() ||
      (funcCoreType.value() != TFuncCoreType::MIX)) {
    return;
  }
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (hacc::utils::isMemBasedArch(moduleOp)) {
    // get && set ffts base addr
    auto baseAddr = getFFTSBaseAddrFromFunc(funcOp);
    assert(baseAddr.has_value() &&
           "The mix kernel parameter must have a ffts_addr value");
    insertSetFFTSBaseAddrOp(baseAddr.value());
  }
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
