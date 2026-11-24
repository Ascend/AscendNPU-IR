//===- PlanMemory.cpp ----Plan Buffer Memory Address ----------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/PlanMemory.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/AllocToPointerCast.h"
#include "bishengir/Dialect/HIVM/Transforms/MemoryDisplay.h"
#include "bishengir/Dialect/HIVM/Transforms/TcbPoolPlan.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExtImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#define DEBUG_TYPE "hivm-plan-memory"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X)

namespace mlir {
#define GEN_PASS_DEF_PLANMEMORY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace util;

namespace {

bool hasExplicitCubeOps(func::FuncOp func) {
  return func
      .walk([](Operation *op) {
        return isa<MmadL0Op, L12L0Op>(op) ? WalkResult::interrupt()
                                          : WalkResult::advance();
      })
      .wasInterrupted();
}

// Select buffers requiring explicit cube validation and independent storage.
// The pass selects the policy once per function, before any planning retries.
bool requiresExplicitCubePlanning(std::optional<AddressSpaceAttr> space,
                                  bool enableMmadL0Planning) {
  if (!enableMmadL0Planning || !space)
    return false;
  switch (space->getAddressSpace()) {
  case AddressSpace::L0A:
  case AddressSpace::L0B:
  case AddressSpace::BiasBUF:
  case AddressSpace::L0C:
    return true;
  default:
    return false;
  }
}

// TODO: Move these A5-only requirements into a shared target-specific
// abstraction when extending standalone MMAD L0 support to A6 and A2/A3.
// Planner sizes are in bits, while pointer_cast addresses are in bytes.
constexpr int kA5L0ABAlignBits = 512 * 8;
constexpr int kA5L0CAlignBits = 1024 * 8;
constexpr int kA5BTAlignBits = 64 * 8;
// HACC has no BT capacity spec yet; this matches CATLASS Arch::AtlasA5.
constexpr int kA5BTSizeBits = 4096 * 8;

LogicalResult validateExplicitCubeAllocs(func::FuncOp func, ModuleOp module) {
  // The caller has already enabled MMAD L0 planning and checked the target.
  auto result = func.walk([&](memref::AllocOp alloc) {
    auto space = GetBufferSpaceAttr(alloc.getResult());
    if (!requiresExplicitCubePlanning(space, /*enableMmadL0Planning=*/true))
      return WalkResult::advance();

    auto type = alloc.getType();
    if (!type.hasStaticShape() || !type.getLayout().isIdentity()) {
      alloc.emitOpError("explicit A5 cube allocations require static identity "
                        "layouts; express offsets through views");
      return WalkResult::interrupt();
    }
    if (!type.getElementType().isIntOrFloat() ||
        type.getElementTypeBitWidth() % utils::kBitsToByte != 0) {
      alloc.emitOpError("explicit A5 cube allocations require byte-addressable "
                        "integer or floating-point elements");
      return WalkResult::interrupt();
    }

    int64_t capacity = kA5BTSizeBits;
    int64_t alignment = kA5BTAlignBits;
    if (space->getAddressSpace() != AddressSpace::BiasBUF) {
      auto spec = hacc::utils::getNPUTargetSpec(module);
      if (!spec) {
        alloc.emitOpError("explicit A5 cube allocation requires device specs; "
                          "run hacc-append-device-spec first");
        return WalkResult::interrupt();
      }
      auto identifier = space->getAddressSpace() == AddressSpace::L0A
                            ? hacc::DeviceSpec::L0A_SIZE
                        : space->getAddressSpace() == AddressSpace::L0B
                            ? hacc::DeviceSpec::L0B_SIZE
                            : hacc::DeviceSpec::L0C_SIZE;
      auto size = spec->getSpecForIdentifierEnum(identifier);
      if (!size || !isa<IntegerAttr>(size.getValue()) ||
          cast<IntegerAttr>(size.getValue()).getInt() <= 0) {
        alloc.emitOpError("explicit A5 cube allocation requires a positive "
                          "device capacity spec");
        return WalkResult::interrupt();
      }
      capacity = cast<IntegerAttr>(size.getValue()).getInt();
      alignment = space->getAddressSpace() == AddressSpace::L0C
                      ? kA5L0CAlignBits
                      : kA5L0ABAlignBits;
    }
    // Do not silently discard an alloc alignment stronger than this planner's
    // space alignment. Views do not change the backing allocation's alignment.
    if (auto requested = alloc.getAlignment()) {
      if (*requested != 0 &&
          (alignment / utils::kBitsToByte) % *requested != 0) {
        alloc.emitOpError("unsupported explicit A5 cube allocation alignment");
        return WalkResult::interrupt();
      }
    }
    // Divide before multiplying dimensions so malformed shapes cannot overflow
    // getStaticTotalSize or constBits during liveness analysis.
    int64_t remainingElements = capacity / type.getElementTypeBitWidth();
    if (remainingElements == 0) {
      alloc.emitOpError(
          "explicit A5 cube element exceeds memory-space capacity");
      return WalkResult::interrupt();
    }
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || dim > remainingElements) {
        alloc.emitOpError("explicit A5 cube allocation must have positive "
                          "dimensions and fit the memory-space capacity");
        return WalkResult::interrupt();
      }
      remainingElements /= dim;
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

bool isCVModule(ModuleOp op) {
  bool hasAIV = false;
  bool hasAIC = false;
  op->walk([&](func::FuncOp funcOp) {
    if (queryFuncCoreType(funcOp) == TFuncCoreType::AIV) {
      hasAIV = true;
    } else if (queryFuncCoreType(funcOp) == TFuncCoreType::AIC) {
      hasAIC = true;
    }
  });
  return hasAIV && hasAIC;
}

bool isReusableCastOp(hivm::VCastOp &castOp, Value output, Value input) {
  auto rank = dyn_cast<MemRefType>(output.getType()).getRank();
  if (rank > 1 || !isLastDimContiguous(output) || !isLastDimContiguous(input)) {
    // can only reuse 1d cast library
    return false;
  }
  return true;
}

bool isReusableVSelOp(hivm::VSelOp &selOp, const Value &genBuffer,
                      const Value &killBuffer) {
  auto condType = getElementTypeOrSelf(selOp.getSrc()[0].getType());
  auto srcType = getElementTypeOrSelf(selOp.getSrc()[1].getType());
  if (srcType.isInteger(64) && condType.isInteger(1)) {
    Value condOperand = utils::tracebackMemRef(selOp.getSrc()[0]);
    return killBuffer != condOperand;
  }
  return true;
}

bool isReusableNarrowWidth(Operation *op, Value output, Value input) {
  if (auto castOp = dyn_cast_if_present<hivm::VCastOp>(op)) {
    return isReusableCastOp(castOp, output, input);
  }
  return true;
}

static std::optional<int64_t> getStaticOffset(Value v) {
  auto memrefTy = dyn_cast<MemRefType>(v.getType());
  if (!memrefTy)
    return std::nullopt;
  if (auto strided = dyn_cast<StridedLayoutAttr>(memrefTy.getLayout()))
    return strided.getOffset();
  return 0;
}

// Memory can reuse by offset if the offset of the output and input are the
// same.
static bool isReusableByOffset(HIVMStructuredOp &hivmOp) {
  auto output = hivmOp.getDpsInits().front();
  auto outputOffset = getStaticOffset(output);
  if (!outputOffset) {
    return false;
  }
  for (Value in : hivmOp.getDpsInputs()) {
    auto inOffset = getStaticOffset(in);
    // if the input is not a memref, skip
    if (!inOffset)
      continue;
    if (*inOffset != *outputOffset) {
      return false;
    }
  }
  return true;
}

/// Memory can inplace with rules as follows:
///   Scene1: Output has same width with input.
///   Scene2: Output width is smaller than input except VCastOp.
///   Scene3: VCastOp output width is smaller than input, and rank equal to 1
///          with the last dim contiguous. If the last dim is not contiguous, it
///          may be lifted to rank2.
bool isReusableOperands(Operation *op, HIVMStructuredOp &hivmOp) {
  auto output = hivmOp.getDpsInits().front();
  auto outputBitWidth =
      getElementTypeOrSelf(output.getType()).getIntOrFloatBitWidth();
  for (auto input : hivmOp.getDpsInputs()) {
    auto inputBitWidth =
        getElementTypeOrSelf(input.getType()).getIntOrFloatBitWidth();
    if (inputBitWidth == outputBitWidth)
      return true;
    if (inputBitWidth % outputBitWidth == 0)
      return isReusableNarrowWidth(op, output, input);
  }
  return false;
}

inline bool hasInlineBroadcastOrTransposeAttr(Operation *op) {
  auto hivmOp = cast<hivm::HIVMStructuredOp>(op);
  return hivmOp.existInlineBroadcastLoopDims() ||
         hivmOp.existInlineTransposeLoopDims();
}

/// ExtraBuffer can inplace with only one extra buffer generated for scalar or
/// OTF Brc. Otherwise, extra buffer may be generated by compound instructions.
inline bool isReusableExtraBuffer(Operation *op) {
  auto extraBufferOp = dyn_cast_if_present<ExtraBufferOpInterface>(op);
  if (!extraBufferOp || extraBufferOp.getExtraBuffers().empty()) {
    return true;
  }
  if (extraBufferOp.getExtraBuffers().size() > 1) {
    return false;
  }
  return extraBufferOp.shouldAllocExtraBufferForScalarOrOTFBrc();
}

bool isStaticShapeSame(Operation *op, const Value &genBuffer,
                       const Value &killBuffer) {
  Value genOperand = genBuffer;
  Value killOperand = killBuffer;
  for (auto operand : op->getOperands()) {
    if (genBuffer == utils::tracebackMemRef(operand)) {
      genOperand = operand;
    }
    if (killBuffer == utils::tracebackMemRef(operand)) {
      killOperand = operand;
    }
  }
  auto genOperandShapes = cast<ShapedType>(genOperand.getType()).getShape();
  auto killOperandShapes = cast<ShapedType>(killOperand.getType()).getShape();
  for (auto [genShape, killShape] :
       llvm::zip(genOperandShapes, killOperandShapes)) {
    if (genShape != killShape)
      return false;
  }
  return true;
}

/// Get allocOp and CVMixId from markOp. If failed, it will return std::nullopt.
std::optional<memref::AllocOp>
getCVMixIdAndAllocOpFromMarkOp(annotation::MarkOp markOp, int32_t &cvMixId) {
  if (!markOp->hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name)) {
    return std::nullopt;
  }
  // get cvMixId
  auto attr = dyn_cast<hivm::HIVMTightlyCoupledBufferAttr>(
      markOp->getAttr(hivm::HIVMTightlyCoupledBufferAttr::name));
  if (!attr || !attr.getId().has_value()) {
    LDBG("MarkOp for : " << markOp.getSrc());
    LDBG("  Parse tightly_coupled_buffer attribute failed. No input or input "
         "fromat error. Correct format should be: hivm.tightly_coupled_buffer"
         " = #hivm.tightly_coupled_buffer<$id>\n");
    return std::nullopt;
  }
  // get allocOp
  if (!isa<MemRefType>(markOp.getSrc().getType())) {
    return std::nullopt;
  }
  auto maybeAlloc = utils::tracebackMemRefToAlloc(markOp.getSrc());
  if (!maybeAlloc.has_value()) {
    return std::nullopt;
  }
  cvMixId = attr.getId().value();
  return maybeAlloc;
}

} // namespace

void MemLivenessAnalysis::build() {
  Region &funcRegion = func_.getBody();
  Liveness live(func_);
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion, live);
  // Extend preload buffer lifetime from scope to parent for.
  PreprocessPreloadBuffersBufferLife();
  // the lifetime of the buffer.
  GenerateBufferLife();
  InitializeInplacePairList();
}

bool MemLivenessAnalysis::isLocalMemPlan() const {
  return planMode == MemPlanMode::LOCAL_MEM_PLAN;
}

bool MemLivenessAnalysis::isGlobalWorkSpaceMemPlan() const {
  return planMode == MemPlanMode::GLOBAL_WORKSPACE_PLAN;
}

void MemLivenessAnalysis::RecursionIR(Region *region, const Liveness &live) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // recursive control flow
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      RecursiveIfOp(ifOp, live);
      return WalkResult::skip();
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      RecursiveForOp(forOp, live);
      return WalkResult::skip();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      RecursiveWhileOp(whileOp, live);
      return WalkResult::skip();
    } else if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      RecursiveScopeOp(scopeOp, live);
      return WalkResult::skip();
    }

    // process operation
    auto *curOpInfo = UpdateLinearOperation(op);
    auto aliasPairs = getOperationAliasInfo(op);
    if (!aliasPairs.empty() && !isa<arith::SelectOp>(op)) {
      for (auto aliasPair : aliasPairs) {
        UpdateBufferAlias(aliasPair.first, aliasPair.second);
      }
    } else if (isLocalMemPlan() && dyn_cast<memref::AllocOp>(op)) {
      if (failed(CheckLocalBufferAllocOp(op))) {
        return WalkResult::interrupt();
      }
      UpdateOpBufferInfo(op, op->getResults());
      return WalkResult::advance();
    } else if (isGlobalWorkSpaceMemPlan() &&
               dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
      UpdateOpBufferInfo(op, op->getResults());
      return WalkResult::advance();
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateStoreOpInfo(curOpInfo, storeOp.getMemRef(), live);
    } else if (auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      // Process the operation of hivm instructions as follows:
      // hivm.hir.copy ins(%0 : memref<16xf16, #hivm.address_space<gm>>)
      //              outs(%1 : memref<16xxf16, #hivm.address_space<ub>>)
      // need to handle kill buffer.
      UpdateInitAndResAlias(dstStyleOp);
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(dstStyleOp.getDpsInits()));
      UpdateOpTempGenInfo(curOpInfo);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      UpdateBufferAlias(selectOp.getResult(), selectOp.getTrueValue(),
                        /*hasCond=*/true, /*isIgnoreInplace=*/true);
      UpdateBufferAlias(selectOp.getResult(), selectOp.getFalseValue(),
                        /*hasCond=*/true, /*isIgnoreInplace=*/true);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto markOp = dyn_cast<annotation::MarkOp>(op)) {
      ProcessMarkOp(markOp, curOpInfo, live);
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(op)) {
      UpdateBranchOpAlias(condBrOp.getTrueDest(),
                          condBrOp.getTrueDestOperands());
      UpdateBranchOpAlias(condBrOp.getFalseDest(),
                          condBrOp.getFalseDestOperands());
    } else if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
      UpdateBranchOpAlias(brOp.getDest(), brOp.getDestOperands());
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(callOp->getOperands()));
      UpdateOpTempGenInfo(curOpInfo);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto gpuLaunchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(gpuLaunchOp->getOperands()));
      UpdateOpTempGenInfo(curOpInfo);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto debugOp = dyn_cast<hivm::DebugOp>(op)) {
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (failed(CheckIfUnknownOpTouchBuffer(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm::report_fatal_error("PlanMemory Traverse IR Failed! ");
  }
}

void MemLivenessAnalysis::UpdateInitAndResAlias(
    DestinationStyleOpInterface dstStyleOp) {
  auto results = dstStyleOp->getResults();
  if (results.empty()) {
    return;
  }
  for (auto [res, init] :
       llvm::zip(dstStyleOp->getResults(), dstStyleOp.getDpsInits())) {
    auto *iter = buffer2AliasVec.find(init);
    if (iter == buffer2AliasVec.end()) {
      continue;
    }
    auto tensorType = dyn_cast_or_null<TensorType>(res.getType());
    if (tensorType) {
      UpdateBufferAlias(res, init);
    }
  }
}

OpInfo *MemLivenessAnalysis::UpdateLinearOperation(Operation *op) {
  auto opInfo = std::make_unique<OpInfo>(op, seqIndex++);
  auto *curOpInfo = opInfo.get();
  linearOperation.push_back(std::move(opInfo));
  return curOpInfo;
}

void MemLivenessAnalysis::UpdateForOpBufferAlias(scf::ForOp forOp) {
  if (forOp.getResults().empty()) {
    return;
  }
  if (!forOp.getRegionIterArgs().empty()) {
    assert(forOp.getYieldedValues().size() == forOp.getRegionIterArgs().size());
    assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      // yielded values alias region iter args.
      UpdateBufferAlias(forOp.getYieldedValues()[i], arg);
    }
  }
  assert(forOp->getResults().size() == forOp.getYieldedValues().size());
  for (auto [i, arg] : llvm::enumerate(forOp.getYieldedValues())) {
    // forOp result values alias region iter yielded values.
    UpdateBufferAlias(forOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::RecursiveForOp(scf::ForOp forOp,
                                         const Liveness &live) {
  // Process the operation of ForOp as follows:
  // alloca %allocA
  // %0 = scf.for %arg4 = %c0 to %c1024 step %c128 iter_args(%arg5 = %4)->
  //      (memref<16x16x16xf16, #hivm.address_space<ub>>):
  //          def(allocA)
  //          ...
  //          scf.yield %alloc0 : memref<16xf16,#hivm.address_space<ub>>
  // need to handle kill buffer.
  auto *forBeginSeq = UpdateLinearOperation(forOp.getOperation());
  UpdateOpGenInfo(forBeginSeq, GetLiveBuffersInLoop(forOp, live));
  UpdateForOpInitArgsAlias(forOp);
  UpdateForOpBufferAlias(forOp);
  RecursionIR(&forOp.getRegion(), live);

  // Select-init loop-carried values use hasCond aliases, so
  // InitializeInplacePairList will not mark the yielded dest ignoreInplace.
  // Extra inplace of that dest onto a sibling loop-local src is still unsafe:
  // the next iteration reads the dest as the iter_arg.
  for (auto [init, yield] :
       llvm::zip(forOp.getInitArgs(), forOp.getYieldedValues())) {
    if (!init.getDefiningOp<arith::SelectOp>())
      continue;
    auto maybeAlloc = utils::tracebackMemRefToAlloc(yield);
    if (!maybeAlloc.has_value())
      continue;
    auto *it = bufferInfos.find(maybeAlloc->getResult());
    if (it != bufferInfos.end())
      it->second.ignoreInplace = true;
  }
  auto *forEndSeq = UpdateLinearOperation(forOp.getOperation());
  OpKillHandle(forEndSeq, live, forOp->getBlock());
}

void MemLivenessAnalysis::UpdateWhileOpInitArgsAlias(scf::WhileOp whileOp) {
  if (whileOp.getInits().empty()) {
    return;
  }
  assert(whileOp.getInits().size() == whileOp.getRegionIterArgs().size());
  for (auto [iterArg, init] :
       llvm::zip(whileOp.getRegionIterArgs(), whileOp.getInits())) {
    // init args alias region iter args.
    UpdateBufferAlias(iterArg, init);
  }
}

void MemLivenessAnalysis::UpdateWhileOpBufferAlias(scf::WhileOp whileOp) {
  if (whileOp.getResults().empty()) {
    return;
  }
  if (!whileOp.getRegionIterArgs().empty()) {
    assert(whileOp.getYieldedValues().size() ==
           whileOp.getRegionIterArgs().size());
    assert(whileOp.getInits().size() == whileOp.getRegionIterArgs().size());
    for (auto [yieldVal, iterArg] :
         llvm::zip(whileOp.getYieldedValues(), whileOp.getRegionIterArgs())) {
      // yielded values alias region iter args.
      UpdateBufferAlias(yieldVal, iterArg);
    }
  }
  assert(whileOp->getResults().size() == whileOp.getAfterArguments().size());
  for (auto [i, arg] : llvm::enumerate(whileOp.getAfterArguments())) {
    // whileOp result values alias after-region args.
    UpdateBufferAlias(whileOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::UpdateConditionOpBufferAlias(
    scf::ConditionOp conditionOp) {
  auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp());
  if (!whileOp || whileOp.getAfterArguments().empty()) {
    return;
  }
  assert(whileOp.getAfterArguments().size() == conditionOp.getArgs().size());
  for (auto [afterArg, condArg] :
       llvm::zip(whileOp.getAfterArguments(), conditionOp.getArgs())) {
    // whileOp after arguments alias condition operands.
    UpdateBufferAlias(afterArg, condArg);
  }
}

void MemLivenessAnalysis::RecursiveWhileOp(scf::WhileOp whileOp,
                                           const Liveness &live) {
  // Process the operation of WhileOp as follows:
  // alloca %allocA
  // %0 = scf.while (%arg0 = %init)
  //          %1 = arith.cmpi eq, %arg1, %const
  //          scf.condition(%1) %shared
  //      do (%shared)
  //          def(allocA)
  //          ...
  //          scf.yield %allocA
  // need to handle kill buffer.
  auto *whileBeginSeq = UpdateLinearOperation(whileOp.getOperation());
  UpdateOpGenInfo(whileBeginSeq, GetLiveBuffersInLoop(whileOp, live));
  UpdateWhileOpInitArgsAlias(whileOp);
  UpdateWhileOpBufferAlias(whileOp);
  UpdateConditionOpBufferAlias(whileOp.getConditionOp());
  RecursionIR(&whileOp.getBefore(), live);
  RecursionIR(&whileOp.getAfter(), live);
  auto *whileEndSeq = UpdateLinearOperation(whileOp.getOperation());
  OpKillHandle(whileEndSeq, live, whileOp->getBlock());
}

void MemLivenessAnalysis::UpdateForOpInitArgsAlias(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty()) {
    return;
  }
  assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
  for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
    // init args alias region iter args.
    UpdateBufferAlias(forOp.getRegionIterArgs()[i], arg);
  }
}

void MemLivenessAnalysis::UpdateIfOpBufferAlias(scf::IfOp ifOp,
                                                scf::YieldOp yieldOp) {
  if (ifOp.getResults().empty()) {
    return;
  }
  assert(ifOp->getResults().size() == yieldOp->getOperands().size());
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOperands())) {
    // Multiple buffers involved, requiring one-to-one correspondence.
    UpdateBufferAlias(ifOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::RecursiveIfOp(scf::IfOp ifOp, const Liveness &live) {
  // Process the operation of IfOp as follows:
  // %0 = scf.if %cond -> (memref<16xf16, #hivm.address_space<ub>>)
  //        scf.yield %alloc0: memref<16xf16, #hivm.address_space<ub>>
  //      else:
  //        scf.yield %alloc1 : memref<16xf16, #hivm.address_space<ub>>
  (void)UpdateLinearOperation(ifOp.getOperation());
  UpdateIfOpBufferAlias(ifOp, ifOp.thenYield());
  if (ifOp.elseBlock()) {
    UpdateIfOpBufferAlias(ifOp, ifOp.elseYield());
  }
  RecursionIR(&ifOp.getThenRegion(), live);
  auto *curIfElse = UpdateLinearOperation(ifOp.getOperation());

  auto *curIfEnd = curIfElse;
  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion(), live);
    curIfEnd = UpdateLinearOperation(ifOp.getOperation());
  }
  OpKillHandle(curIfEnd, live, ifOp->getBlock());
}

void MemLivenessAnalysis::UpdateBranchOpAlias(Block *brBlock,
                                              OperandRange destOperands) {
  if (destOperands.empty()) {
    return;
  }
  assert(brBlock->getArguments().size() == destOperands.size());
  for (auto [destArg, destOperand] :
       llvm::zip(brBlock->getArguments(), destOperands)) {
    // BranchOp dest operands alias dest block arguments.
    UpdateBufferAlias(destArg, destOperand, /*hasCond=*/true);
  }
}

void MemLivenessAnalysis::UpdateScopeOpBufferAlias(scope::ScopeOp scopeOp,
                                                   scope::ReturnOp returnOp) {
  if (scopeOp.getResults().empty()) {
    return;
  }
  for (auto [res, arg] :
       llvm::zip_equal(scopeOp->getResults(), returnOp->getOperands())) {
    // Multiple buffers involved, requiring one-to-one correspondence.
    UpdateBufferAlias(res, arg);
  }
}

void MemLivenessAnalysis::RecursiveScopeOp(scope::ScopeOp scopeOp,
                                           const Liveness &live) {
  (void)UpdateLinearOperation(scopeOp.getOperation());
  auto &scopeRegion = scopeOp.getRegion();
  auto returnOp = cast<scope::ReturnOp>(scopeRegion.front().getTerminator());
  UpdateScopeOpBufferAlias(scopeOp, returnOp);
  RecursionIR(&scopeRegion, live);
  auto *scopeEndSeq = UpdateLinearOperation(scopeOp.getOperation());
  OpKillHandle(scopeEndSeq, live, scopeOp->getBlock());
}

SmallVector<Value>
MemLivenessAnalysis::GetLiveBuffersInLoop(LoopLikeOpInterface loopOp,
                                          const Liveness &live) {
  SmallVector<Value> allocBeforeLoopBuffers;
  const auto *liveBlockInfo = live.getLiveness(loopOp->getBlock());
  assert(liveBlockInfo != nullptr);

  SetVector<Value> currentLiveValues =
      currentlyLiveValuesOrdered(liveBlockInfo, loopOp.getOperation());
  if (currentLiveValues.empty()) {
    return allocBeforeLoopBuffers;
  }

  // TODO: Remove once plan-memory no longer depends on traversal order.
  auto currentLiveValuesShuffled =
      getShuffledRange(currentLiveValues.takeVector());
  for (const Value &operand : currentLiveValuesShuffled) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto buffer : aliasBuffers) {
      auto iter = buffer2status.find(buffer);
      if (iter != buffer2status.end())
        allocBeforeLoopBuffers.push_back(buffer);
    }
  }
  return allocBeforeLoopBuffers;
}

void MemLivenessAnalysis::UpdateMultiBufferInfo(annotation::MarkOp markOp,
                                                Value memrefVal) {
  if (!markOp->hasAttr(hivm::MultiBufferAttr::name)) {
    return;
  }
  auto multiBufferValAttr = markOp->getAttr(hivm::MultiBufferAttr::name);
  assert(isa<IntegerAttr>(multiBufferValAttr) &&
         "multi buffer value must be integer!");
  auto valAttr = cast<IntegerAttr>(multiBufferValAttr);
  if (valAttr.getInt() == 1) {
    // Num is 1, which is a singebuffer and does not require any processing.
    return;
  }
  buffer2MultiNum[memrefVal] = static_cast<uint64_t>(valAttr.getInt());
}

LogicalResult
MemLivenessAnalysis::CheckLocalBufferAllocOp(Operation *op) const {
  auto allocOp = dyn_cast<memref::AllocOp>(op);
  assert(allocOp && "must be alloc op");
  auto memorySpaceAttr = GetBufferSpaceAttr(allocOp.getResult());

  // TODO: Remove this after plan mem support SSBUF alloc
  assert(!isSsbuffer(memorySpaceAttr) &&
         "Alloc on SSBUFFER is not yet supported!");
  if (isLocalBufferForMemoryPlanning(memorySpaceAttr, enableMmadL0Planning)) {
    return success();
  }
  allocOp.getOperation()->emitError("Alloc buffer not at UB space! ");
  return failure();
}

bool MemLivenessAnalysis::isSkippableOp(Operation *op) const {
  // TODO: Remove this after plan mem support SSBUF alloc
  if (llvm::isa<PointerCastOp>(op) && isOpTouchSsbufferOnly(op)) {
    return true;
  }

  return isa<func::ReturnOp, scf::YieldOp, scf::ConditionOp, memref::DimOp,
             hivm::DCCIOp, scope::ReturnOp>(op);
}

LogicalResult
MemLivenessAnalysis::CheckIfUnknownOpTouchBuffer(Operation *op) const {
  if (isSkippableOp(op) || isGlobalWorkSpaceMemPlan()) {
    // This scene can be ignored.
    return success();
  }
  if (llvm::any_of(getOpTouchBuffer(op), [&](Value value) {
        return isLocalBufferForMemoryPlanning(GetBufferSpaceAttr(value),
                                              enableMmadL0Planning);
      })) {
    op->emitError("PlanMemory Fail : Unrecognized type of Operation touches "
                  "local buffer!");
    return failure();
  }
  return success();
}

void MemLivenessAnalysis::UpdateExtraBufferIgnoreInplace(
    const ValueRange &results) {
  // mark the temp buffer as ignoring inplace
  for (Value operand : results) {
    auto *it = bufferInfos.find(operand);
    // The global planner tracks workspace, not the operation's UB scratch.
    if (it == bufferInfos.end() && isGlobalWorkSpaceMemPlan())
      continue;
    assert(it != bufferInfos.end());
    it->second.ignoreInplace = true;
  }
}

void MemLivenessAnalysis::UpdateOpTempGenInfo(OpInfo *opInfo) {
  // Handld Temp buffer Scenarios:
  SmallVector<Value, 1> resultVec;
  if (auto extraBufferOp =
          dyn_cast<ExtraBufferOpInterface>(opInfo->operation)) {
    auto extraBuffers = extraBufferOp.getExtraBuffers();
    resultVec.insert(resultVec.end(), extraBuffers.begin(), extraBuffers.end());
    UpdateExtraBufferIgnoreInplace(resultVec);
    UpdateOpGenInfo(opInfo, resultVec);
  }
}

void MemLivenessAnalysis::UpdateBuffer2AliasVec(
    Value buffer, Value aliasBuffer,
    const SetVector<Value> &aliasBuffersOfBuffer,
    const SetVector<Value> &aliasBuffersOfAliasBuffer, bool hasCond) {
  for (auto aliasBufferOfBuffer : aliasBuffersOfBuffer) {
    auto bufferCondPair = FindBufferCondPair(buffer, aliasBufferOfBuffer);
    auto bufferHasCond = bufferCondPair ? (*bufferCondPair)->second : false;
    for (auto aliasBufferOfAliasBuffer : aliasBuffersOfAliasBuffer) {
      auto aliasBufferCondPair =
          FindBufferCondPair(aliasBuffer, aliasBufferOfAliasBuffer);
      auto aliasBufferHasCond =
          aliasBufferCondPair ? (*aliasBufferCondPair)->second : false;
      bool resultCond = hasCond || bufferHasCond || aliasBufferHasCond;
      auto newBufferPair =
          FindBufferCondPair(aliasBufferOfBuffer, aliasBufferOfAliasBuffer);
      if (newBufferPair) {
        (*newBufferPair)->second = (*newBufferPair)->second || resultCond;
      } else {
        buffer2AliasVec[aliasBufferOfBuffer].push_back(
            std::make_pair(aliasBufferOfAliasBuffer, resultCond));
      }
    }
  }
}

void MemLivenessAnalysis::UpdateBufferAlias(Value buffer, Value aliasBuffer,
                                            bool hasCond,
                                            bool isIgnoreInplace) {
  SetVector<Value> buffers = GetAliasBuffers(buffer);
  SetVector<Value> aliasBuffers = GetAliasBuffers(aliasBuffer);
  buffers.insert(buffer);
  aliasBuffers.insert(aliasBuffer);

  // update alias map info for each buffer
  // e.g. if A alias B, C alias D, now update:
  // A alias B,C,D; B alias A,C,D; and update C,D's condition
  UpdateBuffer2AliasVec(buffer, aliasBuffer, buffers, aliasBuffers, hasCond);

  // e.g. if A alias B, C alias D, now update:
  // C alias A,B,D; D alias A,B,C; and update A,B's condition
  UpdateBuffer2AliasVec(aliasBuffer, buffer, aliasBuffers, buffers, hasCond);

  // AllocOp is DEFFINED, not AllocOp is UNDEFFINED.
  // The buffer of UNDEFFINED is only used as an alias buffer to
  // participate in life interval judgment.
  if (!memref_ext::isDefiningOpAllocLike(buffer)) {
    buffer2status[buffer] = BufferStatus::UNDEFFINED;
  }

  // mark the alias buffer as ignoring Inplace if it is not generated by
  // memref.alloc.
  auto *it = bufferInfos.find(aliasBuffer);
  if (isIgnoreInplace && it != bufferInfos.end()) {
    it->second.ignoreInplace = true;
  }
}

std::optional<BufferCondPair *>
MemLivenessAnalysis::FindBufferCondPair(Value buffer, Value aliasValue) {
  for (BufferCondPair bufferCondPair : buffer2AliasVec[buffer]) {
    if (bufferCondPair.first == aliasValue) {
      return &bufferCondPair;
    }
  }
  return std::nullopt;
}

SmallVector<BufferCondPair>
MemLivenessAnalysis::GetAliasBufferCondPairs(Value aliasBuffer) const {
  auto *trueVar = buffer2AliasVec.find(aliasBuffer);
  if (trueVar != buffer2AliasVec.end()) {
    return trueVar->second;
  }
  return {};
}

SetVector<Value> MemLivenessAnalysis::GetAliasBuffers(Value aliasBuffer) const {
  SetVector<Value> aliasBuffers;
  auto aliasBufferPairVec = GetAliasBufferCondPairs(aliasBuffer);
  for (auto aliasBufferPair : aliasBufferPairVec) {
    aliasBuffers.insert(aliasBufferPair.first);
  }
  return aliasBuffers;
}

void MemLivenessAnalysis::UpdateStoreOpInfo(OpInfo *opInfo,
                                            const Value storeValue,
                                            const Liveness &live) {
  // The src of memref store may also serve as a gen buffer.
  SmallVector<Value, 1> storeValues;
  storeValues.push_back(storeValue);
  UpdateOpGenInfo(opInfo, storeValues);
  // Collect kill buffers corresponding to operation.
  OpKillHandle(opInfo, live, opInfo->operation->getBlock());
}

void MemLivenessAnalysis::UpdateOpBufferInfo(Operation *op,
                                             const ValueRange &results) {
  for (const Value &operand : results) {
    auto it = buffer2status.find(operand);
    if (it != buffer2status.end()) {
      continue;
    }
    bufferInfos[operand] = GenerateBufferInfo(op, operand);
    buffer2status[operand] = BufferStatus::DEFFINED;
  }
}

void MemLivenessAnalysis::UpdateOpGenInfo(OpInfo *opInfo,
                                          const ValueRange &results) {
  if (results.empty()) {
    return;
  }
  for (Value operand : results) {
    if (skipMemPlan.find(operand) != skipMemPlan.end()) {
      continue;
    }
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto buffer : aliasBuffers) {
      UpdateOperandGenInfo(opInfo, buffer);
    }
  }
}

void MemLivenessAnalysis::UpdateOperandGenInfo(OpInfo *opInfo, Value operand) {
  auto iterBuffer = buffer2status.find(operand);
  if (iterBuffer == buffer2status.end())
    return;
  if (iterBuffer->second == BufferStatus::DEFFINED) {
    if (IsPreloadBuffer(operand))
      preloadBufferReuseableInfo[operand].genInfo = opInfo;
    genKillMap[opInfo].gen.push_back(operand);
    buffer2status[iterBuffer->first] = BufferStatus::GENED;
  } else if (iterBuffer->second == BufferStatus::KILLED) {
    llvm::report_fatal_error(
        "The buffer memory has been released and cannot be used again! ");
  }
}
bool MemLivenessAnalysis::IsPreloadBuffer(Value operand) const {
  auto aliasBuffers = GetAliasBuffers(operand);
  aliasBuffers.insert(operand);
  return llvm::any_of(aliasBuffers, [&](Value buffer) {
    return preloadBuffers.contains(buffer);
  });
}

void MemLivenessAnalysis::UpdatePreloadBuffers(annotation::MarkOp markOp,
                                               memref::AllocOp allocOp) {
  // Find marked buffer used in multi scope operations.
  auto attr = markOp->getAttr(hivm::PreloadLocalBufferAttr::name);
  if (!attr) {
    return;
  }
  auto loopOp = markOp->getParentOfType<LoopLikeOpInterface>();
  if (!loopOp) {
    loopOp = allocOp->getParentOfType<LoopLikeOpInterface>();
  }
  if (!loopOp) {
    llvm::report_fatal_error(
        "preload local buffer must be inside a loop-like op");
  }

  auto allocBuffer = allocOp.getResult();
  preloadBuffers.insert(allocBuffer);
  preloadLoop2Buffers[loopOp.getOperation()].insert(allocBuffer);
}

void MemLivenessAnalysis::ProcessMarkOp(annotation::MarkOp markOp,
                                        OpInfo *curOpInfo,
                                        const Liveness &live) {
  if (!isa<MemRefType>(markOp.getSrc().getType())) {
    return;
  }
  // global workspace plan
  if (!isLocalMemPlan()) {
    UpdateMultiBufferInfo(markOp, markOp.getSrc());
    return;
  }
  // get allocOp
  auto maybeAlloc = utils::tracebackMemRefToAlloc(markOp.getSrc());
  if (!maybeAlloc.has_value()) {
    return;
  }
  UpdateMultiBufferInfo(markOp, maybeAlloc.value().getResult());
  UpdateBufferPoolInfo(markOp, maybeAlloc.value());
  UpdatePreloadBuffers(markOp, maybeAlloc.value());
  UpdateMemoryUniqueBufferInfo(markOp, maybeAlloc.value());
  if (ProcessMarkOpForTightlyCoupledCV(markOp, maybeAlloc.value())) {
    UpdateOpGenInfo(curOpInfo, maybeAlloc.value()->getResults());
  }
  // TODO: Update buffer kill time when RecursionIR visits any user of the
  // buffer, rather than only getting buffer kill time in last user
  OpKillHandle(curOpInfo, live, markOp->getBlock());
}

void MemLivenessAnalysis::UpdateBufferPoolInfo(annotation::MarkOp markOp,
                                               memref::AllocOp allocOp) {
  if (!markOp.isAnnotatedBy(hivm::MemPoolSizeAttr::name))
    return;
  auto *bufferInfoIt = bufferInfos.find(allocOp.getResult());
  if (bufferInfoIt == bufferInfos.end()) {
    llvm::report_fatal_error("Use allocOp before defined!");
  }
  auto poolIdAttr =
      markOp->getAttrOfType<IntegerAttr>(hivm::MemPoolIdAttr::name);
  if (!poolIdAttr) {
    llvm::report_fatal_error(
        "hivm.mempool_id and hivm.mempool_size must be specified together");
  }
  bufferInfoIt->second.bufferPoolId = static_cast<int32_t>(poolIdAttr.getInt());
  // Explicitly set each pool slot member's own memoryUnique flag to true.
  bufferInfoIt->second.memoryUnique = true;
}

void MemLivenessAnalysis::UpdateMemoryUniqueBufferInfo(
    annotation::MarkOp markOp, memref::AllocOp allocOp) {
  if (!markOp->hasAttr(hivm::HIVMMemoryUniqueAttr::getMnemonic())) {
    return;
  }
  auto allocValue = allocOp.getResult();
  auto *it = bufferInfos.find(allocValue);
  if (it == bufferInfos.end()) {
    llvm::report_fatal_error("Use allocOp before defined!");
  }
  it->second.memoryUnique = true;
}

bool MemLivenessAnalysis::ProcessMarkOpForTightlyCoupledCV(
    annotation::MarkOp markOp, memref::AllocOp allocOp) {
  // Check CV mix id
  if (!markOp->hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name)) {
    return false;
  }
  auto attr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
      hivm::HIVMTightlyCoupledBufferAttr::name);
  if (!attr.getId().has_value()) {
    LDBG("MarkOp for : " << markOp.getSrc());
    LDBG(" Parse tightly_coupled_buffer attribute failed. No input or input "
         "fromat error. Correct format should be: hivm.tightly_coupled_buffer"
         " = #hivm.tightly_coupled_buffer<$id>\n");
    return false;
  }
  auto allocValue = allocOp.getResult();
  auto *it = bufferInfos.find(allocValue);
  if (it == bufferInfos.end()) {
    llvm::report_fatal_error("Use allocOp before defined!");
  }
  it->second.cvMixId = attr.getId().value();
  auto addressSpace = getHIVMAddressSpace(allocValue.getType());
  auto funcType = queryFuncCoreType(func_);
  if (disableTightlyCoupledBufferReuse && funcType == TFuncCoreType::MIX) {
    it->second.memoryUnique = true;
  }

  if ((funcType == TFuncCoreType::AIC && addressSpace == AddressSpace::UB) ||
      (funcType == TFuncCoreType::AIV && addressSpace == AddressSpace::L1)) {
    skipMemPlan.insert(allocValue);
    LDBG(allocValue << " Skip mem plan\n");
  }
  if ((funcType == TFuncCoreType::AIV && addressSpace == AddressSpace::UB) ||
      (funcType == TFuncCoreType::AIC && addressSpace == AddressSpace::L1)) {
    if (disableTightlyCoupledBufferReuse) {
      it->second.memoryUnique = true;
      LDBG(allocValue << "set memory unique\n");
    }
    // AllocOp will be writed in another scope and read in current scope, so we
    // will update its genInfo in markOp.
    LDBG(allocValue << " Manually run UpdateOpGenInfo\n");
    return true;
  }
  return false;
}

void MemLivenessAnalysis::UpdatePreloadBuffersGenInfo(OpInfo *opInfo) {
  for (auto preloadBuffer : preloadLoop2Buffers.at(opInfo->operation)) {
    auto aliasBuffers = GetAliasBuffers(preloadBuffer);
    aliasBuffers.insert(preloadBuffer);
    for (auto buffer : aliasBuffers) {
      auto iterBuffer = buffer2status.find(buffer);
      if (iterBuffer == buffer2status.end())
        continue;
      if (iterBuffer->second == BufferStatus::DEFFINED) {
        genKillMap[opInfo].gen.push_back(buffer);
        buffer2status[iterBuffer->first] = BufferStatus::GENED;
      }
    }
  }
}

void MemLivenessAnalysis::UpdatePreloadBuffersKillInfo(OpInfo *opInfo) {
  for (auto preloadBuffer : preloadLoop2Buffers.at(opInfo->operation)) {
    auto aliasBuffers = GetAliasBuffers(preloadBuffer);
    aliasBuffers.insert(preloadBuffer);
    for (auto buffer : aliasBuffers) {
      auto iterBuffer = buffer2status.find(buffer);
      if (iterBuffer == buffer2status.end())
        continue;
      if (iterBuffer->second == BufferStatus::GENED) {
        genKillMap[opInfo].kill.push_back(buffer);
        buffer2status[iterBuffer->first] = BufferStatus::KILLED;
      }
    }
  }
}

/// Example IR for preload buffer reuse:
/// 1:  func.func {
/// 2:    scf.for {
/// 3:      %alloc = memref.alloc()
/// 4:      annotation.mark %alloc {hivm.preload_local_buffer}
/// 5:      scope.scope : () -> () {
/// 6:        ops
/// 7:        hivm.hir.load outs(%alloc)
/// 8:        ops
/// 9:      } {preload_num = 1}
/// 10:     scope.scope : () -> () {
/// 11:       %alloc_0 = memref.alloc()
/// 12:       func.call @vf_reuse_direct(%alloc, %alloc_0)
/// 13:       read %alloc_0
/// 14:       scope.return
/// 15:     } {preload_num = 0}
/// 16:   }
/// 17:   return
/// 18: }
///
/// reuseableBufferLifeBeforeGen:
/// 1. allocTime: scope start index(line 6)
/// 2. freeTime: scope end index(line 7)
/// 3. genScopePreloadNum: preload num:1
/// reuseableBufferLifeAfterKill:
/// 1. allocTime: scope start index(line 12)
/// 2. freeTime: scope end index(line 14)
/// 3. genScopePreloadNum: preload num:0
void MemLivenessAnalysis::GeneratePreloadBufferReuseableInfo() {
  int scopeStart = -1;
  SmallVector<Value> killBuffers;
  int64_t preloadNum = -1;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto *opInfo = linearOperation[i].get();
    auto parentScope = opInfo->operation->getParentOfType<scope::ScopeOp>();
    // Independent Load scopes do not carry a compute preload stage. Keep
    // extending their buffer lifetimes without deriving scope reuse windows.
    if (parentScope && parentScope->hasAttr(hivm::LoadPreloadAttr::name))
      parentScope = nullptr;
    // Get scope start index(line 6) and preload num. Scope start will be
    // reuseableBufferLifeBeforeGen's allocTime.
    if (parentScope && opInfo->operation == &parentScope.getBody()->front()) {
      auto preloadNumAttr = parentScope->getAttr(hivm::PreloadNumAttr::name);
      assert(llvm::isa_and_nonnull<IntegerAttr>(preloadNumAttr) &&
             "preload_num must be set and its value must be an integer");
      preloadNum = cast<IntegerAttr>(preloadNumAttr).getInt();
      scopeStart = opInfo->index;
    }
    // Save scope end index(line 14) to reuseableBufferLifeAfterKill's freeTime.
    if (llvm::isa_and_nonnull<scope::ReturnOp>(opInfo->operation)) {
      for (auto killBuffer : killBuffers) {
        preloadBufferReuseableInfo[killBuffer]
            .reuseableBufferLifeAfterKill->freeTime = opInfo->index;
      }
      killBuffers.clear();
    }

    auto it = genKillMap.find(opInfo);
    if (it == genKillMap.end()) {
      continue;
    }

    auto &gen = it->second.gen;
    for (auto *iter = gen.begin(); iter != gen.end();) {
      auto buffer = *iter;
      auto infoIt = preloadBufferReuseableInfo.find(buffer);
      if (infoIt == preloadBufferReuseableInfo.end()) {
        ++iter;
        continue;
      }
      // Unique buffers still need loop-wide liveness. Omit their reuse windows.
      auto &info = infoIt->second;
      if (parentScope && info.genInfo == opInfo &&
          !bufferInfos.find(buffer)->second.memoryUnique) {
        std::unique_ptr<BufferLife> bufferLife =
            std::make_unique<BufferLife>(buffer);
        bufferLife->allocTime = scopeStart;
        bufferLife->freeTime = opInfo->index;
        info.reuseableBufferLifeBeforeGen = std::move(bufferLife);
        info.genScopePreloadNum = preloadNum;
      }
      // Reset buffer2status because it will be used when extending preload
      // buffer's lifetime.
      buffer2status[buffer] = BufferStatus::DEFFINED;
      // Remove preload buffer's genInfo from genKillMap because preload
      // buffer's lifetime will be extended to fill whole parent forOp.
      iter = gen.erase(iter);
    }

    auto &kill = it->second.kill;
    for (auto *iter = kill.begin(); iter != kill.end();) {
      auto buffer = *iter;
      auto infoIt = preloadBufferReuseableInfo.find(buffer);
      if (infoIt == preloadBufferReuseableInfo.end()) {
        iter++;
        continue;
      }
      auto &info = infoIt->second;
      if (parentScope && info.killInfo == opInfo &&
          !bufferInfos.find(buffer)->second.memoryUnique) {
        std::unique_ptr<BufferLife> bufferLife =
            std::make_unique<BufferLife>(buffer);
        bufferLife->allocTime = opInfo->index;
        killBuffers.push_back(buffer);
        info.reuseableBufferLifeAfterKill = std::move(bufferLife);
        info.killScopePreloadNum = preloadNum;
      }
      // Remove preload buffer's killInfo from genKillMap because preload
      // buffer's lifetime will be extended to fill whole parent forOp.
      iter = kill.erase(iter);
    }
  }
  DumpPreloadBufferReuseableInfo();
}

void MemLivenessAnalysis::DumpPreloadBufferReuseableInfo() const {
  LDBG("\nPrint preload buffer reuse info\n");
  for (auto [preload, info] : preloadBufferReuseableInfo) {
    LDBG("\npreload buffer " << preload << "\n");
    if (info.reuseableBufferLifeBeforeGen) {
      LDBG("BufferLifeBeforeGen: gen time:"
           << info.reuseableBufferLifeBeforeGen->allocTime << " kill time: "
           << info.reuseableBufferLifeBeforeGen->freeTime << "\n");
    }
    if (info.reuseableBufferLifeAfterKill) {
      LDBG("BufferLifeAfterKill: gen time:"
           << info.reuseableBufferLifeAfterKill->allocTime << " kill time: "
           << info.reuseableBufferLifeAfterKill->freeTime << "\n");
    }
  }
}

void MemLivenessAnalysis::PreprocessPreloadBuffersBufferLife() {
  if (preloadLoop2Buffers.empty()) {
    return;
  }
  GeneratePreloadBufferReuseableInfo();
  DenseMap<Operation *, unsigned> loopVisitCount;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto *opInfo = linearOperation[i].get();
    assert(opInfo && "linearOperation should not be null.");
    auto loopIt = preloadLoop2Buffers.find(opInfo->operation);
    if (loopIt == preloadLoop2Buffers.end()) {
      continue;
    }
    unsigned &count = loopVisitCount[opInfo->operation];
    if (count == 0) {
      UpdatePreloadBuffersGenInfo(opInfo);
    } else if (count == 1) {
      UpdatePreloadBuffersKillInfo(opInfo);
    }
    count++;
  }
}

void MemLivenessAnalysis::OpKillHandle(OpInfo *opInfo, const Liveness &live,
                                       Block *block) {
  const auto *liveBlockInfo = live.getLiveness(block);
  assert(liveBlockInfo != nullptr && opInfo != nullptr);

  SetVector<Value> currentLiveValues =
      currentlyLiveValuesOrdered(liveBlockInfo, opInfo->operation);
  if (currentLiveValues.empty()) {
    return;
  }

  // TODO: Remove once plan-memory no longer depends on traversal order.
  auto currentLiveValuesShuffled =
      getShuffledRange(currentLiveValues.takeVector());
  for (const Value &operand : currentLiveValuesShuffled) {
    UpdateOpKillInfo(opInfo, operand, live);
  }
}

void MemLivenessAnalysis::UpdateOpKillInfo(OpInfo *opInfo, Value operand,
                                           const Liveness &live) {
  auto aliasBuffers = GetAliasBuffers(operand);
  aliasBuffers.insert(operand);
  if (!AllDeadAfter(opInfo->operation, aliasBuffers, live)) {
    return;
  }
  if (llvm::any_of(aliasBuffers, [&](Value aliasBuffer) {
        return buffer2status.contains(aliasBuffer) &&
               buffer2status[aliasBuffer] == BufferStatus::GENED &&
               !isParentOpDominate(aliasBuffer.getDefiningOp(),
                                   opInfo->operation);
      })) {
    return;
  }
  for (Value aliasBuffer : aliasBuffers) {
    auto iterBuffer = buffer2status.find(aliasBuffer);
    if (iterBuffer == buffer2status.end())
      return;
    if (iterBuffer->second == BufferStatus::GENED) {
      if (IsPreloadBuffer(aliasBuffer)) {
        preloadBufferReuseableInfo[aliasBuffer].killInfo = opInfo;
      }
      genKillMap[opInfo].kill.push_back(aliasBuffer);
      buffer2status[aliasBuffer] = BufferStatus::KILLED;
    }
  }
}

bool MemLivenessAnalysis::isParentOpDominate(Operation *op1,
                                             Operation *op2) const {
  assert((op1 != nullptr && op2 != nullptr) && "op must not be nullptr");
  Operation *op1Parent = op1->getParentOp();
  Operation *op2Parent = op2->getParentOp();
  assert(op2Parent != nullptr && op1Parent != nullptr && "must have parent op");
  // enable op kill preload buffer defined out of scopeOp
  if (IsPreloadBuffer(op1->getResult(0)) && isa<scope::ScopeOp>(op2Parent) &&
      op2Parent->hasAttr(hivm::PreloadNumAttr::name)) {
    op2Parent = op2Parent->getParentOp();
  }
  return op2Parent->isAncestor(op1Parent);
}

bool MemLivenessAnalysis::AllDeadAfter(Operation *op, SetVector<Value> aliasVec,
                                       const Liveness &live) const {
  return llvm::all_of(aliasVec, [&](Value aliasBuffer) {
    // isDeadAfter will only check liveOut and last use in op->getBlock(). It
    // will not check ops in other blocks. And if op->getParentOp() don't have
    // opResult, the liveOut set will be empty. So we need to check all users of
    // aliasBuffer to ensure that they are not after op in memory view.
    return live.isDeadAfter(aliasBuffer, op) &&
           llvm::all_of(aliasBuffer.getUsers(), [&](Operation *user) {
             // user is before or equal to (in same op but different block) op.
             return !utils::isBeforeInMemoryView(op, user);
           });
  });
}

BufferInfo MemLivenessAnalysis::GenerateBufferInfo(Operation *op,
                                                   Value operand) {
  auto memorySpaceAttr = GetBufferSpaceAttr(operand);
  // TODO: Remove this after plan mem support SSBUF alloc
  if (isSsbuffer(memorySpaceAttr)) {
    llvm_unreachable("Alloc-like on SSBUF is not yet supported!");
  }

  if (isLocalMemPlan() &&
      isLocalBufferForMemoryPlanning(memorySpaceAttr, enableMmadL0Planning)) {
    assert(memorySpaceAttr.has_value() && "buffer must has space!");
    return GetBufferInfo(op, operand,
                         memorySpaceAttr.value().getAddressSpace());
  } else if (isGlobalWorkSpaceMemPlan() &&
             isa<bishengir::memref_ext::AllocWorkspaceOp>(
                 operand.getDefiningOp())) {
    return GetBufferInfo(op, operand, hivm::AddressSpace::GM);
  }
  llvm::report_fatal_error("buffer must has BufferInfo !");
}

BufferInfo MemLivenessAnalysis::GetBufferInfo(Operation *op, Value operand,
                                              hivm::AddressSpace bufferScope) {
  BufferInfo bufferInfo;
  bufferInfo.operation = op;
  bufferInfo.bufferScope = bufferScope;
  // Explicit slots may be in flight on different pipes after their last SSA
  // use. Preserve their identity instead of inferring safe asynchronous reuse
  // from SSA liveness. Views of a single allocation still share its storage.
  bufferInfo.memoryUnique = requiresExplicitCubePlanning(
      GetBufferSpaceAttr(operand), enableMmadL0Planning);
  // get buffer size, now for static shape
  Value traceValue = utils::tracebackMemRef(operand);
  auto memRefType = cast<MemRefType>(traceValue.getType());
  bufferInfo.bufferType = memRefType.getElementType();
  std::optional<int64_t> totalStaticSize =
      utils::getStaticTotalSize(memRefType.getShape());
  if (!totalStaticSize.has_value()) {
    llvm::report_fatal_error(
        "Failed to obtain op buffer shape size which should be static.");
  }
  bufferInfo.constBits =
      totalStaticSize.value() *
      static_cast<int64_t>(memRefType.getElementTypeBitWidth());
  return bufferInfo;
}

void MemLivenessAnalysis::InitializeInplacePairList() {
  LDBG("-------------------------- AliasPair --------------------------\n");
  for (auto &[buffer, bufferInfo] : bufferInfos) {
    assert(memref_ext::isDefiningOpAllocLike(buffer));
    auto *iter = buffer2AliasVec.find(buffer);
    if (iter == buffer2AliasVec.end()) {
      continue;
    }
    for (auto &aliasBufferPair : iter->second) {
      auto &aliasBuffer = aliasBufferPair.first;
      if (!memref_ext::isDefiningOpAllocLike(aliasBuffer)) {
        continue;
      }
      if (count(inplacePairList.begin(), inplacePairList.end(),
                std::make_pair(buffer, aliasBuffer)) ||
          count(inplacePairList.begin(), inplacePairList.end(),
                std::make_pair(aliasBuffer, buffer))) {
        continue;
      }
      LDBG("inplace pair: (buffer: "
           << buffer << ", alias buffer: " << aliasBuffer << ")"
           << " , condition: " << aliasBufferPair.second << "\n");
      if (aliasBufferPair.second) {
        continue;
      }
      auto *it = bufferInfos.find(aliasBuffer);
      assert(it != bufferInfos.end() && "buffer Info need define before!");
      auto &aliasBufferInfo = it->second;
      if (aliasBufferInfo.memoryUnique || bufferInfo.memoryUnique) {
        continue;
      }
      inplacePairList.emplace_back(std::make_pair(buffer, aliasBuffer));
      aliasBufferInfo.ignoreInplace = true;
      bufferInfo.ignoreInplace = true;
    }
  }
}

std::optional<BankBufferKinds>
MemLivenessAnalysis::getBankBufferKinds(Value root) {
  auto infoIt = bufferInfos.find(root);
  if (infoIt == bufferInfos.end() ||
      infoIt->second.bufferScope != hivm::AddressSpace::UB ||
      skipMemPlan.contains(root) || !buffer2Life.contains(root))
    return std::nullopt;
  // TODO: Support bank planning without changing memory-unique priority.
  if (infoIt->second.memoryUnique) {
    LDBG("[bank-copy] skip memory-unique buffer: " << root << "\n");
    return std::nullopt;
  }
  BankBufferKinds kinds;
  auto multiIt = buffer2MultiNum.find(root);
  if (multiIt != buffer2MultiNum.end() && multiIt->second > 1)
    kinds.insert(BankBufferKind::Multi);
  if (infoIt->second.cvMixId >= 0)
    kinds.insert(BankBufferKind::TCB);
  if (IsPreloadBuffer(root))
    kinds.insert(BankBufferKind::Preload);
  return kinds;
}

/// Collect static, contiguous 1D UB-to-UB copy information.
void MemLivenessAnalysis::collectUBCopyBankCopies(bool preserveTcbRoots) {
  bankCopies.clear();
  bankTcbRoots.clear();
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::OFF ||
      !isLocalMemPlan()) {
    LDBG("[bank-copy] skip collection: mode="
         << getUBCopyBankModeName(planMemoryUBCopyBankMode)
         << " localMemPlan=" << isLocalMemPlan() << "\n");
    return;
  }
  if (preserveTcbRoots)
    for (auto &[buffer, info] : bufferInfos)
      if (info.cvMixId >= 0)
        bankTcbRoots.insert(buffer);

  // `linearOperation` can visit one operation more than once when control flow
  // is involved. Keep IR order while recording each copy once.
  DenseSet<Operation *> visitedCopies;
  unsigned inspectedCopies = 0;
  for (const std::unique_ptr<OpInfo> &opInfo : linearOperation) {
    auto copyOp = dyn_cast_or_null<hivm::CopyOp>(opInfo->operation);
    if (!copyOp || !visitedCopies.insert(copyOp.getOperation()).second)
      continue;
    ++inspectedCopies;

    auto srcScope =
        hivm::getOptionalHIVMAddressSpace(copyOp.getSrcOperandType());
    auto dstScope =
        hivm::getOptionalHIVMAddressSpace(copyOp.getDstOperandType());
    if (srcScope != hivm::AddressSpace::UB ||
        dstScope != hivm::AddressSpace::UB) {
      LDBG("[bank-copy] skip non-UB: " << copyOp << "\n");
      continue;
    }

    if (!copyOp.hasPureBufferSemantics() || copyOp.getPadMode() ||
        copyOp.getPadValue() || copyOp.getCollapseReassociation()) {
      LDBG("[bank-copy] skip non-pure/padded: " << copyOp << "\n");
      continue;
    }

    auto srcType = dyn_cast<MemRefType>(copyOp.getSrcOperandType());
    auto dstType = dyn_cast<MemRefType>(copyOp.getDstOperandType());
    if (!srcType || !dstType || srcType.getRank() != 1 ||
        dstType.getRank() != 1 || !srcType.hasStaticShape() ||
        !dstType.hasStaticShape() || !utils::hasStaticUnitStride(srcType) ||
        !utils::hasStaticUnitStride(dstType)) {
      LDBG("[bank-copy] skip non-1D-static-unit-stride: " << copyOp << "\n");
      continue;
    }

    if (srcType.getElementType().isInteger(1)) {
      LDBG("[bank-copy] skip i1: " << copyOp << "\n");
      continue;
    }

    int64_t copyBits;
    if (llvm::MulOverflow(
            srcType.getNumElements(),
            static_cast<int64_t>(srcType.getElementTypeBitWidth()), copyBits) ||
        copyBits < utils::kUBCopyBankSizeInBits ||
        copyBits % utils::kUBAlignSizeInBits) {
      LDBG("[bank-copy] skip invalid size: " << copyOp << "\n");
      continue;
    }

    auto srcAccess = utils::traceStaticMemRefAccess(copyOp.getSrc());
    auto dstAccess = utils::traceStaticMemRefAccess(copyOp.getDst());
    if (!srcAccess || !dstAccess) {
      LDBG("[bank-copy] skip untraceable access: " << copyOp << "\n");
      continue;
    }

    auto srcKinds = getBankBufferKinds(srcAccess->first);
    auto dstKinds = getBankBufferKinds(dstAccess->first);
    if (!srcKinds || !dstKinds) {
      LDBG("[bank-copy] skip invalid buffer kind: " << copyOp << "\n");
      continue;
    }

    if (srcAccess->second % utils::kUBAlignSizeInBits != 0 ||
        dstAccess->second % utils::kUBAlignSizeInBits != 0) {
      LDBG("[bank-copy] skip unaligned bias: " << copyOp << "\n");
      continue;
    }

    int64_t dstBits;
    if (llvm::MulOverflow(
            dstType.getNumElements(),
            static_cast<int64_t>(dstType.getElementTypeBitWidth()), dstBits) ||
        copyBits != dstBits) {
      LDBG("[bank-copy] skip size mismatch: " << copyOp << "\n");
      continue;
    }

    LDBG("[bank-copy] collect #"
         << bankCopies.size() << ": src " << srcAccess->first << " + "
         << srcAccess->second << " bits -> dst " << dstAccess->first << " + "
         << dstAccess->second << " bits: " << copyOp << "\n");
    bankCopies.push_back({copyOp, srcAccess->first, dstAccess->first,
                          srcAccess->second, dstAccess->second, copyBits,
                          *srcKinds, *dstKinds});
  }

  LDBG("[bank-copy] inspected "
       << inspectedCopies << " copy op(s), collected " << bankCopies.size()
       << ", skipped " << (inspectedCopies - bankCopies.size()) << "\n");
}

void MemLivenessAnalysis::GenerateBufferLife() {
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto it = genKillMap.find(linearOperation[i].get());
    if (it == genKillMap.end()) {
      continue;
    }
    // Time given to buffer start.
    for (const Value &genBuffer : it->second.gen) {
      std::unique_ptr<BufferLife> bufferLife =
          std::make_unique<BufferLife>(genBuffer);
      bufferLife->allocTime = it->first->index;
      buffer2Life[genBuffer] = std::move(bufferLife);
    }
    // Time given to buffer end.
    for (const Value &killBuffer : it->second.kill) {
      auto iter = buffer2Life.find(killBuffer);
      assert(iter != buffer2Life.end() &&
             "buffer has not been generated before! ");
      iter->second->freeTime = it->first->index;
    }
  }
}

//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/Liveness.cpp
//===----------------------------------------------------------------------===//
// Similar to Liveness::currentlyLiveValues but using SetVector instead of
// ValueSetT=SmallPtrSet. Needed to make the pass output deterministic.
mlir::SetVector<Value> MemLivenessAnalysis::currentlyLiveValuesOrdered(
    const LivenessBlockInfo *livenessInfo, Operation *op) const {
  mlir::SetVector<Value> liveSet;

  // Given a value, check which ops are within its live range. For each of
  // those ops, add the value to the set of live values as-of that op.
  auto addValueToCurrentlyLiveSets = [&](Value value) {
    // Determine the live range of this value inside this block.
    Operation *startOfLiveRange = value.getDefiningOp();
    Operation *endOfLiveRange = nullptr;
    // If it's a live in or a block argument, then the start is the beginning
    // of the block.
    if (livenessInfo->isLiveIn(value) || isa<BlockArgument>(value))
      startOfLiveRange = &livenessInfo->getBlock()->front();
    else
      startOfLiveRange =
          livenessInfo->getBlock()->findAncestorOpInBlock(*startOfLiveRange);

    // If it's a live out, then the end is the back of the block.
    if (livenessInfo->isLiveOut(value))
      endOfLiveRange = &livenessInfo->getBlock()->back();

    // We must have at least a startOfLiveRange at this point. Given this, we
    // can use the existing getEndOperation to find the end of the live range.
    if (startOfLiveRange && !endOfLiveRange)
      endOfLiveRange = livenessInfo->getEndOperation(value, startOfLiveRange);

    assert(endOfLiveRange && "Must have endOfLiveRange at this point!");
    // If this op is within the live range, insert the value into the set.
    if (!(op->isBeforeInBlock(startOfLiveRange) ||
          endOfLiveRange->isBeforeInBlock(op)))
      liveSet.insert(value);
  };

  // Handle block arguments if any.
  for (Value arg : livenessInfo->getBlock()->getArguments())
    addValueToCurrentlyLiveSets(arg);

  // Handle live-ins. Between the live ins and all the op results that gives us
  // every value in the block.
  for (Value in : livenessInfo->in())
    addValueToCurrentlyLiveSets(in);

  // Now walk the block and handle all values used in the block and values
  // defined by the block.
  for (Operation &walkOp :
       llvm::make_range(livenessInfo->getBlock()->begin(), ++op->getIterator()))
    for (auto result : walkOp.getResults())
      addValueToCurrentlyLiveSets(result);

  return liveSet;
}

std::optional<BufferLifeView>
StorageEntry::GetBufferLifeByValue(Value v) const {
  return bufferLifeVec.find(v);
}

bool MemPlan::IsInplaceMultiBuffer(
    Value src, const SmallVector<ValuePair> &inplaceList) const {
  // Check if src or its paired buffer in inplaceList has multi-buffer attribute
  // (buffer2MultiNum > 1)
  auto srcMultiBufferIter = buffer2MultiNum.find(src);
  if (srcMultiBufferIter != buffer2MultiNum.end() &&
      srcMultiBufferIter->second > 1) {
    return true;
  }
  for (const auto &pair : inplaceList) {
    Value pairedBuffer;
    if (pair.first == src) {
      pairedBuffer = pair.second;
    } else if (pair.second == src) {
      pairedBuffer = pair.first;
    } else {
      continue;
    }
    auto multiBufferIter = buffer2MultiNum.find(pairedBuffer);
    if (multiBufferIter != buffer2MultiNum.end() &&
        multiBufferIter->second > 1) {
      return true;
    }
  }
  return false;
}

template <typename DstOpType>
bool MemPlan::VisitDstOpTypeReachable(
    Value src, DenseSet<Value> &visited,
    const SmallVector<ValuePair> &inplaceList) const {
  if (visited.contains(src)) {
    return false;
  }
  visited.insert(src);

  auto srcBuffer = utils::tracebackMemRefToAlloc(src);
  if (!srcBuffer.has_value()) {
    LDBG("inplace-reuse src allocOp is not found: " << src << "\n");
    return false;
  }
  for (Operation *user : src.getUsers()) {
    if (isa<DstOpType>(user)) {
      // successfully reach `DstOpType`
      LDBG("inplace-reuse reachable to op: " << *user << "\n");
      return true;
    }

    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(user)) {
      // recursively visit ViewLikeOp result users
      return VisitDstOpTypeReachable<DstOpType>(
          viewLikeOp.getOperation()->getResult(0), visited, inplaceList);
    }
  }
  // check alias and inplace buffers
  for (auto pair : inplaceList) {
    if (pair.first == pair.second) {
      continue;
    }
    if (pair.first == srcBuffer.value()) {
      if (VisitDstOpTypeReachable<DstOpType>(pair.second, visited,
                                             inplaceList)) {
        return true;
      }
    }
    if (pair.second == srcBuffer.value()) {
      if (VisitDstOpTypeReachable<DstOpType>(pair.first, visited,
                                             inplaceList)) {
        return true;
      }
    }
  }

  // not reachable for all paths
  LDBG("not inplace-reuse reachable from: "
       << src << ", to: " << DstOpType::getOperationName() << "\n");
  return false;
}

/// Determines whether the value `src` is reachable to an operand of a
/// `DstOpType` operation.
///
/// Recursively traces the use chain of `src` through:
/// 1. ViewLikeOpInterface operations (e.g. `memref.subview`)
/// 2. alias buffers
/// 3. Other op's operands that support inplace reuse
template <typename DstOpType>
bool MemPlan::HasUser(Value src,
                      const SmallVector<ValuePair> &inplaceList) const {
  LDBG("-- start visiting inplace-reuse path from: "
       << src << ", to: " << DstOpType::getOperationName() << "\n");
  DenseSet<Value> visited;
  return VisitDstOpTypeReachable<DstOpType>(src, visited, inplaceList);
}

/// check if gen-kill inplace will break up the pipeline.
/// eg. here vector op inplacement will break up the load-vector-store pipeline
/// when gen and kill are optimized by multi-buffers.
///      load outs(%gen)
///      vector op ins(%gen) outs(%kill)
///      store ins(%kill)
bool MemPlan::InplaceStallPipeline(Value gen, Value kill,
                                   const SmallVector<ValuePair> &inplaceList) {
  // 1. trace alloc op for gen and kill
  auto genAlloc = utils::tracebackMemRefToAlloc(gen);
  auto killAlloc = utils::tracebackMemRefToAlloc(kill);
  if (!genAlloc.has_value() || !killAlloc.has_value()) {
    return true;
  }

  // 2. check if inplace multi buffers
  if (!IsInplaceMultiBuffer(genAlloc.value(), inplaceList) ||
      !IsInplaceMultiBuffer(killAlloc.value(), inplaceList)) {
    return false;
  }

  // 3. check if gen has load op and kill has store op
  // When `gen` reaches `store` and `kill` reaches `load`, inplace-reuse for
  // `gen` and `kill` can cause mte2/mte3 pipeline stalls, because we need extra
  // synchronization on the same ub address between loop interations, which
  // means multi-buffer will be not useful.
  if (HasUser<hivm::StoreOp>(genAlloc.value(), inplaceList) &&
      HasUser<hivm::LoadOp>(killAlloc.value(), inplaceList)) {
    // Record the pair of buffers that can inplace but should not be reused due
    // to pipeline stalls
    inplacableBufferPairs.emplace_back(genAlloc.value(), killAlloc.value());
    return true;
  }
  return false;
}

bool MemPlan::IsReuseHIVMOp(Operation *op, const Value &genBuffer,
                            const Value &killBuffer) const {
  auto hivmOp = dyn_cast<hivm::HIVMStructuredOp>(op);
  if (!hivmOp || !isReusableByOffset(hivmOp))
    return false;

  if (mlir::isa<hivm::VAddOp, hivm::VSubOp, hivm::VMaxOp, hivm::VMinOp,
                hivm::VOrOp, hivm::VAndOp, hivm::VMulOp>(op) &&
      !hasInlineBroadcastOrTransposeAttr(op)) {
    // above ops can be inplaced in isa
    return true;
  }

  if (restrictInplaceAsISA)
    return false;

  // not in ISA but confirmed with hardware developers:
  // elementwise ops with the same shape and the same bitwidth operands can also
  // do memory inplace for src and dst

  if (!isReusableExtraBuffer(op))
    return false;

  if (!mlir::hivm::detail::isElemwiseNaryOpImpl(op))
    return false;

  // Tranpose operands can not inplace even with same shape
  if (hivmOp.existInlineTransposeLoopDims())
    return false;

  if (hivmOp.existInlineBroadcastLoopDims())
    return isStaticShapeSame(op, genBuffer, killBuffer);

  if (op->hasTrait<OpTrait::SameOperandsElementType>())
    return true;

  if (auto selOp = dyn_cast_if_present<hivm::VSelOp>(op)) {
    return isReusableVSelOp(selOp, genBuffer, killBuffer);
  }

  return isReusableOperands(op, hivmOp);
}

template <PIPE Pipe> bool MemPlan::IsInplaceReuseReachable(Value allocValue) {
  if (disableVFReachableCheck) {
    return false;
  }
  if (auto computedReachable = reachableMap.get<Pipe>(allocValue)) {
    return computedReachable.value();
  }
  if (dmaFirstPipelineOpt.IsDmaBuffer(Pipe, allocValue)) {
    reachableMap.put<Pipe>(allocValue, true);
    LDBG(allocValue << " is reachable to " << stringifyPIPE(Pipe) << "\n");
    return true;
  }
  reachableMap.put<Pipe>(allocValue, false);
  LDBG(allocValue << " is not reachable to " << stringifyPIPE(Pipe) << "\n");
  return false;
}

bool MemPlan::IsReuseVFStallPipeline(Value gen, Value kill) {
  // `gen` and `kill` are root allocs from genKillMap.
  auto genReachStore = IsInplaceReuseReachable<PIPE::PIPE_MTE3>(gen);
  auto killReachLoad = IsInplaceReuseReachable<PIPE::PIPE_MTE2>(kill);
  auto loopParent = gen.getDefiningOp()->getParentOfType<LoopLikeOpInterface>();

  // When `gen` reaches a store and `kill` reaches a load inside a loop, reuse
  // needs extra mte2/mte3 synchronization on the same ub address between loop
  // iterations, which stalls the pipeline.
  if (genReachStore && killReachLoad && loopParent) {
    LDBG("can't reuse " << gen << " and " << kill << "\n");
    return true;
  }
  LDBG("can reuse " << gen << " and " << kill << "\n");
  return false;
}

bool MemPlan::IsReuseVFCall(Operation *op, Value genBuffer, Value killBuffer) {
  return vfInplaceReuseInfo && hivm::isVFCall(op) &&
         !VFCallInplaceReuseInfo::hasAliasArgRisk(op) &&
         vfInplaceReuseInfo->isInplaceReusable(op, genBuffer, killBuffer);
}

SmallVector<ValuePair> MemPlan::GenerateInplaceList() {
  SmallVector<ValuePair> inplaceList;
  DenseMap<Operation *, bool> hasTouchOp;
  inplaceList.insert(inplaceList.end(), inplacePairList.begin(),
                     inplacePairList.end());
  for (auto &operationSeq : linearOperation) {
    auto it = genKillMap.find(operationSeq.get());
    if (it == genKillMap.end())
      continue;
    if (hasTouchOp[operationSeq->operation]) {
      continue;
    }
    Operation *op = it->first->operation;

    SmallVector<Value> sortedGen(it->second.gen.begin(), it->second.gen.end());
    llvm::stable_sort(sortedGen, [&](Value a, Value b) {
      return bufferInfos.find(a)->second.constBits >
             bufferInfos.find(b)->second.constBits;
    });

    DenseSet<Value> reusedKill;
    for (const Value &genBuffer : sortedGen) {
      auto *genBufferIter = bufferInfos.find(genBuffer);
      assert(genBufferIter != bufferInfos.end() &&
             "genBuffer should be find in bufferInfos");
      if (genBufferIter->second.ignoreInplace ||
          genBufferIter->second.memoryUnique) {
        continue;
      }
      for (const Value &killBuffer : it->second.kill) {
        auto *killBufferIter = bufferInfos.find(killBuffer);
        assert(killBufferIter != bufferInfos.end() &&
               "killBuffer should be find in bufferInfos");
        if (killBufferIter->second.ignoreInplace ||
            killBufferIter->second.memoryUnique) {
          continue;
        }
        if (genBuffer == killBuffer) {
          continue;
        }
        if (reusedKill.contains(killBuffer))
          continue;

        int64_t genBits = genBufferIter->second.constBits;
        int64_t killBits = killBufferIter->second.constBits;
        if (killBits < genBits) {
          continue;
        }

        if (IsReuseVFCall(op, genBuffer, killBuffer) &&
            !IsReuseVFStallPipeline(genBuffer, killBuffer)) {
          reusedKill.insert(killBuffer);
          reachableMap.unite(genBuffer, killBuffer);
          inplaceList.emplace_back(genBuffer, killBuffer);
          break;
        }

        if (IsReuseHIVMOp(op, genBuffer, killBuffer) &&
            !InplaceStallPipeline(genBuffer, killBuffer, inplaceList)) {
          reusedKill.insert(killBuffer);
          reachableMap.unite(genBuffer, killBuffer);
          inplaceList.emplace_back(genBuffer, killBuffer);
          break;
        }
      }
    }
    // Nodes in inplace are only processed once.
    hasTouchOp[operationSeq->operation] = true;
  }
  return inplaceList;
}

void MemPlan::EmitPlanMemoryFailureInfo() {
  if (failApplyBufferInfo.empty())
    return;
  for (auto &iter : failApplyBufferInfo) {
    AddressSpace space = iter.first;
    std::string error =
        stringifyEnum(space).str() + " overflow, requires " +
        std::to_string(iter.second) + " bits while " +
        std::to_string(GetBufferSpaceInfo(space).second) +
        " bits available! (possible reason: tiling basic block is too large or "
        "block number is more than what user expect due to multi-buffer "
        "feature is enabled and some ops need extra local buffer.)";
    func_.emitError() << error;
    errorInfo[space] = error;
  }
}

bool MemPlan::failedOnlyInUB() const {
  return failApplyBufferInfo.size() == 1 &&
         failApplyBufferInfo.count(AddressSpace::UB);
}

// Plan Memory algorithm.
LogicalResult MemPlan::plan(bool emitErrors) {
  // Construct StorageEntry structure.
  GenerateStorageEntry();
  // Plan memory address.
  PlanStatus as = planMode == MemPlanMode::LOCAL_MEM_PLAN
                      ? PlanLocalMemAddress()
                      : PlanWorkSpaceMemAddress();
  if (as == PlanStatus::PLAN_FAILED) {
    if (emitErrors) {
      EmitPlanMemoryFailureInfo();
    }
    if (enableMemoryDisplay) {
      // Update the address information of each buffer after memory buffer.
      UpdateBuffer2Offsets();
    }
    return failure();
  }
  // Update the address information of each buffer after memory buffer.
  UpdateBuffer2Offsets();
  if (enableMemoryDisplay) {
    PrintSuccessfulAllocatedMaxBits();
  }
  return success();
}

void MemPlan::GenerateStorageEntry() {
  DenseMap<int64_t, StorageEntry *> poolId2storageEntry;
  // create new storage entry.
  for (auto &operation : linearOperation) {
    auto it = genKillMap.find(operation.get());
    if (it == genKillMap.end())
      continue;
    for (const Value &genBuffer : it->second.gen) {
      auto *iter = bufferInfos.find(genBuffer);
      if (iter == bufferInfos.end()) {
        continue;
      }
      const std::shared_ptr<BufferLife> &bufLife = buffer2Life.at(genBuffer);
      if (planMode == MemPlanMode::LOCAL_MEM_PLAN &&
          iter->second.bufferPoolId >= 0) {
        auto poolSlotEntry =
            poolId2storageEntry.find(iter->second.bufferPoolId);
        if (poolSlotEntry != poolId2storageEntry.end()) {
          StorageEntry *entry = poolSlotEntry->second;
          entry->bufferLifeVec.append(bufLife->buffer, bufLife->allocTime,
                                      bufLife->freeTime);
          entry->inplaceBuffers.push_back(genBuffer);
          buffer2storageEntry[genBuffer] = entry;
          continue;
        }
      }
      std::unique_ptr<StorageEntry> entry =
          std::make_unique<StorageEntry>(nextEntryMemoId++);
      entry->bufInfo = &iter->second;
      entry->bufferLifeVec.append(bufLife->buffer, bufLife->allocTime,
                                  bufLife->freeTime);
      entry->inplaceBuffers.emplace_back(iter->first);
      auto multiBuffer = buffer2MultiNum.find(genBuffer);
      if (multiBuffer != buffer2MultiNum.end()) {
        entry->multiBufferNum = multiBuffer->second;
      }
      buffer2storageEntry[genBuffer] = entry.get();
      if (planMode == MemPlanMode::LOCAL_MEM_PLAN &&
          iter->second.bufferPoolId >= 0)
        poolId2storageEntry[iter->second.bufferPoolId] = entry.get();
      // Verify the validity of parameters after initialization.
      ValidateParameters(entry);
      StorageEntryVec.emplace_back(std::move(entry));
    }
  }
}

void MemPlanPreferUBConflict::buildBankGraph() {
  // Prefer only constrains copies with at least one normal endpoint.
  buildEnabledBankGraph();
}

void MemPlanPreferUBConflict::PlanBuffersWithoutReuse(StorageEntry *root,
                                                      size_t align) {
  commitBankNoReuseLayout(root, align);
}

void MemPlanRequireUBConflict::buildBankGraph() {
  // Require also includes copies between special buffers.
  buildEnabledBankGraph();
  buildBankComponents();
}

void MemPlanRequireUBConflict::PlanBuffersWithoutReuse(StorageEntry *root,
                                                       size_t align) {
  commitBankNoReuseLayout(root, align);
}

// Find one path while ignoring the specified copy edges.
static SmallVector<const BankEdge *>
findBankPath(const BankGraph &graph, StorageEntry *src, StorageEntry *dst,
             Operation *excluded, Operation *alsoExcluded = nullptr) {
  DenseMap<StorageEntry *, std::pair<StorageEntry *, const BankEdge *>> parent;
  SmallVector<StorageEntry *> worklist{src};
  DenseSet<StorageEntry *> visited{src};
  while (!worklist.empty() && !visited.contains(dst)) {
    StorageEntry *node = worklist.pop_back_val();
    auto it = graph.find(node);
    if (it == graph.end())
      continue;
    for (const BankEdge &edge : it->second) {
      if (edge.copyOp == excluded ||
          (alsoExcluded && edge.copyOp == alsoExcluded) ||
          !visited.insert(edge.peer).second)
        continue;
      parent[edge.peer] = {node, &edge};
      worklist.push_back(edge.peer);
    }
  }
  SmallVector<const BankEdge *> path;
  if (!visited.contains(dst))
    return path;
  for (StorageEntry *node = dst; node != src;) {
    auto step = parent.find(node);
    assert(step != parent.end() && "missing bank path predecessor");
    path.push_back(step->second.second);
    node = step->second.first;
  }
  return path;
}

void MemPlan::pruneBankCycles() {
  while (true) {
    Operation *removed = nullptr;
    for (BankCopy &copy : llvm::reverse(bankCopies)) {
      StorageEntry *src = buffer2storageEntry.lookup(copy.srcRoot);
      StorageEntry *dst = buffer2storageEntry.lookup(copy.dstRoot);
      auto it = bankGraph.find(src);
      if (!src || !dst || src == dst || it == bankGraph.end() ||
          llvm::none_of(it->second, [&copy](const BankEdge &edge) {
            return edge.copyOp == copy.copyOp.getOperation();
          }))
        continue;
      auto path = findBankPath(bankGraph, src, dst, copy.copyOp.getOperation());
      if (path.empty())
        continue;
      removed = copy.copyOp.getOperation();
      LDBG("[bank-cycle] found cycle involving " << copy.copyOp << "\n");
      // A second path plus the current copy forms another cycle.
      bool multipleCycles = llvm::any_of(path, [&](const BankEdge *edge) {
        return edge->copyOp &&
               !findBankPath(bankGraph, src, dst, copy.copyOp.getOperation(),
                             edge->copyOp)
                    .empty();
      });
      if (multipleCycles) {
        LDBG("[bank-cycle] multiple cycles; remove shared copy " << copy.copyOp
                                                                 << "\n");
      } else {
        uint64_t smallest = copy.copyBits;
        for (const BankEdge *edge : path)
          if (edge->copyOp && edge->copyBits < smallest) {
            removed = edge->copyOp;
            smallest = edge->copyBits;
          }
        LDBG("[bank-cycle] single cycle; remove smallest copy " << *removed
                                                                << "\n");
      }
      break;
    }
    if (!removed)
      return;
    removed->emitWarning("ignoring UB copy bank constraint in a cycle");
    SmallVector<StorageEntry *> emptyNodes;
    for (auto &node : bankGraph) {
      llvm::erase_if(node.second, [removed](const BankEdge &edge) {
        return edge.copyOp == removed;
      });
      if (node.second.empty())
        emptyNodes.push_back(node.first);
    }
    for (StorageEntry *node : emptyNodes)
      bankGraph.erase(node);
    llvm::erase_if(bankCopies, [removed](BankCopy &copy) {
      return copy.copyOp.getOperation() == removed;
    });
  }
}

bool MemPlan::isBankMemoryUniqueEntry(const StorageEntry *entry) const {
  if (entry->bufInfo->memoryUnique)
    return true;
  return llvm::any_of(entry->inplaceBuffers, [this](Value buffer) {
    auto info = bufferInfos.find(buffer);
    return info != bufferInfos.end() && info->second.memoryUnique;
  });
}

void MemPlan::buildEnabledBankGraph() {
  bankGraph.clear();
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::OFF) {
    return;
  }
  if (bankCopies.empty()) {
    LDBG("[bank-graph] skip: no eligible copies\n");
    return;
  }
  for (BankCopy &copy : bankCopies) {
    auto *srcSE = buffer2storageEntry.lookup(copy.srcRoot);
    auto *dstSE = buffer2storageEntry.lookup(copy.dstRoot);
    if (!srcSE || !dstSE) {
      LDBG("[bank-graph] skip storage entry lookup: src="
           << copy.srcRoot << ", dst=" << copy.dstRoot << "\n");
      continue;
    }

    // Pool and inplace processing can mark a merged entry memory-unique.
    if (isBankMemoryUniqueEntry(srcSE) || isBankMemoryUniqueEntry(dstSE)) {
      LDBG("[bank-graph] skip memory-unique copy: " << copy.copyOp << "\n");
      continue;
    }

    // TCB pool fallback can clear metadata, so retain the collected kinds.
    bool srcNormal = copy.srcKinds.empty() && !isBankSpecialEntry(srcSE);
    bool dstNormal = copy.dstKinds.empty() && !isBankSpecialEntry(dstSE);
    if (srcSE == dstSE) {
      // A shared base cannot change the relative access offsets.
      LDBG("[bank-graph] skip self edge: " << copy.copyOp << "\n");
      continue;
    }
    if (!srcNormal && !dstNormal &&
        planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::PREFER) {
      LDBG("[bank-graph] skip: both endpoints are special: " << copy.copyOp
                                                             << "\n");
      continue;
    }

    bankGraph[srcSE].push_back({dstSE, static_cast<uint64_t>(copy.srcBiasBits),
                                static_cast<uint64_t>(copy.dstBiasBits),
                                static_cast<uint64_t>(copy.copyBits),
                                copy.copyOp, srcNormal, dstNormal});
    bankGraph[dstSE].push_back({srcSE, static_cast<uint64_t>(copy.dstBiasBits),
                                static_cast<uint64_t>(copy.srcBiasBits),
                                static_cast<uint64_t>(copy.copyBits),
                                copy.copyOp, dstNormal, srcNormal});
  }
  pruneBankCycles();
  size_t addedEdges = 0;
  for (const auto &node : bankGraph)
    addedEdges += node.second.size();
  LDBG("[bank-graph] built " << bankGraph.size() << " node(s), " << addedEdges
                             << " edge(s) from " << bankCopies.size()
                             << " copy(s)\n");
  return;
}

// Add one committed address to the accumulated residue constraint.
static void updateBankResidue(BankResidueResult &result, uint64_t residue) {
  if (result.status == BankResidueStatus::Unconstrained) {
    result.status = BankResidueStatus::Known;
    result.residue = result.preferredResidue = residue;
  } else if (result.residue != residue) {
    result.status = BankResidueStatus::Conflict;
  }
}

void MemPlan::forEachBankNeighbor(
    StorageEntry *entry,
    llvm::function_ref<void(StorageEntry *, uint64_t, uint64_t)> visit) const {
  auto graphIt = bankGraph.find(entry);
  if (graphIt != bankGraph.end())
    for (const BankEdge &edge : graphIt->second)
      visit(edge.peer, edge.thisBiasBits, edge.peerBiasBits);
  if (entry->firstBufferRelationEntry)
    visit(entry->firstBufferRelationEntry, 0, 0);
  else
    for (StorageEntry *slot : entry->otherBufferRelationEntries)
      visit(slot, 0, 0);
}

size_t MemPlan::getBankDegree(StorageEntry *entry) const {
  auto graphIt = bankGraph.find(entry);
  size_t copyDegree = graphIt == bankGraph.end() ? 0 : graphIt->second.size();
  size_t slotDegree = entry->firstBufferRelationEntry
                          ? 1
                          : entry->otherBufferRelationEntries.size();
  return copyDegree + slotDegree;
}

bool MemPlan::hasBankConflict() const {
  return planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::REQUIRE
             ? !bankMember2Anchor.empty()
             : !bankGraph.empty();
}

void MemPlan::buildBankComponents() {
  bankMember2Anchor.clear();
  bankAnchor2Members.clear();
  DenseSet<StorageEntry *> visited;
  for (const auto &entry : StorageEntryVec) {
    StorageEntry *start = entry.get();
    if (start->bufInfo->bufferScope != AddressSpace::UB ||
        isBankMemoryUniqueEntry(start) ||
        (bankGraph.find(start) == bankGraph.end() &&
         !start->firstBufferRelationEntry &&
         start->otherBufferRelationEntries.empty()) ||
        !visited.insert(start).second)
      continue;
    SmallVector<StorageEntry *> members;
    SmallVector<StorageEntry *> worklist{start};
    StorageEntry *anchor = start;
    while (!worklist.empty()) {
      StorageEntry *current = worklist.pop_back_val();
      members.push_back(current);
      size_t currentDegree = getBankDegree(current);
      size_t anchorDegree = getBankDegree(anchor);
      if (currentDegree > anchorDegree ||
          (currentDegree == anchorDegree && current->memoId < anchor->memoId))
        anchor = current;
      forEachBankNeighbor(current, [&visited, &worklist](StorageEntry *peer,
                                                         uint64_t thisBias,
                                                         uint64_t peerBias) {
        (void)thisBias;
        (void)peerBias;
        if (visited.insert(peer).second)
          worklist.push_back(peer);
      });
    }
    for (StorageEntry *member : members)
      bankMember2Anchor[member] = anchor;
    bankAnchor2Members[anchor] = std::move(members);
  }
}

uint64_t MemPlan::bankResidueFromPeer(uint64_t peerOffset, uint64_t thisBias,
                                      uint64_t peerBias) {
  // Match the accessed addresses: this base = peer base + peer bias - this
  // bias.
  constexpr uint64_t mask = utils::kUBCopyBankSizeInBits - 1;
  return (peerOffset + peerBias + utils::kUBCopyBankSizeInBits - thisBias) &
         mask;
}

BankResidueResult
MemPlanRequireUBConflict::analyzeBankResidue(const StorageEntry *entry,
                                             BankOffsetLookup lookupPeerOffset,
                                             uint64_t placementHint) const {
  (void)placementHint;
  auto component = bankMember2Anchor.find(entry);
  if (component == bankMember2Anchor.end())
    return {};
  auto group = bankAnchor2Members.find(component->second);
  assert(group != bankAnchor2Members.end() && "missing bank component");
  auto start = llvm::find(group->second, entry);
  assert(start != group->second.end() && "missing bank component member");

  // Traverse the component during allocation; history supplies only offsets
  // that remain valid after rollback.
  DenseMap<StorageEntry *, uint64_t> relative;
  SmallVector<StorageEntry *> worklist{*start};
  relative[*start] = 0;
  while (!worklist.empty()) {
    StorageEntry *node = worklist.pop_back_val();
    uint64_t delta = relative.lookup(node);
    if (auto offset = lookupPeerOffset(node)) {
      BankResidueResult result;
      result.status = BankResidueStatus::Known;
      result.residue = bankResidueFromPeer(*offset, delta, 0);
      return result;
    }
    forEachBankNeighbor(node, [&relative, &worklist, delta](StorageEntry *peer,
                                                            uint64_t thisBias,
                                                            uint64_t peerBias) {
      uint64_t next = bankResidueFromPeer(delta, peerBias, thisBias);
      if (relative.try_emplace(peer, next).second)
        worklist.push_back(peer);
    });
  }
  return {};
}

BankResidueResult MemPlan::analyzeBankResidue(const StorageEntry *entry,
                                              BankOffsetLookup lookupPeerOffset,
                                              uint64_t placementHint) const {
  (void)entry;
  (void)lookupPeerOffset;
  (void)placementHint;
  return {};
}

BankResidueResult MemPlanPreferUBConflict::analyzeBankResidue(
    const StorageEntry *e,
    llvm::function_ref<std::optional<uint64_t>(const StorageEntry *)>
        lookupPeerOffset,
    uint64_t placementHint) const {
  constexpr uint64_t bankMask = utils::kUBCopyBankSizeInBits - 1;
  auto graphIt = bankGraph.find(e);
  if (graphIt == bankGraph.end())
    return {};

  BankResidueResult result;
  uint64_t preferredCopyBits = 0;
  for (const BankEdge &edge : graphIt->second) {
    if (!edge.thisIsNormal) {
      LDBG("[bank-residue] skip non-normal: " << edge.copyOp << "\n");
      continue;
    }
    auto peerOffset = lookupPeerOffset(edge.peer);
    if (!peerOffset) {
      LDBG("[bank-residue] skip unplanned peer: " << edge.copyOp << "\n");
      continue;
    }

    // PREFER anchors one slot and leaves other runtime slots unchanged.
    uint64_t edgeResidue =
        bankResidueFromPeer(*peerOffset, edge.thisBiasBits, edge.peerBiasBits);
    uint64_t bestPadding = (edgeResidue - placementHint) & bankMask;
    // Slot choice is a local padding heuristic; actual allocation still
    // performs all lifetime, alignment, and capacity checks.
    for (const StorageEntry *slot : edge.peer->otherBufferRelationEntries) {
      auto offset = lookupPeerOffset(slot);
      if (!offset)
        continue;
      uint64_t residue =
          bankResidueFromPeer(*offset, edge.thisBiasBits, edge.peerBiasBits);
      uint64_t padding = (residue - placementHint) & bankMask;
      if (padding < bestPadding) {
        edgeResidue = residue;
        bestPadding = padding;
      }
    }
    if (result.status == BankResidueStatus::Unconstrained) {
      LDBG("[bank-residue] anchor: " << edge.copyOp << "\n");
      result.status = BankResidueStatus::Known;
      result.residue = edgeResidue;
      result.preferredResidue = edgeResidue;
      preferredCopyBits = edge.copyBits;
      continue;
    }
    updateBankResidue(result, edgeResidue);
    if (edgeResidue != result.residue)
      LDBG("[bank-residue] conflict: " << edge.copyOp << "\n");
    else
      LDBG("[bank-residue] consistent: " << edge.copyOp << "\n");

    // Prefer the largest copy; keep the first constraint on ties.
    if (edge.copyBits > preferredCopyBits) {
      result.preferredResidue = edgeResidue;
      preferredCopyBits = edge.copyBits;
    }
  }
  if (result.status == BankResidueStatus::Conflict)
    result.residue = result.preferredResidue;
  // Choose locally among planned peers; this does not solve the global graph.
  LDBG("[bank-residue] final: status="
       << static_cast<int>(result.status) << " residue=" << result.residue
       << " preferredResidue=" << result.preferredResidue << "\n");
  return result;
}

void MemPlan::PrintSuccessfulAllocatedMaxBits() {
  // TODO: This print may be merged into the existing debug log output and
  // could be redundant.
  auto it = memscope2rootStorageEntry.find(hivm::AddressSpace::UB);
  if (it != memscope2rootStorageEntry.end()) {
    assert(it->second != nullptr);
    uint64_t ubAllocBits =
        it->second->alignedConstBits + it->second->bitsOffset;
    for (auto &child : it->second->mergedChildren) {
      ubAllocBits =
          std::max(ubAllocBits, child->bitsOffset + child->alignedConstBits);
    }
    llvm::outs() << "[AscendNPU IR] Allocated UB size = " << ubAllocBits
                 << " bits \n";
  }
}

void MemPlan::ValidateParameters(std::unique_ptr<StorageEntry> &e) const {
  assert(e->bufInfo->operation && "Unrecognized legal define operation !");
  assert(e->bufInfo->constBits >= 0U && "recognized illegal memory sizes !");
  assert(!e->bufferLifeVec.empty() && "Unrecognized buffer's life time !");
}

void MemPlan::UpdateBuffer2Offsets() {
  for (const auto &e : StorageEntryVec) {
    for (Value &buffer : e->inplaceBuffers) {
      // MultiBuffer can cause multiple addrs.
      buffer2Offsets[buffer].push_back(
          (e->bitsOffset + utils::kBitsToByte - 1) / utils::kBitsToByte);
      // Collect local buffers folded onto a preload storage entry that need
      // a new preload mark. The preload TCB itself (in
      // preloadBufferReuseableInfo) is excluded — it already has the mark.
      if (IsPreloadStorageEntry(e.get()) &&
          !preloadBufferReuseableInfo.contains(buffer))
        preloadLocalBuffers.insert(buffer);
    }
  }
  // In the MultiBuffer scenario, single reuse db will result in additional
  // storageEntry.
  UpdateMultiBufferReuseExtraOffset();
}

void MemPlan::UpdateMultiBufferReuseExtraOffset() {
  if (firstBufferEntry2RelationOtherBufferEntry.empty()) {
    return;
  }

  for (auto &relationEntry : firstBufferEntry2RelationOtherBufferEntry) {
    for (const std::unique_ptr<StorageEntry> &otherBufferEntry :
         relationEntry.second) {
      if (!otherBufferEntry)
        continue;
      for (Value &buffer : otherBufferEntry->inplaceBuffers) {
        buffer2Offsets[buffer].push_back(
            (otherBufferEntry->bitsOffset + utils::kBitsToByte - 1) /
            utils::kBitsToByte);
      }
    }
  }
}

void MemPlan::MergeInplaceSE() {
  // get the list of inplace value pair.
  SmallVector<ValuePair> inplaceList = GenerateInplaceList();
  // try to merge storage entries. genSE is replaced by KillSE.
  for (const auto &pairIter : inplaceList) {
    const StorageEntry *genSE = buffer2storageEntry[pairIter.first];
    StorageEntry *killSE = buffer2storageEntry[pairIter.second];
    if (genSE == killSE) {
      // already same storageEntry, no need to inplace.
      continue;
    }
    assert(genSE != nullptr && killSE != nullptr &&
           " genSE and killSE should be valid");
    BufferLifeVec mergedBufferLifeVec;
    mergedBufferLifeVec.append(genSE->bufferLifeVec);
    mergedBufferLifeVec.append(killSE->bufferLifeVec);
    // append merges touching intervals. Endpoint buffers stay visible to
    // inplace checks; an interior buffer is dropped.
    killSE->bufferLifeVec.swap(mergedBufferLifeVec);
    //  merge allocs of two storage entry and update inplaceBuffers.
    killSE->inplaceBuffers.insert(killSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.end());

    // Take the maximum value for the inplace scene.
    killSE->multiBufferNum =
        std::max(genSE->multiBufferNum, killSE->multiBufferNum);

    // all buffers have same storage entry after merging
    for (auto &buffer : genSE->inplaceBuffers) {
      buffer2storageEntry[buffer] = killSE;
    }
    // remove the alloc info of dst after successful merging
    auto *e = std::find_if(StorageEntryVec.begin(), StorageEntryVec.end(),
                           [genSE](std::unique_ptr<StorageEntry> &se) {
                             return se.get() == genSE;
                           });
    StorageEntryVec.erase(e);
  }
}

bool MemPlan::IsPreloadStorageEntry(const StorageEntry *storageEntry) {
  return llvm::any_of(storageEntry->inplaceBuffers, [&](Value buffer) {
    return preloadBufferReuseableInfo.contains(buffer);
  });
}

bool MemPlan::IsPreloadBufferReuseable(PreloadBufferReuseableInfo &info,
                                       const BufferLifeVec &lives) {
  auto preloadBuffer = info.preloadBuffer;
  // TODO: support reuse hivmOp
  auto isInplaceReuseable = [this](Operation *op, Value gen, Value kill) {
    auto genType = cast<ShapedType>(gen.getType());
    auto killType = cast<ShapedType>(kill.getType());
    return genType.getShape() == killType.getShape() &&
           IsReuseVFCall(op, gen, kill);
  };
  // A life sits in the window when it starts at or after the window start and
  // ends at or before the window end.
  auto coveredBy = [](const BufferLife &window, const BufferLifeView &life) {
    return window.allocTime <= life.allocTime &&
           window.freeTime >= life.freeTime;
  };

  // Before-gen: the buffers that die at the merged end must die on the preload
  // gen and be inplace with it. An earlier endpoint only has to be covered.
  auto &beforeGenLife = info.reuseableBufferLifeBeforeGen;
  bool beforeGenOk =
      beforeGenLife && lives.allOfEndpoints([&](const BufferLifeView &life,
                                                bool /*atHead*/, bool atTail) {
        if (!coveredBy(*beforeGenLife, life))
          return false;
        if (!atTail)
          return true;
        return beforeGenLife->freeTime == life.freeTime &&
               isInplaceReuseable(info.genInfo->operation, preloadBuffer,
                                  life.buffer);
      });
  if (beforeGenOk)
    return true;

  // After-kill: the buffers born at the merged start must be born on the
  // preload kill and be inplace with it. A later endpoint only has to be
  // covered. One window has to accept every endpoint; the two windows are not
  // mixed across endpoints.
  auto &afterKillLife = info.reuseableBufferLifeAfterKill;
  return afterKillLife &&
         lives.allOfEndpoints(
             [&](const BufferLifeView &life, bool atHead, bool /*atTail*/) {
               if (!coveredBy(*afterKillLife, life))
                 return false;
               if (!atHead)
                 return true;
               return afterKillLife->allocTime == life.allocTime &&
                      isInplaceReuseable(info.killInfo->operation, life.buffer,
                                         preloadBuffer);
             });
}

void MemPlan::GeneratePreloadReuseableSE() {
  LDBG("\n-------localSE2ReuseablePreloadSE info--------\n\n");
  for (auto &[preloadBuffer, reusableInfo] : preloadBufferReuseableInfo) {
    auto *preloadBufferSE = buffer2storageEntry[preloadBuffer];
    // Preload folding bypasses the normal memoryUnique conflict check. Keep
    // explicit cube slots out of this reuse path as well.
    if (requiresExplicitCubePlanning(GetBufferSpaceAttr(preloadBuffer),
                                     enableMmadL0Planning))
      continue;
    reusableInfo.preloadBuffer = preloadBuffer;
    // A preload buffer's genOp and killOp cannot reside in the same scope
    // (otherwise it would not be a preload buffer). Thus equal preload_num
    // here must come from two distinct scopes that happen to share the same
    // preload_num; in this case the buffer cannot be reused, so skip it.
    if (reusableInfo.killScopePreloadNum == reusableInfo.genScopePreloadNum) {
      continue;
    }
    for (auto &e : StorageEntryVec) {
      if (e->bufInfo->memoryUnique || IsPreloadStorageEntry(e.get()) ||
          preloadBufferSE->bufInfo->bufferScope != e->bufInfo->bufferScope ||
          preloadBufferSE->bufInfo->constBits < e->bufInfo->constBits) {
        continue;
      }
      if (IsPreloadBufferReuseable(reusableInfo, e->bufferLifeVec)) {
        MemLifeDebugInfo(e.get());
        MemLifeDebugInfo(preloadBufferSE);

        localSE2ReuseablePreloadSE.insert({e.get(), preloadBufferSE});
        break;
      }
    }
  }
}

PlanStatus MemPlan::PlanLocalMemAddress() {
  dmaFirstPipelineOpt.build(func_, enableMmadL0Planning);
  // merge from the first storage entry
  MergeInplaceSE();
  // collect local storage entry and preload's first storage entry, so this
  // should be called before ExpandMultiBufferStorageEntry.
  GeneratePreloadReuseableSE();
  ExpandMultiBufferStorageEntry();
  buildBankGraph();
  MergeSameScopeSE();
  return PlanMemAddressOfWholeLocalBuffer();
}

PlanStatus MemPlan::PlanWorkSpaceMemAddress() {
  // merge from the first storage entry
  MergeInplaceSE();
  ExpandMultiBufferStorageEntry();
  // Construct root StorageEntry and collect the same work space arg
  // StorageEntry.
  MergeSameWorkSpaceArgSE();
  // Start plan
  return PlanMemOffsetOfWholeWorkSpace();
}

void MemPlan::MergeSameWorkSpaceArgSE() {
  for (auto &iter : StorageEntryVec) {
    auto allocWorkspaceOp = dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(
        iter->bufInfo->operation);
    assert(allocWorkspaceOp && "only allocWorkspaceOp will plan");
    Value workspaceArg = allocWorkspaceOp.getWorkspaceArg();
    auto *iterArg = workSpaceArg2rootStorageEntry.find(workspaceArg);
    if (iterArg == workSpaceArg2rootStorageEntry.end()) {
      workSpaceArg2rootStorageEntry[workspaceArg] = iter.get();
    } else {
      iterArg->second->mergedChildren.push_back(iter.get());
    }
  }
}

PlanStatus MemPlan::PlanMemOffsetOfWholeWorkSpace() {
  for (auto &it : workSpaceArg2rootStorageEntry) {
    StorageEntry *rootStorageEntry = it.second;
    assert(rootStorageEntry && "Root StorageEntry should not be null");
    if (!enableGlobalReuse) {
      GlobalWorkspaceNoReuse(rootStorageEntry);
      continue;
    }
    MemBoundList outline;
    PlanRecHis history;
    SpecInfo si;
    // Can be reuse without conflicting life intervals.
    si.specLevel = si.minLevel;
    int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
    outline.push_back(std::make_shared<MemoryBound>(
        BufferLifeVec(), 0, std::numeric_limits<uint64_t>::max(), nullptr,
        nextBoundMemoId++));

    // The initial value is rootStorageEntry.
    StorageEntry *curEntry = rootStorageEntry;
    while (si.childIdx < childrenNum) {
      // TODO: Alignment by type can be considered in the future.
      curEntry->alignedConstBits = static_cast<uint64_t>(
          AlignUp(curEntry->bufInfo->constBits, workSpaceAlignSize));
      curEntry->childIdx = si.childIdx;
      LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
      if (failed(planResult)) {
        return PlanStatus::PLAN_FAILED;
      }
      if (si.childIdx >= childrenNum) {
        break;
      }
      curEntry = rootStorageEntry->mergedChildren[si.childIdx];
    }
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

void MemPlan::GlobalWorkspaceNoReuse(StorageEntry *rootStorageEntry) {
  rootStorageEntry->bitsOffset = 0;
  // TODO: Alignment by type can be considered in the future.
  uint64_t offset = static_cast<uint64_t>(
      AlignUp(rootStorageEntry->bufInfo->constBits, workSpaceAlignSize));
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    // TODO: Alignment by type can be considered in the future.
    offset += static_cast<uint64_t>(
        AlignUp(child->bufInfo->constBits, workSpaceAlignSize));
  }
}

void MemPlan::ExpandMultiBufferStorageEntry() {
  // StorageEntry that needs to be expanded.
  size_t size = StorageEntryVec.size();
  for (size_t i = 0; i < size; i++) {
    if (StorageEntryVec[i]->multiBufferNum > 1) {
      // Create multiBufferNum - 1 additional entries
      for (uint32_t j = 1; j < StorageEntryVec[i]->multiBufferNum; j++) {
        std::unique_ptr<StorageEntry> entry =
            std::make_unique<StorageEntry>(nextEntryMemoId++);
        entry->bufInfo = StorageEntryVec[i]->bufInfo;
        entry->bufferLifeVec = StorageEntryVec[i]->bufferLifeVec;
        entry->alignedConstBits = StorageEntryVec[i]->alignedConstBits;
        entry->inplaceBuffers = StorageEntryVec[i]->inplaceBuffers;
        entry->multiBufferNum = StorageEntryVec[i]->multiBufferNum;
        entry->firstBufferRelationEntry = StorageEntryVec[i].get();
        // Store the entry pointer
        StorageEntryVec[i]->otherBufferRelationEntries.push_back(entry.get());
        StorageEntryVec.push_back(std::move(entry));
      }
    }
  }
}

void MemPlan::LayoutNoReuseOffsets(
    StorageEntry *rootStorageEntry, size_t alignUnit,
    DenseMap<const StorageEntry *, uint64_t> &offsets, uint64_t &requiredSize) {
  constexpr uint64_t bankMask = utils::kUBCopyBankSizeInBits - 1;
  SmallVector<StorageEntry *> entries{rootStorageEntry};
  entries.append(rootStorageEntry->mergedChildren.begin(),
                 rootStorageEntry->mergedChildren.end());
  assert(alignUnit != 0 && "buffer alignment must be nonzero");
  uint64_t cursor = 0;
  for (StorageEntry *entry : entries) {
    uint64_t candidate = AlignUp(cursor, alignUnit);
    assert(candidate >= cursor && "layout alignment overflows");
    auto result = analyzeBankResidue(
        entry,
        [&offsets](const StorageEntry *peer) -> std::optional<uint64_t> {
          auto it = offsets.find(peer);
          return it == offsets.end() ? std::nullopt
                                     : std::optional<uint64_t>(it->second);
        },
        candidate);
    if (result.status != BankResidueStatus::Unconstrained) {
      uint64_t residue = result.residue;
      uint64_t adjustment =
          (residue + utils::kUBCopyBankSizeInBits - (candidate & bankMask)) &
          bankMask;
      assert(candidate <= UINT64_MAX - adjustment && "bank padding overflows");
      candidate += adjustment;
      assert(candidate % alignUnit == 0 && "bank residue violates alignment");
    }
    assert(candidate <= UINT64_MAX - entry->alignedConstBits &&
           "no-reuse layout overflows");
    offsets[entry] = candidate;
    cursor = candidate + entry->alignedConstBits;
  }
  requiredSize = cursor;
}

void MemPlan::PlanBuffersWithoutReuse(StorageEntry *rootStorageEntry,
                                      size_t alignUnit) {
  uint offset = 0;
  rootStorageEntry->bitsOffset = offset;
  offset += rootStorageEntry->alignedConstBits;
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    offset += child->alignedConstBits;
  }
}

void MemPlan::prepareNoReuseLayout() {
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::OFF ||
      !hasBankConflict())
    return;
  auto it = memscope2rootStorageEntry.find(AddressSpace::UB);
  if (it == memscope2rootStorageEntry.end())
    return;
  auto *root = GetReorderRootStorageEntry(it->second);
  it->second = root;
  size_t align = GetBufferSpaceInfo(it->first).first;
  uint64_t requiredSize = 0;
  bankNoReuseOffsets.clear();
  LayoutNoReuseOffsets(root, align, bankNoReuseOffsets, requiredSize);
  bufferScope2RequiredSize[it->first] = requiredSize;
}

void MemPlan::commitBankNoReuseLayout(StorageEntry *rootStorageEntry,
                                      size_t alignUnit) {
  if (rootStorageEntry->bufInfo->bufferScope != AddressSpace::UB ||
      !hasBankConflict()) {
    MemPlan::PlanBuffersWithoutReuse(rootStorageEntry, alignUnit);
    return;
  }
  auto rootOffset = bankNoReuseOffsets.find(rootStorageEntry);
  assert(rootOffset != bankNoReuseOffsets.end() &&
         "bank no-reuse layout is missing the root");
  rootStorageEntry->bitsOffset = rootOffset->second;
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    auto childOffset = bankNoReuseOffsets.find(child);
    assert(childOffset != bankNoReuseOffsets.end() &&
           "bank no-reuse layout is missing a child");
    child->bitsOffset = childOffset->second;
  }
}

void MemPlan::MergeSameScopeSE() {
  // Construct root StorageEntry and collect the same scope StorageEntry
  for (auto &iter : StorageEntryVec) {
    auto *iterScope =
        memscope2rootStorageEntry.find(iter->bufInfo->bufferScope);
    if (iterScope == memscope2rootStorageEntry.end()) {
      memscope2rootStorageEntry[iter->bufInfo->bufferScope] = iter.get();
    } else {
      iterScope->second->mergedChildren.push_back(iter.get());
    }
  }

  // set alignedConstBits for all StorageEntry and calculate the required size
  // for each buffer scope.
  for (auto [memScope, rootStorageEntry] : memscope2rootStorageEntry) {
    auto [align, maxBits] = GetBufferSpaceInfo(memScope);
    rootStorageEntry->alignedConstBits = AlignUp(
        static_cast<uint64_t>(rootStorageEntry->bufInfo->constBits), align);
    size_t accumulateSize = rootStorageEntry->alignedConstBits;
    for (auto &childStorageEntry : rootStorageEntry->mergedChildren) {
      childStorageEntry->alignedConstBits = AlignUp(
          static_cast<uint64_t>(childStorageEntry->bufInfo->constBits), align);
      accumulateSize += childStorageEntry->alignedConstBits;
    }
    bufferScope2RequiredSize[memScope] = accumulateSize;
  }
  // Bank-aware planners may add padding before the shared capacity check.
  // The base implementation keeps the original no-reuse sizes.
  prepareNoReuseLayout();
}

uint64_t MemPlan::PlanMemAddressForSingleLevel(StorageEntry *rootStorageEntry,
                                               int specLevel) {
  size_t maxBits = UINT64_MAX;
  rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
  // memory outline in a given buffer scope.
  MemBoundList outline;
  PlanRecHis history;
  SpecInfo si;
  si.specLevel = specLevel;
  si.maxLevel = specLevel;
  si.minLevel = specLevel;
  int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
  outline.push_back(std::make_shared<MemoryBound>(BufferLifeVec(), 0, maxBits,
                                                  nullptr, nextBoundMemoId++));

  // The initial value is rootStorageEntry.
  StorageEntry *curEntry = rootStorageEntry;
  while (si.childIdx < childrenNum) {
    curEntry->childIdx = si.childIdx;
    LogicalResult result = MultiSpecPlan(si, outline, history, curEntry);
    // LEVEL 1 cannot yet split a bound at an interior bank-aligned address.
    // Use the prepared no-reuse size if the fixed-level search cannot proceed.
    if (planMemoryUBCopyBankMode != PlanMemoryUBCopyBankMode::OFF &&
        failed(result)) {
      auto required =
          bufferScope2RequiredSize.find(rootStorageEntry->bufInfo->bufferScope);
      assert(required != bufferScope2RequiredSize.end() &&
             "missing no-reuse required size");
      LDBG("[PlanMemAddressForSingleLevel] use no-reuse size after "
           "fixed-level allocation failed\n");
      return required->second;
    }
    if (si.childIdx >= childrenNum) {
      break;
    }
    curEntry = rootStorageEntry->mergedChildren[si.childIdx];
  }
  // Find the max appled bits from all children and root, which is the max
  // memory applied in this buffer space.
  uint64_t maxAllocBits = rootStorageEntry->alignedConstBits;
  auto children = rootStorageEntry->mergedChildren;
  for (auto *child : children) {
    maxAllocBits =
        std::max(maxAllocBits, child->bitsOffset + child->alignedConstBits);
  }
  return maxAllocBits;
}

PlanStatus MemPlan::PlanMemAddressOfWholeLocalBuffer() {
  // Start plan
  for (auto [memScope, rootStorageEntry] : memscope2rootStorageEntry) {
    assert(rootStorageEntry && "Root StorageEntry should not be null");
    // get the buffer info for a given scope.
    auto bufferSpaceInfo = GetBufferSpaceInfo(memScope);
    size_t align = bufferSpaceInfo.first;
    size_t maxBits = bufferSpaceInfo.second;
    // No reuse plan
    auto iter = bufferScope2RequiredSize.find(memScope);
    assert(iter != bufferScope2RequiredSize.end());
    if (iter->second <= maxBits) {
      ReportMemLifeDebugInfo(rootStorageEntry);
      PlanBuffersWithoutReuse(rootStorageEntry, align);
      LDBG("\nApply no reuse plan strategy for " << memScope << " memScope\n");
      continue;
    } else if (rootStorageEntry->mergedChildren.empty()) {
      failApplyBufferInfo[memScope] = iter->second;
      return PlanStatus::PLAN_FAILED;
    }

    MemBoundList outline;
    PlanRecHis history;
    SpecInfo si;
    si.specLevel = si.maxLevel;
    // No reuse dma buffer plan
    rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
    ReportMemLifeDebugInfo(rootStorageEntry);
    LDBG("\n---------Try no pipe stall plan strategy for "
         << memScope << " memScope---------\n\n");
    auto maxAllocBits =
        PlanMemAddressForSingleLevel(rootStorageEntry, si.maxLevel);
    if (maxAllocBits <= maxBits) {
      ReportAllocatedEntryDebugInfo(rootStorageEntry, false);
      continue;
    }
    memscope2allocatedEntry.erase(memScope);
    // memory outline in a given buffer scope.
    LDBG("\n---------Try multi level plan strategy for "
         << memScope << " memScope---------\n\n");
    // Enable sectional outline from the first multi-level attempt. Waiting
    // until ApplyFailStrategy restarts is too late for L1 pong registration
    // when a later buffer swallows an exact pong start.
    splitOutline = true;
    int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
    outline.push_back(std::make_shared<MemoryBound>(
        BufferLifeVec(), 0, maxBits, nullptr, nextBoundMemoId++));

    // The initial value is rootStorageEntry.
    StorageEntry *curEntry = rootStorageEntry;
    while (si.childIdx < childrenNum) {
      curEntry->childIdx = si.childIdx;
      LDBG("\n");
      LDBG("----------Need-Plan-CurEntry---------\n");
      ReportCurEntryDebugInfo(curEntry);
      LDBG("\n");
      LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
      if (failed(planResult)) {
        StatusWrapper statusWrapper = {false,   curEntry->alignedConstBits,
                                       &si,     outline,
                                       history, rootStorageEntry};
        LDBG("\n");
        LDBG("----------ApplyFailStrategy---------\n");
        ReportCurEntryDebugInfo(curEntry);
        LDBG("\n");
        PlanStatus as = ApplyFailStrategy(statusWrapper, maxBits);
        if (as == PlanStatus::RESTART_NEW_PLAN) {
          // Restart plan.
          LDBG("[PlanLocal] ApplyFailStrategy -> RESTART_NEW_PLAN "
               "(splitOutline enabled); reset SpecInfo to defaults\n");
          si = SpecInfo();
          curEntry = rootStorageEntry;
          continue;
        }
        if (as == PlanStatus::PLAN_FAILED) {
          LDBG("[PlanLocal] ApplyFailStrategy -> PLAN_FAILED.\n");
          ReportAllocatedEntryDebugInfo(rootStorageEntry, true);
          failApplyBufferInfo[memScope] =
              PlanMemAddressForSingleLevel(rootStorageEntry, SPEC_LEVEL_0);
          return as;
        }
        LDBG("[PlanLocal] ApplyFailStrategy -> CONTINUE_PLAN "
             "specLevel="
             << si.specLevel << " childIdx=" << si.childIdx << "\n");
      }
      if (si.childIdx >= childrenNum) {
        break;
      }
      curEntry = rootStorageEntry->mergedChildren[si.childIdx];
    }
    ReportAllocatedEntryDebugInfo(rootStorageEntry, false);
    llvm::WithColor::warning(llvm::outs())
        << "[hivm-plan-memory] There reused some dma buffers in " << memScope
        << " address space, which may stall pipe. Not reusing dma buffer needs "
        << maxAllocBits << " bits while " << maxBits << " bits available!\n";
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

void MemPlan::ReportMemLifeDebugInfo(const StorageEntry *rootStorageEntry) {
  LDBG("\n");
  LDBG("-------------------------- Buffer2Life --------------------------\n");
  MemLifeDebugInfo(rootStorageEntry);
  for (auto &storageEntry : rootStorageEntry->mergedChildren) {
    MemLifeDebugInfo(storageEntry);
  }
}

void MemPlan::MemLifeDebugInfo(const StorageEntry *storageEntry) const {
  for (auto &buffer : storageEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("Buffer : " << allocOp.getResult() << "\n");
      }
    }
  }
  LLVM_DEBUG({
    storageEntry->bufferLifeVec.forEach([](BufferLifeView bufferLife) {
      llvm::dbgs() << "bufferLife : " << "allocTime : " << bufferLife.allocTime
                   << " , freeTime : " << bufferLife.freeTime << "\n";
    });
  });
}

void MemPlan::ReportCurEntryDebugInfo(const StorageEntry *curEntry) const {
  for (auto &buffer : curEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("buffer : ");
        LDBG(allocOp.getResult());
      }
    }
  }
}

static SmallVector<StorageEntry *>
appendStorageEntries(std::initializer_list<ArrayRef<StorageEntry *>> groups) {
  SmallVector<StorageEntry *> result;
  for (ArrayRef<StorageEntry *> group : groups)
    result.append(group.begin(), group.end());
  return result;
}

StorageEntry *
MemPlan::GetReorderRootStorageEntry(StorageEntry *rootStorageEntry) {
  assert(rootStorageEntry->bufInfo->operation &&
         "Unrecognized legal define operation !");
  auto moduleOp =
      rootStorageEntry->bufInfo->operation->getParentOfType<ModuleOp>();
  if (hacc::utils::isRegBasedArch(moduleOp) &&
      rootStorageEntry->bufInfo->bufferScope != hivm::AddressSpace::UB) {
    return rootStorageEntry;
  }
  SmallVector<StorageEntry *> origStorageEntryVec = {rootStorageEntry};
  origStorageEntryVec.insert(origStorageEntryVec.end(),
                             rootStorageEntry->mergedChildren.begin(),
                             rootStorageEntry->mergedChildren.end());

  // reorder storage entrys: mem unique buffers + dma touched buffers + other
  // buffers + scalar touched buffers. Dma touched buffers and scalar touched
  // buffers will only exist in UB.
  SmallVector<StorageEntry *> memUniqueStorageEntryVec;
  SmallVector<StorageEntry *> touchDmaStorageEntryVec;
  SmallVector<StorageEntry *> touchPipeScalarStorageEntryVec;
  SmallVector<StorageEntry *> otherStorageEntryVec;
  for (auto *storageEntry : origStorageEntryVec) {
    if (!storageEntry) {
      continue;
    }
    bool isOtherBuffer =
        llvm::all_of(storageEntry->inplaceBuffers, [&](Value &buffer) {
          auto *it = bufferInfos.find(buffer);
          assert(it != bufferInfos.end() &&
                 "buffer should be find in bufferInfos");
          if (it->second.memoryUnique) {
            memUniqueStorageEntryVec.push_back(storageEntry);
            return false;
          }
          // L1 tightly coupled buffers are not MTE2/MTE3/FIX, so IsDmaBuffer
          // is false and they used to be planned after every load. Give them
          // the same reorder priority as those loads.
          if (dmaFirstPipelineOpt.IsDmaBuffer(buffer) ||
              (it->second.bufferScope == AddressSpace::L1 &&
               it->second.cvMixId >= 0)) {
            touchDmaStorageEntryVec.push_back(storageEntry);
            return false;
          }
          if (dmaFirstPipelineOpt.IsScalarBuffer(buffer)) {
            touchPipeScalarStorageEntryVec.push_back(storageEntry);
            return false;
          }
          return true;
        });
    if (isOtherBuffer) {
      otherStorageEntryVec.push_back(storageEntry);
    }
  }
  if (planMemoryStrategy == PlanMemoryStrategy::LARGEST_FIRST) {
    auto storageEntryVecOrderCmp = [](StorageEntry *a, StorageEntry *b) {
      return a->bufInfo->constBits > b->bufInfo->constBits;
    };
    llvm::stable_sort(touchDmaStorageEntryVec, storageEntryVecOrderCmp);
    llvm::stable_sort(touchPipeScalarStorageEntryVec, storageEntryVecOrderCmp);
    llvm::stable_sort(otherStorageEntryVec, storageEntryVecOrderCmp);
  }

  auto reorderedStorageEntryVec =
      orderBankEntries(memUniqueStorageEntryVec, touchDmaStorageEntryVec,
                       otherStorageEntryVec, touchPipeScalarStorageEntryVec);

  // Ensure that firstbuffer and otherbuffer are continuously plan mem in the
  // multi buffer.
  ReorderContinuousFirstBufferOtherBufferEntry(reorderedStorageEntryVec);
  StorageEntry *reorderedRootStorageEntry = reorderedStorageEntryVec[0];
  reorderedRootStorageEntry->mergedChildren.clear();
  for (size_t j = 1; j < reorderedStorageEntryVec.size(); ++j) {
    reorderedRootStorageEntry->mergedChildren.push_back(
        reorderedStorageEntryVec[j]);
  }
  return reorderedRootStorageEntry;
}

SmallVector<StorageEntry *>
MemPlan::orderBankEntries(ArrayRef<StorageEntry *> memUnique,
                          SmallVectorImpl<StorageEntry *> &dma,
                          SmallVectorImpl<StorageEntry *> &other,
                          SmallVectorImpl<StorageEntry *> &scalar) {
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::REQUIRE)
    return orderRequiredBankEntries(memUnique, dma, other, scalar);
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::OFF)
    return appendStorageEntries({memUnique, dma, other, scalar});

  assert(planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::PREFER);
  SmallVector<StorageEntry *> special;
  collectBankSpecialEntries(dma, special);
  collectBankSpecialEntries(other, special);
  collectBankSpecialEntries(scalar, special);
  return appendStorageEntries({memUnique, special, dma, other, scalar});
}

SmallVector<StorageEntry *>
MemPlan::orderRequiredBankEntries(ArrayRef<StorageEntry *> memUnique,
                                  SmallVectorImpl<StorageEntry *> &dma,
                                  SmallVectorImpl<StorageEntry *> &other,
                                  SmallVectorImpl<StorageEntry *> &scalar) {
  SmallVector<StorageEntry *> anchors;
  SmallVector<StorageEntry *> members;
  collectBankComponentEntries(dma, anchors, members);
  collectBankComponentEntries(other, anchors, members);
  collectBankComponentEntries(scalar, anchors, members);
  llvm::stable_sort(anchors, [this](StorageEntry *lhs, StorageEntry *rhs) {
    return getBankDegree(lhs) > getBankDegree(rhs);
  });
  return appendStorageEntries(
      {memUnique, anchors, dma, members, other, scalar});
}

void MemPlan::collectBankComponentEntries(
    SmallVectorImpl<StorageEntry *> &entries,
    SmallVectorImpl<StorageEntry *> &anchors,
    SmallVectorImpl<StorageEntry *> &members) const {
  SmallVector<StorageEntry *> remaining;
  for (StorageEntry *entry : entries) {
    auto info = bankMember2Anchor.find(entry);
    if (info == bankMember2Anchor.end())
      remaining.push_back(entry);
    else if (info->second == entry)
      anchors.push_back(entry);
    else
      members.push_back(entry);
  }
  entries.assign(remaining.begin(), remaining.end());
}

void MemPlan::collectBankSpecialEntries(
    SmallVectorImpl<StorageEntry *> &entries,
    SmallVectorImpl<StorageEntry *> &special) const {
  SmallVector<StorageEntry *> remaining;
  for (StorageEntry *entry : entries) {
    if (bankGraph.find(entry) != bankGraph.end() && isBankSpecialEntry(entry))
      special.push_back(entry);
    else
      remaining.push_back(entry);
  }
  entries.assign(remaining.begin(), remaining.end());
}

void MemPlan::ReorderContinuousFirstBufferOtherBufferEntry(
    SmallVector<StorageEntry *> &storageEntryVec) {
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  for (auto &storageEntry : storageEntryVec) {
    auto *it = std::find(reorderedStorageEntryVec.begin(),
                         reorderedStorageEntryVec.end(), storageEntry);
    if (it == reorderedStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
      // Add all relation entries for otherbuffer
      for (StorageEntry *relationEntry :
           storageEntry->otherBufferRelationEntries) {
        if (!relationEntry) {
          continue;
        }
        auto *relationIt =
            std::find(reorderedStorageEntryVec.begin(),
                      reorderedStorageEntryVec.end(), relationEntry);
        if (relationIt == reorderedStorageEntryVec.end()) {
          reorderedStorageEntryVec.push_back(relationEntry);
        }
      }
    }
  }
  reorderedStorageEntryVec.swap(storageEntryVec);
}

std::pair<size_t, size_t>
MemPlan::GetBufferSpaceInfo(hivm::AddressSpace &space) const {
  switch (space) {
  case hivm::AddressSpace::UB:
    return std::make_pair(ubAlignSize, ubSpaceSize);
  case hivm::AddressSpace::L1:
    return std::make_pair(l1AlignSize, l1SpaceSize);
  case hivm::AddressSpace::L0C:
    return std::make_pair(l0cAlignSize, l0cSpaceSize);
  case hivm::AddressSpace::L0A:
    if (enableMmadL0Planning)
      return std::make_pair(kA5L0ABAlignBits, l0aSpaceSize);
    break;
  case hivm::AddressSpace::L0B:
    if (enableMmadL0Planning)
      return std::make_pair(kA5L0ABAlignBits, l0bSpaceSize);
    break;
  case hivm::AddressSpace::BiasBUF:
    if (enableMmadL0Planning)
      return std::make_pair(kA5BTAlignBits, kA5BTSizeBits);
    break;
  default:
    break;
  }
  llvm::report_fatal_error("Temporarily unsupported memory buffer space !");
}

void MemPlan::LocalBufferReusePreloadBuffer(StorageEntry *entry,
                                            PlanRecHis &history) {
  if (!localSE2ReuseablePreloadSE.contains(entry) ||
      entry->alignedConstBits == 0) {
    return;
  }
  auto *preloadSE = localSE2ReuseablePreloadSE[entry];
  bool planed = llvm::any_of(
      history, [&](PlanRecord record) { return record.entry == preloadSE; });
  // If preload SE is not planned, we still can reuse these two buffers.
  if (!planed) {
    return;
  }
  preloadSE->inplaceBuffers.insert(preloadSE->inplaceBuffers.end(),
                                   entry->inplaceBuffers.begin(),
                                   entry->inplaceBuffers.end());
  for (auto *otherSE : preloadSE->otherBufferRelationEntries) {
    otherSE->inplaceBuffers.insert(otherSE->inplaceBuffers.end(),
                                   entry->inplaceBuffers.begin(),
                                   entry->inplaceBuffers.end());
  }
  // we have merge local SE into preload SE, so we can clear the inplaceBuffers
  // of local SE and set its alignedConstBits to 0 to avoid double counting in
  // the plan.
  entry->inplaceBuffers.clear();
  entry->alignedConstBits = 0;
}

LogicalResult MemPlan::trySpecLevels(SpecInfo &si, MemBoundList &outline,
                                     PlanRecHis &history, StorageEntry *entry,
                                     UBCopyBankAllocStrategy strategy,
                                     bool allowPreloadReuse) {
  for (int level = si.specLevel; level >= si.minLevel; --level) {
    // TODO: Support cross-bound allocation for LEVEL 1 related slots.
    if (level == SPEC_LEVEL_1 &&
        strategy == UBCopyBankAllocStrategy::CrossBoundSplit)
      continue;
    // Single local buffers cannot reuse preload buffers at SPEC_LEVEL_0.
    if (level == SPEC_LEVEL_1 && allowPreloadReuse)
      LocalBufferReusePreloadBuffer(entry, history);
    if (succeeded(SpecAlloc(outline, history, entry, si, level, strategy))) {
      LDBG("[MultiSpecPlan] SUCCESS at level="
           << level << " offsetByte=" << entry->bitsOffset / 8 << "\n\n");
      if (si.childIdx == si.rollBackStopIdx) {
        // Resume the maximum level after catching up with the rollback.
        LDBG("[MultiSpecPlan] reached rollBackStopIdx="
             << si.rollBackStopIdx
             << ", reset specLevel to maxLevel=" << si.maxLevel << "\n");
        si.specLevel = si.maxLevel;
      }
      ++si.childIdx;
      return success();
    }
    LDBG("[MultiSpecPlan] FAIL at level=" << level << "\n");
  }
  return failure();
}

LogicalResult MemPlan::MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                                     PlanRecHis &history, StorageEntry *entry) {
  // Stage-0 conflict verdicts are reused only while trying this entry.
  stage0ConflictMemo.clear();
  LDBG("[MultiSpecPlan] try entry childIdx="
       << si.childIdx << " needBytes=" << entry->alignedConstBits / 8
       << " from level=" << si.specLevel << " down to minLevel=" << si.minLevel
       << " historySize=" << history.size() << "\n");
  LogicalResult planResult = failure();
  if (!IsEntryBankConflict(entry)) {
    planResult = trySpecLevels(si, outline, history, entry,
                               UBCopyBankAllocStrategy::Normal, true);
  } else {
    for (UBCopyBankAllocStrategy strategy :
         {UBCopyBankAllocStrategy::BoundChange,
          UBCopyBankAllocStrategy::CrossBoundSplit}) {
      planResult = trySpecLevels(si, outline, history, entry, strategy, false);
      if (succeeded(planResult))
        break;
    }
  }
  if (failed(planResult))
    LDBG("[MultiSpecPlan] ALL levels failed for this entry; "
         "will invoke ApplyFailStrategy\n");
  return planResult;
}

void MemPlan::RecordAllocatedEntry(const StorageEntry *e) {
  assert(e && e->bufInfo);
  auto it = memscope2allocatedEntry.find(e->bufInfo->bufferScope);
  if (it == memscope2allocatedEntry.end()) {
    SmallVector<const StorageEntry *> allocatedEntry;
    allocatedEntry.push_back(e);
    memscope2allocatedEntry.insert(
        std::make_pair(e->bufInfo->bufferScope, allocatedEntry));
    return;
  }
  auto &allocatedEntry = it->second;
  bool needRecord = allocatedEntry.end() ==
                    std::find(allocatedEntry.begin(), allocatedEntry.end(), e);
  if (needRecord) {
    allocatedEntry.push_back(e);
  }
}

LogicalResult MemPlan::SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                                 StorageEntry *e, const SpecInfo &si,
                                 int localLevel,
                                 UBCopyBankAllocStrategy strategy) {
  if (std::any_of(his.begin(), his.end(),
                  [e](PlanRecord &r) { return r.entry && r.entry == e; }))
    return success();
  assert(e && "StorageEntry should not be null");
  if (e->alignedConstBits == 0) {
    e->bitsOffset = 0;
    return success();
  }

  std::optional<uint64_t> residue;
  const bool hasBankConstraint = IsEntryBankConflict(e);
  if (hasBankConstraint) {
    // History excludes peers that have been rolled back.
    auto result = analyzeBankResidue(
        e,
        [&his](const StorageEntry *peer) -> std::optional<uint64_t> {
          auto it = llvm::find_if(his, [peer](const PlanRecord &record) {
            return record.entry == peer;
          });
          if (it == his.end())
            return std::nullopt;
          return it->firstMemBound->offset;
        },
        outline.empty() ? 0 : outline.front()->offset);
    if (result.status != BankResidueStatus::Unconstrained)
      residue = result.residue;
  }
  // Without a known peer residue, allocate normally so later peers can use it.
  if (!residue)
    return SpecAllocNormal(outline, his, e, si, localLevel);

  switch (strategy) {
  case UBCopyBankAllocStrategy::Normal:
    return SpecAllocNormal(outline, his, e, si, localLevel);
  case UBCopyBankAllocStrategy::BoundChange:
    return SpecAllocConflictBoundChange(outline, his, e, si, localLevel,
                                        *residue);
  case UBCopyBankAllocStrategy::CrossBoundSplit:
    return SpecAllocConflictCrossBoundSplit(outline, his, e, si, localLevel,
                                            *residue);
  }
  llvm_unreachable("unknown UB copy bank allocation strategy");
}

MemPlan::SpecAllocStep MemPlan::TrySpecAllocSection(
    MemBoundList &outline, PlanRecHis &his, StorageEntry *e, const SpecInfo &si,
    int localLevel, MemBoundListConstIter &start, MemBoundListConstIter end,
    uint64_t size, std::optional<uint64_t> candidate,
    SmallVector<ValuePair> &stallPipelineInplacePairs) {
  const auto &bound = *end;
  // Stage-0 results are stable for the current entry and bound.
  auto key = std::make_pair(e->memoId, bound->memoId);
  auto [it, inserted] = stage0ConflictMemo.try_emplace(key, false);
  if (inserted)
    it->second = VerifyConflictStage0(e, bound, stallPipelineInplacePairs);

  const uint64_t offset = candidate.value_or((*start)->offset);
  // Do not repeat the same entry and address after rollback.
  if (IsSamePlanAsLastRollBack(offset, e->childIdx, si) || it->second)
    return SpecAllocStep::SkipCoveredBounds;
  if (size < e->alignedConstBits)
    return SpecAllocStep::Continue;

  SmallVector<uint64_t, 3> otherBufferOffsets;
  OutlineSectionInfo section(start, end, size, false);
  // Level 1 also allocates the associated other-buffer addresses.
  if (localLevel == SPEC_LEVEL_1 &&
      VerifyConflictStage1(outline, his, e, section, otherBufferOffsets))
    return SpecAllocStep::NextStart;
  if (VerifyConflictStage2(his, e, localLevel, start, outline, candidate) ||
      VerifyConflictStage3(his, e, localLevel, start, outline, candidate))
    return candidate ? SpecAllocStep::NextCandidate : SpecAllocStep::NextStart;

  e->bitsOffset = offset;
  UpdateOutline(outline, his, e, section, localLevel);
  if (localLevel == SPEC_LEVEL_1) {
    // Record the compatible other-buffer addresses in the outline as well.
    assert(!otherBufferOffsets.empty() &&
           "otherBufferOffsets should not be empty at SPEC_LEVEL_1");
    PlanRelationOtherBufferEntryAddress(otherBufferOffsets, e);
    for (uint64_t otherOffset : otherBufferOffsets)
      SpecAllocRelationOtherBufferEntry(outline, his, e, otherOffset);
  }
  if (localLevel == SPEC_LEVEL_0) {
    for ([[maybe_unused]] const auto &pair : stallPipelineInplacePairs)
      LDBG("Store/Load inplace reuse buffer pair: " << pair.first << " and "
                                                    << pair.second << "\n");
  }
  RecordAllocatedEntry(e);
  return SpecAllocStep::Success;
}

LogicalResult MemPlan::SpecAllocNormal(MemBoundList &outline, PlanRecHis &his,
                                       StorageEntry *e, const SpecInfo &si,
                                       int localLevel) {
  SmallVector<ValuePair> stallPipelineInplacePairs;
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      SpecAllocStep step =
          TrySpecAllocSection(outline, his, e, si, localLevel, start, end, size,
                              std::nullopt, stallPipelineInplacePairs);
      if (step == SpecAllocStep::Continue)
        continue;
      if (step == SpecAllocStep::Success)
        return success();
      if (step == SpecAllocStep::SkipCoveredBounds)
        start = end;
      break;
    }
  }
  return failure();
}

LogicalResult
MemPlan::SpecAllocConflictBoundChange(MemBoundList &outline, PlanRecHis &his,
                                      StorageEntry *e, const SpecInfo &si,
                                      int localLevel, uint64_t residue) {
  constexpr uint64_t bankMask = utils::kUBCopyBankSizeInBits - 1;
  SmallVector<ValuePair> stallPipelineInplacePairs;
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    const uint64_t allocOffset = (*start)->offset;
    if ((allocOffset & bankMask) != residue)
      continue;
    uint64_t size = 0;
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      if (end == start)
        size = last->offset + last->extent - allocOffset;
      else
        size += last->extent;
      SpecAllocStep step =
          TrySpecAllocSection(outline, his, e, si, localLevel, start, end, size,
                              std::nullopt, stallPipelineInplacePairs);
      if (step == SpecAllocStep::Continue)
        continue;
      if (step == SpecAllocStep::Success)
        return success();
      if (step == SpecAllocStep::SkipCoveredBounds)
        start = end;
      break;
    }
  }
  return failure();
}

static std::optional<uint64_t>
findBankCandidate(const MemoryBound &bound, uint64_t residue, uint64_t align) {
  constexpr uint64_t bankMask = utils::kUBCopyBankSizeInBits - 1;
  uint64_t alignedOffset = AlignUp(bound.offset, align);
  uint64_t adjustment =
      (residue + utils::kUBCopyBankSizeInBits - (alignedOffset & bankMask)) &
      bankMask;
  auto candidate = llvm::checkedAddUnsigned(alignedOffset, adjustment);
  if (!candidate || *candidate >= bound.offset + bound.extent ||
      *candidate % align != 0)
    return std::nullopt;
  return candidate;
}

LogicalResult MemPlan::SpecAllocConflictCrossBoundSplit(
    MemBoundList &outline, PlanRecHis &his, StorageEntry *e, const SpecInfo &si,
    int localLevel, uint64_t residue) {
  // Try the first candidate and at most two additional bank periods.
  constexpr unsigned kMaxAdditionalCandidates = 2;
  SmallVector<ValuePair> stallPipelineInplacePairs;
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    auto candidate = findBankCandidate(**start, residue, ubAlignSize);
    if (!candidate)
      continue;

    for (unsigned attempt = 0; attempt <= kMaxAdditionalCandidates; ++attempt) {
      if (*candidate < (*start)->offset ||
          *candidate - (*start)->offset >= (*start)->extent)
        break;
      if (!llvm::checkedAddUnsigned(*candidate, e->alignedConstBits))
        break;

      bool tryNextCandidate =
          *candidate % ubAlignSize != 0 ||
          IsSamePlanAsLastRollBack(*candidate, e->childIdx, si);
      if (tryNextCandidate) {
        candidate = llvm::checkedAddUnsigned(
            *candidate, static_cast<uint64_t>(utils::kUBCopyBankSizeInBits));
        if (!candidate)
          break;
        continue;
      }
      uint64_t size = 0;
      for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
        std::shared_ptr<MemoryBound> last = *end;
        if (end == start)
          size = last->offset + last->extent - *candidate;
        else
          size += last->extent;
        SpecAllocStep step =
            TrySpecAllocSection(outline, his, e, si, localLevel, start, end,
                                size, *candidate, stallPipelineInplacePairs);
        if (step == SpecAllocStep::Continue)
          continue;
        if (step == SpecAllocStep::Success)
          return success();
        if (step == SpecAllocStep::SkipCoveredBounds)
          start = end;
        else if (step == SpecAllocStep::NextCandidate)
          tryNextCandidate = true;
        break;
      }
      if (!tryNextCandidate)
        break;
      candidate = llvm::checkedAddUnsigned(
          *candidate, static_cast<uint64_t>(utils::kUBCopyBankSizeInBits));
      if (!candidate)
        break;
    }
  }
  return failure();
}

LoopLikeOpInterface
MemPlan::GetBufferParentLoop(const SmallVector<Value> &buffers) {
  llvm::SmallSet<LoopLikeOpInterface, 1> parentLoopVec;
  for (auto buffer : buffers) {
    if (!buffer.getDefiningOp()) {
      assert(isa<scf::ForOp>(buffer.getParentBlock()->getParentOp()) ||
             isa<scf::WhileOp>(buffer.getParentBlock()->getParentOp()));
      // Init args and region iter arg are inplace, ignore Region Iter Arg
      // without DefineOp.
      continue;
    }
    LoopLikeOpInterface bufferParentLoop = getParentLoop(buffer);
    if (bufferParentLoop) {
      parentLoopVec.insert(bufferParentLoop);
    } else {
      return nullptr;
    }
  }
  if (parentLoopVec.size() == 1) {
    return *parentLoopVec.begin();
  }
  return nullptr;
}

bool MemPlan::VerifyConflictStage1(
    MemBoundList &outline, PlanRecHis &his, StorageEntry *e,
    const OutlineSectionInfo &outlineInfo,
    SmallVectorImpl<uint64_t> &otherBufferOffsets) {
  if (outlineInfo.mem_start != outlineInfo.mem_end) {
    return true;
  }
  auto *reuseBoundStorageEntry = (*outlineInfo.mem_start)->lastStorageEntry;
  if (!reuseBoundStorageEntry) {
    // This area has not been planed, so there is no need to consider it.
    return true;
  }

  // Collect all available otherbuffer entries for the current reuse bound
  // storage entry. In the multi-buffer case, there may be multiple otherbuffer
  // entries (stored in otherBufferRelationEntries).
  SmallVector<StorageEntry *> otherBufferEntries;
  if (reuseBoundStorageEntry->multiBufferNum > 1) {
    for (StorageEntry *relationEntry :
         reuseBoundStorageEntry->otherBufferRelationEntries) {
      if (relationEntry && relationEntry->bitsOffset != 0) {
        otherBufferEntries.push_back(relationEntry);
      }
    }
  } else {
    auto iter =
        firstBufferEntry2RelationOtherBufferEntry.find(reuseBoundStorageEntry);
    if (iter != firstBufferEntry2RelationOtherBufferEntry.end()) {
      for (const std::unique_ptr<StorageEntry> &otherBufferEntry :
           iter->second) {
        if (otherBufferEntry && otherBufferEntry->bitsOffset != 0)
          otherBufferEntries.push_back(otherBufferEntry.get());
      }
    }
  }

  if (otherBufferEntries.empty()) {
    // No valid otherbuffer entry has been allocated on this reuse bound
    // storage entry, SPEC_LEVEL_1 reuse is not applicable.
    return true;
  }

  // The buffer to be planned and the reuse bound storage entry must belong to
  // the same loop, otherwise they cannot participate in firstbuffer-otherbuffer
  // reuse.
  auto parentLoop1 = GetBufferParentLoop(e->inplaceBuffers);
  auto parentLoop2 =
      GetBufferParentLoop(reuseBoundStorageEntry->inplaceBuffers);
  if (!(parentLoop1 != nullptr && parentLoop2 != nullptr &&
        parentLoop1 == parentLoop2)) {
    // Cannot be reused under the same loop.
    return true;
  }

  // Two situations:
  // Single buffer reuse multi buffer
  // Multi-buffer reuse multi buffer

  // Multi-buffer case: require enough multibuffer entries for all buffer
  // instances, and only first buffer(not other relation entries) can reuse in
  // level1.
  auto otherBufferEntriesSize = otherBufferEntries.size();
  if (e->multiBufferNum > 1 &&
      (otherBufferEntriesSize < e->multiBufferNum - 1 ||
       e->otherBufferRelationEntries.empty())) {
    // Not enough historical multibuffer entries to match current multi-buffer
    // requirement, or current entry is not first buffer.
    return true;
  }

  // Check each required multibuffer entry one by one. If any multibuffer
  // entry conflicts with historical records at its offset, the whole
  // multi-buffer reuse fails. Only when all required multibuffer entries are
  // conflict-free can we reuse (return false).
  for (uint32_t i = 0; i < otherBufferEntriesSize; ++i) {
    StorageEntry *multiRelationMultiBufferEntry = otherBufferEntries[i];
    if (!multiRelationMultiBufferEntry) {
      return true;
    }
    uint64_t multiBufferOffset = multiRelationMultiBufferEntry->bitsOffset;
    bool conflict = std::any_of(
        his.begin(), his.end(), [multiBufferOffset, e, this](PlanRecord &r) {
          return this->IsBufferLifeVecConflict(r, multiBufferOffset, e);
        });
    if (!conflict) {
      otherBufferOffsets.push_back(multiBufferOffset);
    } else {
      // As long as one instance conflicts, the whole multi-buffer reuse fails.
      otherBufferOffsets.clear();
      return true;
    }
  }

  // All required multibuffer entries are conflict-free; otherBufferOffsets was
  // already filled in the loop above.
  return false;
}

void MemPlan::SpecAllocRelationOtherBufferEntry(MemBoundList &outline,
                                                PlanRecHis &his,
                                                StorageEntry *e,
                                                uint64_t offset) {
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    // Find the MemBound corresponding to the multibuffer offset.
    if ((*start)->offset != offset) {
      continue;
    }
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      if (size < e->alignedConstBits) {
        continue;
      }
      StorageEntry *otherBufferStorageEntry = nullptr;
      auto iter = firstBufferEntry2RelationOtherBufferEntry.find(e);
      if (iter != firstBufferEntry2RelationOtherBufferEntry.end()) {
        for (const std::unique_ptr<StorageEntry> &otherBufferEntry :
             iter->second) {
          if (otherBufferEntry && otherBufferEntry->bitsOffset == offset) {
            otherBufferStorageEntry = otherBufferEntry.get();
            break;
          }
        }
      }
      if (!otherBufferStorageEntry && e->multiBufferNum > 1) {
        for (StorageEntry *relationEntry : e->otherBufferRelationEntries) {
          if (relationEntry && relationEntry->bitsOffset == offset) {
            otherBufferStorageEntry = relationEntry;
            break;
          }
        }
      }
      assert(otherBufferStorageEntry && "otherBuffer Storage Entry not found!");
      UpdateOutline(outline, his, otherBufferStorageEntry,
                    OutlineSectionInfo(start, end, size, true), SPEC_LEVEL_1);
      return;
    }
  }
}

bool MemPlan::IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                                      const StorageEntry *e) const {
  if ((r.firstMemBound->offset + r.allExtent <= offset) ||
      (r.firstMemBound->offset >= offset + e->alignedConstBits)) {
    return false;
  }
  return r.entry->bufferLifeVec.intersects(e->bufferLifeVec);
}

void MemPlan::PlanRelationOtherBufferEntryAddress(
    llvm::ArrayRef<uint64_t> otherBufferOffsets, StorageEntry *e) {
  // First loop: assign offsets to existing otherBufferRelationEntries.
  // For each multibuffer index from 1 to multiBufferNum, the corresponding
  // relation entry (at index i-1) gets the offset at index i-1.
  for (size_t i = 1; i < e->multiBufferNum; ++i) {
    if (i - 1 >= e->otherBufferRelationEntries.size() ||
        i - 1 >= otherBufferOffsets.size()) {
      continue;
    }
    if (StorageEntry *re = e->otherBufferRelationEntries[i - 1]) {
      re->bitsOffset = otherBufferOffsets[i - 1];
    }
  }

  // Second loop: create new StorageEntries for the remaining offsets (from
  // index multiBufferNum-1 to the end of otherBufferOffsets) and store them
  // in firstBufferEntry2RelationOtherBufferEntry as a vector.
  auto &vec = firstBufferEntry2RelationOtherBufferEntry[e];
  vec.clear();
  vec.reserve(otherBufferOffsets.size());
  for (size_t i = e->multiBufferNum - 1; i < otherBufferOffsets.size(); ++i) {
    auto entry = std::make_unique<StorageEntry>(nextEntryMemoId++);
    entry->bufInfo = e->bufInfo;
    entry->bufferLifeVec = e->bufferLifeVec;
    entry->alignedConstBits = e->alignedConstBits;
    entry->inplaceBuffers = e->inplaceBuffers;
    entry->multiBufferNum = e->multiBufferNum;
    entry->bitsOffset = otherBufferOffsets[i];
    vec.push_back(std::move(entry));
  }
}

bool MemPlan::VerifyConflictStageCommon(
    PlanRecHis &his, const StorageEntry *e, MemBoundListConstIter &start,
    const MemBoundList &outline,
    std::function<bool(const StorageEntry *, const StorageEntry *)>
        conflictChecker,
    std::optional<uint64_t> candidate) {
  // Check the actual proposed range against previously allocated entries.
  auto hasConflict = [&his, e, &conflictChecker](uint64_t begin, uint64_t end) {
    return std::any_of(
        his.begin(), his.end(),
        [begin, end, e, &conflictChecker](PlanRecord &r) {
          return (r.firstMemBound->offset + r.allExtent > begin) &&
                 (r.firstMemBound->offset < end) && conflictChecker(r.entry, e);
        });
  };
  if (candidate) {
    const uint64_t entryEnd = *candidate + e->alignedConstBits;
    // Check each covered part of the outline against the planned entries.
    for (auto iter = start; iter != outline.end(); ++iter) {
      const uint64_t boundBegin = (*iter)->offset;
      if (boundBegin >= entryEnd)
        break;
      const uint64_t begin = std::max(*candidate, boundBegin);
      const uint64_t end = std::min(entryEnd, boundBegin + (*iter)->extent);
      assert(begin < end && "candidate must intersect the outline bound");
      if (hasConflict(begin, end))
        return true;
    }
    return false;
  }

  bool touchMemCanUse = false;
  MemBoundListConstIter foundMem;

  for (auto iter = start; iter != outline.end(); ++iter) {
    bool conflict =
        hasConflict((*iter)->offset, (*iter)->offset + e->alignedConstBits);
    // if conflict, continue finding the first bound that has no conflict
    // if last bound do not meet the size, continue
    if (conflict ||
        ((*iter == outline.back()) && (*iter)->extent < e->alignedConstBits)) {
      continue;
    }
    touchMemCanUse = true;
    foundMem = iter;
    break;
  }

  if (touchMemCanUse) {
    bool conflict = (foundMem != start);
    start = conflict ? --foundMem : start;
    return conflict;
  }
  // if cannot find a bound that has no conflict with current entry,
  return true;
}

bool MemPlan::VerifyConflictStage3(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline,
                                   std::optional<uint64_t> candidate) {
  if (specLevel != SPEC_LEVEL_3) {
    return false;
  }
  return VerifyConflictStageCommon(
      his, e, start, outline,
      [this](const StorageEntry *e1, const StorageEntry *e2) {
        return this->PipeConflict(e1, e2, this->pipeDmaConflictMap);
      },
      candidate);
}

bool MemPlan::VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline,
                                   std::optional<uint64_t> candidate) {
  if (specLevel != SPEC_LEVEL_2) {
    return false;
  }
  return VerifyConflictStageCommon(
      his, e, start, outline,
      [this](const StorageEntry *e1, const StorageEntry *e2) {
        return this->PipeConflictInSameLoop(e1, e2);
      },
      candidate);
}

bool MemPlan::PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                           DenseMap<StorageEntryIdPair, bool> &conflictMap) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  auto sePair = std::make_pair(e1->memoId, e2->memoId);
  auto [iter, isInserted] = conflictMap.try_emplace(sePair, false);
  if (!isInserted) {
    return iter->second;
  }

  for (const Value var1 : e1->inplaceBuffers) {
    for (const Value var2 : e2->inplaceBuffers) {
      bool conflict = dmaFirstPipelineOpt.BufferPipeConflict(var1, var2);
      if (conflict) {
        iter->second = true;
        return true;
      }
    }
  }
  return false;
}

bool MemPlan::PipeConflictInSameLoop(const StorageEntry *e1,
                                     const StorageEntry *e2) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  auto sePair = std::make_pair(e1->memoId, e2->memoId);
  auto [iter, isInserted] = sameLoopConflictMap.try_emplace(sePair, false);
  if (!isInserted) {
    return iter->second;
  }
  auto parentLoop1 = GetBufferParentLoop(e1->inplaceBuffers);
  auto parentLoop2 = GetBufferParentLoop(e2->inplaceBuffers);
  if (parentLoop1 != parentLoop2) {
    return false;
  }
  // Cannot be reused under the same region.
  iter->second = true;
  return true;
}

void MemPlan::UpdateOutline(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            int localLevel) const {
  if (e == nullptr) {
    return;
  }
  auto sectionStart = outlineInfo.mem_start;
  const auto &firstBound = *sectionStart;
  MemBoundListConstIter end = outlineInfo.mem_end;
  // The selected bounds are [sectionStart, end] (inclusive). The entry may
  // start inside the first bound and end inside the last bound:
  //
  // firstBound->offset   e->bitsOffset           entryEnd        sectionEnd
  //        |                   |                     |                 |
  //        v                   v                     v                 v
  //        |------ head -------|------ entry --------|----- tail ------|
  //        |---- first bound ----| ... |--------- last bound ---------|
  //                            |<--------- outlineInfo.size ---------->|
  //                            |<-- e->alignedConstBits -->|<- tailSize ->|
  //
  // The caller computes size from e->bitsOffset, not firstBound->offset.
  // Thus tailSize == sectionEnd - entryEnd and does not include the head.
  const uint64_t entryEnd = e->bitsOffset + e->alignedConstBits;
  std::shared_ptr<MemoryBound> last = *end;
  const uint64_t sectionEnd = last->offset + last->extent;
  assert(e->bitsOffset >= firstBound->offset && entryEnd >= e->bitsOffset &&
         entryEnd <= sectionEnd && "split section must contain the entry");
  assert(outlineInfo.size >= e->alignedConstBits &&
         "section must be large enough for the entry");
  assert(outlineInfo.size == sectionEnd - e->bitsOffset &&
         "section size must start at the entry offset");
  const uint64_t tailSize = outlineInfo.size - e->alignedConstBits;
  ++end;
  std::shared_ptr<MemoryBound> bound;
  SmallVector<std::shared_ptr<MemoryBound>> splitBound;
  // split e, to get Boundbound
  if (splitOutline) {
    // add splitBound by splitting e to section
    // Each entry part merges E with only the lifetime of the bound it covers:
    // old:    |-- bound 1: L1 --|-- bound 2: L2 --|
    // entry:        |------------ E --------|
    // result: |head |-- E + L1 -|-- E + L2 -| tail|
    AddMemBoundInSectionalWay(e, sectionStart, end, e->bitsOffset, entryEnd,
                              splitBound);
  } else {
    // Without sectional splitting, merge all covered lifetimes into one
    // entry bound (head and tail retain their original lifetimes):
    // old:    |-- bound 1: L1 --|-- bound 2: L2 --|
    // entry:        |------------ E --------|
    // result: |head |---- E + L1 + L2 ------| tail|
    BufferLifeVec life = e->bufferLifeVec;
    MergeBufferLife(sectionStart, end, life);
    splitBound.emplace_back(std::make_shared<MemoryBound>(
        life, e->bitsOffset, e->alignedConstBits, e, nextBoundMemoId++));
  }
  assert(!splitBound.empty() && "entry must cover at least one bound");

  // Keep the part of the first bound before e as a head.
  const uint64_t headSize = e->bitsOffset - firstBound->offset;
  if (headSize > 0) {
    LDBG("[UpdateOutline] split head, entry=" << e->memoId << "\n");
    std::shared_ptr<MemoryBound> head = std::make_shared<MemoryBound>(
        firstBound->bufferLifeVec, firstBound->offset, headSize,
        firstBound->lastStorageEntry, nextBoundMemoId++);
    outline.insert(sectionStart, head);
  }
  // insert tail memory bound
  if (tailSize > 0) {
    bound = std::make_shared<MemoryBound>(
        last->bufferLifeVec, last->offset + last->extent - tailSize, tailSize,
        last->lastStorageEntry, nextBoundMemoId++);
    end = outline.insert(end, bound);
  }
  // insert split e memory bound
  for (int i = static_cast<int>(splitBound.size()) - 1; i >= 0; --i) {
    end = outline.insert(end, splitBound[i]);
  }
  // record the current plan of first split entry in his
  his.emplace_back(PlanRecord{localLevel,
                              e->childIdx,
                              tailSize > 0,
                              headSize > 0,
                              splitBound.size(),
                              e,
                              e->alignedConstBits,
                              splitBound[0],
                              {},
                              outlineInfo.isDirectlyRollback});
  e->appliedSpecLevel = localLevel;
  PlanRecord &r = his.back();
  r.replaced.splice(r.replaced.begin(), outline, sectionStart, end);
}

void MemPlan::AddMemBoundInSectionalWay(
    StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
    uint64_t candidate, uint64_t entryEnd,
    SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const {
  // At the bound start:
  // old:   |------ bound 1 ------|------ bound 2 ------|
  // entry: |---------------- e -------------|
  // parts: |------ e part 1 -----|-- e part 2|  tail   |
  //        ^                     ^          ^         ^
  //        begin 1               begin 2    entryEnd  boundEnd 2
  //                              boundEnd 1 splitEnd 2
  //                              splitEnd 1
  //
  // At an interior candidate:
  // old:   |------ bound 1 ------|------ bound 2 ------|
  // entry:        |------------- e ---------|
  // parts: | head |-- e part 1 --|-- e part 2|  tail   |
  //               ^              ^          ^         ^
  //               begin 1       begin 2    entryEnd  boundEnd 2
  //                              boundEnd 1 splitEnd 2
  //                              splitEnd 1
  for (auto iter = start; iter != end; ++iter) {
    // Each covered part retains the original bound's lifetimes.
    BufferLifeVec life = e->bufferLifeVec;
    life.append((*iter)->bufferLifeVec);
    // Intersect the entry with this bound; head and tail are handled by
    // UpdateOutline separately.
    const uint64_t begin = std::max(candidate, (*iter)->offset);
    const uint64_t boundEnd = (*iter)->offset + (*iter)->extent;
    const uint64_t splitEnd = std::min(entryEnd, boundEnd);
    if (begin >= splitEnd)
      continue;
    splitBound.emplace_back(std::make_shared<MemoryBound>(
        life, begin, splitEnd - begin, e, nextBoundMemoId++));
  }
}

inline void MemPlan::MergeBufferLife(MemBoundList::const_iterator start,
                                     MemBoundList::const_iterator end,
                                     BufferLifeVec &newLife) const {
  for (auto it = start; it != end; ++it)
    newLife.append((*it)->bufferLifeVec);
}

bool MemPlan::IsSamePlanAsLastRollBack(uint64_t allocOffset, int curChildIdx,
                                       const SpecInfo &si) const {
  return curChildIdx == si.rollbackIdx && allocOffset == si.rollbackAddr;
}

// spec_level == SPEC_LEVEL_0
inline bool MemPlan::VerifyConflictStage0(
    StorageEntry *e, const std::shared_ptr<MemoryBound> &last,
    SmallVector<ValuePair> &stallPipelineInplacePairs) {
  if (e->bufferLifeVec.empty() || last->bufferLifeVec.empty()) {
    return false;
  }
  // if buffer in b1 has mem_unique, b1 will have only one element.
  auto bufferInfoIt1 = bufferInfos.find(e->bufferLifeVec.frontBuffer());
  auto bufferInfoIt2 = bufferInfos.find(last->bufferLifeVec.frontBuffer());
  assert(bufferInfoIt1 != bufferInfos.end() &&
         bufferInfoIt2 != bufferInfos.end());
  if (bufferInfoIt1->second.memoryUnique ||
      bufferInfoIt2->second.memoryUnique) {
    return true;
  }
  // Wider than a point is a real overlap. A point touch, including a [t, t]
  // interval sitting inside a longer one, is inplace only when the buffers
  // alive at that time are an inplacable pair.
  for (auto [lo, hi] : e->bufferLifeVec.timeOverlaps(last->bufferLifeVec)) {
    if (lo != hi)
      return true;
    SmallVector<BufferLifeView> left = e->bufferLifeVec.buffersAt(lo);
    SmallVector<BufferLifeView> right = last->bufferLifeVec.buffersAt(lo);
    // Empty -> Intersection point is not at the edge of interval -> it is a
    // real overlap.
    if (left.empty() || right.empty())
      return true;
    for (const BufferLifeView &life1 : left) {
      for (const BufferLifeView &life2 : right) {
        if (!IsInplacableBufferPairMatched(life1.buffer, life2.buffer,
                                           stallPipelineInplacePairs))
          return true;
      }
    }
  }
  // Conflict pairs are all in inplacableBufferPairs, which means can reuse
  // memory, return not conflict. But it will cause pipeline stalls.
  return false;
}

bool MemPlan::IsInplacableBufferPairMatched(
    Value buffer, Value conflictBuffer,
    SmallVector<ValuePair> &matchedPairs) const {
  auto iter = buffer2storageEntry.find(buffer);
  auto iter2 = buffer2storageEntry.find(conflictBuffer);
  assert(iter != buffer2storageEntry.end() &&
         iter2 != buffer2storageEntry.end());
  auto *storageEntry = iter->second;
  auto *conflictStorageEntry = iter2->second;
  assert(storageEntry != nullptr && conflictStorageEntry != nullptr);
  for (const auto &inplaceBuffer : storageEntry->inplaceBuffers) {
    for (const auto &conflictInplaceBuffer :
         conflictStorageEntry->inplaceBuffers) {
      for (const auto &pair : inplacableBufferPairs) {
        if ((inplaceBuffer == pair.first &&
             conflictInplaceBuffer == pair.second) ||
            (inplaceBuffer == pair.second &&
             conflictInplaceBuffer == pair.first)) {
          matchedPairs.emplace_back(inplaceBuffer, conflictInplaceBuffer);
          return true;
        }
      }
    }
  }
  return false;
}

PlanStatus MemPlan::ApplyFailStrategy(StatusWrapper &statusWrapper,
                                      const size_t maxBits) {
  RollBackForAllocFail(statusWrapper, maxBits);
  // second class rollback, level 1 --> 0
  if (statusWrapper.si->specLevel > SPEC_LEVEL_0 &&
      statusWrapper.si->childIdx >= 0) {
    statusWrapper.si->specLevel--;
    return PlanStatus::CONTINUE_PLAN;
  }
  if (!splitOutline) {
    // roll back to origin again, enable split outline.
    splitOutline = true;
    return PlanStatus::RESTART_NEW_PLAN;
  }
  return PlanStatus::PLAN_FAILED;
}

void MemPlan::ReportAllocatedEntryDebugInfo(
    const StorageEntry *rootStorageEntry, bool isFail) const {
#ifndef NDEBUG
  auto printRecord = [this](const StorageEntry *entry) {
    uint64_t needByte =
        (entry->alignedConstBits + utils::kBitsToByte - 1) / utils::kBitsToByte;
    uint64_t offsetByte =
        (entry->bitsOffset + utils::kBitsToByte - 1) / utils::kBitsToByte;
    ReportCurEntryDebugInfo(entry);
    LDBG(", offset: " << offsetByte);
    LDBG(", extent: " << needByte);
    LDBG(", buffer life: ");
    entry->bufferLifeVec.forEach([](BufferLifeView bufferLife) {
      LDBG("[" << bufferLife.allocTime << "-" << bufferLife.freeTime << "], ");
    });
  };
  LDBG("--------------------------BUFFER ALLOCATE "
       "START-------------------------- "
       << "\n"
       << "\n");
  LDBG("  BUFFER ALLOCATE START: " << rootStorageEntry->bufInfo->bufferScope
                                   << "\n");

  auto it =
      memscope2allocatedEntry.find(rootStorageEntry->bufInfo->bufferScope);
  if (it != memscope2allocatedEntry.end() && !it->second.empty()) {
    const auto &allocatedEntry = it->second;
    for (auto &entry : allocatedEntry) {
      printRecord(entry);
      LDBG("\n");
    }
    if (isFail) {
      size_t num = allocatedEntry.size() - 1;
      assert(rootStorageEntry->mergedChildren.size() > num);
      const StorageEntry *failedSe = rootStorageEntry->mergedChildren[num];
      printRecord(failedSe);
      LDBG("alloc fail,because exceed bound of memory "
           << rootStorageEntry->bufInfo->bufferScope << "\n");
    }
  }
  LDBG("  BUFFER ALLOCATE END \n");
  LDBG("\n"
       << "--------------------------BUFFER ALLOCATE "
          "END-------------------------- "
       << "\n");
#endif
}

LogicalResult MemPlan::DynamicSetUbSpaceSize(
    hacc::HACCTargetDeviceSpecInterface specInterface, func::FuncOp funcOp) {
  auto aUBSize =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::UB_SIZE);
  ubSpaceSize = cast<IntegerAttr>(aUBSize.getValue()).getInt();
  if (!hacc::utils::isRegBasedArch(funcOp->getParentOfType<ModuleOp>())) {
    return success();
  }
  if (const auto vfModeAttr =
          funcOp->getAttrOfType<VFModeAttr>(VFModeAttr::name)) {
    if (vfModeAttr.getValue() == VFMode::SIMT ||
        vfModeAttr.getValue() == VFMode::MIX) {
      constexpr int numBitsInByte = 8;
      constexpr int numByteInKB = 1024;
      auto simtVFUbSize = this->simtVFDynamicSize * numByteInKB * numBitsInByte;
      auto minimalDCacheSize =
          cast<IntegerAttr>(specInterface
                                .getSpecForIdentifierEnum(
                                    hacc::DeviceSpec::MINIMAL_D_CACHE_SIZE)
                                .getValue())
              .getInt();
      auto maximumDCacheSize =
          cast<IntegerAttr>(specInterface
                                .getSpecForIdentifierEnum(
                                    hacc::DeviceSpec::MAXIMUM_D_CACHE_SIZE)
                                .getValue())
              .getInt();
      if (simtVFUbSize < (ubSpaceSize - maximumDCacheSize) ||
          simtVFUbSize > (ubSpaceSize - minimalDCacheSize)) {
        funcOp->emitError()
            << "PlanMemory Fail : SimtVFDynamicSize have to in range ["
            << (ubSpaceSize - maximumDCacheSize) / numByteInKB / numBitsInByte
            << ", "
            << (ubSpaceSize - minimalDCacheSize) / numByteInKB / numBitsInByte
            << "], but got " << simtVFDynamicSize << "!";
        return failure();
      }
      ubSpaceSize = simtVFUbSize;
      return success();
    }
  }
  return success();
}

LogicalResult MemPlan::InitMemSpecsFromModule(func::FuncOp funcOp) {
  auto moduleOp = utils::getTopLevelModuleOp(funcOp);
  auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
  if (!maybeSpecInterface.has_value()) {
    return failure();
  }
  auto specInterface = maybeSpecInterface.value();

  if (failed(DynamicSetUbSpaceSize(specInterface, funcOp))) {
    return failure();
  }

  auto aL1Size =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L1_SIZE);
  l1SpaceSize = cast<IntegerAttr>(aL1Size.getValue()).getInt();
  auto aL0CSize =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L0C_SIZE);
  l0cSpaceSize = cast<IntegerAttr>(aL0CSize.getValue()).getInt();
  auto aUBAlignSize =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::UB_ALIGN_SIZE);
  ubAlignSize = cast<IntegerAttr>(aUBAlignSize.getValue()).getInt();
  auto aL1AlignSize =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L1_ALIGN_SIZE);
  l1AlignSize = cast<IntegerAttr>(aL1AlignSize.getValue()).getInt();
  auto aL0cAlignSize =
      specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L0C_ALIGN_SIZE);
  l0cAlignSize = cast<IntegerAttr>(aL0cAlignSize.getValue()).getInt();
  if (enableMmadL0Planning) {
    auto aL0ASize =
        specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L0A_SIZE);
    l0aSpaceSize = cast<IntegerAttr>(aL0ASize.getValue()).getInt();
    auto aL0BSize =
        specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::L0B_SIZE);
    l0bSpaceSize = cast<IntegerAttr>(aL0BSize.getValue()).getInt();
    l0cAlignSize = std::max(l0cAlignSize, kA5L0CAlignBits);
  }
  return success();
}

void MemPlan::RollBackForAllocFail(StatusWrapper &statusWrapper,
                                   const size_t maxBits) {
  while (ContinueRollBack(statusWrapper)) {
    RollBackForAllocFailInner(statusWrapper, maxBits);
  }
}

bool MemPlan::ContinueRollBack(const StatusWrapper &statusWrapper) const {
  return (!statusWrapper.hasEnoughRollBackSize) &&
         (!statusWrapper.history.empty() && (!statusWrapper.outline.empty()));
}

// Level-1 jointly allocated other slots (isDirectlyRollback) always roll
// with the first buffer. Independently planned mb-other slots roll with
// the first buffer only when both were planned at the same level; if the
// levels differ they may be retried independently (e.g. other at L2 while
// first already at L0).
bool MemPlan::ShouldRollbackMuiltiBuffer(const PlanRecord &r) const {
  if (r.isDirectlyRollback) {
    return true;
  }
  if (r.entry->multiBufferNum > 1 &&
      r.entry->otherBufferRelationEntries.empty()) {
    StorageEntry *first = r.entry->firstBufferRelationEntry;
    if (first && first->appliedSpecLevel >= SPEC_LEVEL_0 &&
        r.specLevel == first->appliedSpecLevel) {
      return true;
    }
  }
  return false;
}

void MemPlan::RollBackForAllocFailInner(StatusWrapper &statusWrapper,
                                        const size_t maxBits) {
  auto &si = statusWrapper.si;
  if (si->childIdx > si->rollBackStopIdx) {
    si->rollBackStopIdx = si->childIdx;
  }
  // Check whether the container is empty before accessing "history"
  while (!statusWrapper.history.empty()) {
    PlanRecord r =
        RollbackOutline(statusWrapper.history, statusWrapper.outline);
    auto iter = firstBufferEntry2RelationOtherBufferEntry.find(r.entry);
    if (iter != firstBufferEntry2RelationOtherBufferEntry.end()) {
      firstBufferEntry2RelationOtherBufferEntry.erase(iter);
    }
    if (ShouldRollbackMuiltiBuffer(r)) {
      continue;
    }
    si->childIdx = r.childIdx;
    si->specLevel = r.specLevel;
    if (si->specLevel > si->minLevel) {
      // record rollback info: index and address
      si->rollbackAddr =
          si->childIdx == -1
              ? UINT64_MAX
              : statusWrapper.RootE->mergedChildren[si->childIdx]->bitsOffset;
      si->rollbackIdx = si->childIdx;
      if (statusWrapper.si->rollbackAddr + statusWrapper.alignedConstBits >
          maxBits) {
        continue;
      }
      statusWrapper.hasEnoughRollBackSize = true;
      break;
    }
  }
}

PlanRecord MemPlan::RollbackOutline(PlanRecHis &history,
                                    MemBoundList &outline) const {
  auto r = history.back();
  auto it = std::find(outline.begin(), outline.end(), r.firstMemBound);
  // |--head--|--split entry--|--tail--|
  // erase head
  if (r.headed) {
    LDBG("[RollbackOutline] restore head, entry=" << r.entry->memoId << "\n");
    it--;
    it = outline.erase(it);
  }
  // erase split entry
  for (size_t i = 0; i < r.splitNums; i++) {
    it = outline.erase(it);
  }
  // erase tail
  if (r.tailed) {
    it = outline.erase(it);
  }
  // restore outline and replaced
  outline.splice(it, r.replaced);
  history.pop_back();
  return r;
}

namespace {

/// Outcome of one memory-planning attempt.
enum class PlanAttemptStatus {
  /// The attempt produced a valid plan.
  Succeeded,
  /// The attempt failed, but a later attempt may still succeed.
  Retryable,
  /// The attempt ended early in a way that no retry can recover from.
  EarlyEnd
};

/// Describes one memory-planning retry stage. The fallback is applied once
/// before planning is retried with different deterministic shuffle seeds.
struct PlanRetryStage {
  std::function<void(func::FuncOp &)> fallback;
  int attemptCount;
};

constexpr int kDefaultAttemptsPerStage = 3;

struct PlanMemoryPass : public impl::PlanMemoryBase<PlanMemoryPass> {
public:
  explicit PlanMemoryPass(const PlanMemoryOptions &planMemoryOption)
      : PlanMemoryBase(planMemoryOption) {}

  void runOnOperation() override;

private:
  LogicalResult validateUBCopyBankArch(ModuleOp moduleOp);
  std::unique_ptr<MemPlan> initPlanner() const;
  void markTempBufForMemoryDisplay(func::FuncOp funcOp);

  // Whether the current function uses standalone MMAD L0 planning. Selected
  // once per function before planning starts.
  bool enableMmadL0Planning = false;

  // VF inplace-reuse analysis for the planned module, built once at the
  // start of runOnOperation.
  VFInplaceReuseAnalysis vfInplaceReuseAnalysis;

  // Returning nullopt means that memory plan for funcOp failed. On success,
  // first is buffer2Offsets and second is preload-local alloc roots.
  std::optional<PlanResult> planMemoryForFuncOp(func::FuncOp &funcOp);

  // Runs a single planning attempt. Writes the plan to `planned` and returns
  // Succeeded on success, Retryable when the attempt failed but can be
  // retried, and EarlyEnd when the failure is not recoverable.
  PlanAttemptStatus planMemoryAttempt(func::FuncOp &funcOp, int attempt,
                                      int totalAttemptCount,
                                      TcbPoolPlan &tcbPoolPlan,
                                      bool startTcbPoolPlan,
                                      PlanResult &planned);

  // Retry fallback: remove the multi-buffer marks of AIV functions in mix
  // modules, but keep those of preload local buffers.
  void disableAIVMultibuffer(func::FuncOp &funcOp);

  void updateId2Offsets(func::FuncOp &funcOp, Buffer2Offsets &buffer2Offsets,
                        DenseMap<int32_t, SmallVector<uint64_t>> &id2Offsets);

  void updateBuffer2OffsetsForFuncOp(
      func::FuncOp &funcOp, Buffer2Offsets &buffer2Offsets,
      DenseMap<int32_t, SmallVector<uint64_t>> &id2Offsets);

  LogicalResult
  populateBufferAddressToAllocOp(func::FuncOp &funcOp,
                                 const Buffer2Offsets &buffer2Offsets,
                                 const DenseSet<Value> &preloadLocalBuffers) {
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN) {
      // Convert every memref.alloc into an hivm.hir.pointer_cast bound to its
      // planned address(es).
      return walkAllocToPointerCast(funcOp, buffer2Offsets,
                                    preloadLocalBuffers);
    }
    assert(this->memMode == MemPlanMode::GLOBAL_WORKSPACE_PLAN);
    // Attach the planned offset(s) to every memref_ext.alloc_workspace.
    return walkUpdateAllocWorkspaceOffset(funcOp, buffer2Offsets);
  }
  bool checkSimilarPointerCastOps(hivm::PointerCastOp op1,
                                  hivm::PointerCastOp op2) const;
  void fixMultibufferEnabledPointerCastOps(Operation *funcOp) const;
};
} // namespace

void PlanMemoryPass::disableAIVMultibuffer(func::FuncOp &funcOp) {
  funcOp.walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttr(hivm::MultiBufferAttr::name) &&
        !markOp->hasAttr(hivm::PreloadLocalBufferAttr::name)) {
      markOp->removeAttr(hivm::MultiBufferAttr::name);
      if (markOp.isAttrEmpty()) {
        markOp.erase();
      }
    }
  });
  LDBG("Disabled MultiBuffer in AIV function by removing all multiBuffer "
       << "attrs as a retry fallback\n");
}

std::optional<PlanResult>
PlanMemoryPass::planMemoryForFuncOp(func::FuncOp &funcOp) {
  // Add a attr to the memref alloc for the tempbuf.
  if (this->enableMemoryDisplay) {
    markTempBufForMemoryDisplay(funcOp);
  }

  // Each retry stage owns one fallback, which is applied once when the stage
  // starts and stays in effect for all later stages. Within a stage, planning
  // is retried with different shuffle seeds.
  SmallVector<PlanRetryStage> retryStages = {
      // First try without applying a fallback.
      {[](func::FuncOp &) {}, kDefaultAttemptsPerStage},
  };
  auto moduleOp = getOperation();
  // Only AIV functions in mix (CV) modules can drop multi-buffering.
  if (isMixModule(moduleOp) &&
      queryFuncCoreType(funcOp) == TFuncCoreType::AIV) {
    retryStages.push_back(
        {[this](func::FuncOp &funcOp) { disableAIVMultibuffer(funcOp); },
         kDefaultAttemptsPerStage});
  } else {
    retryStages.push_back({[](func::FuncOp &) {}, kDefaultAttemptsPerStage});
  }
  // Disable the VF reachable check on reg-based arch AIV functions as the
  // last resort, to allow more aggressive reuse of VF buffers.
  if (hacc::utils::isRegBasedArch(moduleOp) &&
      queryFuncCoreType(funcOp) == TFuncCoreType::AIV) {
    retryStages.push_back(
        {[this](func::FuncOp &) { this->disableVFReachableCheck = true; },
         kDefaultAttemptsPerStage});
  } else {
    retryStages.push_back({[](func::FuncOp &) {}, kDefaultAttemptsPerStage});
  }

  TcbPoolPlan tcbPoolPlan;
  bool startTcbPoolPlan = false;
  if (memMode == MemPlanMode::LOCAL_MEM_PLAN &&
      TcbPoolPlan::canTryPool(funcOp) &&
      succeeded(tcbPoolPlan.prepareAttempt(funcOp, 0))) {
    retryStages.push_back({[&](func::FuncOp &) { startTcbPoolPlan = true; },
                           /*attemptCount=*/1});
  }

  // TLA mode requires a deterministic plan: disable the fallback stages and
  // the shuffle-seed retries, and plan exactly once.
  if (this->enableTLAmode)
    retryStages = {{[](func::FuncOp &) {}, /*attemptCount=*/1}};

  int totalAttemptCount = 0;
  for (const PlanRetryStage &stage : retryStages)
    totalAttemptCount += stage.attemptCount;
  // The current plan-memory algorithm is sensitive to the order in which
  // some candidate buffers are considered. We retry planning with different
  // deterministic shuffle seeds to improve the chance of finding a valid
  // plan without making the pass behavior non-reproducible.
  // TODO: Remove the retry loop once the plan-memory algorithm is improved
  // to produce a stable valid plan in a single attempt.
  // The outer loop walks the fallback stages; the inner loop retries the
  // current stage with different shuffle seeds.
  int attempt = 0;
  PlanResult planned;
  for (PlanRetryStage &stage : retryStages) {
    stage.fallback(funcOp);
    for (int trial = 0; trial < stage.attemptCount; ++trial, ++attempt) {
      LDBG("Memory planning attempt " << attempt + 1 << "/" << totalAttemptCount
                                      << "\n");
      auto status = planMemoryAttempt(funcOp, attempt, totalAttemptCount,
                                      tcbPoolPlan, startTcbPoolPlan, planned);
      if (status == PlanAttemptStatus::EarlyEnd)
        return std::nullopt;
      else if (status == PlanAttemptStatus::Succeeded)
        return planned;
    }
  }
  return std::nullopt;
}

std::unique_ptr<MemPlan> PlanMemoryPass::initPlanner() const {
  std::unique_ptr<MemPlan> planner;
  if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::OFF)
    planner = std::make_unique<MemPlan>(
        memMode, enableGlobalReuse, enableMemoryDisplay, restrictInplaceAsISA,
        simtVFDynamicSize, disableVFReachableCheck, planMemoryUBCopyBankMode,
        planMemoryStrategy, enableMmadL0Planning);
  else if (planMemoryUBCopyBankMode == PlanMemoryUBCopyBankMode::PREFER)
    planner = std::make_unique<MemPlanPreferUBConflict>(
        memMode, enableGlobalReuse, enableMemoryDisplay, restrictInplaceAsISA,
        simtVFDynamicSize, disableVFReachableCheck, planMemoryUBCopyBankMode,
        planMemoryStrategy, enableMmadL0Planning);
  else
    planner = std::make_unique<MemPlanRequireUBConflict>(
        memMode, enableGlobalReuse, enableMemoryDisplay, restrictInplaceAsISA,
        simtVFDynamicSize, disableVFReachableCheck, planMemoryUBCopyBankMode,
        planMemoryStrategy, enableMmadL0Planning);
  return planner;
}

PlanAttemptStatus PlanMemoryPass::planMemoryAttempt(
    func::FuncOp &funcOp, int attempt, int totalAttemptCount,
    TcbPoolPlan &tcbPoolPlan, bool startTcbPoolPlan, PlanResult &planned) {

  MemLivenessAnalysis memLiveness(
      funcOp, this->memMode, this->disableTightlyCoupledBufferReuse,
      this->planMemoryUBCopyBankMode,
      /*randomSeed=*/startTcbPoolPlan ? 0 : attempt, enableMmadL0Planning);
  memLiveness.build();
  // Preserve copy and TCB metadata before pool fallback changes liveness.
  memLiveness.collectUBCopyBankCopies(startTcbPoolPlan);
  if (startTcbPoolPlan && failed(tcbPoolPlan.adaptLiveness(memLiveness)))
    return PlanAttemptStatus::EarlyEnd;

  std::unique_ptr<MemPlan> planner = initPlanner();
  MemPlan &memPlan = *planner;
  if (failed(memPlan.InitMemSpecsFromModule(funcOp)))
    return PlanAttemptStatus::EarlyEnd;
  memPlan.func_ = funcOp;
  memPlan.SetLinearOperation(memLiveness.linearOperation);
  memPlan.SetBufferInfos(memLiveness.bufferInfos);
  memPlan.SetBuffer2Life(memLiveness.buffer2Life);
  memPlan.SetGenKillMap(memLiveness.genKillMap);
  memPlan.SetBuffer2MultiNum(memLiveness.buffer2MultiNum);
  memPlan.SetInplacePairList(memLiveness.inplacePairList);
  memPlan.SetVFInplaceReuseInfo(
      vfInplaceReuseAnalysis.getVFCallInplaceReuseInfo(funcOp));
  memPlan.SetBankCopiesInfo(std::move(memLiveness.bankCopies));
  memPlan.SetBankTcbRoots(std::move(memLiveness.bankTcbRoots));
  memPlan.SetPreloadBufferReuseableInfo(memLiveness.preloadBufferReuseableInfo);

  bool isLastAttempt = attempt == totalAttemptCount - 1;
  if (succeeded(memPlan.plan(/*emitErrors=*/isLastAttempt))) {
    if (startTcbPoolPlan)
      tcbPoolPlan.recordPlan(getOperation());
    if (memPlan.enableMemoryDisplay) {
      SmallVector<MemoryDisplayInfo> memoryDisplayInfoList;
      // Collect plan success memory info for memory display tools.
      collectMemoryInfoForDebug(
          memoryDisplayInfoList, memPlan.GetMemscope2rootStorageEntry(),
          memPlan.GetBuffer2Offsets(), memPlan.errorInfo, false);
      createJsonForMemoryDisplay(funcOp, memoryDisplayInfoList);
    }
    planned = std::make_pair(memPlan.GetBuffer2Offsets(),
                             memPlan.GetPreloadLocalBuffers());
    return PlanAttemptStatus::Succeeded;
  }

  bool isEarlyEnd = startTcbPoolPlan && !memPlan.failedOnlyInUB();
  if ((isEarlyEnd || isLastAttempt) && memPlan.enableMemoryDisplay) {
    SmallVector<MemoryDisplayInfo> memoryDisplayInfoList;
    // Collect plan fail memory info for memory display tools.
    collectMemoryInfoForDebug(
        memoryDisplayInfoList, memPlan.GetMemscope2rootFailStorageEntry(),
        memPlan.GetBuffer2Offsets(), memPlan.errorInfo, true);
    memPlan.SetMemscope2rootSuccessStorageEntry();
    // Collect plan success memory info for memory display tools.
    collectMemoryInfoForDebug(
        memoryDisplayInfoList, memPlan.GetMemscope2rootSuccessStorageEntry(),
        memPlan.GetBuffer2Offsets(), memPlan.errorInfo, false);
    createJsonForMemoryDisplay(funcOp, memoryDisplayInfoList);
  }
  return isEarlyEnd ? PlanAttemptStatus::EarlyEnd
                    : PlanAttemptStatus::Retryable;
}

void PlanMemoryPass::updateId2Offsets(
    func::FuncOp &funcOp, Buffer2Offsets &buffer2Offsets,
    DenseMap<int32_t, SmallVector<uint64_t>> &id2Offsets) {
  auto funcType = queryFuncCoreType(funcOp);

  funcOp->walk([&](annotation::MarkOp markOp) {
    int32_t cvMixId;
    auto maybeAlloc = getCVMixIdAndAllocOpFromMarkOp(markOp, cvMixId);
    if (!maybeAlloc.has_value()) {
      return WalkResult::advance();
    }
    auto allocOpTypeValue = maybeAlloc.value().getResult();
    auto addressSpace = getHIVMAddressSpace(allocOpTypeValue.getType());
    if ((funcType == TFuncCoreType::AIV && addressSpace == AddressSpace::UB) ||
        (funcType == TFuncCoreType::AIC && addressSpace == AddressSpace::L1)) {
      id2Offsets[cvMixId] = buffer2Offsets[allocOpTypeValue];
      LDBG("id2Offsets add key: " << cvMixId << " vector_size: "
                                  << id2Offsets[cvMixId].size() << "\n");
    }
    return WalkResult::advance();
  });
}

void PlanMemoryPass::updateBuffer2OffsetsForFuncOp(
    func::FuncOp &funcOp, Buffer2Offsets &buffer2Offsets,
    DenseMap<int32_t, SmallVector<uint64_t>> &id2Offsets) {
  auto funcType = queryFuncCoreType(funcOp);

  funcOp->walk([&](annotation::MarkOp markOp) {
    int32_t cvMixId;
    auto maybeAlloc = getCVMixIdAndAllocOpFromMarkOp(markOp, cvMixId);
    if (!maybeAlloc.has_value()) {
      return WalkResult::advance();
    }
    auto allocOpTypeValue = maybeAlloc.value().getResult();
    auto addressSpace = getHIVMAddressSpace(allocOpTypeValue.getType());
    if ((funcType == TFuncCoreType::AIC && addressSpace == AddressSpace::UB) ||
        (funcType == TFuncCoreType::AIV && addressSpace == AddressSpace::L1)) {
      buffer2Offsets[allocOpTypeValue] = id2Offsets[cvMixId];
      LDBG("id2Offsets use key: " << cvMixId << " vector_size: "
                                  << id2Offsets[cvMixId].size() << "\n");
    }
    return WalkResult::advance();
  });
}

bool PlanMemoryPass::checkSimilarPointerCastOps(hivm::PointerCastOp op1,
                                                hivm::PointerCastOp op2) const {
  return (op1->getNumResults() == op2->getNumResults() &&
          llvm::equal(op1->getResultTypes(), op2->getResultTypes())) &&
         (op1->getNumOperands() == op2->getNumOperands() &&
          llvm::equal(op1->getOperands(), op2->getOperands()));
}

void PlanMemoryPass::fixMultibufferEnabledPointerCastOps(
    Operation *funcOp) const {
  llvm::MapVector<PointerCastOp, annotation::MarkOp> markedOps;

  funcOp->walk<WalkOrder::PreOrder>([&](annotation::MarkOp markOp) {
    auto attrDict = markOp->getAttrDictionary();
    if (attrDict.contains(hivm::MultiBufferAttr::name)) {
      if (auto pointerCastOp =
              dyn_cast<hivm::PointerCastOp>(markOp.getSrc().getDefiningOp())) {
        markedOps[pointerCastOp] = markOp;
      }
    }
  });

  for (auto [pointerCastOp, markedOp] : markedOps) {
    auto loopOp = pointerCastOp->getParentOfType<LoopLikeOpInterface>();
    if (!loopOp)
      continue;
    Block *targetBlock = nullptr;
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation())) {
      targetBlock = forOp.getBody();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
      // Hoist within the while region that already owns the pointer_cast.
      // Moving a before-region cast into after (or vice versa) breaks dominance
      // for uses that stay in the original region.
      if (whileOp.getAfter().isAncestor(pointerCastOp->getParentRegion())) {
        targetBlock = whileOp.getAfterBody();
      } else {
        targetBlock = whileOp.getBeforeBody();
      }
    } else {
      llvm::report_fatal_error("Unsupported loop parent for pointer cast op "
                               "with multibuffer attribute");
    }
    pointerCastOp->moveBefore(&targetBlock->front());
    markedOp->moveAfter(pointerCastOp);
  }

  std::vector<PointerCastOp> visitedOps;
  funcOp->walk<WalkOrder::PreOrder>([&](hivm::PointerCastOp pointerCastOp) {
    if (markedOps.contains(pointerCastOp)) {
      visitedOps.push_back(pointerCastOp);
    } else {
      for (auto it = visitedOps.rbegin(); it != visitedOps.rend(); it++) {
        if (!it->getOperation()->getParentOp()->isAncestor(pointerCastOp)) {
          continue;
        }
        if (checkSimilarPointerCastOps(pointerCastOp, *it)) {
          pointerCastOp.replaceAllUsesWith(it->getOperation());
          break;
        }
      }
    }
  });
}

void PlanMemoryPass::markTempBufForMemoryDisplay(func::FuncOp funcOp) {
  funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto extraBufferOp = dyn_cast_if_present<ExtraBufferOpInterface>(op)) {
      for (auto value : extraBufferOp.getExtraBuffers()) {
        auto maybeAlloc = utils::tracebackMemRefToAlloc(value);
        if (maybeAlloc.has_value()) {
          maybeAlloc.value()->setAttr(
              "is_tmpbuf", mlir::BoolAttr::get(funcOp.getContext(), true));
        }
      }
    }
  });
}

void MemPlan::SetMemscope2rootSuccessStorageEntry() {
  for (auto &it : memscope2rootStorageEntry) {
    if (memscope2rootFailStorageEntry.find(it.first) ==
        memscope2rootFailStorageEntry.end()) {
      memscope2rootSuccessStorageEntry[it.first] = it.second;
    }
  }
}

LogicalResult PlanMemoryPass::validateUBCopyBankArch(ModuleOp moduleOp) {
  if (planMemoryUBCopyBankMode != PlanMemoryUBCopyBankMode::OFF &&
      hacc::utils::isMemBasedArch(moduleOp))
    return moduleOp.emitError("UB copy bank planning is not supported on "
                              "memory-based architectures");
  return success();
}

void PlanMemoryPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (memMode == MemPlanMode::LOCAL_MEM_PLAN &&
      failed(validateUBCopyBankArch(moduleOp)))
    return signalPassFailure();
  vfInplaceReuseAnalysis = VFInplaceReuseAnalysis(moduleOp);
  // Cache per-func plan results (buffer2Offsets + preload-local roots) from
  // the first walk; the second walk uses them to populate addresses on
  // allocOps.
  DenseMap<func::FuncOp, PlanResult> planResultMap;
  // map cvMixId to the address will be used in both aic and aiv.
  DenseMap<int32_t, SmallVector<uint64_t>> id2Offsets;
  // planMem for variables and update Id2Offsets.
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (hacc::utils::isHost(funcOp))
      continue;
    if (hivm::isVF(funcOp))
      continue;
    // Select once per function. Standalone L1-to-L0 copies also require
    // explicit L0 allocation planning, even without an MMAD L0 user.
    this->enableMmadL0Planning = this->memMode == MemPlanMode::LOCAL_MEM_PLAN &&
                                 hasExplicitCubeOps(funcOp);
    if (this->enableMmadL0Planning) {
      if (!hacc::utils::isAscend950(moduleOp)) {
        funcOp.emitError("explicit cube memory planning supports only A5");
        signalPassFailure();
        return;
      }
      if (failed(validateExplicitCubeAllocs(funcOp, moduleOp))) {
        signalPassFailure();
        return;
      }
    }
    LDBG("\n-----funcOp " << funcOp.getName() << " mem plan start !! -----\n");
    auto planned = planMemoryForFuncOp(funcOp);
    if (planned.has_value()) {
      planResultMap[funcOp] = std::move(planned.value());
    } else {
      signalPassFailure();
      return;
    }
    LDBG("\n-----------------Update Id2Offsets start !! ------------------\n");
    updateId2Offsets(funcOp, planResultMap[funcOp].first, id2Offsets);
  };

  // Update buffer2Offsets and populate bufferAddress to allocOp.
  LDBG("\n----------------Second traversal of func !! --------------------\n");
  for (auto &[funcOp, planResult] : planResultMap) {
    auto &buffer2Offsets = planResult.first;
    auto &preloadLocalBuffers = planResult.second;
    LDBG("\n------------funcOp : " << funcOp.getName() << "---------------\n");
    updateBuffer2OffsetsForFuncOp(funcOp, buffer2Offsets, id2Offsets);
    if (failed(populateBufferAddressToAllocOp(funcOp, buffer2Offsets,
                                              preloadLocalBuffers))) {
      signalPassFailure();
      return;
    }
    fixMultibufferEnabledPointerCastOps(funcOp);
  }
}

std::unique_ptr<Pass>
mlir::hivm::createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption) {
  return std::make_unique<PlanMemoryPass>(planMemoryOption);
}
