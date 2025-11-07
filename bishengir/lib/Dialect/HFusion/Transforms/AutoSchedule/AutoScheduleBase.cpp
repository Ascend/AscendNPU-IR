//===- AutoScheduleBase.cpp -- Auto-schedule fused kernels ------*- C++ -*-===//
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
//
// This file implements auto scheduler's basic functionality.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfoCollector.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/PureElemwiseSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ShallowCVSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/SingleCubeSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/CacheFuncIO.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/ReorderOpsByBFS.h"
#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "AutoScheduleAttrDefs.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Base Scheduler] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_AUTOSCHEDULE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// By convention, tiling key is the first tiling data.
constexpr size_t kTilingKeyPos = 0;

inline bool compareTopologicalOrdering(const std::pair<Value, size_t> &v1,
                                       const std::pair<Value, size_t> &v2) {
  auto *v1DefiningOp = std::get<0>(v1).getDefiningOp();
  auto *v2DefiningOp = std::get<0>(v2).getDefiningOp();
  assert(v1DefiningOp != nullptr);
  assert(v2DefiningOp != nullptr);
  assert(v1DefiningOp->getBlock() == v2DefiningOp->getBlock());
  return v1DefiningOp->isBeforeInBlock(v2DefiningOp);
}

SmallVector<int64_t> getReturnValueTopologicalOrdering(
    const SmallVectorImpl<Value> &funcReturnValues) {
  // Get the topological ordering of the kernel outputs
  SmallVector<int64_t> sequence =
      llvm::to_vector(llvm::seq<int64_t>(0, funcReturnValues.size()));
  SmallVector<std::pair<Value, size_t>> funcReturnValuesAndOrdering;
  for (auto [value, idx] : llvm::zip(funcReturnValues, sequence)) {
    funcReturnValuesAndOrdering.push_back({value, idx});
  }
  llvm::sort(funcReturnValuesAndOrdering, compareTopologicalOrdering);
  return llvm::map_to_vector(
      funcReturnValuesAndOrdering,
      [](const std::pair<Value, int64_t> &v) { return std::get<1>(v); });
}

transform::SequenceOp initScheduleSequence(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  // create transform sequence op with name
  auto seqOp = opBuilder.create<transform::SequenceOp>(
      opBuilder.getUnknownLoc(), TypeRange(),
      transform::FailurePropagationMode::Propagate,
      opBuilder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });
  return seqOp;
}

/// Collect shaped type arguments used by reshape op.
SmallVector<Value> getMaybeReshapedInputs(ArrayRef<BlockArgument> inputs) {
  SmallVector<Value> result;
  result.append(inputs.begin(), inputs.end());
  for (auto [idx, arg] : llvm::enumerate(inputs)) {
    Type argType = arg.getType();
    if (!isa<ShapedType>(argType)) {
      continue;
    }
    auto maybeArgReshaped =
        hfusion::traceReshapeOrSliceSingleConsumerOrSelf(arg);
    LDBG("maybeArgReshaped [" << idx << "]: " << maybeArgReshaped);
    if (!hfusion::isReshapeOrSliceOp(maybeArgReshaped.getDefiningOp())) {
      continue;
    }
    result[idx] = maybeArgReshaped;
  }
  return result;
}

SmallVector<Value> getMaybeOutputsBeforeReshape(func::FuncOp funcOp) {
  SmallVector<Value> result;
  funcOp->walk([&result](func::ReturnOp retOp) {
    result = llvm::to_vector(retOp->getOperands());
  });
  for (auto [idx, output] : llvm::enumerate(result)) {
    auto maybeOutputsBeforeReshape =
        hfusion::traceReshapeOrSliceSingleProducerOrSelf(output);
    LDBG("maybeOutputsBeforeReshape [" << idx
                                       << "]: " << maybeOutputsBeforeReshape);
    result[idx] = maybeOutputsBeforeReshape;
  }
  return result;
}

static DictionaryAttr createHACCInputArgAttrs(MLIRContext *ctx,
                                              unsigned inputIdx) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(NamedAttribute(
      StringAttr::get(ctx, hacc::KernelArgTypeAttr::name),
      hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kInput)));
  attrs.push_back(NamedAttribute(StringAttr::get(ctx, hacc::InputIdxAttr::name),
                                 hacc::InputIdxAttr::get(ctx, inputIdx)));
  return DictionaryAttr::get(ctx, attrs);
}

/// Tiling function have the same arguments as the kernel
/// Inside the body computation related to tiling and shapes will be done
/// it returns tiling information that can be called from the host side
/// For dynamic dimension shapes of the input will be processed and computed
/// For static dimension shapes there are other pass that will propagate the
/// value to optimize it.

func::FuncOp createEmptyHostTilingFunction(func::FuncOp deviceFunc,
                                           OpBuilder &opBuilder) {
  MLIRContext *ctx = opBuilder.getContext();
  FunctionType t =
      FunctionType::get(ctx,
                        /*inputs=*/deviceFunc.getFunctionType().getInputs(),
                        /*results=*/
                        SmallVector<Type>());
  // Clone the arg attributes as well.
  auto hostTilingFunc = opBuilder.create<func::FuncOp>(
      deviceFunc.getLoc(),
      /*sym_name=*/
      opBuilder.getStringAttr(hacc::constructHostFunctionName(
          deviceFunc.getSymName().str(), hacc::HostFuncType::kTilingFunction)),
      /*function_type=*/TypeAttr::get(t),
      /*sym_visibility=*/StringAttr(),
      /*arg_attrs=*/deviceFunc.getArgAttrsAttr(),
      /*res_attrs=*/ArrayAttr());
  hostTilingFunc.addEntryBlock();

  hacc::utils::setHost(hostTilingFunc);
  hacc::utils::setHostFuncType(hostTilingFunc,
                               hacc::HostFuncType::kTilingFunction);
  return hostTilingFunc;
}
} // namespace

//===----------------------------------------------------------------------===//
// SchedulerBase
//===----------------------------------------------------------------------===//

/// Init static data members.
AutoScheduleOptions SchedulerBase::options_ = AutoScheduleOptions();

SchedulerBase::SchedulerBase(func::FuncOp f, FusionKind kind) {
  kernelInfo_ = std::make_unique<KernelInfo>();
  tilingInfo_ = std::make_unique<TilingInfo>();
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kind;
  kernelTilingMap_ = std::make_unique<IRMapping>();
}

SchedulerBase::SchedulerBase(func::FuncOp f,
                             std::unique_ptr<KernelInfo> &&kernelInfo,
                             std::unique_ptr<TilingInfo> &&tilingInfo) {
  kernelInfo_ = std::move(kernelInfo);
  tilingInfo_ = std::move(tilingInfo);
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kernelInfo_->getFusionKind();
  kernelTilingMap_ = std::make_unique<IRMapping>();
}

SchedulerBase::~SchedulerBase() {
  kernelInfo_.reset();
  tilingInfo_.reset();
  handleRecord_.reset();
}

LogicalResult SchedulerBase::runPreScheduleProcedure(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  if (failed(cacheIO(opBuilder)))
    return currentFunc.emitError() << "Failed to cache inputs/outputs.";

  if (failed(analyzeAndVerifyKernel()))
    return currentFunc.emitError() << "Failed to analyze and verify kernel.";
  return success();
}

LogicalResult SchedulerBase::runPostScheduleProcedure(OpBuilder &opBuilder) {
  return success();
}

LogicalResult SchedulerBase::runScheduleProcedure(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  if (failed(calculateTiling(opBuilder)))
    return currentFunc->emitWarning("Failed to calculate tiling.");

  if (failed(selectTiling()))
    return currentFunc->emitWarning("Failed to select tiling.");

  if (failed(createAndApplySchedules(opBuilder)))
    return currentFunc->emitWarning("Failed to create and apply schedule.");
  return success();
}

LogicalResult SchedulerBase::runNopScheduleProcedure(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  TilingInfo *tilingInfo = getTilingInfo();

  // Create empty host tiling function.
  func::FuncOp originalKernel = getOriginalKernel();
  opBuilder.setInsertionPoint(originalKernel);
  func::FuncOp hostTilingFunc =
      createEmptyHostTilingFunction(originalKernel, opBuilder);
  opBuilder.setInsertionPointToEnd(&hostTilingFunc.getFunctionBody().front());
  opBuilder.create<func::ReturnOp>(originalKernel.getLoc());

  // Record host tiling function.
  tilingInfo->setHostTilingFunc(hostTilingFunc);
  return success();
}

bool SchedulerBase::isNopSchedule() const {
  const auto &kernelInfo = getKernelInfo();
  // If anchor's rank is 0, we don't need to schedule
  return kernelInfo->getAnalyzer()->getAnchorRank() == 0;
}

LogicalResult SchedulerBase::runOnOperation(OpBuilder &opBuilder) {
  if (failed(runPreScheduleProcedure(opBuilder)))
    return getOriginalKernel().emitOpError()
           << "Failed to run pre schedule procedure";

  if (isNopSchedule())
    return runNopScheduleProcedure(opBuilder);

  if (failed(runScheduleProcedure(opBuilder)))
    return failure();

  if (failed(runPostScheduleProcedure(opBuilder)))
    return failure();
  return success();
}

// Get the list of block arguments that are used as dps init operands, and
// whose tied result value is also the kernel return value.
// The return value should enable `cache_write_to_output_init` option when
// performing cache write.
void tieFuncArgsToReturnValue(KernelInfo *info) {
  auto funcArgs = info->originalKernel.getArguments();
  for (auto [idx, ba] : llvm::enumerate(funcArgs)) {
    bool funcArgIsReshaped = false;
    bool funcResultIsReshaped = false;
    if (auto resultIdx = hfusion::getFuncArgTiedResultReturnIdx(
            ba, funcArgIsReshaped, funcResultIsReshaped)) {
      LDBG("Arg number : " << idx << " is tied to result number: "
                           << resultIdx.value());
      info->funcArgIdxWithTiedReturnValue.insert(idx);
      info->returnValueIdx2TiedFuncArg.insert({resultIdx.value(), idx});
      if (funcResultIsReshaped) {
        LDBG("Result number : " << idx << " is reshaped before return");
        info->returnValueWithReshapeIndices.insert(resultIdx.value());
      }
    }
    if (funcArgIsReshaped) {
      LDBG("Arg number : " << idx << " is reshaped before use");
      info->funcArgWithReshapeIndices.insert(idx);
    }
  }
}

LogicalResult SchedulerBase::analyzeAndVerifyKernel() {
  KernelInfo *info = getKernelInfo();
  info->originalKernel = getOriginalKernel();
  if (failed(info->initializeDimensionAnalyzer()))
    return failure();

  auto funcArgs = info->originalKernel.getArguments();
  FunctionType funcType = info->originalKernel.getFunctionType();
  info->numInputs = funcType.getNumInputs();
  info->baseKernelName = info->originalKernel.getSymName().str();
  info->numOutputs = funcType.getNumResults();
  // TODO: use reshape analyzer
  info->inputValues = getMaybeReshapedInputs(funcArgs);
  info->outputValues = getMaybeOutputsBeforeReshape(info->originalKernel);
  info->outputOrdering = getReturnValueTopologicalOrdering(info->outputValues);
  auto getType = [](Value v) { return v.getType(); };
  info->inputTypes = llvm::map_to_vector(info->inputValues, getType);
  info->outputTypes = llvm::map_to_vector(info->outputValues, getType);
  info->blockDim = options_.blockDim;
  info->cubeTilingTuning = options_.cubeTilingTuning;
  // Multi core reduced is enabled when deterministic computing is disabled.
  info->enableMultiCoreReduce = !options_.enableDeterministicComputing;

  // 1. record block args used as dps inits
  // 2. tie return value index to dps inits block args
  // 3. record args reshaped before use
  tieFuncArgsToReturnValue(info);

  // The block arguments that need to do cache read are:
  //   - shaped arguments
  //   - arguments that are not tied to results
  auto funcArgIndices = llvm::to_vector(llvm::seq<int64_t>(0, info->numInputs));
  auto filteredIndices =
      llvm::make_filter_range(funcArgIndices, [&info](int64_t idx) {
        return !info->funcArgIdxWithTiedReturnValue.contains(idx) &&
               isa<ShapedType>(info->inputTypes[idx]);
      });
  info->cacheReadFuncArgIndices =
      SetVector<int64_t>{filteredIndices.begin(), filteredIndices.end()};

  return analyzeAndVerifyKernelImpl();
}

LogicalResult SchedulerBase::analyzeAndVerifyKernelImpl() {
  return KernelInfoCollector(getKernelInfo(), getAutoScheduleOptions()).run();
}

LogicalResult SchedulerBase::cacheIO(OpBuilder &opBuilder) {
  auto originalKernel = getOriginalKernel();
  /// Merge consecutive extract slices to make sure CacheRead is inserted to
  /// the right position
  if (failed(applyMergeConsecutiveInsertExtractSlice(originalKernel))) {
    return failure();
  }

  /// Perform cache IO.
  ///
  /// This needs to be done before `tensor-results-to-out-params` because
  /// of the following case:
  ///
  /// ```mlir
  ///   func.func @foo(%arg0: tensor<?x256xf32>) -> (tensor<?xf32>)
  ///     %c0 = arith.constant 0 : index
  ///     %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
  ///     %2 = tensor.empty(%dim) : tensor<?xf32>
  ///     %3 = linalg.fill ins(...) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  ///     return %3 : tensor<?xf32>
  /// ```
  ///
  /// If we perform `tensor-results-to-out-params` first, we'll lose the
  /// information on the shape relationship between the result tensor and the
  /// original input func arguments:
  ///
  /// ```mlir
  ///   func.func @foo(%arg0: tensor<?x256xf32>, %arg1: tensor<?xf32>)
  ///                                                       -> (tensor<?xf32>)
  ///     %1 = linalg.fill ins(...) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
  ///     return %1 : tensor<?xf32>
  /// ```
  cacheFuncIO(originalKernel, /*annotate=*/false, /*writeUnique=*/true);

  /// Perform canonicalization here mainly for dynamic shape kernels. A lot
  /// of shape-related ops (such as `tensor.dim`) will be generated by CacheIO
  /// and we need to simplify them before scheduling.
  std::vector<std::string> disabledPatterns;
  disabledPatterns.emplace_back(
      "(anonymous "
      "namespace)::FoldFillWithTensorReshape<mlir::tensor::CollapseShapeOp>");
  disabledPatterns.emplace_back(
      "(anonymous "
      "namespace)::FoldFillWithTensorReshape<mlir::tensor::ExpandShapeOp>");
  disabledPatterns.emplace_back("FoldTransposeWithTranspose");
  if (failed(
          applyCSEAndCanonicalizePass(getOriginalKernel(), disabledPatterns)))
    return failure();

  /// Move result's tied init operands to function arguments.
  if (failed(applyTensorResultToOutParamsPass(originalKernel)))
    return failure();

  reorderOpsByBFS(originalKernel);

  /// Apply aggressive bubble up extract slice to make sure recache unaligned
  /// access works as expected.
  if (failed(applyAggressiveBubbleUpExtractSlice(originalKernel)))
    return failure();

  if (failed(applyReCacheIOPass(originalKernel)))
    return failure();

  return success();
}

LogicalResult SchedulerBase::markHACCInputArgAttr(func::FuncOp func) {
  for (unsigned inputIdx = 0; inputIdx < func.getNumArguments(); inputIdx++) {
    if (func.getArgAttrOfType<hacc::KernelArgTypeAttr>(
            inputIdx, hacc::KernelArgTypeAttr::name))
      continue;
    func.setArgAttrs(inputIdx,
                     createHACCInputArgAttrs(func->getContext(), inputIdx));
  }
  return success();
}

/// This is the main function that orchestrates the tiling function making
LogicalResult SchedulerBase::calculateTiling(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  TilingInfo *tilingInfo = getTilingInfo();
  MLIRContext *ctx = getContext();

  // Step 1. Get tiling compute function.
  // This returns a function which takes a kernel info and builder
  //  that said function will return std::pair<TilingCases, TilingStruct>;
  TilingComputeFn fn = calculateTilingImpl();
  // Bail out if the derived scheduler does not require tiling
  if (!fn)
    return success();

  // Step 2. Create host tiling function.
  func::FuncOp originalKernel = getOriginalKernel();
  opBuilder.setInsertionPoint(originalKernel);
  func::FuncOp hostTilingFunc =
      createEmptyHostTilingFunction(originalKernel, opBuilder);

  opBuilder.setInsertionPointToEnd(&hostTilingFunc.getFunctionBody().front());

  getKernelTilingMap()->map(originalKernel.getArguments(),
                            hostTilingFunc.getArguments());
  getKernelTilingMap()->map(hostTilingFunc.getArguments(),
                            originalKernel.getArguments());

  for (auto &op : originalKernel.front().without_terminator()) {
    auto *newOp = opBuilder.clone(op, *kernelTilingMap_);
    getKernelTilingMap()->map(newOp, &op);
    getKernelTilingMap()->map(newOp->getResults(), op.getResults());
  }

  // Step 3. Record host tiling function.
  tilingInfo->setHostTilingFunc(hostTilingFunc);

  // Step 4. Construct ExprBuilder and set insertion point into host tiling
  // func.
  StmtExprBuilder exprBuilder(tilingInfo, getKernelInfo(), ctx);
  exprBuilder.setInsertionPointToEnd(&hostTilingFunc.getFunctionBody().front());

  // Step 5. Evaluate tiling computation function and produce IR.
  FailureOr<SmallVector<Value>> returns =
      tilingInfo->evaluateTilingComputation(fn, getKernelInfo(), &exprBuilder);
  if (failed(returns)) {
    return originalKernel->emitError()
           << "Failed to evaluate tiling computation function!";
  }

  // Step 6: Return tiling data.
  opBuilder.setInsertionPointToEnd(&hostTilingFunc.getFunctionBody().front());
  opBuilder.create<func::ReturnOp>(originalKernel.getLoc(), *returns);

  // Step 7: Update function type because for some fusion kind, the number of
  // tiling keys is kernel-dependent.
  hostTilingFunc.setFunctionType(hostTilingFunc.getFunctionType().clone(
      /*inputs=*/originalKernel.getFunctionType().getInputs(),
      /*results=*/SmallVector<Type>((*returns).size(),
                                    opBuilder.getIntegerType(64))));
  // Tag the tiling function's result with arg attributes so that we can
  // cross-check with the device function's signature later.
  for (unsigned i = 0; i < (*returns).size(); ++i) {
    hostTilingFunc.setResultAttr(
        i, hacc::KernelArgTypeAttr::name,
        hacc::KernelArgTypeAttr::get(getContext(),
                                     i == kTilingKeyPos
                                         ? hacc::KernelArgType::kTilingKey
                                         : hacc::KernelArgType::kTilingData));
  }
  LDBG("--Generated Tiling Func: \n" << *hostTilingFunc);
  return success();
}

LogicalResult SchedulerBase::selectTiling() const {
  TilingInfo *tilingInfo = getTilingInfo();

  // Try to simplify host tiling func
  if (failed(tilingInfo->trySimplifyTilingFunc())) {
    LDBG("Failed to simplify host tiling func");
    return success();
  }

  LDBG("--Simplified Tiling Func: \n" << *tilingInfo->getHostTilingFunc());

  TilingData *tilingKey = tilingInfo->getTilingKey();
  // Cannot constantize tiling, all cases need to be generated
  if (!tilingKey->isConst()) {
    LDBG("Cannot constantize tiling");
    return success();
  }

  int64_t selectedTilingKey = tilingKey->getConst();
  LDBG("Selected tiling key: " << selectedTilingKey);
  // Prune tiling
  tilingInfo->pruneTilingExcept(selectedTilingKey);
  return success();
}

LogicalResult SchedulerBase::createAndApplySchedules(OpBuilder &opBuilder) {
  // iterate over tiling cases
  TilingInfo *info = getTilingInfo();

  for (TilingKey key : info->getTilingCases()) {
    LDBG("Creating schedule for tiling key: " << key);
    if (failed(initSchedule(key, opBuilder)))
      return failure();

    if (failed(createScheduleImpl(key, opBuilder)))
      return failure();

    LDBG("Dumping kernel and schedule for tiling key: " << key);
    LLVM_DEBUG(dumpKernelAndSchedule());

    if (failed(applyScheduleImpl(opBuilder)))
      return failure();

    cleanUpAfterSchedule();
  }

  if (failed(fixCallSitesAndCaller(opBuilder)))
    return failure();

  LDBG("Removing original func...");
  getOriginalKernel()->erase();
  return success();
}

LogicalResult SchedulerBase::applySchedule(func::FuncOp &funcOp,
                                           OpBuilder &opBuilder) {
  auto fusionKindAttr =
      funcOp->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr || !hacc::utils::isDevice(funcOp)) {
    LDBG("Unknown kernel fusion kind");
    return success();
  }

  auto fusionKind = fusionKindAttr.getFusionKind();
  std::unique_ptr<SchedulerBase> scheduler;
  funcOp->setAttr(utils::kEnableAutoMarkBufferSize, opBuilder.getUnitAttr());
  switch (fusionKind) {
  case FusionKind::PureElemwise:
  case FusionKind::AnyPB:
  case FusionKind::LastAxisPBR:
  case FusionKind::AnyPBR:
    scheduler = std::make_unique<AnyPBRScheduler>(funcOp);
    break;
  case FusionKind::SingleCube:
    scheduler = std::make_unique<SingleCubeScheduler>(funcOp);
    break;
  case FusionKind::ShallowCV:
    scheduler = std::make_unique<ShallowCVScheduler>(funcOp);
    break;
  case FusionKind::ShallowVV:
    return success();
  case FusionKind::Unknown:
  default:
    return funcOp.emitError("Unknown kernel fusion kind");
  }
  return scheduler->runOnOperation(opBuilder);
}

LogicalResult SchedulerBase::applyScheduleImpl(OpBuilder &opBuilder) {
  PassManager pm(getContext());
  transform::TransformOptions options;
  options.enableExpensiveChecks(false);
  pm.addPass(hfusion::createAutoScheduleInterpreterPass(
      getToBeScheduledKernelName(), options));
  pm.addPass(
      hfusion::createEraseAutoSchedulePass(getToBeScheduledKernelName()));

  if (failed(pm.run(getModule())))
    return failure();

  return success();
}

void SchedulerBase::dumpKernelAndSchedule() {
  Operation *transformSeqOp = nullptr;
  getModule().walk([&](Operation *nestedOp) {
    if (!nestedOp->hasAttrOfType<StringAttr>(
            transform::TransformDialect::kTargetTagAttrName))
      return WalkResult::advance();

    if (isa<transform::TransformOpInterface>(nestedOp) &&
        nestedOp->getAttrOfType<StringAttr>(
                    transform::TransformDialect::kTargetTagAttrName)
                .str() ==
            auto_schedule::getTransformRootTag(getToBeScheduledKernelName())) {
      assert(transformSeqOp == nullptr &&
             "transform seq with duplicate tags found!");
      transformSeqOp = nestedOp;
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  assert(transformSeqOp && "cannot find target transform seq");
  LDBG("---Current to-be-scheduled kernel func: \n"
       << *getToBeScheduledKernel());
  LDBG("---Current transform sequence: \n" << *transformSeqOp);
}

void SchedulerBase::cleanUpAfterSchedule() {
  getHandleRecord()->clear();
  setToBeScheduledKernel(nullptr);
  setTransformSeqHandle(Value());
  for (auto *td : getTilingInfo()->getTilingStruct())
    td->setHandle(nullptr);
}

void insertTilingDataArgument(OpBuilder &opBuilder, func::FuncOp func,
                              TilingInfo *tilingInfo) {
  auto argIdx = func.getNumArguments();
  for (auto *iter = tilingInfo->tilingDataBegin();
       iter != tilingInfo->tilingDataEnd(); iter++) {
    TilingData *td = iter->get();
    td->setPos(argIdx);
    SmallVector<NamedAttribute> argAttrs = {hacc::createHACCKernelArgAttr(
        opBuilder.getContext(), iter == tilingInfo->tilingDataBegin()
                                    ? hacc::KernelArgType::kTilingKey
                                    : hacc::KernelArgType::kTilingData)};
    func.insertArgument(argIdx, td->getType(),
                        opBuilder.getDictionaryAttr(argAttrs), func.getLoc());
    argIdx++;
  }
}

LogicalResult SchedulerBase::initSchedule(TilingKey key, OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  // Step 1: Construct new kernel name with tiling key post-fix.
  func::FuncOp originalKernel = getOriginalKernel();
  auto toBeScheduledKernelName =
      originalKernel.getSymName().str() + "_" + std::to_string(key);
  auto module = getModule();
  if (module.lookupSymbol(opBuilder.getStringAttr(toBeScheduledKernelName))) {
    return originalKernel->emitError(
        "Duplicate kernel name during auto-scheduling process");
  }

  // Step 2: Clone original func and set it as the to-be-scheduled function.
  opBuilder.setInsertionPoint(originalKernel);
  func::FuncOp toBeScheduleKernel =
      cast<func::FuncOp>(opBuilder.clone(*originalKernel));
  toBeScheduleKernel.setSymName(toBeScheduledKernelName);
  this->setToBeScheduledKernel(toBeScheduleKernel);
  // Bind tiling key to the to-be-scheduled kernel.
  tilingInfo->recordKernelFunc(key, toBeScheduleKernel);

  // Step 3. Insert tiling data to to-be-scheduled kernel function and bind
  // tiling data to kernel argument
  insertTilingDataArgument(opBuilder, toBeScheduleKernel, tilingInfo);

  // Step 4. Insert transform sequence right after the to-be-scheduled kernel.
  opBuilder.setInsertionPointAfter(toBeScheduleKernel);
  auto seqOp = initScheduleSequence(opBuilder);
  auto *transformBody = seqOp.getBodyBlock();
  // Set insertion point to transform sequence body
  opBuilder.setInsertionPointToStart(transformBody);
  // Record transform sequence block argument
  setTransformSeqHandle(transformBody->getArguments().front());

  // Step 5. Set attributes to various functions
  auto blockDimIntAttr = opBuilder.getIntegerAttr(opBuilder.getIntegerType(64),
                                                  getKernelInfo()->blockDim);
  toBeScheduleKernel->setAttr(hacc::BlockDimAttr::name, blockDimIntAttr);
  toBeScheduleKernel->setAttr(
      hacc::TilingFunctionAttr::name,
      hacc::TilingFunctionAttr::get(
          opBuilder.getContext(),
          tilingInfo->getHostTilingFunc().getSymName()));

  // Set transform root and payload root tags
  toBeScheduleKernel->setAttr(
      transform::TransformDialect::kTargetTagAttrName,
      opBuilder.getStringAttr(
          auto_schedule::getPayloadRootTag(toBeScheduledKernelName)));
  seqOp->setAttr(transform::TransformDialect::kTargetTagAttrName,
                 opBuilder.getStringAttr(auto_schedule::getTransformRootTag(
                     toBeScheduledKernelName)));

  return success();
}

SmallVector<Value> SchedulerBase::getNewArgsForCallSite(
    func::FuncOp caller, func::CallOp oldCallSite,
    const SchedulerBase::CallSiteArgBuilderInfo &info, OpBuilder &opBuilder) {
  auto oldCallArgs = oldCallSite->getOperands();
  size_t oldArgCount = oldCallArgs.size();
  size_t tilingStructSize = info.tilingIdx2TilingData.size();
  size_t newArgCount = oldArgCount + tilingStructSize;

  SmallVector<Value> newCallArgs;
  newCallArgs.reserve(newArgCount);
  // By convention, tiling data is appended after existing args, but the order
  // to which they're added matches the tiling struct order
  newCallArgs.append(oldCallArgs.begin(), oldCallArgs.end());
  newCallArgs.append(SmallVector<Value>(tilingStructSize, Value()));

  for (size_t idx = oldArgCount; idx < newArgCount; idx++) {
    auto tilingIdx = idx - oldArgCount;
    assert(info.tilingIdx2TilingData.contains(tilingIdx));
    newCallArgs[idx] = info.tilingIdx2TilingData.at(tilingIdx);
  }

  return newCallArgs;
}

void SchedulerBase::doFixCallSite(CallerInfo &callerInfo, func::CallOp callSite,
                                  CallSiteArgBuilderInfo &builderInfo,
                                  DenseMap<Operation *, Operation *> &irMap,
                                  OpBuilder &opBuilder) const {
  OpBuilder::InsertionGuard g(opBuilder);
  opBuilder.setInsertionPoint(callSite);
  auto newArgs = getNewArgsForCallSite(callerInfo.caller, callSite, builderInfo,
                                       opBuilder);

  // If the callee is not the original kernel, generate a new call
  // with the same callee but new args and bail out.
  if (!builderInfo.calleeIsOriginalKernel) {
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), callSite.getCallee(),
        newArgs);
    irMap.insert(std::make_pair(callSite, newCallSite));
    return;
  }
  // Otherwise, create switch case to call kernels of different tiling cases
  // using the tiling key.
  // By convention, the first tiling data is the tiling key.
  Value tilingKey = builderInfo.tilingIdx2TilingData.at(0);
  return generateDeviceCallers(callSite, tilingKey, newArgs, irMap, opBuilder);
}

void SchedulerBase::generateDeviceCallers(
    func::CallOp callSite, Value tilingKey,
    const SmallVector<Value> &newCallArgs,
    DenseMap<Operation *, Operation *> &irMap, OpBuilder &opBuilder) const {
  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());
  auto tilingKeys = tilingInfo->getTilingCases();
  // If the tiling are fully static, generate a new call with new args and bail
  // out. Because we are guaranteed that all tiling data can be constantize and
  // nothing will be packed.
  if (tilingInfo->isTilingFullyStatic()) {
    func::FuncOp kernelFunc = tilingKey2Kernel.at(tilingKeys[0]);
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), kernelFunc.getSymName(),
        newCallArgs);
    irMap.insert(std::make_pair(callSite, newCallSite));
    return;
  }
  // Otherwise, create switch cases to different tiling cases.
  tilingKey = castToIndex(tilingKey, opBuilder);
  scf::IndexSwitchOp switchOp = opBuilder.create<scf::IndexSwitchOp>(
      callSite.getLoc(), callSite.getResultTypes(), tilingKey,
      tilingKeys.getRef().getArrayRef(), tilingKeys.size());
  for (Region &region : switchOp.getCaseRegions())
    region.emplaceBlock();

  // annotated the switch op's result with tiling function information
  auto markOp = opBuilder.create<annotation::MarkOp>(switchOp->getLoc(),
                                                     switchOp->getResult(0));
  markOp->setAttr(hacc::TilingFunctionAttr::name,
                  hacc::TilingFunctionAttr::get(
                      opBuilder.getContext(),
                      tilingInfo->getHostTilingFunc().getSymName()));

  for (size_t i = 0, e = switchOp.getNumCases(); i != e; ++i) {
    opBuilder.setInsertionPointToStart(&switchOp.getCaseBlock(i));
    func::FuncOp kernelFunc = tilingKey2Kernel.at(tilingKeys[i]);
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), kernelFunc.getSymName(),
        newCallArgs);
    opBuilder.create<scf::YieldOp>(callSite->getLoc(),
                                   newCallSite.getResults());
  }

  switchOp.getDefaultRegion().emplaceBlock();
  opBuilder.setInsertionPointToStart(&switchOp.getDefaultBlock());
  Value constFalse = opBuilder.create<arith::ConstantOp>(
      callSite.getLoc(), opBuilder.getI1Type(), opBuilder.getBoolAttr(false));
  opBuilder.create<cf::AssertOp>(callSite.getLoc(), constFalse,
                                 "Invalid tiling key");

  // For the default branch, yield the function arguments that are used as
  // outputs instead of `ub.poison` so that `memref.copy` won't be
  // inserted after bufferization.
  SmallVector<Value> defaultYieldedValues;
  // [Assumptions]:
  //  1. All tiling case's kernel function has the same argument ordering.
  //  2. All call sites have the same operand ordering.
  func::FuncOp kernelFunc = tilingKey2Kernel.at(tilingKeys[0]);
  for (size_t i = 0, e = kernelFunc.getNumArguments(); i != e; ++i) {
    if (hacc::utils::isKernelArg(kernelFunc, i, hacc::KernelArgType::kOutput))
      defaultYieldedValues.push_back(newCallArgs[i]);
  }
  opBuilder.create<scf::YieldOp>(callSite.getLoc(), defaultYieldedValues);
  irMap.insert(std::make_pair(callSite, switchOp));
}

SchedulerBase::CallSite2TilingIdx2TilingData
SchedulerBase::getTilingDataForCallSite(func::FuncOp caller,
                                        TilingInfo *tilingInfo,
                                        const CallerInfo &callerInfo,
                                        OpBuilder &opBuilder) {
  CallSite2TilingIdx2TilingData callSite2TilingIdx2TilingData;
  if (hacc::utils::isHost(caller) && getEnableHostResourceMgmt()) {
    // If the caller is a host function and we enabled host resource mgmt,
    // construct a call to the tiling function at each call site.
    for (func::CallOp callSite : callerInfo.callSites) {
      opBuilder.setInsertionPoint(callSite);
      // Construct call to the tiling function.
      func::FuncOp tilingFunc = tilingInfo->getHostTilingFunc();
      auto tilingFuncCallOp = opBuilder.create<func::CallOp>(
          callSite.getLoc(), tilingFunc.getSymName(),
          tilingFunc.getFunctionType().getResults(), callSite.getOperands());
      for (auto result : tilingFuncCallOp.getResults()) {
        callSite2TilingIdx2TilingData[callSite].insert(
            {result.getResultNumber(), result});
      }
    }
    return callSite2TilingIdx2TilingData;
  }
  // Insert tiling data arguments to caller's enclosing func.
  // Assume for now that within the same caller, all callee with the same
  // tiling function receive the same tiling data arguments.
  // TODO: Revisit this later.
  size_t callerArgCount = caller.getNumArguments();
  size_t tilingStructSize = tilingInfo->size();
  for (size_t idx = callerArgCount; idx < callerArgCount + tilingStructSize;
       idx++) {
    SmallVector<NamedAttribute> argAttrs = {hacc::createHACCKernelArgAttr(
        opBuilder.getContext(), idx == callerArgCount
                                    ? hacc::KernelArgType::kTilingKey
                                    : hacc::KernelArgType::kTilingData)};
    caller.insertArgument(idx, opBuilder.getI64Type(),
                          opBuilder.getDictionaryAttr(argAttrs),
                          caller.getLoc());
    BlockArgument tilingDataArg = caller.getArgument(idx);
    for (func::CallOp callSite : callerInfo.callSites) {
      callSite2TilingIdx2TilingData[callSite].insert(
          {idx - callerArgCount, tilingDataArg});
    }
  }
  return callSite2TilingIdx2TilingData;
}

/// Fix direct and indirect callers of the unscheduled kernel.
///
/// Original IR:
/// \code
/// func.func private @unschedule_kernel()
/// func.func private @schedule_kernel_1(%tiling_key, %td0, %td1)
/// func.func private @schedule_kernel_2(%tiling_key, %td0, %td1)
///
/// func.func @nested_caller() {
///    func.call @unschedule_kernel()
/// }
/// func.func @caller() {
///   func.call @nested_caller()
/// }
/// \endcode
///
/// After fixing call sites:
/// \code
/// func.func private @unschedule_kernel()
/// func.func private @schedule_kernel_1(%tiling_key, %td0, %td1)
/// func.func private @schedule_kernel_2(%tiling_key, %td0, %td1)
///
/// func.func @nested_caller(%tiling_key, %td0, %td1) {
///    %cst_tiling_key_1 = arith.const ...
///    %cst_tiling_key_2 = arith.const ...
///    scf.index_switch %tiling_key
///      case %cst_tiling_key_1 :
///        func.call @schedule_kernel_1(%tiling_key, %td0, %td1)
///      case %cst_tiling_key_2 :
///        func.call @schedule_kernel_1(%tiling_key, %td0, %td1)
/// }
/// func.func @caller(%tiling_key, %td0, %td1) {
///   func.call @nested_caller(%tiling_key, %td0, %td1)
/// }
/// \endcode
///
/// When `-enable-manage-host-resources=true`, a tiling function call is
/// inserted at the first Host caller.
///
/// After fixing call sites:
/// \code
//  func.func private @tiling_func()
/// func.func private @unschedule_kernel()
/// func.func private @schedule_kernel_1(%tiling_key, %td0, %td1)
/// func.func private @schedule_kernel_2(%tiling_key, %td0, %td1)
///
/// func.func @caller() {Host} {
///    %tiling_key, %td0, %td1 = func.call @tiling_func()
///    scf.index_switch %tiling_key
///      case %cst_tiling_key_1 :
///        func.call @schedule_kernel_1(%tiling_key, %td0, %td1)
///      case %cst_tiling_key_2 :
///        func.call @schedule_kernel_1(%tiling_key, %td0, %td1)
/// }
/// \endcode
LogicalResult SchedulerBase::fixCallSitesAndCaller(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  LDBG("Fixing call sites of un-scheduled func...");

  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());

  // Get callers of the original, unscheduled kernel
  DenseMap<func::FuncOp, CallerInfo> workList;
  tiling::getCallerInfo(getOriginalKernel(), getModule(), workList);

  // Bail out on trivial case where there is no caller
  if (workList.empty()) {
    // TODO: If there is only one tiling case, don't modify the kernel name for
    // now. Modify this after we fully switch to support dynamic/static shape +
    // multiple tiling
    auto tilingCases = tilingInfo->getTilingCases();
    if (tilingCases.size() == 1)
      tilingKey2Kernel[tilingCases[0]].setSymName(getOriginalKernelName());
    return success();
  }

  // Repeatedly modify the caller and call site, until there is no caller.
  DenseMap<Operation *, Operation *> irMap;
  DenseSet<func::FuncOp> processedCaller;
  while (!workList.empty()) {
    auto &[caller, callerInfo] = *(workList.begin());
    if (processedCaller.contains(caller)) {
      LDBG("Cyclic call detected");
      return failure();
    }
    LDBG("Fixing call site in: \n" << *caller);

    auto callSite2TilingIdx2TilingData =
        getTilingDataForCallSite(caller, tilingInfo, callerInfo, opBuilder);

    // Fix the call sites
    bool calleeIsOriginalKernel = callerInfo.callee == getOriginalKernel();
    for (func::CallOp callSite : callerInfo.callSites) {
      CallSiteArgBuilderInfo builderInfo{
          callSite2TilingIdx2TilingData.at(callSite), calleeIsOriginalKernel};
      doFixCallSite(callerInfo, callSite, builderInfo, irMap, opBuilder);
    }

    // If we created a tiling function call within the caller, there is no need
    // to propagate to the callers of the caller because the function signature
    // is not modified.
    if (!(hacc::utils::isHost(caller) && getEnableHostResourceMgmt()))
      tiling::getCallerInfo(caller, getModule(), workList);

    processedCaller.insert(caller);
    workList.erase(caller);
  }

  for (auto &[oldOp, newOp] : irMap) {
    oldOp->replaceAllUsesWith(newOp);
    oldOp->erase();
  }

  return success();
}

NamedValueHandle SchedulerBase::recordImpl(Value target, OpBuilder &opBuilder,
                                           const NamedValueHandleArgs &args) {
  // If the identifier type is operation name, then it's already unique.
  std::string uniqueName =
      args.isNameUnique ? args.name.str()
                        : getHandleRecord()->getAndRecordAttrName(args.name);

  if (args.needsReverse)
    target = opBuilder.create<transform::ReverseOp>(
        target.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/target);

  if (args.needsAnnotate)
    opBuilder.create<transform::AnnotateOp>(
        target.getLoc(),
        /*target=*/target,
        /*name=*/opBuilder.getStringAttr(uniqueName),
        /*param=*/Value{});

  return NamedValueHandle(target, uniqueName, args.type, HandleStatus::kValid,
                          args.needsReverse);
}

RegularValueHandle
SchedulerBase::recordImpl(Value target, [[maybe_unused]] OpBuilder &opBuilder) {
  return RegularValueHandle(target, HandleStatus::kValid);
}

FuncArgHandle SchedulerBase::recordImpl(Value target,
                                        [[maybe_unused]] OpBuilder &opBuilder,
                                        size_t funcArgNum) {
  return FuncArgHandle(target, funcArgNum, HandleStatus::kValid);
}

LogicalResult
SchedulerBase::applyPatternSets(Operation *op,
                                const FrozenRewritePatternSet &patterns) const {
  if (failed(applyPatternsGreedily(op, patterns))) {
    return failure();
  }
  return success();
}

LogicalResult
SchedulerBase::applyOpFlattenPass(Operation *target,
                                  const FlattenOpsOptions &options) const {
  PassManager pm(getContext());
  pm.addPass(hfusion::createFlattenOpsPass(options));

  if (failed(pm.run(target)))
    return failure();

  RewritePatternSet patterns(getContext());
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (failed(applyPatternsGreedily(target, std::move(patterns))))
    return failure();

  return success();
}

FailureOr<SmallVector<func::FuncOp>> SchedulerBase::applyOpFusionOutline(
    func::FuncOp target, const HFusionOpFusionOptions &options) const {
  SmallVector<func::FuncOp> outlinedFuncs;
  if (failed(outlineFusedFuncs(target, options, outlinedFuncs)))
    return failure();

  HFusionOpFusionOptions singleOutlineOptions = options;
  singleOutlineOptions.fusionMode = this->kind_;
  if (failed(
          outlineSingleFusedFuncs(target, singleOutlineOptions, outlinedFuncs)))
    return failure();

  return outlinedFuncs;
}

LogicalResult
SchedulerBase::applyTensorResultToOutParamsPass(func::FuncOp target) {
  if (failed(markHACCInputArgAttr(target))) {
    target->emitWarning("Failed to mark input argument attribute.");
    return failure();
  }
  PassManager pm(getContext());
  SmallVector<std::string> includeSymbols = {target.getSymName().str()};
  TensorResToOutParamsOptions resToOutParamOptions;
  resToOutParamOptions.includeSymbols = includeSymbols;
  resToOutParamOptions.enableManageHostResources =
      options_.enableManageHostResources;
  pm.addPass(hfusion::createTensorResToOutParamsPass(resToOutParamOptions));
  if (failed(pm.run(getModule())))
    return failure();
  return success();
}

LogicalResult SchedulerBase::applyPackTilingDataPass(func::FuncOp target) {
  PassManager pm(getContext());
  SmallVector<std::string> includeSymbols = {target.getSymName().str()};
  PackTilingDataOptions packTilingDataOptions;
  packTilingDataOptions.includeSymbols = includeSymbols;
  pm.addPass(hfusion::createPackTilingDataPass(packTilingDataOptions));
  if (failed(pm.run(getModule())))
    return failure();
  return success();
}

LogicalResult SchedulerBase::applyCSEAndCanonicalizePass(
    func::FuncOp target, ArrayRef<std::string> disabledPatterns) const {
  PassManager pm(target->getContext());
  CanonicalizerOptions options;
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.addPass(createCSEPass());
  if (failed(pm.run(target))) {
    return target->emitError("Apply Canonicalizer && CSE error");
  }
  return success();
}

LogicalResult SchedulerBase::applyReCacheIOPass(func::FuncOp target) const {
  PassManager pm(getContext());
  pm.addPass(mlir::hfusion::createReCacheIO());
  if (failed(pm.run(target)))
    return failure();
  return success();
}

LogicalResult
SchedulerBase::applyAggressiveBubbleUpExtractSlice(func::FuncOp target) const {
  PassManager pm(getContext());
  BubbleUpExtractSliceOptions bubbleUpOptions;
  bubbleUpOptions.aggressive = true;
  pm.addPass(tensor::createBubbleUpExtractSlicePass(bubbleUpOptions));
  if (failed(pm.run(target)))
    return failure();
  return success();
}

LogicalResult SchedulerBase::applyMergeConsecutiveInsertExtractSlice(
    func::FuncOp target) const {
  PassManager pm(getContext());
  pm.addPass(tensor::createMergeConsecutiveInsertExtractSlicePass());
  if (failed(pm.run(target)))
    return failure();
  return success();
}

namespace {
struct AutoSchedulePass : public impl::AutoScheduleBase<AutoSchedulePass> {
  using AutoScheduleBase<AutoSchedulePass>::AutoScheduleBase;

public:
  explicit AutoSchedulePass(const AutoScheduleOptions &options)
      : AutoScheduleBase(options) {}

  void runOnOperation() override;

private:
  void setOptionsForFunc(AutoScheduleOptions &options, func::FuncOp func);
};

} // namespace

void AutoSchedulePass::runOnOperation() {
  AutoScheduleOptions options;
  SmallVector<func::FuncOp> funcList;
  getOperation()->walk([&](func::FuncOp func) { funcList.push_back(func); });

  for (auto &func : funcList) {
    OpBuilder opBuilder(&getContext());
    // set options individually for each function
    setOptionsForFunc(options, func);
    if (succeeded(SchedulerBase::applySchedule(func, opBuilder)))
      continue;

    func->emitOpError("Failed to create and apply schedule.");
    return signalPassFailure();
  }
}

void AutoSchedulePass::setOptionsForFunc(AutoScheduleOptions &options,
                                         func::FuncOp func) {
  options.enableAutoMultiBuffer = this->enableAutoMultiBuffer;
  options.enableDeterministicComputing = this->enableDeterministicComputing;
  options.maxBufferCntTuning = this->maxBufferCntTuning;
  options.enableCountBufferDmaOpt = this->enableCountBufferDmaOpt;
  options.enableManageHostResources = this->enableManageHostResources;
  options.cubeTilingTuning = this->cubeTilingTuning;

  auto maybeFusionKind = hfusion::tryGetFusionKind(func);
  // For cube and mix fusion kind, the block dim is set to half because cube
  // and vector is 1:2 for now.
  if (maybeFusionKind.has_value() &&
      ((*maybeFusionKind) == FusionKind::MixCV ||
       (*maybeFusionKind) == FusionKind::SingleCube ||
       (*maybeFusionKind) == FusionKind::ShallowCV)) {
    options.blockDim = std::max(this->blockDim / 2, (unsigned int)1);
  } else {
    options.blockDim = this->blockDim;
  }
  SchedulerBase::setAutoScheduleOptions(options);
}

std::unique_ptr<Pass> mlir::hfusion::createHFusionAutoSchedulePass(
    const AutoScheduleOptions &options) {
  return std::make_unique<AutoSchedulePass>(options);
}
