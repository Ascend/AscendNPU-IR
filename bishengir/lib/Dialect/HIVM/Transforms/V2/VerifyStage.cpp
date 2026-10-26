//===- VerifyStage.cpp - V2 pipeline stage verifier ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compile-time MIX stage verifier; never rewrites IR.
// Stages: tensor, bufferized, planned, solved, split, tail.
//
// TODO(v2): this is a debug aid for bringing up the v2 pipeline. It asserts
// the intermediate IR contracts (tensor vs buffer, unsplit vs split, etc.)
// and must not stay in the final pipeline; remove it once the stages are
// stable. The stage mechanism is meant to be as short-lived as decompose.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Pipelines/v2/TargetUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_V2VERIFYSTAGE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Function-level view of the module at one stage.
struct StageState {
  StageState(ModuleOp module, StringRef stageName)
      : module(module), stageName(stageName) {}

  ModuleOp module;
  StringRef stageName;
  SmallVector<func::FuncOp> deviceFuncs;
  SmallVector<func::FuncOp> mixFuncs;
  SmallVector<func::FuncOp> partOfMixFuncs;
  unsigned entryCount = 0;
};

static LogicalResult fail(StageState &state, const Twine &message) {
  state.module.emitError() << "hivm-v2-verify-stage[" << state.stageName
                           << "]: " << message;
  return failure();
}

/// Walk nested regions and block arguments.
static bool hasTypeMatching(Operation *op,
                            llvm::function_ref<bool(Type)> matches) {
  WalkResult result = op->walk([&](Operation *inner) {
    for (Value operand : inner->getOperands())
      if (matches(operand.getType()))
        return WalkResult::interrupt();
    for (Value result : inner->getResults())
      if (matches(result.getType()))
        return WalkResult::interrupt();
    for (Region &region : inner->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (matches(arg.getType()))
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool hasTensorType(Operation *op) {
  return hasTypeMatching(op, [](Type type) { return isa<TensorType>(type); });
}

static bool hasMemRefType(Operation *op) {
  return hasTypeMatching(op,
                         [](Type type) { return isa<BaseMemRefType>(type); });
}

/// Every pre-split stage expects one unsplit MIX definition.
static LogicalResult verifyUnsplitMix(StageState &state) {
  if (state.mixFuncs.size() != 1 || !state.partOfMixFuncs.empty())
    return fail(state, "expected one unsplit MIX function and no part_of_mix "
                       "clones");
  return success();
}

static LogicalResult verifyTensorStage(StageState &state) {
  if (failed(verifyUnsplitMix(state)))
    return failure();
  Operation *func = state.mixFuncs.front().getOperation();
  if (hasMemRefType(func) && !hasTensorType(func))
    return fail(state, "tensor phase produced memref IR before bufferization");
  return success();
}

static LogicalResult verifyBufferizedStage(StageState &state) {
  if (failed(verifyUnsplitMix(state)))
    return failure();
  for (func::FuncOp func : state.deviceFuncs)
    if (hasTensorType(func.getOperation()))
      return fail(state, "tensor types survived bufferization in @" +
                             func.getSymName());
  return success();
}

static LogicalResult verifySplitStage(StageState &state) {
  if (!state.mixFuncs.empty())
    return fail(state, "MIX function survived the split");
  if (state.partOfMixFuncs.size() != 2)
    return fail(state, "expected two part_of_mix clones, found " +
                           Twine(state.partOfMixFuncs.size()));

  bool hasAic = false;
  bool hasAiv = false;
  for (func::FuncOp func : state.partOfMixFuncs) {
    auto coreType = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
        hivm::TFuncCoreTypeAttr::name);
    if (!coreType)
      return fail(state, "part_of_mix clone @" + func.getSymName() +
                             " has no func_core_type");
    hasAic |= coreType.getFuncCoreType() == hivm::TFuncCoreType::AIC;
    hasAiv |= coreType.getFuncCoreType() == hivm::TFuncCoreType::AIV;
  }
  if (!hasAic || !hasAiv)
    return fail(state, "split did not produce one AIC and one AIV clone");
  return success();
}

static LogicalResult verifyTailStage(StageState &state) {
  if (hasTensorType(state.module.getOperation()))
    return fail(state, "tensor types survived the per-core tail");
  return success();
}

static LogicalResult verifyStage(StageState &state) {
  if (state.stageName == "tensor")
    return verifyTensorStage(state);
  if (state.stageName == "bufferized")
    return verifyBufferizedStage(state);
  if (state.stageName == "planned" || state.stageName == "solved")
    return verifyUnsplitMix(state);
  if (state.stageName == "split")
    return verifySplitStage(state);
  if (state.stageName == "tail")
    return verifyTailStage(state);
  return fail(state,
              "unknown stage name (expected tensor, bufferized, planned, "
              "solved, split or tail)");
}

struct V2VerifyStagePass : public impl::V2VerifyStageBase<V2VerifyStagePass> {
  using Base::Base;
  void runOnOperation() override {
    StageState state(getOperation(), stage.getValue());

    auto target = hacc::utils::getTargetDevice(state.module);
    if (!target || !mlir::hivm::v2::isSupportedTarget(*target)) {
      state.module.emitError()
          << "hivm-v2-verify-stage expects a reg-based on-chip CV target, got "
          << (target ? hacc::stringifyTargetDeviceEnum(*target)
                     : StringRef("<unset>"));
      return signalPassFailure();
    }

    state.module.walk([&](func::FuncOp func) {
      if (hacc::utils::isHost(func))
        return;
      // Ignore the private MIX declaration emitted by the split.
      if (func.isExternal())
        return;

      state.deviceFuncs.push_back(func);
      if (hacc::utils::isDeviceEntry(func))
        ++state.entryCount;

      auto coreType = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
          hivm::TFuncCoreTypeAttr::name);
      if (coreType && coreType.getFuncCoreType() == hivm::TFuncCoreType::MIX)
        state.mixFuncs.push_back(func);
      if (func->hasAttr(hivm::TPartOfMixAttr::name))
        state.partOfMixFuncs.push_back(func);
    });

    // Both clones are entries after the split; one entry before it.
    if (state.stageName != "split" && state.stageName != "tail" &&
        state.entryCount != 1) {
      state.module.emitError() << "hivm-v2-verify-stage: expected exactly one "
                                  "device entry function, found "
                               << state.entryCount;
      return signalPassFailure();
    }

    if (failed(verifyStage(state)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::hivm::createV2VerifyStagePass(const V2VerifyStageOptions &options) {
  return std::make_unique<V2VerifyStagePass>(options);
}
