//===- OpFusion.cpp -- Outline fusible ops into kernels -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements op fusion algorithm and outline into functions.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockOutliner.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "hfusion-op-fusion"

namespace mlir {
#define GEN_PASS_DEF_HFUSIONOPFUSION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {

inline std::optional<HFusionOpFusionOptions>
getOptionFromLabel(func::FuncOp func, const HFusionOpFusionOptions &options) {
  HFusionOpFusionOptions newOptions = options;
  auto fusionKindAttr =
      func->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr)
    return std::nullopt;
  auto fusionKind = fusionKindAttr.getFusionKind();
  newOptions.fusionMode = fusionKind;
  return newOptions;
}

static const SmallVector<FusionKind> kMultiKernelOrder = {
    FusionKind::ShallowCV, FusionKind::ShallowVV,   FusionKind::MixCV,
    FusionKind::MixC2,     FusionKind::LastAxisPBR, FusionKind::AnyPB};

LogicalResult outlineMultiKernel(func::FuncOp entryFunc,
                                 const HFusionOpFusionOptions &options,
                                 SmallVector<func::FuncOp> &outlinedFuncs) {
  std::optional<HFusionOpFusionOptions> newOption =
      hfusion::getOptionFromLabel(entryFunc, options);
  bool enableShallowFusion =
      newOption.has_value() &&
      FusibleHelper::isShallowFusion(newOption.value().fusionMode);
  // TODO: refactor ShallowCV and remove enableShallowCV
  bool enableShallowCV = newOption.has_value() &&
                         newOption.value().fusionMode == FusionKind::ShallowCV;
  for (auto fusionMode : kMultiKernelOrder) {
    if (enableShallowFusion && FusibleHelper::isShallowFusion(fusionMode))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Trying fusion mode "
                            << stringifyFusionKind(fusionMode) << "\n";);

    // TODO: For outline multi kernel, mix cv output mode should be single
    // This asserts no horizontal fusion is happening
    auto usedOutputMode = options.outputMode;
    auto horizontalMax = options.maxHorizontalFusionSize;
    if (fusionMode == FusionKind::MixCV) {
      horizontalMax = 0;
      usedOutputMode = OutputMode::Single;
    }

    FusibleHelper fusibleHelper(fusionMode, options.moveOutToParam,
                                horizontalMax);
    FusibleBlocks fusibleBlocks = getFusibleBlocks(entryFunc, fusibleHelper);
    if (fusibleBlocks.empty())
      continue;

    /// For multi-kernel mode, we assume that the runtime has the ability to
    /// analyze and manage host resources, including analyzing and optimizing
    /// tensor alias. So we don't need to store an extra result.

    HFusionOpFusionOptions newOptions = options;
    newOptions.outputMode = usedOutputMode;
    FusibleBlockOutliner outliner(fusibleBlocks,
                                  OutlineFuncOptions(newOptions, entryFunc),
                                  /*shouldRemoveDuplicateAliasOuts=*/true);

    const std::string prefixMulti =
        "_multi_" + stringifyFusionKind(fusionMode).str();
    if (!outliner.outline(prefixMulti))
      return failure();

    // TODO: refactor ShallowCV and remove enableShallowCV
    if (FusibleHelper::isShallowFusion(fusionMode) && !enableShallowCV) {
      for (const func::FuncOp func : outliner.getOutlinedFuncs()) {
        if (failed(outlineMultiKernel(func, options, outlinedFuncs)))
          return failure();
        return success();
      }
    } else {
      outlinedFuncs.append(outliner.getOutlinedFuncs());
    }
  }
  return success();
}
} // namespace

LogicalResult outlineFusedFuncs(func::FuncOp entryFunc,
                                const HFusionOpFusionOptions &options,
                                SmallVector<func::FuncOp> &outlinedFuncs) {
  if (options.fusionMode == FusionKind::Unknown)
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Outlining function "
                          << (hacc::utils::isHost(entryFunc) ? "" : "Not")
                          << " Heterogeneous\n";);
  FusibleHelper fusibleHelper(options.fusionMode, options.moveOutToParam,
                              options.maxHorizontalFusionSize);

  // RVO will optimize this
  FusibleBlocks fusibleBlocks = getFusibleBlocks(entryFunc, fusibleHelper);
  if (fusibleBlocks.empty())
    return success();

  FusibleBlockOutliner outliner(fusibleBlocks,
                                OutlineFuncOptions(options, entryFunc));

  if (!outliner.outline())
    return failure();

  // TODO : refactor shallow cv and add shallow cv here
  if (FusibleHelper::isShallowFusion(options.fusionMode) &&
      options.fusionMode != FusionKind::ShallowCV) {
    for (func::FuncOp func : outliner.getOutlinedFuncs()) {
      if (failed(outlineMultiKernel(func, options, outlinedFuncs)))
        return failure();
    }
  } else {
    outlinedFuncs.append(outliner.getOutlinedFuncs());
  }

  return success();
}

LogicalResult
outlineSingleFusedFuncs(func::FuncOp entryFunc,
                        const HFusionOpFusionOptions &options,
                        SmallVector<func::FuncOp> &outlinedFuncs) {
  size_t funcCnt = 0;
  auto outlineSingleOp = [&options, &funcCnt, &outlinedFuncs,
                          &entryFunc](func::FuncOp &func,
                                      Operation *op) -> LogicalResult {
    OpBuilder builder(func.getContext());
    builder.setInsertionPoint(func);
    // RVO will optimize this

    auto singleOpFusionKind = opfusion::FusibleHelper::getSingleFusionKind(op);
    // TODO: Handle this better later
    // Currently, since single cube schedule requires a tiling function,
    // we cannot outline it during ShallowCV fusion.
    if (singleOpFusionKind == FusionKind::SingleCube &&
        options.fusionMode == FusionKind::ShallowCV)
      return success();

    FusibleHelper fusibleHelper(singleOpFusionKind, options.moveOutToParam,
                                options.maxHorizontalFusionSize);

    SmallVector<Operation *> ops = {op};
    FusibleBlocks fusibleBlocks = {FusibleBlock(ops, &fusibleHelper)};
    FusibleBlockOutliner outliner(fusibleBlocks,
                                  OutlineFuncOptions(options, entryFunc));
    if (!outliner.outline("_single_outlined_" + std::to_string(funcCnt++)))
      return failure();
    outlinedFuncs.append(outliner.getOutlinedFuncs());
    return success();
  };

  LLVM_DEBUG(llvm::dbgs() << "Running single op\n";);
  // handle case where the removed dead operations are after the iterated
  // operation so that it doesn't iterate the erased operation.
  DenseSet<Operation *> visitedOperations;
  while (true) {
    bool outlined = false;
    for (Operation &op : entryFunc.getOps()) {
      if (visitedOperations.contains(&op))
        continue;
      visitedOperations.insert(&op);
      LLVM_DEBUG(llvm::dbgs() << "Checking op " << op << "\n";);
      if (opfusion::FusibleHelper::isSingleOutlinable(&op)) {
        LLVM_DEBUG(llvm::dbgs() << "Outlinable\n";);
        if (failed(outlineSingleOp(entryFunc, &op)))
          return failure();
        outlined = true;
        break;
      }
    };
    if (!outlined)
      break;
  }

  return success();
}

} // namespace hfusion

//===---------------------------------------------------------------------===//
// Pass
//===---------------------------------------------------------------------===//

struct HFusionOpFusionPass
    : public impl::HFusionOpFusionBase<HFusionOpFusionPass> {
  explicit HFusionOpFusionPass(const HFusionOpFusionOptions &options)
      : HFusionOpFusionBase(options) {}
  void initOptions() {
    options_.outputMode = this->outputMode;
    options_.fusionMode = this->fusionMode;
    options_.alwaysInline = this->alwaysInline;
    options_.moveOutToParam = this->moveOutToParam;
    options_.maxHorizontalFusionSize = this->maxHorizontalFusionSize;
    options_.enableMultiKernel = this->enableMultiKernel;
  }

  // For all instance, InferFuncFusionKind is run
  // device + multi-kernel=false: fusion[no outline, hard assert infer success]
  // device + multi-kernel=true : fusion[emit error, no such cases]
  // host   + multi-kernel=false: fusion[assert(FusionKind != Unknown) and
  // outline device, keep signature]
  // host   + multi-kernel=true : fusion[if unknown, try to outline multiple
  // device kernels; else: outline according to fusion kind]
  void runOnOperation() override {
    initOptions();
    // This is a module pass to avoid function making and calling issues
    getOperation()->walk([&](func::FuncOp func) -> void {
      if (hacc::utils::isDevice(func) && enableMultiKernel) {
        func->emitOpError("enableMultiKernel not supported in Device mode");
        return signalPassFailure();
      }
      std::optional<HFusionOpFusionOptions> newOption =
          hfusion::getOptionFromLabel(func, options_);
      bool enableShallowFusion =
          newOption.has_value() &&
          FusibleHelper::isShallowFusion(newOption.value().fusionMode);
      // TODO: refactor ShallowCV and remove enableShallowCV
      bool enableShallowCV =
          newOption.has_value() &&
          newOption.value().fusionMode == FusionKind::ShallowCV;

      [[maybe_unused]] SmallVector<func::FuncOp> outlinedFuncs;
      if (enableShallowFusion && !enableShallowCV) {
        if (failed(
                outlineMultiKernel(func, newOption.value(), outlinedFuncs))) {
          func.emitOpError("outline shallow multi kernel failed");
          return signalPassFailure();
        }
      }
      // For non-shallow fusion kinds, there is no need to outline from device
      // functions.
      if (!hacc::utils::isHost(func)) {
        return;
      }

      bool emptyOrUnknownFusionKind = !newOption.has_value();
      emptyOrUnknownFusionKind |=
          newOption.has_value() &&
          newOption.value().fusionMode == FusionKind::Unknown;
      LLVM_DEBUG(llvm::dbgs() << emptyOrUnknownFusionKind << " "
                              << enableMultiKernel << "\n";);
      if (enableMultiKernel && emptyOrUnknownFusionKind) {
        // - Host function
        // - Enable multi kernel
        // - Do multi kernel if failed to infer into single func
        //   (fusion kind is unknown)
        if (failed(outlineMultiKernel(func, options_, outlinedFuncs))) {
          func->emitOpError("outline multi kernel failed");
          return signalPassFailure();
        }
      } else {
        // Return by reference of this outlinedFuncs
        if (newOption &&
            failed(outlineFusedFuncs(func, newOption.value(), outlinedFuncs))) {
          func->emitOpError("outline single kernel failed");
          return signalPassFailure();
        }
      }
    });
  }

private:
  HFusionOpFusionOptions options_;
};
} // namespace mlir

std::unique_ptr<Pass> mlir::hfusion::createHFusionOpFusionPass(
    const HFusionOpFusionOptions &options) {
  return std::make_unique<HFusionOpFusionPass>(options);
}
