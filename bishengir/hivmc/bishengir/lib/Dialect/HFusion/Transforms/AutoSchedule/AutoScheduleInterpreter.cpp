//===- TestTransformDialectInterpreter.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::detail;

namespace {

/// Finds an operation nested in `root` that has the transform dialect tag
/// attribute with the value specified as `tag`. Assumes only one operation
/// may have the tag. Returns nullptr if there is no such operation.
static Operation *findOpWithTag(Operation *root, StringRef tagKey,
                                StringRef tagValue) {
  Operation *found = nullptr;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
      [tagKey, tagValue, &found, root](Operation *op) {
        auto attr = op->getAttrOfType<StringAttr>(tagKey);
        if (!attr || attr.getValue() != tagValue)
          return WalkResult::advance();

        if (found) {
          InFlightDiagnostic diag = root->emitError()
                                    << "more than one operation with " << tagKey
                                    << "=\"" << tagValue << "\" attribute";
          diag.attachNote(found->getLoc()) << "first operation";
          diag.attachNote(op->getLoc()) << "other operation";
          return WalkResult::interrupt();
        }

        found = op;
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return nullptr;

  if (!found) {
    root->emitError() << "could not find the operation with " << tagKey << "=\""
                      << tagValue << "\" attribute";
  }
  return found;
}

class AutoScheduleInterpreterPass
    : public mlir::PassWrapper<AutoScheduleInterpreterPass,
                               OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoScheduleInterpreterPass)

  AutoScheduleInterpreterPass() = default;
  AutoScheduleInterpreterPass &
  operator=(const AutoScheduleInterpreterPass &pass) = delete;
  AutoScheduleInterpreterPass(const AutoScheduleInterpreterPass &pass)
      : PassWrapper(pass) {
    this->kernelName = pass.kernelName;
    this->enforceSingleToplevelTransformOp =
        pass.enforceSingleToplevelTransformOp;
    this->enableExpensiveChecks = pass.enableExpensiveChecks;
    this->debugPayloadRootTag = pass.debugPayloadRootTag;
    this->debugTransformRootTag = pass.debugTransformRootTag;
    this->options = pass.options;
  }
  AutoScheduleInterpreterPass(
      const transform::TransformOptions &transformOptions,
      const std::string &kernelName)
      : PassWrapper() {
    this->kernelName = kernelName;
    this->enforceSingleToplevelTransformOp =
        transformOptions.getEnforceSingleToplevelTransformOp();
    this->enableExpensiveChecks = transformOptions.getExpensiveChecksEnabled();
  }

  StringRef getArgument() const override {
    return "hfusion-auto-schedule-interpreter";
  }

  StringRef getDescription() const override {
    return "apply auto schedule transform sequence to target kernel";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    options
        .enableEnforceSingleToplevelTransformOp(
            enforceSingleToplevelTransformOp)
        .enableExpensiveChecks(enableExpensiveChecks);

    auto payloadRootTag = kernelName.empty()
                              ? debugPayloadRootTag
                              : auto_schedule::getPayloadRootTag(kernelName);
    Operation *payloadRoot = findOpWithTag(
        getOperation(), transform::TransformDialect::kTargetTagAttrName,
        payloadRootTag);
    if (!payloadRoot)
      return signalPassFailure();

    auto transformRootTag =
        kernelName.empty() ? debugTransformRootTag
                           : auto_schedule::getTransformRootTag(kernelName);
    Operation *transformEntryPoint = findOpWithTag(
        getOperation(), transform::TransformDialect::kTargetTagAttrName,
        transformRootTag);
    if (!transformEntryPoint)
      return signalPassFailure();

    if (failed(transform::applyTransformNamedSequence(
            payloadRoot, transformEntryPoint, /*transformModule=*/{},
            options))) {
      return signalPassFailure();
    }
  }

protected:
  Option<std::string> kernelName{
      *this, "kernel-name", llvm::cl::init(""),
      llvm::cl::desc(
          "The kernel function name to apply the schedule sequence.")};

  Option<bool> enableExpensiveChecks{
      *this, "enable-expensive-checks", llvm::cl::init(false),
      llvm::cl::desc("Perform expensive checks to better report errors in the "
                     "transform IR")};
  Option<bool> enforceSingleToplevelTransformOp{
      *this, "enforce-single-top-level-transform-op", llvm::cl::init(true),
      llvm::cl::desc("Ensure that only a single top-level transform op is "
                     "present in the IR.")};

  Option<std::string> debugPayloadRootTag{
      *this, "debug-payload-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as payload IR root. If empty select the pass anchor "
          "operation as the payload IR root.")};
  Option<std::string> debugTransformRootTag{
      *this, "debug-transform-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as container IR for top-level transform ops. This "
          "allows user control on what transformation to apply. If empty, "
          "select the container of the top-level transform op.")};

  /// Transform interpreter options.
  mlir::transform::TransformOptions options{};
};

struct EraseAutoSchedulePass
    : public PassWrapper<EraseAutoSchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseAutoSchedulePass)

  EraseAutoSchedulePass() = default;
  EraseAutoSchedulePass &operator=(const EraseAutoSchedulePass &pass) = delete;
  explicit EraseAutoSchedulePass(const EraseAutoSchedulePass &pass)
      : PassWrapper(pass) {
    this->kernelName = pass.kernelName;
  };
  explicit EraseAutoSchedulePass(const std::string &kernelName) {
    this->kernelName = kernelName;
  }

  StringRef getArgument() const final { return "hfusion-erase-auto-schedule"; }

  StringRef getDescription() const final {
    return "erase auto schedule transform sequence from the IR";
  }

  void runOnOperation() override {
    auto targetPayloadRootTag = auto_schedule::getPayloadRootTag(kernelName);
    auto targetTransformRootTag =
        auto_schedule::getTransformRootTag(kernelName);
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (!nestedOp->hasAttrOfType<StringAttr>(
              transform::TransformDialect::kTargetTagAttrName)) {
        return WalkResult::advance();
      }
      if (isa<transform::TransformOpInterface>(nestedOp) &&
          nestedOp->getAttrOfType<StringAttr>(
                      transform::TransformDialect::kTargetTagAttrName)
                  .str() == targetTransformRootTag) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      if (isa<func::FuncOp>(nestedOp) &&
          nestedOp->getAttrOfType<StringAttr>(
                      transform::TransformDialect::kTargetTagAttrName)
                  .str() == targetPayloadRootTag) {
        nestedOp->removeAttr(transform::TransformDialect::kTargetTagAttrName);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }

  Option<std::string> kernelName{
      *this, "kernel-name", llvm::cl::init(""),
      llvm::cl::desc("Erase transform sequence for target kernel")};
};

} // namespace

namespace mlir {
namespace hfusion {
void registerEraseAutoSchedulePass() {
  PassRegistration<EraseAutoSchedulePass> reg;
}

void registerAutoScheduleInterpreterPass() {
  PassRegistration<AutoScheduleInterpreterPass> reg;
}

std::unique_ptr<Pass>
createAutoScheduleInterpreterPass(const std::string &kernelName,
                                  transform::TransformOptions options) {
  return std::make_unique<AutoScheduleInterpreterPass>(options, kernelName);
}

std::unique_ptr<Pass>
createEraseAutoSchedulePass(const std::string &kernelName) {
  return std::make_unique<EraseAutoSchedulePass>(kernelName);
}

} // namespace hfusion
} // namespace mlir
