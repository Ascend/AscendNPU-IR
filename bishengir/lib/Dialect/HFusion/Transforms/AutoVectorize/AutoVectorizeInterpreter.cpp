//===- AutoVectorizeInterpreter.cpp - Apply vectorize transform sequence --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Attrs.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <optional>

using namespace mlir;
using namespace mlir::hfusion;

namespace {

static StringRef getTargetTagName() {
  return transform::TransformDialect::kTargetTagAttrName;
}

static StringAttr getTargetTag(Operation *op) {
  return op->getAttrOfType<StringAttr>(getTargetTagName());
}

static std::optional<StringRef>
getAutoVectorizeTransformFuncName(transform::SequenceOp seqOp) {
  StringAttr tag = getTargetTag(seqOp);
  if (!tag)
    return std::nullopt;

  StringRef value = tag.getValue();
  if (!value.consume_front(auto_vectorize::kTransformRootTagPrefix))
    return std::nullopt;
  return value;
}

static func::FuncOp findPayloadRoot(ModuleOp module,
                                    transform::SequenceOp seqOp) {
  std::optional<StringRef> funcName = getAutoVectorizeTransformFuncName(seqOp);
  if (!funcName)
    return {};

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(*funcName);
  if (!func) {
    auto diag = module.emitError()
                << "could not find AutoVectorize payload function \""
                << *funcName << "\"";
    diag.attachNote(seqOp.getLoc())
        << "referenced from this transform sequence";
    return {};
  }

  std::string expectedTag = auto_vectorize::getPayloadRootTag(*funcName);
  StringAttr payloadTag = getTargetTag(func);
  if (!payloadTag || payloadTag.getValue() != expectedTag) {
    auto diag = func.emitError() << "expected " << getTargetTagName() << "=\""
                                 << expectedTag << "\"";
    diag.attachNote(seqOp.getLoc())
        << "referenced from this AutoVectorize transform sequence";
    return {};
  }

  return func;
}

class AutoVectorizeInterpreterPass
    : public PassWrapper<AutoVectorizeInterpreterPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoVectorizeInterpreterPass)

  AutoVectorizeInterpreterPass() = default;
  AutoVectorizeInterpreterPass &
  operator=(const AutoVectorizeInterpreterPass &pass) = delete;
  AutoVectorizeInterpreterPass(const AutoVectorizeInterpreterPass &pass)
      : PassWrapper(pass) {
    this->enableExpensiveChecks = pass.enableExpensiveChecks;
  }

  StringRef getArgument() const override {
    return "hfusion-auto-vectorize-interpreter";
  }

  StringRef getDescription() const override {
    return "apply emitted auto vectorize transform sequences";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<transform::SequenceOp> sequences;
    module.walk([&](transform::SequenceOp seqOp) {
      if (getAutoVectorizeTransformFuncName(seqOp))
        sequences.push_back(seqOp);
    });

    if (sequences.empty()) {
      module.emitError()
          << "could not find an AutoVectorize transform sequence";
      return signalPassFailure();
    }

    transform::TransformOptions options;
    options.enableExpensiveChecks(enableExpensiveChecks);

    for (transform::SequenceOp seqOp : sequences) {
      func::FuncOp func = findPayloadRoot(module, seqOp);
      if (!func)
        return signalPassFailure();
      if (failed(transform::applyTransformNamedSequence(
              func, seqOp, /*transformModule=*/{}, options)))
        return signalPassFailure();
    }
  }

protected:
  Option<bool> enableExpensiveChecks{
      *this, "enable-expensive-checks", llvm::cl::init(false),
      llvm::cl::desc("Perform expensive checks to better report errors in the "
                     "transform IR")};
};

} // namespace

namespace mlir {
namespace hfusion {

void registerAutoVectorizeInterpreterPass() {
  PassRegistration<AutoVectorizeInterpreterPass> reg;
}

std::unique_ptr<Pass> createAutoVectorizeInterpreterPass() {
  return std::make_unique<AutoVectorizeInterpreterPass>();
}

} // namespace hfusion
} // namespace mlir
