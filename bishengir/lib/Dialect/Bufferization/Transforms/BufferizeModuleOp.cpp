//===- BufferizeModuleOp.cpp - Bufferize a module with function boundaries ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass rewrites tensor IR to memref IR, bufferizing a module and its
// function boundaries. It is the bufferization (rewrite) phase of
// `one-shot-bufferize`: run `tensor-copy-insertion` first to analyze the IR
// and insert `bufferization.alloc_tensor` copies, then run this pass to
// rewrite the IR without re-running the analysis.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_BUFFERIZEMODULEOP
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;
using namespace mlir::bufferization;

static LayoutMapOption parseLayoutMapOption(const std::string &s) {
  if (s == "fully-dynamic-layout-map")
    return LayoutMapOption::FullyDynamicLayoutMap;
  if (s == "identity-layout-map")
    return LayoutMapOption::IdentityLayoutMap;
  if (s == "infer-layout-map")
    return LayoutMapOption::InferLayoutMap;
  llvm_unreachable("invalid layout map option");
}

namespace {
struct BufferizeModuleOpPass
    : public bishengir::bufferization::impl::BufferizeModuleOpBase<
          BufferizeModuleOpPass> {
  BufferizeModuleOpPass() = default;

  explicit BufferizeModuleOpPass(
      const mlir::bufferization::OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Use externally provided options if any; otherwise fall back to the
    // options specified on the command line.
    OneShotBufferizationOptions opt;
    if (options) {
      opt = *options;
    } else {
      // Make new bufferization options if none were provided when creating
      // the pass.
      opt.allowUnknownOps = allowUnknownOps;
      opt.copyBeforeWrite = copyBeforeWrite;
      opt.setFunctionBoundaryTypeConversion(
          parseLayoutMapOption(functionBoundaryTypeConversion));
      if (mustInferMemorySpace) {
        opt.defaultMemorySpaceFn =
            [](TensorType t) -> std::optional<Attribute> {
          return std::nullopt;
        };
      }
      opt.noAnalysisFuncFilter = noAnalysisFuncFilter;

      // Configure type converter.
      LayoutMapOption unknownTypeConversionOption =
          parseLayoutMapOption(unknownTypeConversion);
      if (unknownTypeConversionOption == LayoutMapOption::InferLayoutMap) {
        emitError(UnknownLoc::get(&getContext()),
                  "Invalid option: 'infer-layout-map' is not a valid value "
                  "for 'unknown-type-conversion'");
        return signalPassFailure();
      }
      opt.unknownTypeConverterFn = [=](Value value, Attribute memorySpace,
                                       const BufferizationOptions &options) {
        auto tensorType = cast<TensorType>(value.getType());
        if (unknownTypeConversionOption == LayoutMapOption::IdentityLayoutMap)
          return bufferization::getMemRefTypeWithStaticIdentityLayout(
              tensorType, memorySpace);
        assert(unknownTypeConversionOption ==
                   LayoutMapOption::FullyDynamicLayoutMap &&
               "invalid layout map option");
        return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                                  memorySpace);
      };

      // Configure op filter.
      OpFilter::Entry::FilterFn filterFn = [&](Operation *op) {
        // Filter may be specified via options.
        if (this->dialectFilter.hasValue())
          return llvm::is_contained(this->dialectFilter,
                                    op->getDialect()->getNamespace());
        // No filter specified: All other ops are allowed.
        return true;
      };
      opt.opFilter.allowOperation(filterFn);
    }

    // This pass bufferizes module-level function boundaries.
    opt.bufferizeFunctionBoundaries = true;

    BufferizationStatistics statistics;
    if (failed(bufferizeModuleOp(moduleOp, opt, &statistics)))
      return signalPassFailure();

    // Set pass statistics.
    this->numBufferAlloc = statistics.numBufferAlloc;
  }

private:
  std::optional<mlir::bufferization::OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass> bishengir::bufferization::createBufferizeModuleOpPass() {
  return std::make_unique<BufferizeModuleOpPass>();
}

std::unique_ptr<Pass> bishengir::bufferization::createBufferizeModuleOpPass(
    const mlir::bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<BufferizeModuleOpPass>(options);
}
