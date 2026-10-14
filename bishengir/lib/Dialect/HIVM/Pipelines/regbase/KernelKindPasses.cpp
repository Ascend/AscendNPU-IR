//===- KernelKindPasses.cpp - Kernel-kind pass filter -----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// HIVM kernel-kind lists and the classify-kernel follow-up. Both are
// generated from HIVMPass arguments on HIVM passes (`kernelKinds`,
// `classifyKernel`); this file only registers that metadata with the generic
// policy in bishengir/Pass/PassExecutionPolicy.h.
//
// A pass does not run when `hivm.module_core_type` maps to a kernel kind
// that is not in its list. The default set is all three kinds, so those
// passes still run for MIX, AIV, and AIC. `classifyKernel = true` runs
// InferFuncCoreType after the pass and replaces an existing core-type
// annotation. A function already split out of a mix kernel
// (`hivm.part_of_mix`) keeps its stamp, and its presence keeps the module
// core-type attribute. A declaration with no body also keeps its stamp:
// InferFuncCoreType skips external call-graph nodes, and a caller of an
// unannotated external function fails. Other functions in that module are
// still cleared.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Pass/PassExecutionPolicy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace hivm {
namespace regbase {
namespace {

void clearFuncCoreTypeAttrs(ModuleOp module) {
  // After SplitMixKernel, members carry hivm.part_of_mix and already have
  // their AIC/AIV core types. Leave those stamps and the module core type
  // alone. That member does not cancel the clear for the other functions.
  bool partOfMix = false;
  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(hivm::TPartOfMixAttr::name)) {
      partOfMix = true;
      return;
    }
    // SplitSimtModule leaves the SIMT VF as a private declaration in the
    // SIMD module. There is no body to re-infer, and dropping the stamp
    // makes the caller's core-type inference fail.
    if (func.isExternal())
      return;
    func->removeAttr(hivm::TFuncCoreTypeAttr::name);
  });
  if (!partOfMix)
    module->removeAttr(hivm::TModuleCoreTypeAttr::name);
}

/// classifyKernel follow-up. maybeClassifyKernel only calls this when the bit
/// is set, so an existing core type is replaced from the current IR.
LogicalResult classifyKernelAfterPass(Operation *op) {
  Operation *moduleOp = op;
  while (moduleOp && !isa<ModuleOp>(moduleOp))
    moduleOp = moduleOp->getParentOp();
  if (!moduleOp)
    return success();

  auto module = cast<ModuleOp>(moduleOp);
  // InferFuncCoreType keeps an existing annotation; drop it so this follow-up
  // replaces the stamp left by an earlier classification.
  clearFuncCoreTypeAttrs(module);

  PassManager nested(module->getContext());
  nested.addPass(createInferFuncCoreTypePass());
  return nested.run(module);
}

} // namespace

void registerKernelKindPassesForRegbase() {
  // Idempotent: the registries are sets and this runs once per process.
  // Called from the public RegBase pipeline builders, so it does not depend
  // on the linker keeping a file-scope initializer.
  static const bool registered = [] {
    // Each pass is registered for every kind in its HIVMPass kernelKinds
    // list. A later module whose core type is not in that list skips the
    // pass. The default list is every kind.
    //
    // Deliberately left at the default set:
    //   - hivm-infer-func-core-type: still does the AIV-side work when the
    //     module core type is AIV.
    // hivm-insert-load-store-for-mix-cv lists Mix and pure AIC. Its Aiv
    // substitute calls setOperationsCoreType with VECTOR instead of inserting
    // the cube/vector boundary.
    // hivm-mark-real-core-type lists Mix and pure AIC. Its Aiv substitute
    // stamps VECTOR on scalar-pipe ops instead of cloning and splitting, and
    // still honors remove-core-type-attrs on the cleanup invocation.
    bishengir::registerKernelKindPasses(bishengir::PipelineMode::Mix, {
#define GEN_MIX_KERNEL_PASS_ARGS
#include "bishengir/Dialect/HIVM/Transforms/PassMetadata.inc"
#undef GEN_MIX_KERNEL_PASS_ARGS
                                                                      });
    bishengir::registerKernelKindPasses(bishengir::PipelineMode::Aiv, {
#define GEN_PURE_AIV_KERNEL_PASS_ARGS
#include "bishengir/Dialect/HIVM/Transforms/PassMetadata.inc"
#undef GEN_PURE_AIV_KERNEL_PASS_ARGS
                                                                      });
    bishengir::registerKernelKindPasses(bishengir::PipelineMode::Aic, {
#define GEN_PURE_AIC_KERNEL_PASS_ARGS
#include "bishengir/Dialect/HIVM/Transforms/PassMetadata.inc"
#undef GEN_PURE_AIC_KERNEL_PASS_ARGS
                                                                      });
    bishengir::registerClassifyKernelPasses({
#define GEN_CLASSIFY_KERNEL_PASS_ARGS
#include "bishengir/Dialect/HIVM/Transforms/PassMetadata.inc"
#undef GEN_CLASSIFY_KERNEL_PASS_ARGS
    });
    bishengir::registerAivSubstitute("hivm-insert-load-store-for-mix-cv",
                                     substituteInsertLoadStoreForMixCV);
    bishengir::registerAivSubstitute("hivm-mark-real-core-type",
                                     substituteMarkRealCoreType);
    return true;
  }();
  (void)registered;
  // Always refresh: unit tests may replace the callback, and pipeline
  // builders call this before each run.
  bishengir::registerClassifyKernelCallback(classifyKernelAfterPass);
}

} // namespace regbase
} // namespace hivm
} // namespace mlir
