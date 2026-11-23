//===- PassExecutionPolicy.cpp - Pipeline-wide pass filtering ---*- C++ -*-===//
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

#include "bishengir/Pass/PassExecutionPolicy.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace bishengir {

//===----------------------------------------------------------------------===//
// PipelineMode detection
//===----------------------------------------------------------------------===//

PipelineMode detectPipelineMode(ModuleOp module) {
  // One attribute lookup. Do not walk functions: the module core type is the
  // summary of the device functions. A missing or unrecognized value does
  // not select a kernel kind.
  auto attr = module->getAttrOfType<hivm::TModuleCoreTypeAttr>(
      hivm::TModuleCoreTypeAttr::name);
  if (!attr)
    return PipelineMode::Unspecified;
  switch (attr.getModuleCoreType()) {
  case hivm::TModuleCoreType::AIV:
    return PipelineMode::Aiv;
  case hivm::TModuleCoreType::AIC:
    return PipelineMode::Aic;
  case hivm::TModuleCoreType::MIX:
    return PipelineMode::Mix;
  }
  return PipelineMode::Unspecified;
}

//===----------------------------------------------------------------------===//
// Kernel-kind applicability
//===----------------------------------------------------------------------===//

namespace {
/// Bit set of PipelineMode values a pass applies to. Unspecified is not a
/// kind and has no bit. Function-local so construction does not depend on
/// static initialization order across the pipeline libraries that register.
llvm::StringMap<unsigned> &mutablePassKernelKinds() {
  static llvm::StringMap<unsigned> passKernelKinds;
  return passKernelKinds;
}

unsigned kernelKindBit(PipelineMode kind) {
  switch (kind) {
  case PipelineMode::Mix:
    return 1u << 0;
  case PipelineMode::Aiv:
    return 1u << 1;
  case PipelineMode::Aic:
    return 1u << 2;
  case PipelineMode::Unspecified:
    return 0;
  }
  return 0;
}

const char *pipelineModeName(PipelineMode mode) {
  switch (mode) {
  case PipelineMode::Unspecified:
    return "unspecified";
  case PipelineMode::Mix:
    return "mix";
  case PipelineMode::Aiv:
    return "aiv";
  case PipelineMode::Aic:
    return "aic";
  }
  return "unspecified";
}
} // namespace

void registerKernelKindPasses(PipelineMode kind,
                              std::initializer_list<StringRef> args) {
  unsigned bit = kernelKindBit(kind);
  if (bit == 0)
    return;
  // Registration happens while a pipeline is being built, before it runs.
  auto &kinds = mutablePassKernelKinds();
  for (StringRef arg : args) {
    if (!arg.empty())
      kinds[arg] |= bit;
  }
}

bool passAppliesToKernelKind(StringRef passArg, PipelineMode kind) {
  unsigned bit = kernelKindBit(kind);
  if (passArg.empty() || bit == 0)
    return false;
  const auto &kinds = mutablePassKernelKinds();
  auto it = kinds.find(passArg);
  return it != kinds.end() && (it->second & bit) != 0;
}

namespace {
llvm::StringSet<> &mutableClassifyKernelPasses() {
  static llvm::StringSet<> classifyKernelPasses;
  return classifyKernelPasses;
}

ClassifyKernelCallback &mutableClassifyKernelCallback() {
  static ClassifyKernelCallback callback = nullptr;
  return callback;
}
} // namespace

void registerClassifyKernelPasses(std::initializer_list<StringRef> args) {
  auto &set = mutableClassifyKernelPasses();
  for (StringRef arg : args)
    set.insert(arg);
}

bool passClassifiesKernel(StringRef passArg) {
  return !passArg.empty() && mutableClassifyKernelPasses().contains(passArg);
}

void registerClassifyKernelCallback(ClassifyKernelCallback callback) {
  mutableClassifyKernelCallback() = callback;
}

ClassifyKernelCallback getClassifyKernelCallback() {
  return mutableClassifyKernelCallback();
}

namespace {
llvm::StringMap<AivSubstituteCallback> &mutableAivSubstitutes() {
  static llvm::StringMap<AivSubstituteCallback> substitutes;
  return substitutes;
}
} // namespace

void registerAivSubstitute(StringRef skippedPassArg,
                           AivSubstituteCallback callback) {
  mutableAivSubstitutes()[skippedPassArg] = callback;
}

AivSubstituteCallback getAivSubstitute(StringRef skippedPassArg) {
  auto &substitutes = mutableAivSubstitutes();
  auto it = substitutes.find(skippedPassArg);
  if (it == substitutes.end())
    return nullptr;
  return it->second;
}

//===----------------------------------------------------------------------===//
// PassExecutionPolicy
//===----------------------------------------------------------------------===//

bool PassExecutionPolicy::shouldSkipGlobally(const Pass &pass,
                                             Operation *op) const {
  // Re-read the module core type on every decision. Classify-kernel follow-up
  // and InferFuncCoreType replace or delete it between passes; a mode frozen
  // on the first action would keep skipping after the module stopped being
  // that kind, or keep running the pass after the module became a kind the
  // pass does not list.
  Operation *root = op;
  while (root && !isa<ModuleOp>(root))
    root = root->getParentOp();
  PipelineMode detected = PipelineMode::Unspecified;
  if (root)
    detected = detectPipelineMode(cast<ModuleOp>(root));
  PipelineMode previous = mode.exchange(detected, std::memory_order_acq_rel);
  if (previous != detected) {
    LLVM_DEBUG(DBGS() << "PassExecutionPolicy: pipeline mode="
                      << pipelineModeName(detected) << "\n");
  }

  // No recognized core type: the kernel kind is not known, so every pass
  // runs. Adaptor passes only orchestrate nested pipelines.
  StringRef passArg = pass.getArgument();
  if (detected == PipelineMode::Unspecified || passArg.empty())
    return false;

  // A pass with no kernel-kind registration is not an HIVMPass list entry.
  const auto &kinds = mutablePassKernelKinds();
  auto it = kinds.find(passArg);
  if (it == kinds.end() || (it->second & kernelKindBit(detected)) != 0)
    return false;

  // Exemption: the delayed cross-core graph sync solver needs the backup
  // functions produced by the mix-CV flow, so an op that is (or a module whose
  // body contains) such a backup function is never kernel-kind-filtered. This
  // is deliberately conservative — it can only run *more* passes, never drop
  // a step of the backup closed loop.
  auto hasBackup = [](Operation *candidate) {
    return candidate->hasAttr("hivm.backup_function");
  };
  if (isa<ModuleOp>(op)) {
    for (Block &block : op->getRegion(0))
      for (Operation &child : block)
        if (hasBackup(&child))
          return false;
  } else if (hasBackup(op)) {
    return false;
  }

  LLVM_DEBUG(DBGS() << "skip '" << passArg << "' reason=KernelKind\n");
  return true;
}

} // namespace bishengir
