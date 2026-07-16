//===------------------------ MarkDisableLoad.cpp -------------------------===//
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
// This pass marks the memref.loads that need to disable dcache.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_MARKDISABLELOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-mark-disable-load"

namespace {

// Trace a value back through ViewLike ops to a function argument.
// Returns {the BlockArgument value, its index in the function arg list},
// or nullopt if the value does not originate from a function argument.
std::optional<std::pair<Value, unsigned>>
traceToFuncArgWithIndex(Value v, func::FuncOp f) {
  for (auto [idx, arg] : llvm::enumerate(f.getBody().getArguments())) {
    if (arg == v)
      return std::make_pair(Value(arg), idx);
  }
  if (auto viewLikeOp = v.getDefiningOp<ViewLikeOpInterface>())
    return traceToFuncArgWithIndex(viewLikeOp.getViewSource(), f);
  return std::nullopt;
}

// Collect all operations with a Write memory effect that transitively use v,
// following ViewLike ops recursively.
std::vector<Operation *> traceWriteEndUsers(Value v) {
  std::vector<Operation *> endUsers;
  for (Operation *userOp : v.getUsers()) {
    if (isa<ViewLikeOpInterface>(userOp)) {
      auto nextLayerEndUsers = traceWriteEndUsers(userOp->getOpResult(0));
      for (Operation *nextLayerEndUser : nextLayerEndUsers) {
        if (hasEffect<MemoryEffects::Write>(nextLayerEndUser))
          endUsers.push_back(nextLayerEndUser);
      }
    } else {
      if (hasEffect<MemoryEffects::Write>(userOp))
        endUsers.push_back(userOp);
    }
  }
  return endUsers;
}

// Given either a _mix_aic or a _mix_aiv function, find the paired counterpart
// in the module by swapping the suffix.  Returns nullopt if the function name
// does not carry either suffix or the paired symbol cannot be found.
std::optional<func::FuncOp> findPairedMixFunc(func::FuncOp func,
                                              ModuleOp module) {
  StringRef name = func.getSymName();
  std::string pairedName;
  if (name.ends_with(kMixFuncAicSuffix))
    pairedName = name.drop_back(kMixFuncAicSuffix.size()).str() +
                 kMixFuncAivSuffix.str();
  else if (name.ends_with(kMixFuncAivSuffix))
    pairedName = name.drop_back(kMixFuncAivSuffix.size()).str() +
                 kMixFuncAicSuffix.str();
  else
    return std::nullopt;
  auto funcOp = dyn_cast_or_null<func::FuncOp>(module.lookupSymbol(pairedName));
  if (!funcOp)
    return std::nullopt;
  return funcOp;
}

// Determine whether a memref.load needs dcache disabled and set the appropriate
// attribute.  The analysis checks:
//   1. Whether the load's source memref originates from a function argument.
//   2. Whether that argument (or the corresponding argument in the paired mix
//      function, _mix_aic <-> _mix_aiv) is also the target of a write
//      operation anywhere in the module.
void processLoadOp(memref::LoadOp loadOp, ModuleOp moduleOp) {
  if (loadOp->hasAttr(kMarkDCacheVisitedAttr))
    return;

  auto i32Zero =
      IntegerAttr::get(IntegerType::get(loadOp->getContext(), 32), 0);

  auto f = loadOp->getParentOfType<func::FuncOp>();
  if (!f) {
    loadOp->setAttr(kMarkDCacheVisitedAttr, i32Zero);
    return;
  }

  auto argInfo = traceToFuncArgWithIndex(loadOp.getMemref(), f);
  if (!argInfo) {
    loadOp->setAttr(kMarkDCacheVisitedAttr, i32Zero);
    return;
  }

  auto [argVal, argIdx] = *argInfo;
  bool hasWriteUser = !traceWriteEndUsers(argVal).empty();

  // Additionally check the paired mix function (_mix_aic <-> _mix_aiv), since
  // both share the same external memory buffers.
  if (!hasWriteUser) {
    auto pairedFunc = findPairedMixFunc(f, moduleOp);
    if (pairedFunc) {
      auto pairedArgs = pairedFunc->getBody().getArguments();
      if (argIdx < pairedArgs.size()) {
        Value pairedArg = pairedArgs[argIdx];
        hasWriteUser = !traceWriteEndUsers(pairedArg).empty();
      }
    }
  }

  if (hasWriteUser)
    loadOp->setAttr(kDisableDCacheAttr, i32Zero);
  loadOp->setAttr(kMarkDCacheVisitedAttr, i32Zero);
}

struct MarkDisableLoad : public impl::MarkDisableLoadBase<MarkDisableLoad> {
  using Base::Base;
  void runOnOperation() override;
};

void MarkDisableLoad::runOnOperation() {
  ModuleOp moduleOp = cast<ModuleOp>(getOperation());
  moduleOp.walk(
      [&](memref::LoadOp loadOp) { processLoadOp(loadOp, moduleOp); });
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkDisableLoadPass() {
  return std::make_unique<MarkDisableLoad>();
}
