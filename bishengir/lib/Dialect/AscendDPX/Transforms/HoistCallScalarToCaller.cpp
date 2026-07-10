//===- HoistCallScalarToCaller.cpp ----------------------------------------===//
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
// This pass hoists ascend_dpx.call_scalar operations from callee functions to
// their call sites.  For each call_scalar:
//
//   - If use_shmem_offset is set: the result is stored to shared memory at
//     that offset in the caller, and replaced with an llvm.load from shared
//     memory in the callee.
//
//   - If use_shmem_offset is not set: the result is passed as a new function
//     argument from the caller to the callee.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace ascend_dpx {

#define GEN_PASS_DEF_HOISTCALLSCALARTOCALLER
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h.inc"

namespace {

class HoistCallScalarToCallerPass
    : public impl::HoistCallScalarToCallerBase<HoistCallScalarToCallerPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    // Collect functions that contain ascend_dpx.call_scalar ops.
    SmallVector<LLVM::LLVMFuncOp> targets;
    mod.walk([&](LLVM::LLVMFuncOp func) {
      bool has = false;
      func.walk([&](mlir::ascend_dpx::CallScalarOp) { has = true; });
      if (has)
        targets.push_back(func);
    });

    if (targets.empty())
      return;

    for (auto callee : targets)
      processCallee(callee, mod, ctx);
  }

private:
  /// Find the function argument annotated with "shared_memory".
  static Value findSmemArg(LLVM::LLVMFuncOp func) {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      if (func.getArgAttr(i, hivm::SharedMemoryAttr::name))
        return func.getArgument(i);
    return {};
  }

  /// Return a pointer to \p smemBase + \p offset bytes.
  static Value getSmemOffsetPtr(OpBuilder &b, Location loc, MLIRContext *ctx,
                                Value smemBase, int32_t offset) {
    auto ptrTy = smemBase.getType();
    auto i8Ty = IntegerType::get(ctx, 8);
    return b.create<LLVM::GEPOp>(loc, ptrTy, i8Ty, smemBase,
                                 ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(offset)});
  }

  /// Recursively clone the defining op of \p v into the caller so that
  /// the cloned call_scalar can reference it.  Block arguments and
  /// ascend_dpx.call_scalar results (handled by the main loop) are skipped.
  void ensureAvailable(Value v, OpBuilder &b, IRMapping &mapping) {
    if (mapping.contains(v))
      return;
    if (isa<BlockArgument>(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp || isa<mlir::ascend_dpx::CallScalarOp>(defOp))
      return;
    for (Value operand : defOp->getOperands())
      ensureAvailable(operand, b, mapping);
    Operation *cloned = b.clone(*defOp, mapping);
    for (unsigned i = 0; i < defOp->getNumResults(); ++i)
      mapping.map(defOp->getResult(i), cloned->getResult(i));
  }

  void processCallee(LLVM::LLVMFuncOp callee, ModuleOp mod, MLIRContext *ctx) {
    // Collect call_scalar ops in program order.
    SmallVector<ascend_dpx::CallScalarOp> callScalarOps;
    callee.walk([&](mlir::ascend_dpx::CallScalarOp op) {
      callScalarOps.push_back(op);
    });
    if (callScalarOps.empty())
      return;

    // Find all call sites that invoke this function.
    SmallVector<hivm_regbaseintrins::LaunchFuncOp> callSites;
    mod.walk([&](hivm_regbaseintrins::LaunchFuncOp call) {
      if (call.getKernel() == callee.getName())
        callSites.push_back(call);
    });
    if (callSites.empty())
      return;

    // Locate the callee's "shared_memory" argument (used for shmem ops).
    Value calleeSmemArg = findSmemArg(callee);
    assert(calleeSmemArg &&
           "Shared Memory base address has not been allocated");
    unsigned origNumArgs = callee.getNumArguments();

    // ------------------------------------------------------------------
    // Phase A: At each call site, clone the call_scalar ops and emit
    //          stores (shmem case) or collect extra arguments.
    // ------------------------------------------------------------------
    struct CallSiteWork {
      hivm_regbaseintrins::LaunchFuncOp callSite;
      SmallVector<Value> extraArgs;
    };
    SmallVector<CallSiteWork> work;

    for (auto callSite : callSites) {
      CallSiteWork w;
      w.callSite = callSite;

      OpBuilder callerB(callSite);
      Location callLoc = callSite.getLoc();

      // Map callee entry-block args → call-site operands.
      IRMapping mapping;
      Block &entry = callee.getBody().front();
      for (unsigned i = 0; i < origNumArgs; ++i)
        // caller side need an offset of 3 due to the 3d block size
        mapping.map(entry.getArgument(i), callSite.getOperands()[i + 3]);

      // Resolve the caller-side shared memory base from the mapping.
      Value callerSmemBase = mapping.lookup(calleeSmemArg);

      for (auto csOp : callScalarOps) {
        // Ensure non-call_scalar operands (e.g. constants) are cloned.
        for (Value operand : csOp.getCallArgs())
          ensureAvailable(operand, callerB, mapping);

        // Clone the call_scalar op into the caller as a llvm.call.
        SmallVector<Value> argMapping;
        for (Value arg : csOp.getCallArgs())
          argMapping.push_back(mapping.lookup(arg));

        LLVM::CallOp clonedCall = callerB.create<LLVM::CallOp>(
            callLoc, csOp->getResultTypes(), csOp.getCallee(), argMapping);

        // Track results for inter-call_scalar dependencies.
        mapping.map(csOp.getResult(0), clonedCall.getResult());

        auto offset = csOp.getUseShmemOffset();
        if (offset && callerSmemBase) {
          // Store result to shared memory.
          for (auto result : clonedCall.getResults()) {
            Value ptr = getSmemOffsetPtr(callerB, callLoc, ctx, callerSmemBase,
                                         static_cast<int32_t>(*offset));
            callerB.create<LLVM::StoreOp>(callLoc, result, ptr);
          }
        } else {
          // Collect result to pass as a new argument.
          for (auto result : clonedCall.getResults())
            w.extraArgs.push_back(result);
        }
      }

      work.push_back(std::move(w));
    }

    // ------------------------------------------------------------------
    // Phase B: In the callee, replace uses of each call_scalar result.
    // ------------------------------------------------------------------
    unsigned nextArgIdx = origNumArgs;

    for (auto csOp : callScalarOps) {
      auto offset = csOp.getUseShmemOffset();
      if (offset && calleeSmemArg) {
        // Replace each use with its own llvm.load from shared memory,
        // inserted immediately before the user operation.
        for (auto result : csOp.getResults()) {
          for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
            OpBuilder b(use.getOwner());
            Location loc = use.getOwner()->getLoc();
            Value ptr = getSmemOffsetPtr(b, loc, ctx, calleeSmemArg,
                                         static_cast<int32_t>(*offset));
            Value loaded = b.create<LLVM::LoadOp>(loc, result.getType(), ptr);
            use.set(loaded);
          }
        }
      } else {
        // Add a new callee argument for each result.
        for (auto result : csOp.getResults()) {
          auto newArg = callee.getBody().front().addArgument(result.getType(),
                                                             csOp.getLoc());
          result.replaceAllUsesWith(newArg);
          nextArgIdx++;
        }
      }
    }

    // Update the callee function type if new arguments were added.
    if (nextArgIdx > origNumArgs) {
      auto oldTy = callee.getFunctionType();
      SmallVector<Type> newInputs(oldTy.getParams());
      for (unsigned i = origNumArgs; i < nextArgIdx; ++i)
        newInputs.push_back(callee.getBody().front().getArgument(i).getType());
      callee.setFunctionType(
          LLVM::LLVMFunctionType::get(oldTy.getReturnType(), newInputs));
    }

    // ------------------------------------------------------------------
    // Phase C: Update call sites that need extra arguments.
    // ------------------------------------------------------------------
    for (auto &w : work) {
      if (w.extraArgs.empty())
        continue;
      OpBuilder b(w.callSite);
      SmallVector<Value> allOperands(w.callSite.getOperands());
      allOperands.append(w.extraArgs);
      b.create<hivm_regbaseintrins::LaunchFuncOp>(
          w.callSite.getLoc(), w.callSite.getKernel(),
          w.callSite.getBlockSizeX(), w.callSite.getBlockSizeY(),
          w.callSite.getBlockSizeZ(), allOperands);
      w.callSite.erase();
    }

    // ------------------------------------------------------------------
    // Phase D: Erase original call_scalar ops from the callee.
    // ------------------------------------------------------------------
    for (auto it = callScalarOps.rbegin(); it != callScalarOps.rend(); ++it)
      (*it)->erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createHoistCallScalarToCallerPass() {
  return std::make_unique<HoistCallScalarToCallerPass>();
}

} // namespace ascend_dpx
} // namespace mlir