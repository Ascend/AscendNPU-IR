//===------------------ InsertInitAndFinishForDebug.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SIMT-mode analogue of HIVM's InsertInitAndFinishForDebug. Brackets the
// SIMD launch wrapper's relevant calls to SIMT kernels with
// _mlir_ciface_init_debug() / _mlir_ciface_finish_debug() so the CCEC
// debug-tunnel handoff is initialized before SIMT prints fire and torn down
// after.
//
// The handoff mechanism is owned by ccelib. This pass only emits init/finish;
// there is no tunnel pointer to thread through the SIMD-to-SIMT launch IR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_SIMTINSERTINITANDFINISHFORDEBUG
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "simt-insert-init-and-finish-for-debug"

namespace {
struct SIMTInsertInitAndFinishForDebug
    : public impl::SIMTInsertInitAndFinishForDebugBase<
          SIMTInsertInitAndFinishForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

// "hivm_regbaseintrins.kernel" marks the SIMD launch wrapper. init/finish
// live on the wrapper because their lib implementations are [aicore] and
// the wrapper is the AICORE-context entry the host launches.
bool isLaunchWrapper(LLVM::LLVMFuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  auto kernelAttr =
      StringAttr::get(context, hivm_regbaseintrins::kDavinciKernelAttrName);
  return funcOp->hasAttr(kernelAttr);
}

bool isDebugRuntimeCall(LLVM::CallOp call) {
  auto callee = call.getCallee();
  return callee && (callee->starts_with("_mlir_ciface_print_") ||
                    callee->starts_with("_mlir_ciface_assert_"));
}

// True iff the module contains any lowered SIMT print calls. ascend_dpx.print
// has already been lowered to _mlir_ciface_print_* LLVM calls by the time
// this pass runs; emit init/finish only when prints exist.
bool moduleHasPrintCalls(ModuleOp mod) {
  bool found = false;
  mod.walk([&](LLVM::CallOp call) {
    if (isDebugRuntimeCall(call)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

bool functionTransitivelyHasDebugCalls(
    LLVM::LLVMFuncOp funcOp, SymbolTableCollection &symbolTables,
    llvm::SmallPtrSetImpl<Operation *> &visitedFunctions) {
  if (!visitedFunctions.insert(funcOp).second)
    return false;

  bool found = false;
  funcOp.walk([&](LLVM::CallOp call) {
    if (isDebugRuntimeCall(call)) {
      found = true;
      return WalkResult::interrupt();
    }
    if (!call.getCallee())
      return WalkResult::advance();

    auto callee = symbolTables.lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        call, call.getCalleeAttr());
    if (callee && functionTransitivelyHasDebugCalls(callee, symbolTables,
                                                    visitedFunctions)) {
      found = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return found;
}

bool functionTransitivelyHasDebugCalls(LLVM::LLVMFuncOp funcOp,
                                       SymbolTableCollection &symbolTables) {
  llvm::SmallPtrSet<Operation *, 16> visitedFunctions;
  return functionTransitivelyHasDebugCalls(funcOp, symbolTables,
                                           visitedFunctions);
}

enum class DebugLifecycleState { None, Valid, Malformed };

enum class DebugLifecycleMarkerKind {
  HivmInit,
  HivmFinish,
  LLVMInit,
  LLVMFinish
};

struct DebugLifecycleMarker {
  Operation *op;
  DebugLifecycleMarkerKind kind;
};

DebugLifecycleState classifyDebugLifecycle(
    LLVM::LLVMFuncOp funcOp,
    ArrayRef<hivm_regbaseintrins::LaunchFuncOp> debugLaunches) {
  assert(!debugLaunches.empty() && "expected at least one debug launch");

  llvm::SmallVector<DebugLifecycleMarker, 2> markers;
  funcOp.walk([&](Operation *op) {
    if (isa<hivm::InitDebugOp>(op)) {
      markers.push_back({op, DebugLifecycleMarkerKind::HivmInit});
      return;
    }
    if (isa<hivm::FinishDebugOp>(op)) {
      markers.push_back({op, DebugLifecycleMarkerKind::HivmFinish});
      return;
    }
    auto call = dyn_cast<LLVM::CallOp>(op);
    if (!call)
      return;
    auto callee = call.getCallee();
    if (callee && *callee == "_mlir_ciface_init_debug")
      markers.push_back({op, DebugLifecycleMarkerKind::LLVMInit});
    if (callee && *callee == "_mlir_ciface_finish_debug")
      markers.push_back({op, DebugLifecycleMarkerKind::LLVMFinish});
  });

  if (markers.empty())
    return DebugLifecycleState::None;
  if (markers.size() != 2)
    return DebugLifecycleState::Malformed;

  auto [initOp, initKind] = markers.front();
  auto [finishOp, finishKind] = markers.back();
  bool isHivmPair = initKind == DebugLifecycleMarkerKind::HivmInit &&
                    finishKind == DebugLifecycleMarkerKind::HivmFinish;
  bool isLLVMPair = initKind == DebugLifecycleMarkerKind::LLVMInit &&
                    finishKind == DebugLifecycleMarkerKind::LLVMFinish;
  if (!isHivmPair && !isLLVMPair)
    return DebugLifecycleState::Malformed;

  Operation *firstLaunch = debugLaunches.front();
  Operation *lastLaunch = debugLaunches.back();
  Block *launchBlock = firstLaunch->getBlock();
  if (initOp->getBlock() != launchBlock || finishOp->getBlock() != launchBlock)
    return DebugLifecycleState::Malformed;
  if (!initOp->isBeforeInBlock(firstLaunch) ||
      !lastLaunch->isBeforeInBlock(finishOp))
    return DebugLifecycleState::Malformed;
  return DebugLifecycleState::Valid;
}

bool launchCallsDebugKernel(SymbolTableCollection &symbolTables,
                            hivm_regbaseintrins::LaunchFuncOp launch) {
  auto callee = symbolTables.lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      launch, launch.getKernelAttr());
  return callee && functionTransitivelyHasDebugCalls(callee, symbolTables);
}

// Materialize a private LLVM decl of init/finish at module scope, with the
// attribute set HIVMToStandard's createLibCall produces. LLVM dialect, not
// func, because the wrapper is already LLVM by the time we run.
static FlatSymbolRefAttr ensureDebugLibFuncDecl(ModuleOp mod, OpBuilder &b,
                                                StringRef name) {
  MLIRContext *ctx = mod.getContext();
  auto fnAttr = SymbolRefAttr::get(ctx, name);
  if (mod.lookupSymbol(fnAttr.getAttr()))
    return fnAttr;
  auto fnTy =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), /*params=*/{});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));
  auto func = b.create<LLVM::LLVMFuncOp>(mod.getLoc(), name, fnTy);
  func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                UnitAttr::get(ctx));
  return fnAttr;
}

void SIMTInsertInitAndFinishForDebug::runOnOperation() {
  ModuleOp mod = cast<ModuleOp>(getOperation());

  if (!moduleHasPrintCalls(mod))
    return;

  struct WrapperWork {
    ModuleOp debugLibScope;
    llvm::SmallVector<hivm_regbaseintrins::LaunchFuncOp> debugLaunches;
  };

  SymbolTableCollection symbolTables;
  llvm::SmallVector<WrapperWork> work;
  WalkResult collectionResult = mod.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (!isLaunchWrapper(funcOp))
      return WalkResult::advance();

    llvm::SmallVector<hivm_regbaseintrins::LaunchFuncOp> debugLaunches;
    funcOp.walk([&](hivm_regbaseintrins::LaunchFuncOp launch) {
      if (launchCallsDebugKernel(symbolTables, launch))
        debugLaunches.push_back(launch);
    });
    if (debugLaunches.empty())
      return WalkResult::advance();

    Block *launchBlock = debugLaunches.front()->getBlock();
    if (llvm::any_of(debugLaunches, [launchBlock](auto launch) {
          return launch->getBlock() != launchBlock;
        })) {
      funcOp.emitError(
          "cannot insert one debug lifecycle around launches in different "
          "blocks");
      return WalkResult::interrupt();
    }

    switch (classifyDebugLifecycle(funcOp, debugLaunches)) {
    case DebugLifecycleState::None:
      work.push_back(
          {funcOp->getParentOfType<ModuleOp>(), std::move(debugLaunches)});
      return WalkResult::advance();
    case DebugLifecycleState::Valid:
      return WalkResult::advance();
    case DebugLifecycleState::Malformed:
      funcOp.emitError(
          "malformed debug lifecycle: expected exactly one same-kind "
          "init/finish pair bracketing all relevant launches");
      return WalkResult::interrupt();
    }
    llvm_unreachable("unhandled debug lifecycle state");
  });

  if (collectionResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }
  if (work.empty())
    return;

  MLIRContext *context = &getContext();
  OpBuilder moduleBuilder(context);

  for (WrapperWork &item : work) {
    FlatSymbolRefAttr initFn = ensureDebugLibFuncDecl(
        item.debugLibScope, moduleBuilder, "_mlir_ciface_init_debug");
    FlatSymbolRefAttr finishFn = ensureDebugLibFuncDecl(
        item.debugLibScope, moduleBuilder, "_mlir_ciface_finish_debug");
    OpBuilder builder(context);
    builder.setInsertionPoint(item.debugLaunches.front());
    builder.create<LLVM::CallOp>(item.debugLaunches.front()->getLoc(),
                                 TypeRange{}, initFn, ValueRange{});
    builder.setInsertionPointAfter(item.debugLaunches.back());
    builder.create<LLVM::CallOp>(item.debugLaunches.back()->getLoc(),
                                 TypeRange{}, finishFn, ValueRange{});
  }
}

} // namespace

std::unique_ptr<Pass>
bishengir::triton::createSIMTInsertInitAndFinishForDebugPass() {
  return std::make_unique<SIMTInsertInitAndFinishForDebug>();
}
