//===- InferSimtVFMemScopeHint.cpp - Infer SIMT VF Mem Scope Hints --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_INFERSIMTVFMEMSCOPEHINT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

AddressSpaceAttr getAddressSpaceAttr(MLIRContext *ctx, hivm::AddressSpace space) {
  return AddressSpaceAttr::get(ctx, space);
}

std::optional<AddressSpaceAttr> getSimtMemScopeHint(func::FuncOp func,
                                                    unsigned argIdx) {
  auto hint = func.getArgAttrOfType<SimtMemScopeHintAttr>(
      argIdx, SimtMemScopeHintAttr::name);
  if (!hint)
    return std::nullopt;
  return getAddressSpaceAttr(func.getContext(), hint.getAddressSpace());
}

LogicalResult setSimtMemScopeHint(func::FuncOp func, unsigned argIdx,
                                  AddressSpaceAttr hint) {
  if (auto existingHint = getSimtMemScopeHint(func, argIdx)) {
    if (existingHint.value() != hint) {
      return func.emitOpError()
             << "Conflicting simt memscope hints for argument #" << argIdx;
    }
    return success();
  }
  func.setArgAttr(
      argIdx, SimtMemScopeHintAttr::name,
      SimtMemScopeHintAttr::get(func.getContext(), hint.getAddressSpace()));
  return success();
}

bool isMemScopeHintAnchor(Value val) {
  // Stop traceback at values whose storage class is already semantically
  // known at the mixed boundary.
  if (utils::isAllocLikeOp(val))
    return true;
  if (auto bbArg = dyn_cast<BlockArgument>(val)) {
    auto *parentOp = bbArg.getOwner()->getParentOp();
    return isa<func::FuncOp>(parentOp);
  }
  return isa_and_nonnull<hivm::PointerCastOp>(val.getDefiningOp());
}

std::optional<AddressSpaceAttr> inferMemScopeFromAnchor(Value root) {
  if (auto bbArg = dyn_cast<BlockArgument>(root)) {
    auto *parentOp = bbArg.getOwner()->getParentOp();
    if (auto func = dyn_cast<func::FuncOp>(parentOp)) {
      if (hacc::utils::isDeviceEntry(func))
        return getAddressSpaceAttr(func.getContext(), hivm::AddressSpace::GM);
    }
    return std::nullopt;
  }

  if (auto allocOp = root.getDefiningOp<memref::AllocOp>())
    return getAddressSpaceAttr(allocOp.getContext(), hivm::AddressSpace::UB);

  if (auto pointerCastOp = root.getDefiningOp<hivm::PointerCastOp>()) {
    if (util::isGMPointerCastOp(pointerCastOp)) {
      return getAddressSpaceAttr(pointerCastOp.getContext(),
                                 hivm::AddressSpace::GM);
    }
  }

  return std::nullopt;
}

struct InferSimtVFMemScopeHintPass
    : public impl::InferSimtVFMemScopeHintBase<
          InferSimtVFMemScopeHintPass> {
  void runOnOperation() override;

private:
  LogicalResult inferHintForArgument(ModuleOp module, func::FuncOp func,
                                     BlockArgument arg);
};

} // namespace

LogicalResult InferSimtVFMemScopeHintPass::inferHintForArgument(
    ModuleOp module, func::FuncOp func, BlockArgument arg) {
  if (!isa<BaseMemRefType>(arg.getType()))
    return success();

  std::optional<AddressSpaceAttr> inferredHint = std::nullopt;
  // Shared placeholders are always expected to land in UB on the SIMT side.
  if (func.getArgAttr(arg.getArgNumber(), SharedMemoryAttr::name)) {
    inferredHint =
        getAddressSpaceAttr(func.getContext(), hivm::AddressSpace::UB);
  }

  auto maybeUses = func.getSymbolUses(module);
  if (!maybeUses.has_value()) {
    return func.emitOpError("failed to query symbol uses for simt vf");
  }

  for (const auto &use : maybeUses.value()) {
    auto callOp = dyn_cast<func::CallOp>(use.getUser());
    if (!callOp) {
      return func.emitOpError(
          "SIMT VF is expected to be used only by func.call");
    }

    Value callOperand = callOp.getOperand(arg.getArgNumber());

    // Infer from the SIMD-side operand roots without mutating types yet.
    auto roots = utils::tracebackMemRefVecByTargetFn(callOperand,
                                                     isMemScopeHintAnchor);
    if (roots.empty())
      roots.push_back(callOperand);

    for (Value root : roots) {
      auto maybeHint = inferMemScopeFromAnchor(root);
      if (!maybeHint.has_value())
        continue;
      if (!inferredHint.has_value()) {
        inferredHint = maybeHint;
        continue;
      }
      if (inferredHint.value() != maybeHint.value()) {
        return func.emitOpError()
               << "conflicting memory scope hints for simt argument #"
               << arg.getArgNumber();
      }
    }
  }

  if (!inferredHint.has_value()) {
    return func.emitOpError()
           << "failed to infer memory scope hint for simt argument #"
           << arg.getArgNumber();
  }
  return setSimtMemScopeHint(func, arg.getArgNumber(), inferredHint.value());
}

void InferSimtVFMemScopeHintPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<func::FuncOp> simtFuncs;
  module.walk([&](func::FuncOp func) {
    if (util::isSIMTVF(func))
      simtFuncs.push_back(func);
  });

  for (func::FuncOp func : simtFuncs) {
    for (BlockArgument arg : func.getArguments()) {
      if (failed(inferHintForArgument(module, func, arg))) {
        signalPassFailure();
        return;
      }
    }
  }
}

std::unique_ptr<Pass> mlir::hivm::createInferSimtVFMemScopeHintPass() {
  return std::make_unique<InferSimtVFMemScopeHintPass>();
}
