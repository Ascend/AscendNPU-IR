//===- InferSimtVFMemEffect.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iterator>

namespace mlir {
#define GEN_PASS_DEF_INFERSIMTVFMEMEFFECT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct InferSimtVFMemEffectPass
    : public impl::InferSimtVFMemEffectBase<InferSimtVFMemEffectPass> {
  void runOnOperation() override;

  void inferFuncArgMemEffect(func::FuncOp funcOp);

  void setFuncArgMemEffect(func::FuncOp funcOp, Value blockArg,
                           hivm::MemoryEffect memEffect);
  template <typename OpTy, typename GetMemRefFn>
  void handleMemOp(OpTy op, func::FuncOp funcOp, GetMemRefFn &&getMemRefFn,
                   hivm::MemoryEffect memEffect);
};

} // namespace

void InferSimtVFMemEffectPass::setFuncArgMemEffect(
    func::FuncOp funcOp, Value blockArg, hivm::MemoryEffect memEffect) {
  auto argList = funcOp.getArguments();
  auto actualArg = llvm::find(argList, blockArg);
  if (!actualArg) {
    return;
  }
  auto idx = std::distance(argList.begin(), actualArg);
  if (auto existMemEffectAttr = funcOp.getArgAttrOfType<hivm::MemoryEffectAttr>(
          idx, hivm::MemoryEffectAttr::name)) {
    if (existMemEffectAttr.getEffect() == memEffect) {
      return;
    }
    auto memEffectAttr = hivm::MemoryEffectAttr::get(
        funcOp->getContext(), hivm::MemoryEffect::READ_WRITE);
    funcOp.setArgAttr(idx, hivm::MemoryEffectAttr::name, memEffectAttr);
  } else {
    auto memEffectAttr =
        hivm::MemoryEffectAttr::get(funcOp->getContext(), memEffect);
    funcOp.setArgAttr(idx, hivm::MemoryEffectAttr::name, memEffectAttr);
  }
}

template <typename OpTy, typename GetMemRefFn>
void InferSimtVFMemEffectPass::handleMemOp(OpTy op, func::FuncOp funcOp,
                                           GetMemRefFn &&getMemRefFn,
                                           hivm::MemoryEffect memEffect) {
  Value memRef = getMemRefFn(op);
  Value blockArg;

  auto blockArgs = utils::tracebackMemRefVecByTargetFn(
      memRef, [](Value val) { return !val.getDefiningOp(); });

  if (blockArgs.empty() && !memRef.getDefiningOp()) {
    blockArg = memRef;
  } else {
    assert((blockArgs.size() == 1) &&
           "tracebackMemRef found multiple sources!");
    blockArg = blockArgs[0];
  }

  setFuncArgMemEffect(funcOp, blockArg, memEffect);
}

void InferSimtVFMemEffectPass::inferFuncArgMemEffect(func::FuncOp funcOp) {
  funcOp->walk([this, &funcOp](Operation *op) {
    if (auto loadOp = llvm::dyn_cast<hivm::LoadOp>(op)) {
      handleMemOp(
          loadOp, funcOp, [](hivm::LoadOp op) { return op.getSrc(); },
          hivm::MemoryEffect::READ);
    } else if (auto gatherLoadOp = llvm::dyn_cast<hivm::GatherLoadOp>(op)) {
      handleMemOp(
          gatherLoadOp, funcOp,
          [](hivm::GatherLoadOp op) { return op.getBase(); },
          hivm::MemoryEffect::READ);
    } else if (auto storeOp = llvm::dyn_cast<hivm::StoreOp>(op)) {
      handleMemOp(
          storeOp, funcOp, [](hivm::StoreOp op) { return op.getDst(); },
          hivm::MemoryEffect::WRITE);
    } else if (auto scatterStoreOp = llvm::dyn_cast<hivm::ScatterStoreOp>(op)) {
      if (isa<MemRefType>(scatterStoreOp.getBase().getType())) {
        handleMemOp(
            scatterStoreOp, funcOp,
            [](hivm::ScatterStoreOp op) { return op.getBase(); },
            hivm::MemoryEffect::WRITE);
      }
    } else if (auto localLoadOp = llvm::dyn_cast<hivm::LocalLoadOp>(op)) {
      handleMemOp(
          localLoadOp, funcOp,
          [](hivm::LocalLoadOp op) { return op.getAddr(); },
          hivm::MemoryEffect::READ);
    } else if (auto localStoreOp = llvm::dyn_cast<hivm::LocalStoreOp>(op)) {
      handleMemOp(
          localStoreOp, funcOp,
          [](hivm::LocalStoreOp op) { return op.getAddr(); },
          hivm::MemoryEffect::WRITE);
    }
  });
}


void InferSimtVFMemEffectPass::runOnOperation() {
  auto mod = getOperation();
  mod->walk([this](func::FuncOp funcOp) {
    if (!util::isSIMTVF(funcOp)) {
      return;
    }
    inferFuncArgMemEffect(funcOp);
  });
}

std::unique_ptr<Pass> mlir::hivm::createInferSimtVFMemEffectPass() {
  return std::make_unique<InferSimtVFMemEffectPass>();
}
