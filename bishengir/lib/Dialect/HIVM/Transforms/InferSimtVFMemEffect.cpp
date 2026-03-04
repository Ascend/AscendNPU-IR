//===- InferSimtVFMemEffect.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
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
          idx, hivm::MemoryEffectAttr::getMnemonic())) {
    if (existMemEffectAttr.getEffect() == memEffect) {
      return;
    }
    auto memEffectAttr = hivm::MemoryEffectAttr::get(
        funcOp->getContext(), hivm::MemoryEffect::READ_WRITE);
    funcOp.setArgAttr(idx, hivm::MemoryEffectAttr::getMnemonic(),
                      memEffectAttr);
  } else {
    auto memEffectAttr =
        hivm::MemoryEffectAttr::get(funcOp->getContext(), memEffect);
    funcOp.setArgAttr(idx, hivm::MemoryEffectAttr::getMnemonic(),
                      memEffectAttr);
  }
}

void InferSimtVFMemEffectPass::inferFuncArgMemEffect(func::FuncOp funcOp) {
  funcOp->walk([this, &funcOp](Operation *op) {
    if (auto loadOp = llvm::dyn_cast<hivm::LoadOp>(op)) {
      Value blockArg;
      auto blockArgs = utils::tracebackMemRefVecByTargetFn(
          loadOp.getSrc(), [](Value val) { return !val.getDefiningOp(); });
      if (blockArgs.empty() && !loadOp.getSrc().getDefiningOp()) {
        blockArg = loadOp.getSrc();
      } else {
        assert(blockArgs.size() == 1 &&
               "tracebackMemRef found multiple sources!");
        blockArg = blockArgs[0];
      }
      setFuncArgMemEffect(funcOp, blockArg, hivm::MemoryEffect::READ);
    } else if (auto loadOp = llvm::dyn_cast<hivm::GatherLoadOp>(op)) {
      Value blockArg;
      auto blockArgs = utils::tracebackMemRefVecByTargetFn(
          loadOp.getBase(), [](Value val) { return !val.getDefiningOp(); });
      if (blockArgs.empty() && !loadOp.getBase().getDefiningOp()) {
        blockArg = loadOp.getBase();
      } else {
        assert(blockArgs.size() == 1 &&
               "tracebackMemRef found multiple sources!");
        blockArg = blockArgs[0];
      }
      setFuncArgMemEffect(funcOp, blockArg, hivm::MemoryEffect::READ);
    } else if (auto storeOp = llvm::dyn_cast<hivm::StoreOp>(op)) {
      Value blockArg;
      auto blockArgs = utils::tracebackMemRefVecByTargetFn(
          storeOp.getDst(), [](Value val) { return !val.getDefiningOp(); });
      if (blockArgs.empty() && !storeOp.getDst().getDefiningOp()) {
        blockArg = storeOp.getDst();
      } else {
        assert(blockArgs.size() == 1 &&
               "tracebackMemRef found multiple sources!");
        blockArg = blockArgs[0];
      }
      setFuncArgMemEffect(funcOp, blockArg, hivm::MemoryEffect::WRITE);
    } else if (auto storeOp = llvm::dyn_cast<hivm::ScatterStoreOp>(op)) {
      Value blockArg;
      auto blockArgs = utils::tracebackMemRefVecByTargetFn(
          storeOp.getBase(), [](Value val) { return !val.getDefiningOp(); });
      if (blockArgs.empty() && !storeOp.getBase().getDefiningOp()) {
        blockArg = storeOp.getBase();
      } else {
        assert(blockArgs.size() == 1 &&
               "tracebackMemRef found multiple sources!");
        blockArg = blockArgs[0];
      }
      setFuncArgMemEffect(funcOp, blockArg, hivm::MemoryEffect::WRITE);
    }
  });
}

void InferSimtVFMemEffectPass::runOnOperation() {
  auto mod = getOperation();
  mod->walk([this](func::FuncOp funcOp) {
    if (!funcOp->hasAttr(hivm::VFModeAttr::getMnemonic())) {
      return;
    }
    auto vfmode = llvm::dyn_cast<hivm::VFModeAttr>(
                      funcOp->getAttr(hivm::VFModeAttr::getMnemonic()))
                      .getValue();
    if (vfmode != hivm::VFMode::SIMT) {
      return;
    }
    inferFuncArgMemEffect(funcOp);
  });
}

std::unique_ptr<Pass> mlir::hivm::createInferSimtVFMemEffectPass() {
  return std::make_unique<InferSimtVFMemEffectPass>();
}
