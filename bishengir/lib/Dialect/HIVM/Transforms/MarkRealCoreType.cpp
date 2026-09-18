//===------ MarkRealCoreType.cpp --------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

namespace mlir {
#define GEN_PASS_DEF_MARKREALCORETYPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "hivm-mark-real-core-type"

using namespace mlir;
using namespace mlir::hivm;

struct MarkRealCoreTypePass
    : public impl::MarkRealCoreTypeBase<MarkRealCoreTypePass> {

  explicit MarkRealCoreTypePass(const MarkRealCoreTypeOptions &options)
      : MarkRealCoreTypeBase(options) {}
  ~MarkRealCoreTypePass() override = default;
  void runOnOperation() override;

  bool isOpTypeToBeMarked(Operation *op) const {
    // scalar-pipe operations.
    if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
            affine::AffineStoreOp, tensor::ExtractOp, tensor::InsertOp,
            tensor::InsertSliceOp, tensor::ExtractSliceOp>(op)) {
      return true;
    }
    if (isa<hivm::CustomOp, hivm::CustomMacroOp>(op)) {
      return false;
    }
    if (isa<hivm::VBrcOp>(op))
      return false;
    if (isa<hivm::InferCoreTypeInterface>(op)) {
      return true;
    }
    return false;
  }
};

void MarkRealCoreTypePass::runOnOperation() {
  auto moduleOp = getOperation();

  if (this->removeCoreTypeAttrs) {
    moduleOp.walk([&](Operation *op) {
      if (isOpTypeToBeMarked(op)) {
        if (op->hasAttr(hivm::TCoreTypeAttr::name)) {
          op->removeAttr(hivm::TCoreTypeAttr::name);
        }
      }
    });
    return;
  }

  // clone moduleOp to moduleClone
  IRMapping mapper;
  ModuleOp moduleClone = cast<ModuleOp>(moduleOp->clone(mapper));

  SmallVector<Operation *> markerOperations;
  SmallVector<hivm::TCoreType> markerCoreTypes;
  static constexpr StringLiteral kInstructionMarkerAttr = "instruction-marker";

  // Annotate each cloned operation in preorder and retain its original
  // counterpart by marker index. The marker is only used after the clone-side
  // pipeline.
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  uint64_t instructionCounter = 0;
  moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ModuleOp>(op))
      return;

    Operation *clonedOp = mapper.lookupOrNull(op);
    assert(clonedOp && "missing cloned operation mapping");
    markerOperations.push_back(op);
    clonedOp->setAttr(kInstructionMarkerAttr,
                      builder.getIndexAttr(instructionCounter));
    instructionCounter++;
  });
  markerCoreTypes.resize(markerOperations.size(),
                         hivm::TCoreType::CUBE_OR_VECTOR);

  // run split mix kernel pass to annotate core type attribute
  PassManager pm(moduleClone.getContext());
  pm.addPass(createSplitMixKernelPass());
  canonicalizationHIVMPipeline(pm);
  if (failed(pm.run(moduleClone))) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "canonicalized splitted kernels:\n" << moduleClone << '\n';
  });

  // get function with aic core type from cloned module.
  moduleClone->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    auto funcOpCoreTypeOpt = queryFuncCoreType(funcOp);
    if (!funcOpCoreTypeOpt.has_value()) {
      return;
    }
    auto funcOpCoreType = funcOpCoreTypeOpt.value();
    if (funcOpCoreType != hivm::TFuncCoreType::AIC &&
        funcOpCoreType != hivm::TFuncCoreType::AIV) {
      return;
    }
    auto opCoreType = funcOpCoreType == hivm::TFuncCoreType::AIC
                          ? hivm::TCoreType::CUBE
                          : hivm::TCoreType::VECTOR;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto instructionCounterAttr =
              op->getAttrOfType<IntegerAttr>(kInstructionMarkerAttr)) {
        uint64_t instructionCounter =
            instructionCounterAttr.getValue().getZExtValue();
        assert(instructionCounter < markerOperations.size() &&
               "instructionCounter not found in map!");
        auto &coreType = markerCoreTypes[instructionCounter];
        if (coreType == hivm::TCoreType::CUBE_OR_VECTOR)
          coreType = opCoreType;
        else if (coreType != opCoreType)
          coreType = hivm::TCoreType::CUBE_AND_VECTOR;
      }
    });
  });
  moduleClone->erase();
  for (auto [instructionCounter, coreType] : llvm::enumerate(markerCoreTypes)) {
    if (coreType == hivm::TCoreType::CUBE_OR_VECTOR)
      continue;
    Operation *op = markerOperations[instructionCounter];
    if (isOpTypeToBeMarked(op)) {
      op->setAttr(hivm::TCoreTypeAttr::name,
                  hivm::TCoreTypeAttr::get(op->getContext(), coreType));
    }
  }

  // Fallback: for ops that were DCE'd in moduleClone (e.g. dead iter_arg cycles),
  // inherit core type from the enclosing scope.scope. Registered kernels only.
  moduleOp.walk([&](Operation *op) {
    if (isOpTypeToBeMarked(op) && !op->hasAttr(hivm::TCoreTypeAttr::name)) {
      auto func = op->getParentOfType<func::FuncOp>();
      if (!func ||
          !allowLoopShapeHeuristics(this->bypassShapeRegistry, func.getName(),
                                    this->enablePreload))
        return;
      if (auto parentScope = op->getParentOfType<scope::ScopeOp>()) {
        if (auto attr = parentScope->getAttrOfType<hivm::TCoreTypeAttr>(
                hivm::kPipelinedLoopCoreTypeAttrName)) {
          op->setAttr(hivm::TCoreTypeAttr::name, attr);
        } else if (auto attr = parentScope->getAttrOfType<hivm::TCoreTypeAttr>(
                       hivm::TCoreTypeAttr::name)) {
          op->setAttr(hivm::TCoreTypeAttr::name, attr);
        }
      }
    }
  });
}

std::unique_ptr<Pass>
mlir::hivm::createMarkRealCoreTypePass(const MarkRealCoreTypeOptions &options) {
  return std::make_unique<MarkRealCoreTypePass>(options);
}
