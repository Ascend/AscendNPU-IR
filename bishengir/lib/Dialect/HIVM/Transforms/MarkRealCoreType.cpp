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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
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
  auto clonedOpMap = mapper.getOperationMap();
  DenseMap<Operation *, Operation *> invClonedOpMap;
  for (auto &[op, clonedOp] : clonedOpMap) {
    invClonedOpMap[clonedOp] = op;
  }

  DenseMap<uint64_t, Operation *> instructionCounterToOpMap;
  DenseMap<Operation *, hivm::TCoreType> instructionCounterToCoreTypeMap;
  static constexpr StringLiteral kInstructionMarkerAttr = "instruction-marker";

  // annotate instruction counter attribute to each op in moduleClone
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  uint64_t instructionCounter = 0;
  moduleClone->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ModuleOp>(op)) {
      return;
    }
    instructionCounterToOpMap[instructionCounter] = invClonedOpMap[op];
    op->setAttr(kInstructionMarkerAttr,
                builder.getIndexAttr(instructionCounter));
    instructionCounter++;
  });

  // run split mix kernel pass to annotate core type attribute
  PassManager pm(moduleClone.getContext());
  pm.addPass(createSplitMixKernelPass());
  canonicalizationHIVMPipeline(pm);
  if (failed(pm.run(moduleClone))) {
    return signalPassFailure();
  }

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
        assert(instructionCounterToOpMap.count(instructionCounter) &&
               "instructionCounter not found in map!");
        Operation *opInOriginalModule =
            instructionCounterToOpMap[instructionCounter];

        auto [it, inserted] = instructionCounterToCoreTypeMap.insert(
            {opInOriginalModule, opCoreType});
        if (!inserted && it->second != opCoreType) {
          it->second = hivm::TCoreType::CUBE_AND_VECTOR;
        }
      }
    });
  });
  moduleClone->erase();
  for (auto &[op, coreType] : instructionCounterToCoreTypeMap) {
    if (isOpTypeToBeMarked(op)) {
      op->setAttr(hivm::TCoreTypeAttr::name,
                  hivm::TCoreTypeAttr::get(op->getContext(), coreType));
    }
  }
}

std::unique_ptr<Pass>
mlir::hivm::createMarkRealCoreTypePass(const MarkRealCoreTypeOptions &options) {
  return std::make_unique<MarkRealCoreTypePass>(options);
}
