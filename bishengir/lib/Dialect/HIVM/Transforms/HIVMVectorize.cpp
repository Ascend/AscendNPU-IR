//===- HIVMVectorize.cpp - tensor-level HIVM vectorization ---------------===//
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
#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/TransformOps/Partition.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeResult.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/HIVMVectorizeBuffers.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMVECTORIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
using namespace mlir;
using namespace mlir::transform;

bool hasUnvectorizedCompute(ModuleOp module) {
  bool remaining = false;
  module.walk([&](Operation *op) {
    if (!mlir::hivm::isVectorizationCompute(*op))
      return;
    if (op->getNumResults() &&
        llvm::all_of(op->getResultTypes(),
                     [](Type type) { return isa<RankedTensorType>(type); }))
      remaining = true;
  });
  return remaining;
}

constexpr llvm::StringLiteral kHIVMVectorizeLibrary =
#include "HIVMVectorize/HIVMVectorize.inc"
    ;

constexpr llvm::StringLiteral kHIVMVectorizeEntryPoint = "hivm_vectorize";
constexpr llvm::StringLiteral kHIVMVectorizeLibraryName =
    "<hivm-vectorize-library>";

using namespace mlir;
using namespace mlir::transform;

OwningOpRef<ModuleOp> parseLibrary(MLIRContext *context) {
  return parseSourceString<ModuleOp>(kHIVMVectorizeLibrary, context,
                                     kHIVMVectorizeLibraryName);
}

transform::NamedSequenceOp findEntry(ModuleOp library) {
  return library.lookupSymbol<transform::NamedSequenceOp>(
      kHIVMVectorizeEntryPoint);
}

RaggedArray<transform::MappedValue>
makeTransformBindings(ModuleOp module, MLIRContext *context) {
  auto *moduleOp = module.getOperation();
  auto options = DictionaryAttr::get(context);

  RaggedArray<transform::MappedValue> bindings;
  bindings.push_back(ArrayRef<Operation *>{moduleOp});
  bindings.push_back(ArrayRef<Attribute>{options});
  return bindings;
}

LogicalResult applyLibrary(ModuleOp module, ModuleOp library,
                           transform::NamedSequenceOp entry) {
  auto bindings = makeTransformBindings(module, module.getContext());
  auto entryPoint = cast<transform::TransformOpInterface>(*entry);
  return transform::applyTransformNamedSequence(
      std::move(bindings), entryPoint, library, transform::TransformOptions());
}

bool needsStaticCallOperandCopy(Value operand) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  auto *source = operand.getDefiningOp();
  return type && type.hasStaticShape() && source &&
         isa<tensor::ExtractSliceOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp>(source);
}

void materializeCallOperandCopies(func::CallOp call) {
  OpBuilder builder(call);
  for (auto &operand : call->getOpOperands()) {
    if (!needsStaticCallOperandCopy(operand.get()))
      continue;
    auto type = cast<RankedTensorType>(operand.get().getType());
    auto copy = builder.create<bufferization::AllocTensorOp>(
        call.getLoc(), type, ValueRange{}, operand.get());
    operand.set(copy);
  }
}

void materializeStaticCallOperandCopies(ModuleOp module) {
  module.walk([&](func::CallOp call) {
    auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (function && function->hasAttr("hivm.vector_function"))
      materializeCallOperandCopies(call);
  });
}

void addFunctionPass(OpPassManager &pipeline, std::unique_ptr<Pass> pass) {
  pipeline.nest<func::FuncOp>().addPass(std::move(pass));
}

void markAllocationLayoutForVectorCall(ModuleOp module, func::CallOp call) {
  auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
  if (!callee || !callee->hasAttr("hivm.vector_function"))
    return;
  for (auto operand : call.getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() < 2)
      continue;
    auto root = hivm::traceToRoot(operand);
    auto *allocation = root.getDefiningOp();
    if (!allocation || !utils::isAllocLikeOp(allocation) ||
        utils::getAnnotateOpWithAttr(root,
                                     hivm::SkipStrideAlignForVLoadAttr::name))
      continue;
    OpBuilder builder(allocation);
    builder.setInsertionPointAfter(allocation);
    auto mark = builder.create<annotation::MarkOp>(call.getLoc(), root);
    mark->setAttr(hivm::SkipStrideAlignForVLoadAttr::name,
                  hivm::SkipStrideAlignForVLoadAttr::get(module.getContext()));
  }
}

void markVectorFunctionAllocationLayouts(ModuleOp module) {
  if (!hacc::utils::isRegBasedArch(module))
    return;
  module.walk([&](func::CallOp call) {
    markAllocationLayoutForVectorCall(module, call);
  });
}

void addPrepareBuffersPhase(OpPassManager &pipeline) {
  pipeline.addPass(hivm::createOutlineAllocInVFPass());
  pipeline.addPass(hivm::createOutlineCopyInVFPass());
  pipeline.addPass(bufferization::createDropEquivalentBufferResultsPass());
}

void addAlignmentPhase(OpPassManager &pipeline) {
  addFunctionPass(pipeline, hivm::createPreMarkStrideAlignPass());
  addFunctionPass(pipeline, hivm::createMarkStrideAlignPass());
}

void addBufferVectorizationPhase(OpPassManager &pipeline) {
  pipeline.addPass(hivm::createHIVMVectorizeBuffersPass());
}

void addTailNormalizationPhase(OpPassManager &pipeline) {
  pipeline.addPass(hivm::createHIVMVectorizePhasePass("backend"));
  addFunctionPass(pipeline, arith::createNormalizeArithPass());
  addFunctionPass(pipeline, arith::createLiftArithIndexCastPass());
  addFunctionPass(pipeline, vector::createPeelLoopsContainingTransposePass());
  pipeline.addPass(createCanonicalizerPass());
  addFunctionPass(pipeline, vector::createNormalizeVectorPass());
  addFunctionPass(pipeline, vector::createHoistConstantMasksPass());
  addFunctionPass(pipeline, createCSEPass());
  addFunctionPass(pipeline, hivm::createArithVectorMaskAnalysisPass());
}

LogicalResult buildVectorizerCompatibilityPipeline(OpPassManager &pipeline,
                                                   StringRef phase) {
  if (phase == "prepare-buffers") {
    addPrepareBuffersPhase(pipeline);
    return success();
  }
  if (phase == "align") {
    addAlignmentPhase(pipeline);
    return success();
  }
  if (phase == "buffers") {
    addBufferVectorizationPhase(pipeline);
    return success();
  }
  if (phase == "tail") {
    addTailNormalizationPhase(pipeline);
    return success();
  }
  return failure();
}

LogicalResult normalizeBackendFunctions(ModuleOp module) {
  WalkResult result = module.walk([&](func::FuncOp function) {
    if (!function->hasAttr("hivm.vector_function"))
      return WalkResult::advance();
    return failed(hivm::normalizeVectorizerBackendForms(function))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult vectorizeTensorPhase(ModuleOp module) {
  auto library = parseLibrary(module.getContext());
  if (!library) {
    module.emitError("failed to load the HIVM vectorization transform library");
    return failure();
  }
  if (failed(verify(*library)))
    return failure();

  auto entry = findEntry(*library);
  if (!entry) {
    module.emitError(
        "missing @hivm_vectorize entry in the HIVM vectorization transform "
        "library");
    return failure();
  }

  if (failed(applyLibrary(module, *library, entry)))
    return failure();
  materializeStaticCallOperandCopies(module);
  if (hasUnvectorizedCompute(module)) {
    module.emitError("HIVM vectorization left tensor compute unvectorized");
    return failure();
  }
  return verify(module);
}

struct HIVMVectorizePass
    : public mlir::impl::HIVMVectorizeBase<HIVMVectorizePass> {
  void setPhase(StringRef value) { phase = value.str(); }

  LogicalResult runCompatibilityPhase(ModuleOp module, StringRef phase) {
    OpPassManager pipeline(ModuleOp::getOperationName());
    if (failed(buildVectorizerCompatibilityPipeline(pipeline, phase)))
      return failure();
    return runPipeline(pipeline, module);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    HIVMVectorizeBase::getDependentDialects(registry);
    registry.insert<mlir::hfusion::HFusionDialect, mlir::hivmave::AVEDialect>();
    mlir::hivm::registerVectorizeTransformDialectExtension(registry);
    mlir::linalg::registerTransformDialectExtension(registry);
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::scf::registerTransformDialectExtension(registry);
    mlir::tensor::registerTransformDialectExtension(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (phase == "align")
      markVectorFunctionAllocationLayouts(module);
    if (phase == "backend") {
      if (failed(normalizeBackendFunctions(module)))
        signalPassFailure();
      return;
    }
    if (phase != "tensor") {
      if (failed(runCompatibilityPhase(module, phase)))
        signalPassFailure();
      return;
    }
    if (failed(vectorizeTensorPhase(module)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::hivm::createHIVMVectorizePass() {
  return std::make_unique<HIVMVectorizePass>();
}

std::unique_ptr<mlir::Pass>
mlir::hivm::createHIVMVectorizePhasePass(llvm::StringRef phase) {
  auto pass = std::make_unique<HIVMVectorizePass>();
  pass->setPhase(phase);
  return pass;
}
