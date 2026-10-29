//===- HIVMPipelineBase.cpp - Shared HIVM pipeline base -------------------===//
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

#include "bishengir/Dialect/HIVM/Pipelines/HIVMPipelineBase.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivm {

void canonicalizationHIVMPipeline(OpPassManager &pm) {
  pm.addPass(createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(scf::createCanonicalizeIterArgPass());
  pm.addPass(bishengir::createExtendedCanonicalizerPass());
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  // No canonicalizer between cse and single-point: its folds are subsumed by
  // the trailing canonicalizer of this cluster.
  pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
  // No canonicalizer between single-point and dead store elimination:
  // dead-store matching does not depend on its rewrites, and the trailing
  // canonicalizer subsumes them.
  pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
  pm.nest<func::FuncOp>().addPass(bishengir::createExtendedCanonicalizerPass());
}

} // namespace hivm
} // namespace mlir
