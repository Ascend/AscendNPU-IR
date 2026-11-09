//===- VerifyVectorFunctionTopLevelCopy.cpp - Check VF top-level copy ----===//
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
// Post-bufferization IR checkpoint that fails when a vector function holds a
// memref.copy in its top-level block, which the downstream lowering pipeline
// does not support.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_VERIFYVECTORFUNCTIONTOPLEVELCOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct VerifyVectorFunctionTopLevelCopyPass
    : public mlir::impl::VerifyVectorFunctionTopLevelCopyBase<
          VerifyVectorFunctionTopLevelCopyPass> {
  void runOnOperation() override;
};

} // namespace

void VerifyVectorFunctionTopLevelCopyPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!hivm::isVF(funcOp) || funcOp.getBody().empty())
    return;
  bool failed = false;
  for (auto copyOp : funcOp.getBody().front().getOps<memref::CopyOp>()) {
    copyOp.emitOpError("is not supported in the top-level block of a vector "
                       "function");
    failed = true;
  }
  if (failed)
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createVerifyVectorFunctionTopLevelCopyPass() {
  return std::make_unique<VerifyVectorFunctionTopLevelCopyPass>();
}
