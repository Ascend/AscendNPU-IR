//===---------- InsertInferSyncBlockLockSizeAndInitFunc.cpp ---------------===//
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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
#define GEN_PASS_DECL_INSERTINFERTASKTYPEFUNC
#define GEN_PASS_DEF_INSERTINFERTASKTYPEFUNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
func::FuncOp insertInferTaskTypeFuncImpl(func::FuncOp funcOp,
                                         StringRef funcName, int8_t taskType) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp);

  FunctionType funcType = FunctionType::get(
      funcOp.getContext(),
      /*input*/ TypeRange{}, /*result*/ TypeRange{builder.getI8Type()});
  auto func =
      builder.create<func::FuncOp>(funcOp.getLoc(),
                                   /*name*/ funcName, /*type*/ funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto taskTypeValue =
      builder.create<arith::ConstantIntOp>(funcOp.getLoc(), taskType, 8);
  builder.create<func::ReturnOp>(funcOp.getLoc(),
                                 ValueRange{taskTypeValue.getResult()});
  return func;
}

void insertInferTaskTypeFunc(func::FuncOp funcOp, int8_t taskType) {
  std::string callbackFuncName = hacc::constructHostFunctionName(
      funcOp.getSymName().str(), hacc::HostFuncType::kInferTaskTypeFunction);
  func::FuncOp callbackFunc =
      insertInferTaskTypeFuncImpl(funcOp, callbackFuncName, taskType);
  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(callbackFunc,
                               hacc::HostFuncType::kInferTaskTypeFunction);
}
} // anonymous namespace

namespace {
class InsertInferTaskTypeFuncPass
    : public impl::InsertInferTaskTypeFuncBase<InsertInferTaskTypeFuncPass> {
public:
  using InsertInferTaskTypeFuncBase<
      InsertInferTaskTypeFuncPass>::InsertInferTaskTypeFuncBase;
  void runOnOperation() override;
};
} // namespace

void InsertInferTaskTypeFuncPass::runOnOperation() {
  ModuleOp module = getOperation();
  int8_t taskType = util::TaskType::Unknown;
  func::FuncOp entryFunc = nullptr;
  module.walk(
      [&](func::FuncOp func) {
        bool isHivmEntry = func->hasAttr("hivm.entry");
        bool isHaccEntry = func->hasAttr("hacc.entry");

        if (!isHaccEntry && !isHivmEntry)
          return WalkResult::advance();

        if (entryFunc) {
          func->emitError()
              << "multiple entry functions detected (previous: '"
              << entryFunc.getName() << "', this: '" << func.getName()
              << "'). Only one function may be marked with hivm.entry or "
                 "hacc.entry.";
          signalPassFailure();
          WalkResult::interrupt();
        }

        entryFunc = func;
        return WalkResult::advance();
      });

  if (!entryFunc) {
    module.emitWarning() << "No function with attribute hivm.entry or "
                            "hacc.entry found in the module. "
                         << "Task type is set to Unknown.";
    taskType = util::TaskType::Unknown;
  } else {
    auto coreAttr = entryFunc->getAttrOfType<hivm::TFuncCoreTypeAttr>(
        hivm::TFuncCoreTypeAttr::name);

    if (!coreAttr) {
      entryFunc->emitError() << "Entry function '" << entryFunc.getName()
                             << "' does not have the required '"
                             << hivm::TFuncCoreTypeAttr::name << "' attribute.";
      signalPassFailure();
      return;
    }
    switch (coreAttr.getFuncCoreType()) {
    case hivm::TFuncCoreType::AIC:
      taskType = util::TaskType::CubeOnly;
      break;
    case hivm::TFuncCoreType::AIV:
      taskType = util::TaskType::VectorOnly;
      break;
    case hivm::TFuncCoreType::MIX:
      taskType = util::TaskType::CubeVectorMix_1_2;
      break;
    default:
      taskType = util::TaskType::Unknown;
      break;
    }
  }

  module.walk([&](func::FuncOp func) {
    if (hacc::utils::isHost(func))
      return WalkResult::advance();
    insertInferTaskTypeFunc(func, taskType);
    return WalkResult::interrupt();
  });
}

std::unique_ptr<Pass> mlir::hivm::createInsertInferTaskTypeFuncPass() {
  return std::make_unique<InsertInferTaskTypeFuncPass>();
}