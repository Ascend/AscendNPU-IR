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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <memory>

#define DEBUG_TYPE "hivm-insert-infer-sync-block-lock-num-and-init-func"

namespace mlir {
#define GEN_PASS_DEF_INSERTINFERSYNCBLOCKLOCKNUMANDINITFUNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
SmallVector<Operation *> collectCreateSyncBlockLockOp(func::FuncOp funcOp) {
  SmallVector<Operation *> candidate;
  funcOp.walk([&](hivm::CreateSyncBlockLockOp op) { candidate.push_back(op); });

  return candidate;
}

func::FuncOp insertInferSyncBlockLockNumFuncImpl(func::FuncOp funcOp,
                                                 StringRef funcName,
                                                 int64_t syncBlockLockNum) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp);

  FunctionType funcType = FunctionType::get(
      funcOp.getContext(),
      /*input*/ TypeRange{}, /*result*/ TypeRange{builder.getI64Type()});
  auto func =
      builder.create<func::FuncOp>(funcOp.getLoc(),
                                   /*name*/ funcName, /*type*/ funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto lockNumVal = builder.create<arith::ConstantIntOp>(funcOp.getLoc(),
                                                         syncBlockLockNum, 64);
  builder.create<func::ReturnOp>(funcOp.getLoc(),
                                 ValueRange{lockNumVal.getResult()});
  return func;
}

void insertInferSyncBlockLockNumFunc(func::FuncOp funcOp,
                                     int64_t syncBlockLockNum) {
  std::string callbackFuncName = hacc::constructHostFunctionName(
      funcOp.getSymName().str(),
      hacc::HostFuncType::kInferSyncBlockLockNumFunction);
  func::FuncOp callbackFunc = insertInferSyncBlockLockNumFuncImpl(
      funcOp, callbackFuncName, syncBlockLockNum);
  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(
      callbackFunc, hacc::HostFuncType::kInferSyncBlockLockNumFunction);
}

func::FuncOp insertInferSyncBlockLockInitFuncImpl(func::FuncOp funcOp,
                                                  StringRef funcName,
                                                  int64_t syncBlockLockInit) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp);

  FunctionType funcType = FunctionType::get(
      funcOp.getContext(),
      /*input*/ TypeRange{}, /*result*/ TypeRange{builder.getI64Type()});
  auto func =
      builder.create<func::FuncOp>(funcOp.getLoc(),
                                   /*name*/ funcName, /*type*/ funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto lockInitVal = builder.create<arith::ConstantIntOp>(
      funcOp.getLoc(), syncBlockLockInit, 64);
  builder.create<func::ReturnOp>(funcOp.getLoc(),
                                 ValueRange{lockInitVal.getResult()});
  return func;
}

void insertInferSyncBlockLockInitFunc(func::FuncOp funcOp,
                                      int64_t syncBlockLockInit) {
  std::string callbackFuncName = hacc::constructHostFunctionName(
      funcOp.getSymName().str(),
      hacc::HostFuncType::kInferSyncBlockLockInitFunction);
  func::FuncOp callbackFunc = insertInferSyncBlockLockInitFuncImpl(
      funcOp, callbackFuncName, syncBlockLockInit);
  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(
      callbackFunc, hacc::HostFuncType::kInferSyncBlockLockInitFunction);
}

} // anonymous namespace

namespace {
class InsertInferSyncBlockLockNumAndInitFuncPass
    : public impl::InsertInferSyncBlockLockNumAndInitFuncBase<
          InsertInferSyncBlockLockNumAndInitFuncPass> {
public:
  using InsertInferSyncBlockLockNumAndInitFuncBase<
      InsertInferSyncBlockLockNumAndInitFuncPass>::
      InsertInferSyncBlockLockNumAndInitFuncBase;
  void runOnOperation() override;
};
} // namespace

void InsertInferSyncBlockLockNumAndInitFuncPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<Operation *> createSyncBlockLockOps =
      collectCreateSyncBlockLockOp(funcOp);
  if (createSyncBlockLockOps.empty())
    return;

  // 1. Calculate total sync block lock num
  auto syncBlockLockNum = createSyncBlockLockOps.size();

  // 2. Insert host callback func to return sync block lock num
  insertInferSyncBlockLockNumFunc(funcOp, syncBlockLockNum);

  // 2. Insert host callback func to return sync block lock init
  insertInferSyncBlockLockInitFunc(funcOp, 0);
}

std::unique_ptr<Pass>
mlir::hivm::createInsertInferSyncBlockLockNumAndInitFuncPass() {
  return std::make_unique<InsertInferSyncBlockLockNumAndInitFuncPass>();
}
