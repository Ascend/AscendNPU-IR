//===- DropSymbols.cpp ----------------------------------------------------===//
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
// This file implements tensor.dim source replacer optimization
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hfusion-eliminate-duplicate-funcs"

namespace mlir {
#define GEN_PASS_DEF_ELIMINATEDUPLICATEFUNCS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;
using namespace llvm;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {

// Compare two funcOps
bool compareFuncOps(func::FuncOp f1, func::FuncOp f2) {
  if (f1.getFunctionTypeAttr() != f2.getFunctionTypeAttr())
    return false;

  if (f1->getAttrs().size() != f2->getAttrs().size())
    return false;

  for (auto [attr1, attr2] : llvm::zip(f1->getAttrs(), f2->getAttrs())) {
    if (attr1.getName() != attr2.getName())
      return false;

    // all attributes' value are matched except the following attributes
    if (llvm::isa<hacc::InferOutputShapeFunctionAttr, hacc::TilingFunctionAttr,
                  hacc::InferWorkspaceShapeFunctionAttr,
                  hacc::GetTilingStructSizeFunctionAttr,
                  hacc::InferSyncBlockLockNumFunctionAttr,
                  hacc::InferSyncBlockLockInitFunctionAttr,
                  hacc::InferTaskTypeFunctionAttr, mlir::StringAttr>(
            attr2.getValue()))
      continue;

    if (attr1.getValue() != attr2.getValue())
      return false;
  }

  auto &body1 = f1.getBody();
  auto &body2 = f2.getBody();

  if (body1.getBlocks().size() != body2.getBlocks().size())
    return false;

  for (auto [b1, b2] : llvm::zip(body1.getBlocks(), body2.getBlocks())) {
    if (b1.getOperations().size() != b2.getOperations().size())
      return false;

    for (auto [op1, op2] : llvm::zip(b1, b2)) {
      if (!OperationEquivalence::isEquivalentTo(
              &op1, &op2, OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::IgnoreLocations)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "op1: " << op1 << "\nop2: " << op2.getName() << "\nDiffer"
                   << "\n");
        return false;
      }
    }
  }

  return true;
}

// Create replacementMap where key: OldFuncOpName, value: NewFuncOpName
void createReplacementMap(
    SmallVector<func::FuncOp, 8> &funcs,
    llvm::DenseMap<func::FuncOp, func::FuncOp> &ReplacementMap) {
  // Double loop to find identical funcOps
  for (auto f1 = funcs.begin(); f1 != funcs.end(); ++f1) {
    for (auto f2 = std::next(f1); f2 != funcs.end(); ++f2) {
      if ((ReplacementMap.find(*f1) == ReplacementMap.end()) &&
          compareFuncOps(*f1, *f2))
        // by default using f1 to replace f2
        ReplacementMap[*f2] = *f1;
    }
  }

  // DBGPrint replacement map
#ifndef NDEBUG
  for (auto &[k, v] : ReplacementMap) {
    LLVM_DEBUG(llvm::dbgs() << "Replace " << k.getSymName() << " with "
                            << v.getSymName() << "\n");
  }
#endif
}

// Replace call found in ReplacementMap from host functions
bool replaceCallFromHost(
    SmallVector<func::FuncOp, 8> &funcs,
    llvm::DenseMap<func::FuncOp, func::FuncOp> &ReplacementMap) {
  for (auto f : funcs) {
    for (auto &[oldOp, newOp] : ReplacementMap) {
      if (oldOp == f || newOp == f)
        continue;
      if (failed(SymbolTable::replaceAllSymbolUses(
              oldOp.getSymNameAttr(), newOp.getSymNameAttr(), f))) {
        f.emitOpError() << "failed to replace all uses of "
                        << oldOp.getSymName() << " with " << newOp.getSymName();
        return false;
      }
    }
  }
  return true;
}

static void eraseHostFuncByName(SmallVector<func::FuncOp, 8> &funcOps,
                                SmallSet<func::FuncOp, 8> &funcToErase,
                                StringRef funcName) {
  for (auto f2 : funcOps) {
    if (f2.getSymName() == funcName) {
      funcToErase.insert(f2);
      break;
    }
  }
}

template <typename AttrType>
void processAttribute(Attribute attr, SmallSet<func::FuncOp, 8> &funcToErase,
                      SmallVector<func::FuncOp, 8> &funcOps) {
  if (auto i = dyn_cast<AttrType>(attr)) {
    eraseHostFuncByName(funcOps, funcToErase, i.getFuncNameStr());
  }
}

// Actual erase redundant functions from both DeviceFuncOps and HostFuncOps
void eraseUnusedFuncs(SmallVector<func::FuncOp, 8> funcOps,
                      DenseMap<func::FuncOp, func::FuncOp> ReplacementMap) {
  SmallSet<func::FuncOp, 8> funcToErase;
  for (auto f : funcOps) {
    if (ReplacementMap.find(f) != ReplacementMap.end()) {
      if (hacc::utils::isDevice(f))
        for (auto attr : f->getAttrs()) {
          processAttribute<hacc::TilingFunctionAttr>(attr.getValue(),
                                                     funcToErase, funcOps);
          processAttribute<hacc::InferOutputShapeFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
          processAttribute<hacc::InferWorkspaceShapeFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
          processAttribute<hacc::GetTilingStructSizeFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
          processAttribute<hacc::InferSyncBlockLockNumFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
          processAttribute<hacc::InferSyncBlockLockInitFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
          processAttribute<hacc::InferTaskTypeFunctionAttr>(
              attr.getValue(), funcToErase, funcOps);
        }
      funcToErase.insert(f);
    }
  }

  for (auto f : funcToErase)
    f.erase();
}

// Pass to erase redundant functions
void eraseRedundantFuncs(ModuleOp moduleOp) {
  SmallVector<func::FuncOp, 8> HostFuncOps;
  SmallVector<func::FuncOp, 8> DeviceFuncOps;
  moduleOp.walk([&](mlir::func::FuncOp funcOp) {
    if (hacc::utils::isDevice(funcOp))
      DeviceFuncOps.push_back(funcOp);

    if (hacc::utils::isHost(funcOp))
      HostFuncOps.push_back(funcOp);
  });

  llvm::DenseMap<func::FuncOp, func::FuncOp> ReplacementMap;
  createReplacementMap(DeviceFuncOps, ReplacementMap);
  createReplacementMap(HostFuncOps, ReplacementMap);
  // if replacement is successful, erase unused functions
  if (replaceCallFromHost(HostFuncOps, ReplacementMap)) {
    SmallVector<func::FuncOp, 8> AllFuncOps = HostFuncOps;
    AllFuncOps.append(DeviceFuncOps.begin(), DeviceFuncOps.end());
    eraseUnusedFuncs(AllFuncOps, ReplacementMap);
  }
}

} // namespace
} // namespace hfusion

} // namespace mlir

struct EliminateDuplicateFuncsPass
    : public impl::EliminateDuplicateFuncsBase<EliminateDuplicateFuncsPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running Eliminate Duplicate Funcs Pass\n");
    auto moduleOp = getOperation();
    hfusion::eraseRedundantFuncs(moduleOp);
  }
};
std::unique_ptr<Pass> mlir::hfusion::createEliminateDuplicateFuncsPass() {
  return std::make_unique<EliminateDuplicateFuncsPass>();
}
