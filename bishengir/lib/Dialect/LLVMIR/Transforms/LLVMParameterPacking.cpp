//===- ParameterPacking.cpp - Pack pointer paramters to array -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/LLVMIR/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "parameter-packing"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_PARAMETERPACKING
#include "bishengir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;

namespace {
void packParameters(LLVM::LLVMFuncOp funcOp) {
  if (funcOp.getBody().empty())
    return;

  auto funcType = funcOp.getFunctionType();
  if (!llvm::all_of(funcType.getParams(), [](Type argType) {
        return isa<LLVM::LLVMPointerType>(argType) || isa<IntegerType>(argType);
      })) {
    assert(false && "Host function contains none packable parameter type. "
                    "Currently only support int and ptr.");
    return;
  }

  auto params = funcType.getParams();
  if (params.empty())
    return;

  OpBuilder builder(funcOp);
  auto *context = funcOp.getContext();

  // Create an array type of pointers
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto arrayTy = LLVM::LLVMArrayType::get(builder.getI64Type(), params.size());
  SmallVector<Type> constructParameterType;
  auto oldArgs = llvm::to_vector(funcOp.getArguments());

  // Construct struct type
  for (auto arg : oldArgs) {
    auto llvmType = arg.getType();
    constructParameterType.push_back(llvmType);
  }
  auto newFuncType = LLVM::LLVMFunctionType::get(funcType.getReturnType(),
                                                 ptrType, funcType.isVarArg());
  funcOp.setType(newFuncType);

  if (!funcOp.getBody().empty()) {
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    auto &entryBlock = funcOp.getBody().front();
    entryBlock.addArgument(ptrType, funcOp.getLoc());
    auto arrayOp = builder.create<LLVM::LoadOp>(
        funcOp->getLoc(), arrayTy, entryBlock.getArguments().back());

    for (const auto &it : llvm::enumerate(oldArgs)) {
      SmallVector<int64_t, 1> position;
      position.push_back(static_cast<int64_t>(it.index()));

      auto &currentType = constructParameterType[it.index()];
      LLVM_DEBUG(llvm::dbgs() << it.index() << " " << currentType << "\n";);
      auto extractOp = builder.create<LLVM::ExtractValueOp>(
          funcOp.getLoc(), builder.getI64Type(), arrayOp, position);
      if (auto intType = dyn_cast<IntegerType>(it.value().getType())) {
        it.value().replaceAllUsesWith(extractOp);
      } else {
        auto castToPtrOp = builder.create<LLVM::IntToPtrOp>(
            extractOp.getLoc(), currentType, extractOp.getRes());
        it.value().replaceAllUsesWith(castToPtrOp);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << funcOp << "\n";);
    funcOp.getBody().front().eraseArguments(0, oldArgs.size());
  }
}

struct ParameterPackingPass
    : public LLVM::impl::ParameterPackingBase<ParameterPackingPass> {
  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      auto hostFuncType = hacc::utils::getHostFuncType(funcOp);
      bool allowedFunction =
          (hostFuncType.has_value() &&
           (hostFuncType.value() == hacc::HostFuncType::kTilingFunction ||
            hostFuncType.value() ==
                hacc::HostFuncType::kInferOutputShapeFunction ||
            hostFuncType.value() ==
                hacc::HostFuncType::kInferWorkspaceShapeFunction ||
            hostFuncType.value() ==
                hacc::HostFuncType::kInferSyncBlockLockNumFunction ||
            hostFuncType.value() ==
                hacc::HostFuncType::kInferSyncBlockLockInitFunction));
      if (hacc::utils::isHost(funcOp) && allowedFunction)
        packParameters(funcOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createParameterPackingPass() {
  return std::make_unique<ParameterPackingPass>();
}
