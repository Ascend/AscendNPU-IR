//===- Utils.cpp - Utilities to support the HACC dialect ------------------===//
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
// This file implements utilities for the HACC dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "bishengir-hacc-utils"

namespace {
constexpr static llvm::StringLiteral kNPUStr = "NPU";
} // namespace

namespace mlir {
namespace hacc {
namespace utils {

//===----------------------------------------------------------------------===//
// Utility functions for HACCFunction
//===----------------------------------------------------------------------===//
constexpr const static uint64_t kBitsInByte = 8;

bool isHost(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return false;

  return haccFunction.isHost();
}

bool isDevice(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return false;

  return haccFunction.isDevice();
}

bool isDeviceEntry(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return false;

  return haccFunction.isDeviceEntry();
}

bool hasNoIOAlias(Operation *func) {
  return func->hasAttr(hacc::NoIOAliasAttr::name);
}

std::optional<HostFuncType> getHostFuncType(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return {};

  return haccFunction.getHostFuncType();
}

bool isKernelArg(Operation *func, unsigned argIdx, KernelArgType argType) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return false;

  return haccFunction.isKernelArg(argIdx, argType);
}

std::optional<BlockArgument> getBlockArgument(func::FuncOp funcOp,
                                              KernelArgType argType) {
  auto blockArgs = funcOp.getArguments();
  for (size_t idx = 0; idx < blockArgs.size(); idx++) {
    if (hacc::utils::isKernelArg(funcOp, idx, argType)) {
      return blockArgs[idx];
    }
  }
  return std::nullopt;
}

bool isTilingArg(Operation *func, unsigned argIdx) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return false;

  return haccFunction.isKernelArg(argIdx, hacc::KernelArgType::kTilingData) ||
         haccFunction.isKernelArg(argIdx, hacc::KernelArgType::kTilingKey);
}

void setDevice(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return;

  haccFunction.setDevice();
}

void setDeviceEntry(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return;

  haccFunction.setDeviceEntry();
}

void setHost(Operation *func) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return;

  haccFunction.setHost();
}

void setHostFuncType(Operation *func, HostFuncType hostFuncType) {
  auto haccFunction = dyn_cast_if_present<hacc::HACCFunction>(func);
  if (!haccFunction)
    return;

  haccFunction.setHostFuncType(hostFuncType);
}

void setAlwaysInline(Operation *func) {
  auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
      hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
  func->setAttr(haccAlwaysInlineAttr,
                OpBuilder(func->getContext()).getUnitAttr());
}

SmallVector<ExternalFuncInfo> collectExternalFuncs(ModuleOp mod) {
  SmallVector<ExternalFuncInfo> externalFuncs;
  llvm::SmallSet<typename llvm::StringRef, 1>
      seenFiles; // Assume 1 external func per host module

  mod.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (auto attr = funcOp->getAttrOfType<StringAttr>(
            hacc::ExternalFunctionPathAttr::name)) {
      StringRef srcPath = attr;
      StringRef funcName = funcOp.getName();

      if (seenFiles.insert(srcPath).second) {
        externalFuncs.push_back({funcName, srcPath});
      }
    }
  });

  return externalFuncs;
}

//===----------------------------------------------------------------------===//
// Data Layout and Target Info
//===----------------------------------------------------------------------===//

std::optional<HACCTargetDeviceSpecInterface> getNPUTargetSpec(ModuleOp op) {
  auto interface = op.getTargetSystemSpec();
  if (!interface)
    return std::nullopt;

  auto deviceSpec = interface.getDeviceSpecForDeviceID(
      StringAttr::get(op->getContext(), kNPUStr));
  if (!deviceSpec)
    return std::nullopt;

  return dyn_cast<HACCTargetDeviceSpecInterface>(deviceSpec.value());
}

void setNPUTargetSpec(ModuleOp op, HACCTargetDeviceSpecInterface spec) {
  MLIRContext *ctx = op->getContext();
  SmallVector<DeviceIDTargetDeviceSpecPair> entries;
  entries.push_back({StringAttr::get(ctx, kNPUStr), spec});
  op->setAttr(TargetSystemSpecAttr::name,
              TargetSystemSpecAttr::get(ctx, entries));
}

int64_t getIntegerSpecValue(DataLayoutEntryInterface entry) {
  return cast<IntegerAttr>(entry.getValue()).getValue().getSExtValue();
}

} // namespace utils

static ModuleOp getDeviceModule(ModuleOp op) {
  ModuleOp module = op.clone();
  LLVM_DEBUG(llvm::dbgs() << "Processing to get device " << op << "\n";);
  module->walk([&](HACCFunction func) {
    if (func.isHost())
      func->erase();
  });
  return module;
}

static void
resetDeclFuncLoc(LLVM::LLVMFuncOp /* don't need reference */ llvmFunc) {
  /// In LLVM IR, there are two types of debug information (!dbg):
  /// (1) distinct, (2) uniqued.
  /// According to llvm/lib/IR/Verifier.cpp:Verifier::visitFunction,
  /// "function declaration may only have a unique !dbg attachment".
  /// We need to set this in addition to making the function's body as
  /// empty.
  if (auto originalLoc =
          llvm::dyn_cast_if_present<FusedLoc>(llvmFunc.getLoc())) {
    auto originalAttr = cast<LLVM::DISubprogramAttr>(originalLoc.getMetadata());
    auto newAttr = LLVM::DISubprogramAttr::get(
        llvmFunc->getContext(), DistinctAttr(), LLVM::DICompileUnitAttr(),
        originalAttr.getScope(), originalAttr.getName(),
        originalAttr.getLinkageName(), originalAttr.getFile(), unsigned(),
        unsigned(), LLVM::DISubprogramFlags::Optimized, originalAttr.getType());
    auto newLoc = FusedLoc::get(originalLoc.getLocations(), newAttr,
                                llvmFunc->getContext());
    llvmFunc->setLoc(newLoc);
  }
}

static ModuleOp getHostModule(ModuleOp op) {
  ModuleOp module = op.clone();

  LLVM_DEBUG(llvm::dbgs() << "Processing to get host module " << op << "\n";);
  module->walk([&](HACCFunction func) {
    if (!func->hasAttr(HACCFuncTypeAttr::name) || func.isDevice()) {
      Region emptyRegion;
      if (auto llvmFunc = utils::dynCastFunc<LLVM::LLVMFuncOp>(func)) {
        llvmFunc.getBody().takeBody(emptyRegion);
        llvmFunc.setLinkage(LLVM::Linkage::External);
        resetDeclFuncLoc(llvmFunc);
      } else if (auto funcOp = utils::dynCastFunc<func::FuncOp>(func)) {
        funcOp.getBody().takeBody(emptyRegion);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
      }
    }
  });
  return module;
}

bool existHost(Operation *module) {
  bool ret = false;
  module->walk([&](HACCFunction op) { ret |= op.isHost(); });
  return ret;
}

static bool isCallLike(Operation *op) {
  return isa<LLVM::CallOp>(op) || isa<func::CallOp>(op);
}

bool existEntryHost(Operation *module) {
  bool ret = false;
  module->walk([&](HACCFunction func) {
    bool isCallingKernelHost = false;
    func->walk([&](Operation *callOp) {
      if (!isCallLike(callOp))
        return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs() << "Found call like OP " << *callOp << "\n";);
      if (auto callOpInf = dyn_cast<CallOpInterface>(callOp)) {
        CallInterfaceCallable callee = callOpInf.getCallableForCallee();

        LLVM_DEBUG(llvm::dbgs() << "got callee name "
                                << callee.get<SymbolRefAttr>() << "\n";);
        auto calleeOp =
            dyn_cast<HACCFunction>(SymbolTable::lookupNearestSymbolFrom(
                func, callee.get<SymbolRefAttr>()));

        LLVM_DEBUG(llvm::dbgs() << "got callee " << *calleeOp << "\n";);
        if (calleeOp.isDevice()) {
          isCallingKernelHost |= true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    isCallingKernelHost &= func.isHost();
    LLVM_DEBUG(llvm::dbgs() << *func << " " << isCallingKernelHost << "\n";);
    ret |= isCallingKernelHost;
    return WalkResult::advance();
  });
  return ret;
}

std::pair<ModuleOp, ModuleOp> separateHostDeviceModule(ModuleOp op) {
  return {getHostModule(op), getDeviceModule(op)};
}

ModuleOp filterFuncsInModule(ModuleOp &op,
                             std::function<bool(Operation *op)> shouldInclude) {
  ModuleOp module = op.clone();
  LLVM_DEBUG(llvm::dbgs() << "Processing to filter module " << op << "\n";);
  module->walk([&](HACCFunction func) {
    if (!shouldInclude(func.getOperation())) {
      func->erase();
    }
  });
  return module;
}

bool isMixEntry(Operation *func) {
  if (!func)
    return false;
  return func->hasAttr(
      hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::MIX_ENTRY));
}

bool notExportedAsDag(Operation *op) {
  return !op->hasAttr(hacc::ExportAsDAGAttr::name);
}

NamedAttribute createHACCKernelArgAttr(MLIRContext *ctx,
                                       KernelArgType argType) {
  return NamedAttribute(StringAttr::get(ctx, hacc::KernelArgTypeAttr::name),
                        hacc::KernelArgTypeAttr::get(ctx, argType));
}

std::optional<unsigned> getHACCOuputIdx(func::FuncOp func, unsigned argIdx) {
  auto dictAttr = func.getArgAttrDict(argIdx);
  if (!dictAttr)
    return std::nullopt;

  Attribute attr = dictAttr.get(hacc::OutputIdxAttr::name);
  auto outputIdxAttr = dyn_cast_or_null<hacc::OutputIdxAttr>(attr);
  if (!outputIdxAttr)
    return std::nullopt;

  return outputIdxAttr.getArgIdx();
}

std::optional<unsigned> getHACCInputIdx(func::FuncOp func, unsigned argIdx) {
  auto dictAttr = func.getArgAttrDict(argIdx);
  if (!dictAttr)
    return std::nullopt;

  Attribute attr = dictAttr.get(hacc::InputIdxAttr::name);
  auto inputIndexAttr = dyn_cast_or_null<hacc::InputIdxAttr>(attr);
  if (!inputIndexAttr)
    return std::nullopt;

  return inputIndexAttr.getArgIdx();
}

std::string constructHostFunctionName(const std::string &kernelName,
                                      HostFuncType type) {
  return llvm::formatv("{0}_{1}", kernelName, stringifyHostFuncType(type));
}

size_t countDeviceArgSizeInByte(ModuleOp modOp) {
  size_t maxArgSizeInBytes = 0;
  modOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (utils::isHost(funcOp))
      return WalkResult::advance();
    size_t curFuncArgSizeInBits = 0;
    for (auto argTypes : funcOp.getArgumentTypes()) {
      if (isa<LLVM::LLVMPointerType>(argTypes)) {
        LLVMTypeConverter llvmTypeConverter(funcOp->getContext());
        curFuncArgSizeInBits += llvmTypeConverter.getPointerBitwidth();
      } else {
        curFuncArgSizeInBits +=
            getElementTypeOrSelf(argTypes).getIntOrFloatBitWidth();
      }
    }
    size_t curFuncArgSizeInBytes = curFuncArgSizeInBits / utils::kBitsInByte;
    if (curFuncArgSizeInBytes > maxArgSizeInBytes) {
      maxArgSizeInBytes = curFuncArgSizeInBytes;
    }
    return WalkResult::advance();
  });
  return maxArgSizeInBytes;
}

} // namespace hacc
} // namespace mlir
