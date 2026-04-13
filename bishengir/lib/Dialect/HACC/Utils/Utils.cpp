//===- Utils.cpp - Utilities to support the HACC dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the HACC dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/FormatVariadic.h"
#include <optional>
#include <unordered_set>

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

std::optional<TargetDevice> getTargetDevice(ModuleOp op) {
  if (auto targetAttr = op->getAttrOfType<TargetAttr>(TargetAttr::name))
    return symbolizeTargetDeviceEnum(targetAttr.getTarget());
  return std::nullopt;
}

void setTargetDevice(ModuleOp op, TargetDevice targetDevice) {
  MLIRContext *ctx = op->getContext();
  op->setAttr(
      TargetAttr::name,
      TargetAttr::get(
          ctx, StringAttr::get(ctx, stringifyTargetDeviceEnum(targetDevice))));
  return;
}

// TODO: we should use td to check the arch automatically
bool isAscend910B(TargetDevice targetDevice) {
  return targetDevice == TargetDevice::Ascend910B1 ||
         targetDevice == TargetDevice::Ascend910B2 ||
         targetDevice == TargetDevice::Ascend910B3 ||
         targetDevice == TargetDevice::Ascend910B4;
}

bool isAscend910_93(TargetDevice targetDevice) {
  return targetDevice == TargetDevice::Ascend910_9362 ||
         targetDevice == TargetDevice::Ascend910_9372 ||
         targetDevice == TargetDevice::Ascend910_9381 ||
         targetDevice == TargetDevice::Ascend910_9382 ||
         targetDevice == TargetDevice::Ascend910_9391 ||
         targetDevice == TargetDevice::Ascend910_9392;
}

bool isMemBasedArch(TargetDevice targetDevice) {
  return isAscend910B(targetDevice) || isAscend910_93(targetDevice);
}

bool isAscend310B(TargetDevice targetDevice) {
  return targetDevice == TargetDevice::Ascend310B1 ||
         targetDevice == TargetDevice::Ascend310B2 ||
         targetDevice == TargetDevice::Ascend310B3 ||
         targetDevice == TargetDevice::Ascend310B4;
}

// use unordered_set to speedup because this func is frequently called
bool isAscend950(TargetDevice targetDevice) {
  static const std::unordered_set<TargetDevice> ascend950Devices = {
      TargetDevice::Ascend910_950z,   TargetDevice::Ascend910_9579,
      TargetDevice::Ascend910_957b,   TargetDevice::Ascend910_957d,
      TargetDevice::Ascend910_9581,   TargetDevice::Ascend910_9589,
      TargetDevice::Ascend910_958a,   TargetDevice::Ascend910_958b,
      TargetDevice::Ascend910_9599,   TargetDevice::Ascend950PR_950z,
      TargetDevice::Ascend950PR_9579, TargetDevice::Ascend950PR_957a,
      TargetDevice::Ascend950PR_957b, TargetDevice::Ascend950PR_957c,
      TargetDevice::Ascend950PR_957d, TargetDevice::Ascend950PR_9589,
      TargetDevice::Ascend950PR_958a, TargetDevice::Ascend950PR_958b,
      TargetDevice::Ascend950PR_958c, TargetDevice::Ascend950PR_958d,
      TargetDevice::Ascend950PR_9599, TargetDevice::Ascend950PR_959a,
      TargetDevice::Ascend950PR_959b, TargetDevice::Ascend950DT_950x,
      TargetDevice::Ascend950DT_950y, TargetDevice::Ascend950DT_9571,
      TargetDevice::Ascend950DT_9572, TargetDevice::Ascend950DT_9573,
      TargetDevice::Ascend950DT_9574, TargetDevice::Ascend950DT_9575,
      TargetDevice::Ascend950DT_9576, TargetDevice::Ascend950DT_9577,
      TargetDevice::Ascend950DT_9578, TargetDevice::Ascend950DT_9581,
      TargetDevice::Ascend950DT_9582, TargetDevice::Ascend950DT_9583,
      TargetDevice::Ascend950DT_9584, TargetDevice::Ascend950DT_9585,
      TargetDevice::Ascend950DT_9586, TargetDevice::Ascend950DT_9587,
      TargetDevice::Ascend950DT_9588, TargetDevice::Ascend950DT_9591,
      TargetDevice::Ascend950DT_9592, TargetDevice::Ascend950DT_9595,
      TargetDevice::Ascend950DT_9596, TargetDevice::Ascend950DT_95A1,
      TargetDevice::Ascend950DT_95A2};

  return ascend950Devices.find(targetDevice) != ascend950Devices.end();
}

bool isRegBasedArch(TargetDevice targetDevice) {
  return isAscend310B(targetDevice) || isAscend950(targetDevice);
}

bool isFFTSSupportedArch(TargetDevice targetDevice) {
  return isAscend910B(targetDevice) || isAscend910_93(targetDevice) ||
         isAscend950(targetDevice);
}

bool isAscend910B(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return true;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend910B(targetDevice);
}

bool isAscend910_93(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return false;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend910_93(targetDevice);
}

bool isMemBasedArch(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return true;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend910B(targetDevice) || isAscend910_93(targetDevice);
}

bool isAscend310B(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return false;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend310B(targetDevice);
}

bool isAscend950(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return false;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend950(targetDevice);
}

bool isRegBasedArch(ModuleOp op) {
  auto maybeTargetDevice = getTargetDevice(op);
  if (!maybeTargetDevice.has_value())
    // Default is 910B. To ensure compatibility, return false
    // if no target device exists in the IR.
    return false;
  auto targetDevice = maybeTargetDevice.value();
  return isAscend310B(targetDevice) || isAscend950(targetDevice);
}

} // namespace utils

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

static bool isCombinerLikeRegion(Region &reg, unsigned expectedNumArgs = 2) {
  if (!reg.hasOneBlock()) {
    return false;
  }
  auto &block = reg.front();
  if (block.getNumArguments() != expectedNumArgs) {
    return false;
  }
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  return yield && yield.getNumOperands() == 1;
}

static bool verifyReduceSupportedParams(linalg::ReduceOp op) {
  if (op.hasIndexSemantics()) {
    return false;
  }
  return isCombinerLikeRegion(op.getRegion());
}

static std::optional<std::pair<RankedTensorType, RankedTensorType>>
getAndVerifyReduceInputs(linalg::ReduceOp op) {
  if (op.getNumDpsInits() != 1 || op.getNumDpsInputs() != 1) {
    return {};
  }

  Value input = op.getDpsInputOperand(0)->get();
  Value outInit = op.getDpsInitOperand(0)->get();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(outInit.getType());
  if (!inputType || !outputType) {
    return {};
  }

  return std::pair{inputType, outputType};
}

static std::optional<SmallVector<int64_t, 2>>
verifyReduceGeometryAndGetReduceDims(linalg::ReduceOp op) {
  auto reduceInputs = getAndVerifyReduceInputs(op);
  if (!reduceInputs) {
    return {};
  }
  auto [inputType, outputType] = *reduceInputs;

  auto inputRank = inputType.getRank();
  if (inputRank < 1) {
    return {};
  }

  auto outRank = outputType.getRank();
  if (outRank >= inputRank || outRank < 0) {
    return {};
  }

  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 2) {
    return {};
  }

  auto inMap = maps[0];
  if (!inMap.isIdentity()) {
    return {};
  }
  if (inMap.getNumDims() != inputRank) {
    return {};
  }

  auto outMap = maps[1];
  if (outMap.getNumDims() != inputRank) {
    return {};
  }
  if (outMap.getNumResults() != outRank) {
    return {};
  }

  auto iters = op.getIteratorTypesArray();
  SmallVector<int64_t, 2> reduceDims;
  for (auto [i, iterType] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (iterType == ::mlir::utils::IteratorType::reduction) {
      reduceDims.push_back(i);
    } else {
      if (iterType != ::mlir::utils::IteratorType::parallel) {
        return {};
      }
    }
  }

  if (inputRank - (int)reduceDims.size() != outRank) {
    return {};
  }
  return reduceDims;
}

bool isSkippable(linalg::ReduceOp op) {
  return op.getNumDpsInits() != 1 || op.getNumDpsInputs() != 1 ||
         isCombinerLikeRegion(op.getRegion(), 4); // reduce with index
}

bool isLegalToAutoVectorizeReduce(linalg::ReduceOp op) {
  assert(isLegalReduceOp(op));
  auto isComplexSingleOp = [](Operation *op) {
    return op->getDialect()->getNamespace() ==
           scf::SCFDialect::getDialectNamespace();
  };
  auto getSingleReduceOp =
      [&isComplexSingleOp](linalg::ReduceOp op) -> std::optional<Operation *> {
    auto &region = op.getRegion();
    assert(isCombinerLikeRegion(region));

    auto &ops = region.front().getOperations();
    if (ops.size() > 2 && !isComplexSingleOp(&ops.front())) {
      return {};
    }
    return &ops.front();
  };

  auto reduceOp = getSingleReduceOp(op);
  if (!reduceOp) {
    return false;
  }

  if (isa<arith::AndIOp, arith::OrIOp>(*reduceOp)) {
    // reduce i1 andi/ori will be normalized to i16 min/max in
    // hfusion-normalize-ops, so it's legal to vectorize
    auto elemType = (*reduceOp)->getOperand(0).getType();
    return elemType.isInteger(1);
  }

  return isa<linalg::YieldOp, arith::AddFOp, arith::AddIOp, arith::MaximumFOp,
             arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
             arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
             arith::XOrIOp>(*reduceOp);
}

bool isLegalReduceOp(linalg::ReduceOp op) {
  return verifyReduceGeometryAndGetReduceDims(op) &&
         verifyReduceSupportedParams(op);
}

} // namespace hacc
} // namespace mlir
