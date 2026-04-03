//===- CostModelInfoUtils.cpp - Implementation of
// CostModelInfoUtils--------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfoUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "hardware_info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {
bool isDeviceInited = false;
std::string finalTargetDevice;
hacc::TargetDevice targetDevice;
const CostInfo DEFAULT_COST_INFO = {1,4};

TypeKind CostModelInfoUtils::getTypeKind(mlir::Type type) {
  if (auto vecType = mlir::dyn_cast<mlir::VectorType>(type)) {
    return getTypeKind(vecType.getElementType());
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF16())
      return TypeKind::F16;
    if (floatType.isF32())
      return TypeKind::F32;
    if (floatType.isF64())
      return TypeKind::F64;
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1)
      return TypeKind::I1;
    if (intType.getWidth() == 32)
      return TypeKind::I32;
    if (intType.getWidth() == 64)
      return TypeKind::I64;
  }
  if (auto indexType = mlir::dyn_cast<mlir::IndexType>(type)) {
    return TypeKind::Index;
  }
  return TypeKind::Unknown;
}

CostInfo CostModelInfoUtils::lookupConfig(const OpConfigMap &targetMap,
                             mlir::TypeID opTypeID, mlir::Type dataType,
                             const llvm::StringRef opName) {
  auto typeIDIter = targetMap.find(opTypeID);
  // fall back for no config of OP
  if (typeIDIter == targetMap.end()) {
    LDBG("No config for op: " << opName);
    return DEFAULT_COST_INFO;
  }
  TypeKind typeKind = getTypeKind(dataType);
  auto &typeToInfoMap = typeIDIter->second;
  auto typeKindIter = typeToInfoMap.find(typeKind);
  // fall back for no config of dataType
  if (typeKindIter == typeToInfoMap.end()) {
    TypeKind fallbackTypeKind = TypeKind::Unknown;
    if (mlir::dyn_cast<mlir::FloatType>(dataType)) {
      fallbackTypeKind = TypeKind::F32;
    } else if (mlir::dyn_cast<mlir::IntegerType>(dataType)) {
      fallbackTypeKind = TypeKind::I32;
    }
    auto fallbackIter = typeToInfoMap.find(fallbackTypeKind);
    if (fallbackIter != typeToInfoMap.end()) {
      LDBG(opName << "<<has no config for Type" << dataType
                  << "fall bach to defalut info"
                  << (fallbackTypeKind == TypeKind::F32 ? "F32" : "I32"));
      return fallbackIter->second;
    }
    // without distinguishing between data types
    fallbackTypeKind = typeToInfoMap.begin()->first;
    fallbackIter = typeToInfoMap.find(fallbackTypeKind);
    LDBG("without distinguishing between data types of op" << opName);
    return fallbackIter->second;
  }
  return typeKindIter->second;
}

OpConfigMap CostModelInfoUtils::getOpConfigMap(hacc::TargetDevice dev,
                                  bool isReduction) {

  if (hacc::utils::isAscend950(dev)) {
    return std::make_unique<CostModelInfo>()->getConfigMap(isReduction);
  } else {
    return OpConfigMap();
  }
}

CostInfo CostModelInfoUtils::getOpCostInfo(mlir::Operation *op, bool isReduction) {
  mlir::TypeID opTypeID = op->getRegisteredInfo()->getTypeID();
  llvm::StringRef opName = op->getName().getStringRef();
  if (!isDeviceInited) {
    ModuleOp moduleOp = utils::getTopLevelModuleOp(op);
    auto maybeTargetDevice = hacc::utils::getTargetDevice(moduleOp);
    if (maybeTargetDevice.has_value()) {
      targetDevice = maybeTargetDevice.value();
      finalTargetDevice = hacc::stringifyTargetDeviceEnum(targetDevice);
      LDBG("TargetDevice is" << finalTargetDevice);
      isDeviceInited = true;
    }
  }
  const auto &targetMap = getOpConfigMap(targetDevice, isReduction);
  auto results = op->getResults();
  mlir::Type dataType = results.front().getType();
  return lookupConfig(targetMap, opTypeID, dataType, opName);
}
} // namespace mlir::analysis