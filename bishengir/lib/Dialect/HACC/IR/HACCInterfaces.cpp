//===- HACCInterfaces.cpp - HACC interfaces implementation ----------------===//
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

#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "hacc-interfaces"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hacc;

#include "bishengir/Dialect/HACC/IR/HACCAttrInterfaces.cpp.inc"
#include "bishengir/Dialect/HACC/IR/HACCInterfaces.cpp.inc"

namespace {

/// Mapping from function type to the set of disallowed attr keys.
const std::map<HACCFuncType, std::set<std::string>> kFuncType2DisallowedAttrs =
    {{HACCFuncType::HOST,
      std::set<std::string>{
          // The following attributes are for device function only. They
          // would affect the calling convention.
          stringifyHACCToLLVMIRTranslateAttr(HACCToLLVMIRTranslateAttr::ENTRY)
              .str(),
          stringifyHACCToLLVMIRTranslateAttr(
              HACCToLLVMIRTranslateAttr::MIX_ENTRY)
              .str(),
          // The following attributes are for device functions to mark the
          // corresponding host function.
          TilingFunctionAttr::name.str(),
          InferOutputShapeFunctionAttr::name.str(),
          InferWorkspaceShapeFunctionAttr::name.str(),
          GetTilingStructSizeFunctionAttr::name.str(),
          InferSyncBlockLockNumFunctionAttr::name.str(),
          InferSyncBlockLockInitFunctionAttr::name.str(),
          InferTaskTypeFunctionAttr::name.str()}},
     {HACCFuncType::DEVICE, {HostFuncTypeAttr::name.str()}}};

void setFuncType(Operation *op, HACCFuncType funcType) {
  op->setAttr(HACCFuncTypeAttr::name,
              HACCFuncTypeAttr::get(op->getContext(), funcType));
}

void filterAttrsByFuncType(Operation *op, HACCFuncType targetFuncTy) {
  assert(op);
  const auto it = kFuncType2DisallowedAttrs.find(targetFuncTy);
  if (it == kFuncType2DisallowedAttrs.cend())
    return;

  const auto disallowedAttrs = it->second;
  for (auto namedAttr : op->getAttrs()) {
    auto name = namedAttr.getName();
    if (disallowedAttrs.find(name.str()) == disallowedAttrs.cend())
      continue;

    LDBG("Erased disallowed attribute: " << name);
    op->removeAttr(name);
  }
}

} // namespace

namespace mlir {
namespace hacc {

template <typename FuncOpTy>
struct HACCFunctionExternalModel
    : public HACCFunction::ExternalModel<HACCFunctionExternalModel<FuncOpTy>,
                                         FuncOpTy> {
  //===------------------------------------------------------------------===//
  // Query HACC Function attributes.
  //===------------------------------------------------------------------===//
  std::optional<HACCFuncType> getHACCFuncType(Operation *op) const;
  bool isHost(Operation *op) const;
  bool isDevice(Operation *op) const;
  bool isDeviceEntry(Operation *op) const;
  std::optional<HostFuncType> getHostFuncType(Operation *op) const;
  //===------------------------------------------------------------------===//
  // Set HACC Function attributes.
  //===------------------------------------------------------------------===//
  void setDevice(Operation *op) const;
  void setDeviceEntry(Operation *op) const;
  void setHost(Operation *op) const;
  void setHostFuncType(Operation *op, HostFuncType funcType) const;
  //===------------------------------------------------------------------===//
  // Query HACC Function argument attributes.
  //===------------------------------------------------------------------===//
  bool isKernelArg(Operation *op, int argIdx, KernelArgType argType) const;

private:
  FunctionOpInterface getFunc(Operation *op) const;
};

template <typename FuncOpTy>
std::optional<HACCFuncType>
HACCFunctionExternalModel<FuncOpTy>::getHACCFuncType(Operation *op) const {
  assert(op);
  auto maybeHACCFuncType =
      op->getAttrOfType<HACCFuncTypeAttr>(HACCFuncTypeAttr::name);
  if (!maybeHACCFuncType)
    return {};

  return maybeHACCFuncType.getFunctionKind();
}

template <typename FuncOpTy>
bool HACCFunctionExternalModel<FuncOpTy>::isHost(Operation *op) const {
  auto maybeHACCFuncType = getHACCFuncType(op);
  if (!maybeHACCFuncType)
    return {};

  return *maybeHACCFuncType == HACCFuncType::HOST;
}

template <typename FuncOpTy>
bool HACCFunctionExternalModel<FuncOpTy>::isDevice(Operation *op) const {
  auto maybeHACCFuncType = getHACCFuncType(op);
  if (!maybeHACCFuncType)
    return {};

  return *maybeHACCFuncType == HACCFuncType::DEVICE;
}

template <typename FuncOpTy>
bool HACCFunctionExternalModel<FuncOpTy>::isDeviceEntry(Operation *op) const {
  return this->isDevice(op) &&
         op->hasAttr(stringifyEnum(HACCToLLVMIRTranslateAttr::ENTRY));
}

template <typename FuncOpTy>
std::optional<HostFuncType>
HACCFunctionExternalModel<FuncOpTy>::getHostFuncType(Operation *op) const {
  if (!this->isHost(op))
    return {};

  auto maybeHostFuncTypeAttr =
      op->getAttrOfType<hacc::HostFuncTypeAttr>(hacc::HostFuncTypeAttr::name);
  if (!maybeHostFuncTypeAttr)
    return {};

  return maybeHostFuncTypeAttr.getHostFuncType();
}

template <typename FuncOpTy>
void HACCFunctionExternalModel<FuncOpTy>::setDevice(Operation *op) const {
  setFuncType(op, HACCFuncType::DEVICE);
  filterAttrsByFuncType(op, /*targetFuncTy=*/HACCFuncType::DEVICE);
}

template <typename FuncOpTy>
void HACCFunctionExternalModel<FuncOpTy>::setDeviceEntry(Operation *op) const {
  this->setDevice(op);
  op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY),
              UnitAttr::get(op->getContext()));
}

template <typename FuncOpTy>
void HACCFunctionExternalModel<FuncOpTy>::setHost(Operation *op) const {
  setFuncType(op, HACCFuncType::HOST);
  filterAttrsByFuncType(op, /*targetFuncTy=*/HACCFuncType::HOST);
}

template <typename FuncOpTy>
void HACCFunctionExternalModel<FuncOpTy>::setHostFuncType(
    Operation *op, HostFuncType funcType) const {
  assert(this->isHost(op) &&
         "Must be a host function to set the host func type");
  op->setAttr(HostFuncTypeAttr::name,
              HostFuncTypeAttr::get(op->getContext(), funcType));
}

template <typename FuncOpTy>
bool HACCFunctionExternalModel<FuncOpTy>::isKernelArg(
    Operation *op, int argIdx, KernelArgType argType) const {
  auto kernelArgTypeAttr =
      getFunc(op).template getArgAttrOfType<hacc::KernelArgTypeAttr>(
          argIdx, hacc::KernelArgTypeAttr::name);
  return kernelArgTypeAttr && kernelArgTypeAttr.getArgType() == argType;
}

template <typename FuncOpTy>
FunctionOpInterface
HACCFunctionExternalModel<FuncOpTy>::getFunc(Operation *op) const {
  return cast<FunctionOpInterface>(op);
}

namespace detail {

static LogicalResult
verifyHACCFunctionAttributesByFuncType(HACCFunction haccFunc) {
  auto maybeHACCFuncType = haccFunc.getHACCFuncType();
  if (!maybeHACCFuncType)
    return success();

  const auto it = kFuncType2DisallowedAttrs.find(*maybeHACCFuncType);
  if (it == kFuncType2DisallowedAttrs.cend())
    return success();

  const auto disallowedAttrs = it->second;
  for (auto namedAttr : haccFunc->getAttrs()) {
    auto name = namedAttr.getName();
    if (disallowedAttrs.find(name.str()) == disallowedAttrs.cend())
      continue;

    return haccFunc.emitError() << "found disallowed attribute: " << name;
  }
  return success();
}

LogicalResult verifyHACCFunctionOpInterface(Operation *op) {
  HACCFunction haccFunc = cast<HACCFunction>(op);

  // Verify whether the function contains disallowed attributes in terms of HACC
  // function type.
  if (failed(verifyHACCFunctionAttributesByFuncType(haccFunc)))
    return failure();

  return success();
}

DataLayoutEntryInterface getSpecImpl(HACCTargetDeviceSpecInterface specEntries,
                                     DeviceSpec identifier) {
  return specEntries.getSpecForIdentifier(
      StringAttr::get(specEntries.getContext(), stringifyEnum(identifier)));
}

} // namespace detail

} // namespace hacc
} // namespace mlir

void mlir::hacc::func_ext::registerHACCDialectExtension(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    func::FuncOp::attachInterface<HACCFunctionExternalModel<func::FuncOp>>(
        *ctx);
  });
}

void mlir::hacc::llvm_ext::registerHACCDialectExtension(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMFuncOp::attachInterface<
        HACCFunctionExternalModel<LLVM::LLVMFuncOp>>(*ctx);
  });
}