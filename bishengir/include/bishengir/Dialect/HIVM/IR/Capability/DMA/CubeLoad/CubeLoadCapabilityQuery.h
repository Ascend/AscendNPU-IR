//===- CubeLoadCapabilityQuery.h - CubeLoad native queries ------*- C++ -*-===//
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
//
// Header-only Native checks for cube DMA creation. Only Native is true.
// A missing #hacc.target is false.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYQUERY_H
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYQUERY_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadCapabilityBinding.h"
#include "bishengir/Target/NPU/NPUArchitectureFeatures.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hivm {

inline std::optional<bishengir::npu::NPUArchitectureKind>
cubeLoadArchitecture(Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  auto device = hacc::utils::getTargetDevice(module);
  if (!device)
    return std::nullopt;
  return hacc::getArchitecture(*device);
}

inline AddressSpace cubeLoadSpaceOr(Type type, AddressSpace fallback) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return fallback;
  auto attr = dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!attr)
    return fallback;
  return attr.getAddressSpace();
}

inline bool isNativeCubeLoadRequest(
    Operation *anchor,
    bishengir::npu::cubeload::CapabilityRequestStorage storage) {
  auto architecture = cubeLoadArchitecture(anchor);
  if (!architecture)
    return false;
  return bishengir::npu::cubeload::queryNativeSupport(*architecture,
                                                      storage.getRequest()) ==
         bishengir::npu::cubeload::NativeSupport::Native;
}

inline bool isNativeNd2NzBatch(Operation *anchor, Type elementType,
                               AddressSpace srcSpace, AddressSpace dstSpace) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor,
      bishengir::npu::cubeload::Nd2NzBatchCapabilityRequestBuilder::create(
          elementType, srcSpace, elementType, dstSpace)
          .build());
}

inline bool isNativeNz2Nd(Operation *anchor, Type elementType,
                          AddressSpace srcSpace, AddressSpace dstSpace) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor, bishengir::npu::cubeload::Nz2NdCapabilityRequestBuilder::create(
                  elementType, srcSpace, elementType, dstSpace)
                  .build());
}

inline bool isNativeL12UB(Operation *anchor, Type elementType,
                          AddressSpace srcSpace, AddressSpace dstSpace) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor, bishengir::npu::cubeload::L12UBCapabilityRequestBuilder::create(
                  elementType, srcSpace, elementType, dstSpace)
                  .build());
}

inline bool isNativeNchw2Nc1hwc0(Operation *anchor, Type elementType,
                                 AddressSpace srcSpace, AddressSpace dstSpace) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor,
      bishengir::npu::cubeload::Nchw2Nc1hwc0CapabilityRequestBuilder::create(
          elementType, srcSpace, elementType, dstSpace)
          .build());
}

inline bool isNativeNchw2C1hwnc0(Operation *anchor, Type elementType,
                                 AddressSpace srcSpace, AddressSpace dstSpace) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor,
      bishengir::npu::cubeload::Nchw2C1hwnc0CapabilityRequestBuilder::create(
          elementType, srcSpace, elementType, dstSpace)
          .build());
}

inline bool isNativeLoadScale(Operation *anchor, Type elementType,
                              AddressSpace srcSpace, AddressSpace dstSpace,
                              bool isTransposed) {
  if (!elementType)
    return false;
  return isNativeCubeLoadRequest(
      anchor,
      bishengir::npu::cubeload::LoadScaleCapabilityRequestBuilder::create(
          elementType, srcSpace, elementType, dstSpace, isTransposed)
          .build());
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYQUERY_H
