//===- MmadCapabilityQuery.h - MMAD native queries -------------*- C++ -*-===//
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
// Header-only Native checks. Only Native is true. A missing target is false.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYQUERY_H
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYQUERY_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapabilityBinding.h"
#include "bishengir/Target/NPU/NPUArchitectureFeatures.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hivm {

inline std::optional<bishengir::npu::NPUArchitectureKind>
mmadArchitecture(Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::hasTargetAttr(module))
    return std::nullopt;
  auto device = hacc::utils::getTargetDevice(module);
  if (!device)
    return std::nullopt;
  return hacc::getArchitecture(*device);
}

inline bool
isNativeMmadRequest(Operation *anchor,
                    bishengir::npu::mmad::CapabilityRequestStorage storage) {
  auto architecture = mmadArchitecture(anchor);
  if (!architecture)
    return false;
  return bishengir::npu::mmad::queryNativeSupport(*architecture,
                                                  storage.getRequest()) ==
         bishengir::npu::mmad::NativeSupport::Native;
}

inline bool isNativeMmadMx(Operation *anchor, Type aElementType,
                           Type bElementType, Type scaleAElementType,
                           Type scaleBElementType, Type cElementType,
                           bool formatsPresent, uint64_t lhsFormat,
                           uint64_t rhsFormat, int64_t k) {
  // For some targets, it's only native when K is multiple of 64.
  bool kMultipleOf64 = !ShapedType::isDynamic(k) && k > 0 && k % 64 == 0;
  Type f32 = Float32Type::get(anchor->getContext());
  return isNativeMmadRequest(
      anchor,
      bishengir::npu::mmad::MmadMxCapabilityRequestBuilder::create(
          aElementType, AddressSpace::L1, bElementType, AddressSpace::L1,
          scaleAElementType, AddressSpace::L1, scaleBElementType,
          AddressSpace::L1, cElementType, AddressSpace::L0C, f32,
          AddressSpace::L1,
          /*aTranspose=*/false, /*bTranspose=*/false, /*biasPresent=*/false,
          llvm::APInt(64, lhsFormat), llvm::APInt(64, rhsFormat),
          formatsPresent, formatsPresent, kMultipleOf64)
          .build());
}

// Layout insertion asks whether this target has mmadmx at all. K alignment is
// checked by the caller that decides software emulation.
inline bool isNativeMmadMxTarget(Operation *anchor) {
  Type f8 = Float8E5M2Type::get(anchor->getContext());
  Type i8 = IntegerType::get(anchor->getContext(), 8);
  Type f32 = Float32Type::get(anchor->getContext());
  return isNativeMmadRequest(
      anchor,
      bishengir::npu::mmad::MmadMxCapabilityRequestBuilder::create(
          f8, AddressSpace::L1, f8, AddressSpace::L1, i8, AddressSpace::L1, i8,
          AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1,
          /*aTranspose=*/false, /*bTranspose=*/false, /*biasPresent=*/false,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          /*lhsFormatPresent=*/false, /*rhsFormatPresent=*/false,
          /*kMultipleOf64=*/true)
          .build());
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYQUERY_H
