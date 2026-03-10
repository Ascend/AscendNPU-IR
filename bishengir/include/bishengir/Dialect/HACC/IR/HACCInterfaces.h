//===- HACCInterfaces.h -----------------------------------------*- C++ -*-===//
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
// This file contains a set of interfaces for HACC ops.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_H
#define BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/Support/VersionTuple.h"

/// Forward declarations. This is because the interface depends on HACC
/// attributes.
namespace mlir {
namespace hacc {
enum class KernelArgType : uint32_t;
enum class HACCFuncType : uint32_t;
enum class HostFuncType : uint32_t;
enum class DeviceSpec : uint32_t;
class HACCTargetDeviceSpecInterface;

namespace detail {
/// Verify that `op` conforms to the invariants of HACCFunctionInterface.
LogicalResult verifyHACCFunctionOpInterface(Operation *op);

/// Get device spec from entries.
DataLayoutEntryInterface getSpecImpl(HACCTargetDeviceSpecInterface specEntries,
                                     DeviceSpec identifier);
} // namespace detail

} // namespace hacc
} // namespace mlir

/// Include the generated interface declarations.
#include "bishengir/Dialect/HACC/IR/HACCAttrInterfaces.h.inc"
#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h.inc"

#endif // BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_H