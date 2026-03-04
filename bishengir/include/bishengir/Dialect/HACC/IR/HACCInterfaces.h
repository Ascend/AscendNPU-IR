//===- HACCInterfaces.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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