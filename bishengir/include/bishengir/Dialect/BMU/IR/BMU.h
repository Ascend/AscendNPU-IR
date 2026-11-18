//===- BMU.h - BMU dialect -------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMU_H
#define BISHENGIR_DIALECT_BMU_IR_BMU_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

// ODS-generated declarations.
#include "bishengir/Dialect/BMU/IR/BMUDialect.h.inc"
#include "bishengir/Dialect/BMU/IR/BMUEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUTypes.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUIntrinOps.h.inc"

namespace mlir::bmu {

/// Buffer metadata: maps buffer enum to hardware properties.
struct BMUBufferInfo {
  uint32_t bufField;     // 3-bit buffer field encoding
  uint32_t addressSpace; // LLVM pointer address space (== HIVM address space)
  StringRef allocPrefix; // intrinsic name prefix for alloc
  StringRef sprSuffix;   // SPR name suffix
};

/// Get buffer info for a given buffer enum value. Returns nullptr if
/// the buffer is not supported.
const BMUBufferInfo *getBMUBufferInfo(Buffer buffer);

/// HIVM address space of the memref handle allocated for `buffer`. Aborts for
/// BT, whose allocations are scalar i64 values with no address space.
hivm::AddressSpace getBufferAddressSpace(Buffer buffer);

/// Number of slices of the BMU ring of `buffer`, i.e. the ring geometry in
/// slices. The count belongs to the BMU hardware unit rather than to the
/// requested target, so it stays a dialect constant; buffers without a BMU ring
/// (L0A, L0B, FB, BT) report 0.
uint64_t getBufferNumSlices(Buffer buffer);

/// Byte size of one slice of `buffer`, derived from the device spec of `module`
/// as `buffer size / getBufferNumSlices(buffer)`: UB = (`UbSize` + the 8 KiB
/// the spec reserves for the compiler) / 48, L1 = `L1Size` / 64, L0C =
/// `L0cSize` / 64. Deriving the value instead of freezing it keeps the dialect
/// in step with the target spec.
///
/// Returns 0 when the module carries no device spec or the spec has no size for
/// the buffer, in which case capacity checks that need the slice size are
/// skipped.
uint64_t getBufferSliceBytes(ModuleOp module, Buffer buffer);

/// True when allocations of `buffer` are addressed by a scalar i64 value
/// instead of a memref in an address space (BT only, proto "ULi").
bool isScalarHandleBuffer(Buffer buffer);

/// Human-readable description of the handle form required by `buffer`, used in
/// verifier diagnostics.
std::string describeBufferHandle(Buffer buffer);

/// Number of segments of a BMU buffer, i.e. the number of tail fields of the
/// segment register, and the width of one field in bits.
constexpr size_t kNumBufferSegments = 4;
constexpr int64_t kSegmentFieldBits = 8;

/// Decode the cumulative segment ends of a segment register value. The ends are
/// the four 8-bit fields, so the value is valid when they are non-decreasing
/// and each fits in a field (design 3.3.12, section 5.3.1). `value` is the
/// register content, not a slice count.
std::optional<std::array<int64_t, kNumBufferSegments>>
decodeSegmentEnds(int64_t value);

/// Same as `decodeSegmentEnds`, plus the requirement that the last end equals
/// `capSlices`, i.e. that the segments tile the whole ring.
std::optional<std::array<int64_t, kNumBufferSegments>>
decodeSegmentTails(int64_t value, int64_t capSlices);

/// Pack cumulative segment ends into a segment register value.
int64_t packSegmentTails(llvm::ArrayRef<int64_t> tails);

/// Capacity of segment `seg` in slices: `tails[seg] - tails[seg - 1]`, with the
/// end before the first segment taken as 0 (design 3.3.12).
int64_t getSegmentCapacity(llvm::ArrayRef<int64_t> tails, int64_t seg);

/// First slice of segment `seg`; its byte base is this value times the slice
/// size of the buffer (design 3.3.12, the reserved window base).
int64_t getSegmentBaseSlices(llvm::ArrayRef<int64_t> tails, int64_t seg);

/// Get the pipe encoding as i64 integer.
int64_t getPipeEncoding(Pipe pipe);

/// True when the target of `module` supports BMU, i.e. when it belongs to the
/// A6 generation (`dav-920r1`). The target comes from `hacc.target`, which
/// bishengir-compile fills from its `--target` option, so this is what ties the
/// feature to the requested generation.
///
/// Returns true when the module carries no target at all: BMU is then only
/// bound by the A6 LLVM feature macro, since a bare `bishengir-opt` run has no
/// `--target` to consult.
bool isBMUTargetSupported(ModuleOp module);

} // namespace mlir::bmu

#endif // BISHENGIR_DIALECT_BMU_IR_BMU_H
