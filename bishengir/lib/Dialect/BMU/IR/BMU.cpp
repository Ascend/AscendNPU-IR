//===- BMU.cpp - BMU dialect implementation -----------------------*- C++
//-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/BMU/IR/BMU.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

// Bring in attr storage type definitions (needed before addAttributes).
#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUTypes.cpp.inc"

#include "bishengir/Dialect/BMU/IR/BMUDialect.cpp.inc"
#include "bishengir/Dialect/BMU/IR/BMUEnums.cpp.inc"

using namespace mlir;
using namespace mlir::bmu;

namespace mlir {
namespace bmu {
namespace detail {
struct BMUStateResource;
} // namespace detail
} // namespace bmu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bmu::detail::BMUStateResource)

//===----------------------------------------------------------------------===//
// Buffer info table.
//===----------------------------------------------------------------------===//

/// LLVM pointer address space of a buffer's HIVM address space. BMU reuses the
/// HIVM address space numbering, so the memref handle
/// `memref<..., #hivm.address_space<X>>` and the `!llvm.ptr<N>` it lowers to
/// share the same number N.
static constexpr uint32_t toLLVMAddressSpace(hivm::AddressSpace space) {
  return static_cast<uint32_t>(space);
}

static const BMUBufferInfo bufferInfoTable[] = {
    /*UB*/ {0b000, toLLVMAddressSpace(hivm::AddressSpace::UB), "ubuf", "ub"},
    /*L1*/ {0b001, toLLVMAddressSpace(hivm::AddressSpace::L1), "cbuf", "l1"},
    /*L0A*/ {0b010, toLLVMAddressSpace(hivm::AddressSpace::L0A), "ca", "l0a"},
    /*L0B*/ {0b011, toLLVMAddressSpace(hivm::AddressSpace::L0B), "cb", "l0b"},
    /*L0C*/ {0b100, toLLVMAddressSpace(hivm::AddressSpace::L0C), "cc", "l0c"},
    /*FB*/
    {0b101, toLLVMAddressSpace(hivm::AddressSpace::FixBUF), "fbuf", "fb"},
    /*BT*/ {0b110, 0, "bt", "BT"}, // BT returns an i64 scalar address and has
                                   // no address space (proto "ULi").
};

const BMUBufferInfo *bmu::getBMUBufferInfo(Buffer buffer) {
  uint32_t idx = static_cast<uint32_t>(buffer);
  if (idx >= std::size(bufferInfoTable))
    return nullptr;
  return &bufferInfoTable[idx];
}

hivm::AddressSpace bmu::getBufferAddressSpace(Buffer buffer) {
  switch (buffer) {
  case Buffer::UB:
    return hivm::AddressSpace::UB;
  case Buffer::L1:
    return hivm::AddressSpace::L1;
  case Buffer::L0A:
    return hivm::AddressSpace::L0A;
  case Buffer::L0B:
    return hivm::AddressSpace::L0B;
  case Buffer::L0C:
    return hivm::AddressSpace::L0C;
  case Buffer::FB:
    return hivm::AddressSpace::FixBUF;
  case Buffer::BT:
    break;
  }
  llvm_unreachable("BT has no address space");
}

bool bmu::isScalarHandleBuffer(Buffer buffer) { return buffer == Buffer::BT; }

std::optional<std::array<int64_t, bmu::kNumBufferSegments>>
bmu::decodeSegmentEnds(int64_t value) {
  if (value < 0)
    return std::nullopt;
  std::array<int64_t, kNumBufferSegments> ends = {};
  for (size_t index = 0; index < kNumBufferSegments; ++index) {
    int64_t end =
        (value >> (kSegmentFieldBits * static_cast<int64_t>(index))) & 0xFF;
    if (index > 0 && end < ends[index - 1])
      return std::nullopt;
    ends[index] = end;
  }
  // Reject values with bits above the four fields: they are not a register
  // content produced by packSegmentTails.
  if ((value >>
       (kSegmentFieldBits * static_cast<int64_t>(kNumBufferSegments))) != 0)
    return std::nullopt;
  return ends;
}

std::optional<std::array<int64_t, bmu::kNumBufferSegments>>
bmu::decodeSegmentTails(int64_t value, int64_t capSlices) {
  auto ends = decodeSegmentEnds(value);
  if (!ends.has_value())
    return std::nullopt;
  if ((*ends)[kNumBufferSegments - 1] != capSlices)
    return std::nullopt;
  return ends;
}

int64_t bmu::packSegmentTails(llvm::ArrayRef<int64_t> tails) {
  int64_t value = 0;
  for (auto [index, tail] : llvm::enumerate(tails))
    value |= (tail & 0xFF) << (kSegmentFieldBits * static_cast<int64_t>(index));
  return value;
}

int64_t bmu::getSegmentCapacity(llvm::ArrayRef<int64_t> tails, int64_t seg) {
  assert(seg >= 0 && static_cast<size_t>(seg) < tails.size());
  int64_t begin = seg == 0 ? 0 : tails[static_cast<size_t>(seg) - 1];
  return tails[static_cast<size_t>(seg)] - begin;
}

int64_t bmu::getSegmentBaseSlices(llvm::ArrayRef<int64_t> tails, int64_t seg) {
  assert(seg >= 0 && static_cast<size_t>(seg) < tails.size());
  return seg == 0 ? 0 : tails[static_cast<size_t>(seg) - 1];
}

/// Ring geometry of the BMU hardware unit: the number of slices each buffer's
/// ring hands out. Only the count is a dialect fact; the byte size of a slice
/// comes from the target device spec, so a corrected spec moves the arithmetic.
uint64_t bmu::getBufferNumSlices(Buffer buffer) {
  switch (buffer) {
  case Buffer::UB:
    return 48;
  case Buffer::L1:
  case Buffer::L0C:
    return 64;
  case Buffer::L0A:
  case Buffer::L0B:
  case Buffer::FB:
  case Buffer::BT:
    break;
  }
  return 0;
}

namespace {
constexpr uint64_t kBitsPerByte = 8;

/// The A6 spec calls out 8 KiB reserved for the compiler on top of `UbSize`,
/// and the BMU ring spans the whole buffer.
constexpr uint64_t kCompilerReservedUbBytes = 8 * 1024;

/// Device spec entry that carries the size of `buffer`, if the buffer has one.
std::optional<hacc::DeviceSpec> getBufferSizeEntry(Buffer buffer) {
  switch (buffer) {
  case Buffer::UB:
    return hacc::DeviceSpec::UB_SIZE;
  case Buffer::L1:
    return hacc::DeviceSpec::L1_SIZE;
  case Buffer::L0C:
    return hacc::DeviceSpec::L0C_SIZE;
  case Buffer::L0A:
  case Buffer::L0B:
  case Buffer::FB:
  case Buffer::BT:
    break;
  }
  return std::nullopt;
}
} // namespace

uint64_t bmu::getBufferSliceBytes(ModuleOp module, Buffer buffer) {
  uint64_t numSlices = getBufferNumSlices(buffer);
  std::optional<hacc::DeviceSpec> entry = getBufferSizeEntry(buffer);
  if (!module || numSlices == 0 || !entry)
    return 0;

  std::optional<hacc::HACCTargetDeviceSpecInterface> spec =
      hacc::utils::getNPUTargetSpec(module);
  if (!spec)
    return 0;

  DataLayoutEntryInterface sizeEntry = spec->getSpecForIdentifierEnum(*entry);
  if (!sizeEntry)
    return 0;
  auto sizeBits = dyn_cast<IntegerAttr>(sizeEntry.getValue());
  if (!sizeBits || sizeBits.getInt() <= 0)
    return 0;

  uint64_t sizeBytes = static_cast<uint64_t>(sizeBits.getInt()) / kBitsPerByte;
  if (buffer == Buffer::UB)
    sizeBytes += kCompilerReservedUbBytes;
  return sizeBytes / numSlices;
}

std::string bmu::describeBufferHandle(Buffer buffer) {
  if (isScalarHandleBuffer(buffer))
    return "an i64 scalar address (BT has no address space)";
  return "a ranked memref with a static shape and an identity layout in "
         "#hivm.address_space<" +
         hivm::stringifyAddressSpace(getBufferAddressSpace(buffer)).str() + ">";
}

int64_t bmu::getPipeEncoding(Pipe pipe) { return static_cast<int64_t>(pipe); }

bool bmu::isBMUTargetSupported(ModuleOp module) {
  if (!module)
    return true;

  // No hacc.target yet: a bare bishengir-opt run has no --target option, so the
  // only gate left is the A6 LLVM feature macro that compiles BMU in at all.
  std::optional<hacc::TargetDevice> target =
      hacc::utils::getTargetDevice(module);
  if (!target.has_value())
    return true;

  return hacc::utils::isAscend960(target.value());
}

//===----------------------------------------------------------------------===//
// Dialect initialization.
//===----------------------------------------------------------------------===//

void BMUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/BMU/IR/BMUTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/BMU/IR/BMUAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/BMU/IR/BMUOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/BMU/IR/BMUIntrinOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Verifiers.
//===----------------------------------------------------------------------===//

/// Check that `handleTy` is a valid allocation handle for `buffer`.
///
/// Every buffer is addressed through a ranked memref in the buffer's HIVM
/// address space, which is what makes a BMU allocation usable by memref/HIVM
/// operations. The memref must have a static shape and an identity layout: the
/// allocation is described by a slice count, not by a shape, so there are no
/// dynamic sizes to attach to the operation.
///
/// BT is the exception: its allocations are scalar i64 addresses (proto "ULi")
/// and carry no address space.
static LogicalResult verifyBufferHandle(Buffer buffer, Type handleTy) {
  auto *info = getBMUBufferInfo(buffer);
  if (!info)
    return failure();

  if (isScalarHandleBuffer(buffer))
    return success(handleTy.isInteger(64));

  auto memRefTy = dyn_cast<MemRefType>(handleTy);
  if (!memRefTy || !memRefTy.hasStaticShape() ||
      !memRefTy.getLayout().isIdentity())
    return failure();

  std::optional<hivm::AddressSpace> space =
      hivm::getOptionalHIVMAddressSpace(handleTy);
  return success(space && *space == getBufferAddressSpace(buffer));
}

/// Byte size of one element of `type`, or std::nullopt when the type is not a
/// fixed-size type whose in-memory size is unambiguous. Sub-byte integers are
/// packed in some address spaces and complex types are laid out by the target,
/// so they are not guessed: an unknown element type skips the capacity check.
static std::optional<uint64_t> getElementByteSize(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() < 8 || intTy.getWidth() % 8 != 0)
      return std::nullopt;
    return intTy.getWidth() / 8;
  }
  if (auto floatTy = dyn_cast<FloatType>(type))
    return (floatTy.getWidth() + 7) / 8;
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    if (vecTy.isScalable())
      return std::nullopt;
    std::optional<uint64_t> elemBytes =
        getElementByteSize(vecTy.getElementType());
    if (!elemBytes)
      return std::nullopt;
    return *elemBytes * static_cast<uint64_t>(vecTy.getNumElements());
  }
  if (isa<IndexType>(type))
    return 8;
  return std::nullopt;
}

/// Number of elements viewed by the identity-layout, statically shaped memref
/// `memRefTy`, or std::nullopt when it cannot be computed without overflow.
static std::optional<uint64_t> getStaticElementCount(MemRefType memRefTy) {
  uint64_t count = 1;
  for (int64_t dim : memRefTy.getShape()) {
    if (dim < 0)
      return std::nullopt;
    uint64_t dimSize = static_cast<uint64_t>(dim);
    if (dimSize != 0 && count > std::numeric_limits<uint64_t>::max() / dimSize)
      return std::nullopt;
    count *= dimSize;
  }
  return count;
}

/// Check that the reservation holds the memref view.
///
/// An allocation reserves `size` slices, i.e. `size * getBufferSliceBytes()`
/// bytes, while `handleTy` views `prod(shape) * sizeof(element)` bytes. The
/// view may be smaller than the reservation -- a caller may reuse one
/// allocation with several views, ask for a round number of slices, or leave
/// room for padding -- but it must never be larger, because the extra bytes
/// would alias neighbouring allocations and the hardware does not diagnose
/// that.
///
/// The slice count is a compile-time constant only for the IMM variants, and
/// the slice size and element size are not always known, so the check is
/// skipped whenever any of them cannot be determined.
static LogicalResult verifyBufferCapacity(Operation *op, ModuleOp module,
                                          Buffer buffer, Type handleTy,
                                          uint64_t size) {
  uint64_t sliceBytes = getBufferSliceBytes(module, buffer);
  auto memRefTy = dyn_cast<MemRefType>(handleTy);
  if (sliceBytes == 0 || !memRefTy)
    return success();

  std::optional<uint64_t> elemBytes =
      getElementByteSize(memRefTy.getElementType());
  std::optional<uint64_t> numElems = getStaticElementCount(memRefTy);
  if (!elemBytes || !numElems)
    return success();

  if (size > std::numeric_limits<uint64_t>::max() / sliceBytes)
    return success();
  if (*numElems != 0 &&
      *elemBytes > std::numeric_limits<uint64_t>::max() / *numElems)
    return success();

  uint64_t capacity = size * sliceBytes;
  uint64_t viewBytes = *elemBytes * *numElems;
  if (capacity >= viewBytes)
    return success();

  return op->emitOpError() << "buffer " << stringifyBuffer(buffer)
                           << " reserves " << capacity
                           << " bytes, but the handle type views " << viewBytes
                           << " bytes";
}

/// Emit the standard diagnostic for an invalid allocation handle.
static InFlightDiagnostic emitBufferHandleError(Operation *op, Buffer buffer,
                                                Type handleTy) {
  return op->emitOpError() << "buffer " << stringifyBuffer(buffer)
                           << " requires " << describeBufferHandle(buffer)
                           << ", but got " << handleTy;
}

LogicalResult BMUAllocRegOp::verify() {
  if (failed(verifyBufferHandle(getBuffer(), getAddr().getType())))
    return emitBufferHandleError(*this, getBuffer(), getAddr().getType());
  return success();
}

LogicalResult BMUAllocImmOp::verify() {
  if (failed(verifyBufferHandle(getBuffer(), getAddr().getType())))
    return emitBufferHandleError(*this, getBuffer(), getAddr().getType());
  return verifyBufferCapacity(*this,
                              getOperation()->getParentOfType<ModuleOp>(),
                              getBuffer(), getAddr().getType(), getSize());
}

LogicalResult BMUFreeRegOp::verify() {
  if (failed(verifyBufferHandle(getBuffer(), getAddr().getType())))
    return emitBufferHandleError(*this, getBuffer(), getAddr().getType());
  return success();
}

LogicalResult BMUFreeImmOp::verify() {
  if (failed(verifyBufferHandle(getBuffer(), getAddr().getType())))
    return emitBufferHandleError(*this, getBuffer(), getAddr().getType());
  return verifyBufferCapacity(*this,
                              getOperation()->getParentOfType<ModuleOp>(),
                              getBuffer(), getAddr().getType(), getSize());
}

/// design: bmu#s.3.3.6
///
/// A region allocation reserves slices of `buffer`, so the buffer must have a
/// BMU ring; BT has none (it is addressed by an i64 scalar allocation, not by
/// the ring), and neither do the buffers the dialect has no ring geometry for.
static LogicalResult verifyRegionBuffer(Operation *op, Buffer buffer) {
  if (!getBMUBufferInfo(buffer))
    return op->emitOpError() << "unknown BMU buffer";
  if (getBufferNumSlices(buffer) == 0)
    return op->emitOpError()
           << "buffer " << stringifyBuffer(buffer)
           << " has no slice size and cannot be allocated as a region";
  return success();
}

LogicalResult BMURegionAllocOp::verify() {
  return verifyRegionBuffer(*this, getBuffer());
}

/// The token of `region_addr` / `region_free` must come from one region
/// allocation, so a member can never be addressed through a foreign region.
static LogicalResult verifyRegionToken(Operation *op, Value region) {
  if (region.getDefiningOp<BMURegionAllocOp>())
    return success();
  return op->emitOpError()
         << "expects a value defined by bmu.region_alloc, got " << region;
}

LogicalResult BMURegionAddrOp::verify() {
  return verifyRegionToken(*this, getRegion());
}

LogicalResult BMURegionFreeOp::verify() {
  return verifyRegionToken(*this, getRegion());
}

LogicalResult BMUAllocRegionImmOp::verify() {
  return verifyRegionBuffer(*this, getBuffer());
}

LogicalResult BMUFreeRegionImmOp::verify() {
  return verifyRegionBuffer(*this, getBuffer());
}

LogicalResult BMUSetSegmOp::verify() { return success(); }
LogicalResult BMUGetSegmOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// Memory effects.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bmu {
namespace detail {
/// The BMU segment/allocation state is a side-effect resource. It lives in a
/// named namespace and carries an explicit TypeID (declared above): the
/// implicit type-name fallback is not valid for anonymous-namespace types and
/// asserts in assertions-enabled builds.
struct BMUStateResource : public SideEffects::Resource::Base<BMUStateResource> {
  StringRef getName() final { return "BMUState"; };
};
} // namespace detail
} // namespace bmu
} // namespace mlir

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::bmu::detail::BMUStateResource)

void BMUAllocRegOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The allocation is carved out of (or returned to) the buffer's segment
  // state, which is tracked as a dedicated side-effect resource.
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  // The handle owns the allocation; no memory is read or written.
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0));
}

void BMUAllocImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The allocation is carved out of (or returned to) the buffer's segment
  // state, which is tracked as a dedicated side-effect resource.
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  // The handle owns the allocation; no memory is read or written.
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0));
}

void BMUFreeRegOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The allocation is carved out of (or returned to) the buffer's segment
  // state, which is tracked as a dedicated side-effect resource.
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  // The handle owns the allocation; no memory is read or written.
  effects.emplace_back(MemoryEffects::Free::get(), &getAddrMutable());
}

void BMUFreeImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The allocation is carved out of (or returned to) the buffer's segment
  // state, which is tracked as a dedicated side-effect resource.
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  // The handle owns the allocation; no memory is read or written.
  effects.emplace_back(MemoryEffects::Free::get(), &getAddrMutable());
}

void BMUAllocRegionImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The region is carved out of the buffer's segment state, and its base
  // address owns the reservation.
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0));
}

void BMURegionAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
}

void BMURegionAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
}

void BMURegionFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
}

void BMUFreeRegionImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Free::get(), &getBaseMutable());
}

void BMUSetSegmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::bmu::detail::BMUStateResource::get());
}

void BMUGetSegmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::bmu::detail::BMUStateResource::get());
}

//===----------------------------------------------------------------------===//
// ODS-generated definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUOps.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUIntrinOps.cpp.inc"
