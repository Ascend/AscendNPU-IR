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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Bring in attr storage type definitions (needed before addAttributes).
#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUAttrs.cpp.inc"

#include "bishengir/Dialect/BMU/IR/BMUDialect.cpp.inc"
#include "bishengir/Dialect/BMU/IR/BMUEnums.cpp.inc"

using namespace mlir;
using namespace mlir::bmu;

//===----------------------------------------------------------------------===//
// Buffer info table.
//===----------------------------------------------------------------------===//

static const BMUBufferInfo bufferInfoTable[] = {
    /*UB*/ {0b000, 6, "ubuf", "ub"},
    /*L1*/ {0b001, 2, "cbuf", "l1"},
    /*L0A*/ {0b010, 3, "ca", "l0a"},
    /*L0B*/ {0b011, 4, "cb", "l0b"},
    /*L0C*/ {0b100, 5, "cc", "l0c"},
    /*FB*/ {0b101, 7, "fbuf", "fb"},
    /*BT*/ {0b110, 0, "bt", "BT"}, // BT returns i64 scalar, not pointer (proto
                                   // "ULi")
};

const BMUBufferInfo *bmu::getBMUBufferInfo(Buffer buffer) {
  uint32_t idx = static_cast<uint32_t>(buffer);
  if (idx >= std::size(bufferInfoTable))
    return nullptr;
  return &bufferInfoTable[idx];
}

int64_t bmu::getPipeEncoding(Pipe pipe) { return static_cast<int64_t>(pipe); }

//===----------------------------------------------------------------------===//
// Dialect initialization.
//===----------------------------------------------------------------------===//

void BMUDialect::initialize() {
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

static LogicalResult verifyBufferAndPointer(Buffer buffer, Type resultTy) {
  auto *info = getBMUBufferInfo(buffer);
  if (!info)
    return failure();
  // BT returns i64 scalar (proto "ULi"), not a pointer — skip address space
  // check.
  if (buffer == Buffer::BT)
    return success();
  // Check pointer address space if it's an LLVM pointer.
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(resultTy)) {
    if (ptrTy.getAddressSpace() != info->addressSpace)
      return failure();
  }
  return success();
}

LogicalResult BMUAllocRegOp::verify() {
  if (failed(verifyBufferAndPointer(getBuffer(), getAddr().getType())))
    return emitOpError("buffer/pointer address space mismatch");
  return success();
}

LogicalResult BMUAllocImmOp::verify() {
  if (failed(verifyBufferAndPointer(getBuffer(), getAddr().getType())))
    return emitOpError("buffer/pointer address space mismatch");
  return success();
}

LogicalResult BMUFreeRegOp::verify() {
  if (failed(verifyBufferAndPointer(getBuffer(), getAddr().getType())))
    return emitOpError("buffer/pointer address space mismatch");
  return success();
}

LogicalResult BMUFreeImmOp::verify() {
  if (failed(verifyBufferAndPointer(getBuffer(), getAddr().getType())))
    return emitOpError("buffer/pointer address space mismatch");
  return success();
}

LogicalResult BMUSetSegmOp::verify() { return success(); }
LogicalResult BMUGetSegmOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// Memory effects.
//===----------------------------------------------------------------------===//

namespace {
struct BMUStateResource : public SideEffects::Resource::Base<BMUStateResource> {
  StringRef getName() final { return "BMUState"; };
};
} // namespace

void BMUAllocRegOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), BMUStateResource::get());
}

void BMUAllocImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), BMUStateResource::get());
}

void BMUFreeRegOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), BMUStateResource::get());
}

void BMUFreeImmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), BMUStateResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), BMUStateResource::get());
}

void BMUSetSegmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), BMUStateResource::get());
}

void BMUGetSegmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), BMUStateResource::get());
}

//===----------------------------------------------------------------------===//
// ODS-generated definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUOps.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUIntrinOps.cpp.inc"
