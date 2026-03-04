//===-- HIVM.cpp - C Interface for HIVM dialect -------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/Dialect/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace llvm;
using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HIVM, hivm, mlir::hivm::HIVMDialect)

bool mlirAttrIsAHivmAddressSpaceAttr(MlirAttribute attr) {
  return isa<hivm::AddressSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirHivmAddressSpaceAttrGet(MlirContext ctx,
                                          enum MLIRHIVMAddressSpace addrSpace) {
  auto addrSpaceAttr = hivm::AddressSpaceAttr::get(
      unwrap(ctx), static_cast<hivm::AddressSpace>(addrSpace));
  return wrap(addrSpaceAttr);
}

enum MLIRHIVMAddressSpace
mlirHivmAddressSpaceAttrGetAddrSpace(MlirAttribute attr) {
  return static_cast<MLIRHIVMAddressSpace>(
      cast<hivm::AddressSpaceAttr>(unwrap(attr)).getAddressSpace());
}

bool mlirAttrIsAHivmPipeAttr(MlirAttribute attr) {
  return isa<hivm::PipeAttr>(unwrap(attr));
}

MlirAttribute mlirHivmPipeAttrGet(MlirContext ctx, enum MLIRHIVMPipe pipe) {
  auto pipeAttr =
      hivm::PipeAttr::get(unwrap(ctx), static_cast<hivm::PIPE>(pipe));
  return wrap(pipeAttr);
}

enum MLIRHIVMPipe mlirHivmPipeAttrGetPipe(MlirAttribute attr) {
  return static_cast<MLIRHIVMPipe>(
      cast<hivm::PipeAttr>(unwrap(attr)).getPipe());
}

bool mlirAttrIsAHivmEventAttr(MlirAttribute attr) {
  return isa<hivm::EventAttr>(unwrap(attr));
}

MlirAttribute mlirHivmEventAttrGet(MlirContext ctx, enum MLIRHIVMEvent event) {
  auto eventAttr =
      hivm::EventAttr::get(unwrap(ctx), static_cast<hivm::EVENT>(event));
  return wrap(eventAttr);
}

enum MLIRHIVMEvent mlirHivmEventAttrGetEvent(MlirAttribute attr) {
  return static_cast<MLIRHIVMEvent>(
      cast<hivm::EventAttr>(unwrap(attr)).getEvent());
}
