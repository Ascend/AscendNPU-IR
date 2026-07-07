//===-- HIVM.cpp - C Interface for HIVM dialect -------------------*- C -*-===//
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

#include "bishengir-c/Dialect/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HIVM, hivm, mlir::hivm::HIVMDialect)
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
