//===--------- HIVM.h - C API for HIVM dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_C_DIALECT_HIVM_H
#define BISHENGIR_C_DIALECT_HIVM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HIVM, hivm);

enum MLIRHIVMAddressSpace {
  HIVM_AddressSpace_Default = 0,
  HIVM_AddressSpace_GM = 1,
  HIVM_AddressSpace_L1 = 2,
  HIVM_AddressSpace_L0A = 3,
  HIVM_AddressSpace_L0B = 4,
  HIVM_AddressSpace_L0C = 5,
  HIVM_AddressSpace_UB = 6
};

MLIR_CAPI_EXPORTED bool mlirAttrIsAHivmAddressSpaceAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirHivmAddressSpaceAttrGet(
    MlirContext ctx, enum MLIRHIVMAddressSpace addrSpace);

MLIR_CAPI_EXPORTED enum MLIRHIVMAddressSpace
mlirHivmAddressSpaceAttrGetAddrSpace(MlirAttribute attr);

enum MLIRHIVMPipe {
  HIVM_PIPE_S,
  HIVM_PIPE_V,
  HIVM_PIPE_M,
  HIVM_PIPE_MTE1,
  HIVM_PIPE_MTE2,
  HIVM_PIPE_MTE3,
  HIVM_PIPE_ALL,
  HIVM_PIPE_MTE4,
  HIVM_PIPE_MTE5,
  HIVM_PIPE_V2,
  HIVM_PIPE_FIX,
  HIVM_VIRTUAL_PIPE_MTE2_L1A,
  HIVM_VIRTUAL_PIPE_MTE2_L1B,
  HIVM_PIPE_NUM,
  HIVM_PIPE_UNASSIGNED
};

MLIR_CAPI_EXPORTED bool mlirAttrIsAHivmPipeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirHivmPipeAttrGet(MlirContext ctx,
                                                     enum MLIRHIVMPipe pipe);

MLIR_CAPI_EXPORTED enum MLIRHIVMPipe
mlirHivmPipeAttrGetPipe(MlirAttribute attr);

enum MLIRHIVMEvent {
  HIVM_EventID0,
  HIVM_EventID1,
  HIVM_EventID2,
  HIVM_EventID3,
  HIVM_EventID4,
  HIVM_EventID5,
  HIVM_EventID6,
  HIVM_EventID7,
};

MLIR_CAPI_EXPORTED bool mlirAttrIsAHivmEventAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirHivmEventAttrGet(MlirContext ctx,
                                                      enum MLIRHIVMEvent id);

MLIR_CAPI_EXPORTED enum MLIRHIVMEvent
mlirHivmEventAttrGetEvent(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#include "bishengir/Dialect/HIVM/Transforms/Passes.capi.h.inc"

#endif // BISHENGIR_C_DIALECT_HIVM_H
