//===- ExtraBufferOpInterface.td - Ops with Temp. Buffer ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the ExtraBufferOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ExtraBufferOpInterface : OpInterface<"ExtraBufferOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Allocate extra temp buffers for the current operation.
      }],
      /*retTy=*/"llvm::LogicalResult",
      /*methodName=*/"allocExtraBuffersIfPossible",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the extra temp buffers.
      }],
      /*retTy=*/"mlir::OperandRange",
      /*methodName=*/"getExtraBuffers",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc*/[{
        Get the size of the extra buffer. Returns std::nullopt if the op doesn't require extra buffer.
      }],
      /*retTy*/"std::optional<int64_t>",
      /*methodName*/"getExtraBufferSize",
      /*args*/(ins),
      /*methodBody*/"",
      /*defaultImplementation*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Should allocate extra buffer for scalar or OTF Brc.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"shouldAllocExtraBufferForScalarOrOTFBrc",
      /*args=*/(ins)
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
