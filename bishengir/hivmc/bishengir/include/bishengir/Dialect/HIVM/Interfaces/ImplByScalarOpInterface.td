//===- ImplByScalarOpInterface.td - Ops to decompose -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the ImplByScalarOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ImplByScalarOpInterface : OpInterface<"ImplByScalarOpInterface"> {
  let description = [{
    Interface for operations that needs to be lowered to scalar loops.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to lower the operation to scalar loops.
      }],
      /*retType=*/"mlir::FailureOr<mlir::SmallVector<mlir::Value>>",
      /*methodName=*/"lowerToLoops",
      /*args=*/(ins
          "RewriterBase &":$b),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the operation should be lowered to scalar loops.
      }],
      /*retType=*/"bool",
      /*methodName=*/"shouldLowerToScalarLoops",
      /*args=*/(ins),
      /*methodBody=*/""
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD
