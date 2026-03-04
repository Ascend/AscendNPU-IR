//===- AggregatedOpInterface.td - Ops to decompose ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the BiShengIR custom AggregatedOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
#define BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def BiShengIRAggregatedOpInterface : OpInterface<"BiShengIRAggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let cppNamespace = "::bishengir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to decompose the operation into simpler operations.
      }],
      /*retType=*/"mlir::FailureOr<llvm::SmallVector<mlir::Value>>",
      /*methodName=*/"decomposeOperation",
      /*args=*/(ins
          "mlir::OpBuilder &":$b),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the phase for decomposing the operation.
      }],
      /*retType=*/"::bishengir::DecomposePhase",
      /*methodName=*/"getDecomposePhase",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::bishengir::DecomposePhase::NO_CONSTRAINT;
      }]
    >
  ];
}

#endif // BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
