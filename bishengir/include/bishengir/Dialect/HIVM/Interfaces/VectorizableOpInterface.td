//===- VectorizableOpInterface.td - Ops to vectorize -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the VectorizableOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def VectorizableOpInterface : OpInterface<"VectorizableOpInterface"> {
  let description = [{
    Interface for operations that needs to be vectorized.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to vectorize the operation.
      }],
      /*retType=*/"::llvm::LogicalResult",
      /*methodName=*/"vectorize",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter, "::llvm::ArrayRef<int64_t>":$vectorSizes),
      /*methodBody=*/""
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD
