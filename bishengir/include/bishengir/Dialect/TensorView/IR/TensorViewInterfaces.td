//===-- TensorViewInterfaces.td - TensorView op interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_INTERFACES
#define TENSORVIEW_INTERFACES

include "mlir/IR/OpBase.td"

def TensorViewEncodingOpInterface
    : OpInterface<"TensorViewEncodingOpInterface"> {
  let description = [{
    Common interface for operations that attach an encoding to a base
    `tv.tensor_view`.
  }];
  let cppNamespace = "::mlir::tv";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the base tensor view.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSource">
  ];
}

#endif // TENSORVIEW_INTERFACES
