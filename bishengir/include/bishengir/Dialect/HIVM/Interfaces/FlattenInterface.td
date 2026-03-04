//===- FlattenInterface.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the FlattenInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD

include "mlir/IR/OpBase.td"

def FlattenInterface : OpInterface<"FlattenInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the HIVM Layout reassociation after flattening.
        Return identity reassociation if not flattenable.
      }],
      /*retTy=*/"llvm::FailureOr<::mlir::hivm::FlattenResult>",
      /*methodName=*/"getFlattened",
      /*args=*/(ins "::mlir::hivm::FlattenOptions":$flattenOptions),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getFlattenedImpl($_op, flattenOptions);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        This returns the axes that cannot be collapsed with its
        adjacent dimension, even though it's a unit dimension.
      }],
      /*retTy=*/"llvm::SmallVector<int64_t>",
      /*methodName=*/"getLimitedAxes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("get limited axes is not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        This function adjust target dimensions which semantics changed after collapsing.
      }],
      /*retTy=*/"void",
      /*methodName=*/"adjustTargetDimensions",
      /*args=*/(ins "::mlir::OpBuilder&":$builder, "const FlattenResult&":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return;
      }]
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD
