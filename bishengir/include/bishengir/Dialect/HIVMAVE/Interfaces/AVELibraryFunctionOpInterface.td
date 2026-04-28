//===- AVELibraryFunctionOpInterface.td -----------------. -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the AVELibraryFunctionOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVMAVE_INTERFACES_AVELIBRARYFUNCTIONOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVMAVE_INTERFACES_AVELIBRARYFUNCTIONOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def AVELibraryFunctionOpInterface : OpInterface<"AVEOpWithLibraryFunction"> {
  let cppNamespace = "::mlir::hivmave";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the name registered for this op when lowering to an external library call.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getOpLibraryCallName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getOpLibraryCallName not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the inputs of the external library call.
      }],
      /*retTy=*/"llvm::SmallVector<mlir::Value>",
      /*methodName=*/"getOpLibraryCallInputs",
      /*args=*/(ins "mlir::OpBuilder&":$b),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: return all operands
        Operation* baseOp = $_op.getOperation();
        return llvm::to_vector(baseOp->getOperands());
      }]
    >
  ];
}


def AVEElementwiseOpInterface : OpInterface<"AVEElementwiseOp"> {
  let cppNamespace = "::mlir::hivmave";
}

def AVEUnaryOpInterface : OpInterface<"AVEUnaryOp", [AVEElementwiseOpInterface]> {
  let cppNamespace = "::mlir::hivmave";
}

def AVEBinaryOpInterface : OpInterface<"AVEBinaryOp", [AVEElementwiseOpInterface]> {
  let cppNamespace = "::mlir::hivmave";
}

def AVETenaryOpInterface : OpInterface<"AVETenaryOp", [AVEElementwiseOpInterface]> {
  let cppNamespace = "::mlir::hivmave";
}

#endif // BISHENGIR_DIALECT_HIVMAVE_INTERFACES_AVELIBRARYFUNCTIONOPINTERFACE_TD
