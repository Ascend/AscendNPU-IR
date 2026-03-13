//===- OpLayoutInterface.td - Ops with Layout Information --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the OpLayoutInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD

include "mlir/IR/OpBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

def OpLayoutInterface : OpInterface<"OpWithLayoutInterface"> {
  let description = [{
     Interface to query characteristics of HIVM Ops that have specific
     data layout requirements.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the target data layouts for memref-typed operands.",
    /*retTy=*/"llvm::SmallDenseMap<mlir::Value, mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsTargetFractalLayout",
    /*args=*/(ins), /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      llvm_unreachable("getOperandsTargetFractalLayout is not implemented.");
    }]
    >,
    InterfaceMethod<
    /*desc=*/"Returns the target data layouts for memref-typed operands.",
    /*retTy=*/"llvm::SmallDenseMap<mlir::Value, mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsTargetLayout",
    /*args=*/(ins)
    >,
    InterfaceMethod<
    /*desc=*/"Returns the current data layouts for memref-typed operands.",
    /*retTy=*/"llvm::SmallDenseMap<mlir::Value, mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsCurrentLayout",
    /*args=*/(ins)
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
