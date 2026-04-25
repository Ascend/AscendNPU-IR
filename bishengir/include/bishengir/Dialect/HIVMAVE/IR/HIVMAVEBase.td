//===- HIVMAVEBase.td - AVE dialect base support -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base Ascend Vector Extension (AVE) support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_AVE_IR_AVEBASE_TD
#define BISHENGIR_DIALECT_AVE_IR_AVEBASE_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "bishengir/Dialect/HIVMAVE/Interfaces/AVELibraryFunctionOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// AVE Dialect.
//===----------------------------------------------------------------------===//

def AVE_Dialect : Dialect {
  let name = "ave";
  let description = [{
    AVE (Ascend Vector Extension) dialect.
  }];
  let cppNamespace = "::mlir::hivmave";
  let dependentDialects = [
    "arith::ArithDialect", "math::MathDialect", "memref::MemRefDialect",
    "vector::VectorDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AVE Dialect operations.
//===----------------------------------------------------------------------===//

class AVE_Op<string mnemonic, list<Trait> traits = []>
    : Op<AVE_Dialect, !strconcat("hir.", mnemonic), traits> {
  string intrinsicName = mnemonic;
  // C++ function that returns the op name.
  code opNameDeclaration = [{
    static StringRef getOpName() { return "}] # mnemonic # [{"; }
    static StringRef getIntrinsicName() { return "}] # intrinsicName # [{"; }
  }];
}

#endif // BISHENGIR_DIALECT_AVE_IR_AVEBASE_TD
