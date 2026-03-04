//===- HIVMBase.td - HIVM dialect base support -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base Hybrid Intelligence Virtual
// Machine (HIVM) support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD

include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect.
//===----------------------------------------------------------------------===//

def HIVM_Dialect : Dialect {
  let name = "hivm";
  let description = [{
    HIVM (Hybrid Intelligence Virtual Machine) dialect.
  }];
  let cppNamespace = "::mlir::hivm";
  let dependentDialects = [
    "arith::ArithDialect", "bishengir::memref_ext::MemRefExtDialect",
    "math::MathDialect", "memref::MemRefDialect", "hacc::HACCDialect",
    "tensor::TensorDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// HIVM Types.
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// HIVM Dialect operations.
//===----------------------------------------------------------------------===//

class HIVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<HIVM_Dialect, !strconcat("hir.", mnemonic), traits> {
  // C++ function that returns the op name.
  code opNameDeclaration = [{
    static StringRef getOpName() { return "}] # mnemonic # [{"; }
  }];
}

class HIVM_StructuredOp<string mnemonic, list<Trait> traits = []>
    : HIVM_Op<mnemonic, 
      !listconcat([
        HIVMStructuredOpInterface,
        MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<FlattenInterface>,
        DeclareOpInterfaceMethods<HIVMStructuredOpInterface>,
        DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
      ], traits)>;

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD
