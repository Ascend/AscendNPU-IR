//===- HIVMBase.td - HIVM dialect base support -------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
    "math::MathDialect", "memref::MemRefDialect",
    "hacc::HACCDialect",
    "tensor::TensorDialect"
  ];
  let hasCanonicalizer = 1;
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
        DeclareOpInterfaceMethods<HIVMStructuredOpInterface>
      ], traits)>;

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD
