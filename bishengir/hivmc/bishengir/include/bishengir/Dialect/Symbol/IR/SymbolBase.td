//===- SymbolBase.td - Symbol dialect base support -----------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SYMBOL_IR_SYMBOLBASE_TD
#define BISHENGIR_DIALECT_SYMBOL_IR_SYMBOLBASE_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Symbol Dialect.
//===----------------------------------------------------------------------===//

def Symbol_Dialect : Dialect {
  let name = "symbol";
  let description = [{
    Symbol dialect to represent and expresses dynamic shape's relationship.
  }];
  let cppNamespace = "::mlir::symbol";
  let dependentDialects = [
    "tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Symbol Dialect operations.
//===----------------------------------------------------------------------===//

class Symbol_Op<string mnemonic, list<Trait> traits = []>
    : Op<Symbol_Dialect, mnemonic, traits>;

#endif // BISHENGIR_DIALECT_Symbol_IR_SymbolBASE_TD
