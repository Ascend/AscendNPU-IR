//===- HFusionBase.td - Base definitions for hybrid fusion dialect --------===//
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
// This is the definition file for base hybrid fusion support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD

include "bishengir/Dialect/HFusion/IR/HFusionEnums.td"
include "bishengir/Dialect/Symbol/IR/SymbolBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def HFusion_Dialect : Dialect {
  let name = "hfusion";
  let description = [{
    Hybrid Fusion (HFusion) dialect.
  }];
  let cppNamespace = "::mlir::hfusion";
  let dependentDialects = [
    "hacc::HACCDialect",
    "linalg::LinalgDialect",
    "mathExt::MathExtDialect",
    "mesh::MeshDialect",
    "symbol::SymbolDialect"
  ];
  let hasCanonicalizer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Define the function attribute enums matching the OpDSL functions.
def UnaryFnAttr : EnumAttr<HFusion_Dialect, UnaryFn, "unary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def BinaryFnAttr : EnumAttr<HFusion_Dialect, BinaryFn, "binary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def CompareFnAttr : EnumAttr<HFusion_Dialect, CompareFn, "compare_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TernaryFnAttr : EnumAttr<HFusion_Dialect, TernaryFn, "ternary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TypeFnAttr : EnumAttr<HFusion_Dialect, TypeFn, "type_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def RoundModeAttr : EnumAttr<HFusion_Dialect, RoundMode, "round_mode"> {
  let assemblyFormat = "`<` $value `>`";
}
def AtomicKindAttr : EnumAttr<HFusion_Dialect, AtomicKind, "atomic_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD
