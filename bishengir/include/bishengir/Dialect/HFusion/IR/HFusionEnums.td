//===- HFusionEnums.td - HFusion dialect base support ------*- tablegen -*-===//
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
// This is the definition file for enums used in hfusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD

include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/IR/EnumAttr.td"

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"relu", 0>,
  I32EnumAttrCase<"sqrt", 1>,
  I32EnumAttrCase<"rsqrt", 2>,
  I32EnumAttrCase<"rec", 3>,
  I32EnumAttrCase<"vnot", 4>,
  I32EnumAttrCase<"tanh", 5>,
  I32EnumAttrCase<"sin", 6>,
  I32EnumAttrCase<"cos", 7>,
  I32EnumAttrCase<"atan", 8>,
  I32EnumAttrCase<"tan", 9>,
  I32EnumAttrCase<"absi", 10>,
  I32EnumAttrCase<"erf", 11>,
  I32EnumAttrCase<"log2", 12>,
  I32EnumAttrCase<"log10", 13>,
  I32EnumAttrCase<"log1p", 14>,
  I32EnumAttrCase<"exp2", 15>,
  I32EnumAttrCase<"expm1", 16>,
  I32EnumAttrCase<"ilogb", 17>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"vor", 0>,
  I32EnumAttrCase<"vand", 1>,
  I32EnumAttrCase<"vxor", 2>,
  I32EnumAttrCase<"minf", 3>,
  I32EnumAttrCase<"maxf", 4>,
  I32EnumAttrCase<"powf", 5>,
  I32EnumAttrCase<"mod",  6>,
  I32EnumAttrCase<"modui", 7>,
  I32EnumAttrCase<"shli",  8>,
  I32EnumAttrCase<"shrsi", 9>,
  I32EnumAttrCase<"shrui", 10>,
  I32EnumAttrCase<"ldexp", 11>,
  I32EnumAttrCase<"ceildivsi", 12>,
  I32EnumAttrCase<"ceildivui", 13>,
  I32EnumAttrCase<"floordivsi", 14>,
  I32EnumAttrCase<"powi", 15>,
  I32EnumAttrCase<"minnumf", 16>,
  I32EnumAttrCase<"maxnumf", 17>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def CompareFn : I32EnumAttr<"CompareFn", "", [
  I32EnumAttrCase<"veq", 0>,
  I32EnumAttrCase<"vne", 1>,
  I32EnumAttrCase<"vle", 2>,
  I32EnumAttrCase<"vlt", 3>,
  I32EnumAttrCase<"vge", 4>,
  I32EnumAttrCase<"vgt", 5>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def TernaryFn : I32EnumAttr<"TernaryFn", "", [
  I32EnumAttrCase<"select", 0>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def TypeFn : I32EnumAttr<"TypeFn", "", [
  I32EnumAttrCase<"cast_signed", 0>,
  I32EnumAttrCase<"cast_unsigned", 1>,
  I32EnumAttrCase<"bitcast", 2>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def RoundMode : I32EnumAttr<"RoundMode", "", [
  I32EnumAttrCase<"RINT", 0, "rint">,
  I32EnumAttrCase<"ROUND", 1, "round">,
  I32EnumAttrCase<"FLOOR", 2, "floor">,
  I32EnumAttrCase<"CEIL", 3, "ceil">,
  I32EnumAttrCase<"TRUNC", 4, "trunc">,
  I32EnumAttrCase<"ODD", 5, "odd">,
  I32EnumAttrCase<"TRUNCWITHOVERFLOW", 6, "truncwithoverflow">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
  let description = [{
    - RINT: round to nearest, tie to even (c language rint)
    - ROUND: round to nearest, tie away from zero (c language round)
    - FLOOR: round to minus infinity (c language floor)
    - CEIL: round to positive infinity (c language ceil)
    - TRUNC: round to zero (c language trunc)
    - ODD: round to odd (Von Neumann rounding)
  }];
}

def CastMode : I32EnumAttr<"CastMode", "", [
  I32EnumAttrCase<"F32TOI8", 0, "F32TOI8">,
  I32EnumAttrCase<"F32TOI16", 1, "F32TOI16">,
  I32EnumAttrCase<"F16TOI8", 2, "F16TOI8">,
  I32EnumAttrCase<"I64TOI32", 3, "I64TOI32">,
  I32EnumAttrCase<"I64TOI16", 4, "I64TOI16">,
  I32EnumAttrCase<"I64TOI8", 5, "I64TOI8">,
  I32EnumAttrCase<"I32TOI16", 6, "I32TOI16">,
  I32EnumAttrCase<"I32TOI8", 7, "I32TOI8">,
  I32EnumAttrCase<"I16TOI8", 8, "I16TOI8">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
  let description = [{
    cast mode, cast from src dtype to dst dtype
  }];
}

//===----------------------------------------------------------------------===//
// OpFusion Related Enums.
//===----------------------------------------------------------------------===//

def TaylerMode : I32EnumAttr<"TaylerMode", "", [
  I32EnumAttrCase<"SIN", 0, "sin">,
  I32EnumAttrCase<"ATAN", 1, "atan">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def AtomicKind : I32EnumAttr<"AtomicKind", "", [
  I32EnumAttrCase<"NONE", 0, "none">,
  I32EnumAttrCase<"ADD", 1, "add">,
  I32EnumAttrCase<"MAX", 2, "max">,
  I32EnumAttrCase<"MIN", 3, "min">,
  I32EnumAttrCase<"AND", 4, "and">,
  I32EnumAttrCase<"OR", 5, "or">,
  I32EnumAttrCase<"XOR", 6, "xor">,
  I32EnumAttrCase<"CAS", 7, "cas">,
  I32EnumAttrCase<"XCHG", 8, "xchg">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

class HFusion_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def HFUSION_FUSION_KIND_PURE_ELEMWISE : I32EnumAttrCase<"PureElemwise", 1, "PURE_ELEMWISE">;
def HFUSION_FUSION_KIND_ANY_PB : I32EnumAttrCase<"AnyPB", 2, "ANY_PB">;
def HFUSION_FUSION_KIND_LAST_AXIS_PBR : I32EnumAttrCase<"LastAxisPBR", 3, "LAST_AXIS_PBR">;
def HFUSION_FUSION_KIND_ANY_PBR : I32EnumAttrCase<"AnyPBR", 4, "ANY_PBR">;
def HFUSION_FUSION_KIND_SINGLE_CUBE : I32EnumAttrCase<"SingleCube", 5, "SINGLE_CUBE">;
def HFUSION_FUSION_KIND_SHALLOW_CV : I32EnumAttrCase<"ShallowCV", 6, "SHALLOW_CV">;
def HFUSION_FUSION_KIND_SHALLOW_VV : I32EnumAttrCase<"ShallowVV", 7, "SHALLOW_VV">;
def HFUSION_FUSION_KIND_MIX_CV : I32EnumAttrCase<"MixCV", 8, "MIX_CV">;
def HFUSION_FUSION_KIND_MIX_C2 : I32EnumAttrCase<"MixC2", 9, "MIX_C2">;
def HFUSION_FUSION_KIND_UNKNOWN : I32EnumAttrCase<"Unknown", 10, "UNKNOWN">;

def HFusion_FusionKindEnum :
  HFusion_I32Enum<"FusionKind", "HFusion fused kernel kind", [
  HFUSION_FUSION_KIND_PURE_ELEMWISE,
  HFUSION_FUSION_KIND_ANY_PB,
  HFUSION_FUSION_KIND_LAST_AXIS_PBR,
  HFUSION_FUSION_KIND_ANY_PBR,
  HFUSION_FUSION_KIND_SINGLE_CUBE,
  HFUSION_FUSION_KIND_SHALLOW_CV,
  HFUSION_FUSION_KIND_SHALLOW_VV,
  HFUSION_FUSION_KIND_MIX_CV,
  HFUSION_FUSION_KIND_MIX_C2,
  HFUSION_FUSION_KIND_UNKNOWN,
]> {}

def HFUSION_FLATTEN_MODE_GREEDY : I32EnumAttrCase<"Greedy", 1>;
def HFUSION_FLATTEN_MODE_TIDY : I32EnumAttrCase<"Tidy", 2>;

def HFusion_FlattenModeEnum :
  HFusion_I32Enum<"FlattenMode", "HFusion flatten mode", [
  HFUSION_FLATTEN_MODE_GREEDY,
  HFUSION_FLATTEN_MODE_TIDY,
]> {}

def MmMapMode : I32EnumAttr<"MmMapMode", "", [
  I32EnumAttrCase<"CoreOp", 0, "core_op">,
  I32EnumAttrCase<"MacroInstr", 1, "macro_instr">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def HFUSION_OUTPUT_MODE_MULTIPLE : I32EnumAttrCase<"Multiple", 1>;
def HFUSION_OUTPUT_MODE_SINGLE : I32EnumAttrCase<"Single", 2>;
def HFUSION_OUTPUT_MODE_SINGLE_AGGR : I32EnumAttrCase<"SingleAggressive", 3>;

def HFusion_OutputModeEnum :
  HFusion_I32Enum<"OutputMode", "HFusion Output mode", [
  HFUSION_OUTPUT_MODE_MULTIPLE,
  HFUSION_OUTPUT_MODE_SINGLE,
  HFUSION_OUTPUT_MODE_SINGLE_AGGR,
]> {}

def ReduceWithIndexKind : I32EnumAttr<"ReduceWithIndexKind", "", [
  I32EnumAttrCase<"MIN", 0, "min">,
  I32EnumAttrCase<"MAX", 1, "max">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def Hfusion_CumOpTypeEnum :
  HFusion_I32Enum<"CumOpType", "HFusion cumulative operation type", [
  I32EnumAttrCase<"UNDEFINED", 0, "undefined">,
  I32EnumAttrCase<"CUMSUM", 1, "cumsum">,
  I32EnumAttrCase<"CUMPROD", 2, "cumprod">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD
