//===- BiShengIRAttrs.td - BiShengIR general attr defs. ----*- tablegen -*-===//
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
// This is the definition file for BiShengIR general attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INTERFACES_BISHENGIRATTRS_TD
#define BISHENGIR_INTERFACES_BISHENGIRATTRS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// BiShengIR general Enums and Attributes.
//===----------------------------------------------------------------------===//

class BiShengIR_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::bishengir";
}

// Decompose phase for AggregateOpInterface
def BiShengIR_DecomposePhase_Enum : BiShengIR_I32Enum<
  "DecomposePhase", "BiShengIR AggregateOpInterface Decompose Phase", [
    I32EnumAttrCase<"NO_CONSTRAINT", 0>,
    I32EnumAttrCase<"AFTER_HFUSION_FLATTEN", 1>,
    I32EnumAttrCase<"BEFORE_HIVM_STRIDE_ALIGNMENT", 2>,
    I32EnumAttrCase<"AFTER_RECOGNIZE_DEINTERLEAVE", 3>,
    I32EnumAttrCase<"AFTER_RECOGNIZE_BROADCAST", 4>,
    I32EnumAttrCase<"AFTER_HIVM_STRIDE_ALIGNMENT", 5>,
    I32EnumAttrCase<"AFTER_INFER_HIVM_DATA_LAYOUT", 6>,
    I32EnumAttrCase<"AFTER_HIVM_FLATTEN_OPS", 7>,
    I32EnumAttrCase<"AFTER_LIFT_LOWEST_STRIDE", 8>
]>;

#endif //BISHENGIR_INTERFACES_BISHENGIRATTRS_TD
