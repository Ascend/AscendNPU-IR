//===- BiShengIRAttrs.td - BiShengIR general attr defs. ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    I32EnumAttrCase<"AFTER_HIVM_FLATTEN_OPS", 7>
]>;

#endif //BISHENGIR_INTERFACES_BISHENGIRATTRS_TD
