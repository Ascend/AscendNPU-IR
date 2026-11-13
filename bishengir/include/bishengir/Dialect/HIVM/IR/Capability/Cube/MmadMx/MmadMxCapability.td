//===- MmadMxCapability.td - MmadMx execution schema --------*- tablegen
//-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// MmadMx execution schema. Rows live in
// Target/NPU/Cube/MmadMx/MmadMxCapabilityProfile.td. Scale element type is
// the IR type (i8 or ui8). The cube library stores that scale as fp8_e8m0.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMADMX_MMADMXCAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMADMX_MMADMXCAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapability.td"

def MmadMxScaleA
    : CapabilitySlot<[ElementType, AddressSpace], "ScaleA", "scale_a">;
def MmadMxScaleB
    : CapabilitySlot<[ElementType, AddressSpace], "ScaleB", "scale_b">;
def MmadMxScaleAType : SlotField<MmadMxScaleA, ElementType>;
def MmadMxScaleASpace : SlotField<MmadMxScaleA, AddressSpace>;
def MmadMxScaleBType : SlotField<MmadMxScaleB, ElementType>;
def MmadMxScaleBSpace : SlotField<MmadMxScaleB, AddressSpace>;

// TableGen domain carrier for EnumAttrCapabilityParameter. This record is
// not an HIVM attribute definition and is not registered on the dialect.
// The binding stores the format value as an integer.
def MmadMxFormatAttr : HIVM_Attr<"MmadMxFormat", "mmadmx_format"> {
  let parameters = (ins EnumParameter<HIVM_LocalMatmulMx_DataFormat>:$value);
  let assemblyFormat = "`<` params `>`";
}
def MmadMxLhsFormat : EnumAttrCapabilityParameter<MmadMxFormatAttr> {
  let CppName = "LhsFormat";
  let Name = "lhs_format";
}
def MmadMxRhsFormat : EnumAttrCapabilityParameter<MmadMxFormatAttr> {
  let CppName = "RhsFormat";
  let Name = "rhs_format";
}
def MmadMxLhsFormatPresent
    : BoolAttrCapabilityParameter<"LhsFormatPresent", "lhs_format_present">;
def MmadMxRhsFormatPresent
    : BoolAttrCapabilityParameter<"RhsFormatPresent", "rhs_format_present">;
// True when real_k is a positive constant multiple of 64. A dynamic K is false.
def MmadMxKMultipleOf64
    : BoolAttrCapabilityParameter<"KMultipleOf64", "k_multiple_of_64">;

def MmadMxSchema
    : RequestSchema<[MmadA, MmadB, MmadMxScaleA, MmadMxScaleB, MmadC, MmadBias],
                    [MmadATranspose, MmadBTranspose, MmadBiasPresent,
                     MmadMxLhsFormat, MmadMxRhsFormat, MmadMxLhsFormatPresent,
                     MmadMxRhsFormatPresent, MmadMxKMultipleOf64]>;

def MmadMx : CapabilityKey<MmadMxSchema> {
  let DocNote = "L1 mmadmx. Typed fp8 rows leave both formats absent. "
                "Format rows accept i8 storage or matching fp8 A/B types. "
                "Scale A and scale B are "
                "the same IR integer type in every row. Every C310 row "
                "requires k_multiple_of_64. HF32 and i4 are not fields.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMADMX_MMADMXCAPABILITY_TD
