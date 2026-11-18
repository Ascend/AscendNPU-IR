//===- MmadCapability.td - Mmad execution schema -----------*- tablegen -*-===//
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
// Mmad and BatchMmad execution schema. Rows live in
// Target/NPU/Cube/Mmad/MmadCapabilityProfile.td. MmadL0 has no catalogue:
// the cube library symbols are the L1 mma_tile and batch_mma_tile entries.
// Layout, rank, and unit flag are not fields.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/Common.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadEmission.td"

def MmadA : CapabilitySlot<[ElementType, AddressSpace], "A", "a">;
def MmadB : CapabilitySlot<[ElementType, AddressSpace], "B", "b">;
def MmadC : CapabilitySlot<[ElementType, AddressSpace], "C", "c">;
def MmadBias : CapabilitySlot<[ElementType, AddressSpace], "Bias", "bias">;

def MmadAType : SlotField<MmadA, ElementType>;
def MmadASpace : SlotField<MmadA, AddressSpace>;
def MmadBType : SlotField<MmadB, ElementType>;
def MmadBSpace : SlotField<MmadB, AddressSpace>;
def MmadCType : SlotField<MmadC, ElementType>;
def MmadCSpace : SlotField<MmadC, AddressSpace>;
def MmadBiasType : SlotField<MmadBias, ElementType>;
def MmadBiasSpace : SlotField<MmadBias, AddressSpace>;

def MmadATranspose : BoolAttrCapabilityParameter<"ATranspose", "a_transpose">;
def MmadBTranspose : BoolAttrCapabilityParameter<"BTranspose", "b_transpose">;
def MmadEnableHF32 : BoolAttrCapabilityParameter<"EnableHF32", "enable_hf32">;
def MmadEnableI4 : BoolAttrCapabilityParameter<"EnableI4", "enable_i4">;
def MmadBiasPresent
    : BoolAttrCapabilityParameter<"BiasPresent", "bias_present">;

def MmadSchema : RequestSchema<[MmadA, MmadB, MmadC, MmadBias],
                               [MmadATranspose, MmadBTranspose, MmadEnableHF32,
                                MmadEnableI4, MmadBiasPresent]>;

def Mmad : CapabilityKey<MmadSchema> {
  let DocNote = "L1 mma_tile. A and B share one element type in every row. "
                "Layout, rank, and unit flag are outside this schema. "
                "A missing A, B, or C memory scope is Unknown until the query "
                "sets it. MmadL0 is not this key.";
}

// Same schema, separate rows. batch_mma_tile has no bias and no i8 or fp8.
def BatchMmad : CapabilityKey<MmadSchema> {
  let DocNote = "L1 batch_mma_tile. Same fields as mmad. "
                "A row here is not a row of mmad.";
}

// A and B stay paired. A later f16 row does not allow f16 x f32.
def MmadF16ToF32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [F16, F16, F32]>;
def MmadBF16ToF32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [BF16, BF16, F32]>;
def MmadF32ToF32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [F32, F32, F32]>;
def MmadI8ToI32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [I8, I8, I32]>;
def MmadF8E4ToF32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [F8E4M3FN, F8E4M3FN,
                                                        F32]>;
def MmadF8E5ToF32
    : TypeSignature<[MmadAType, MmadBType, MmadCType], [F8E5M2, F8E5M2, F32]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITY_TD
