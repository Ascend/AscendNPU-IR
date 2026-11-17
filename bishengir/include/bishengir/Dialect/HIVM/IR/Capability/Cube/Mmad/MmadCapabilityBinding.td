//===- MmadCapabilityBinding.td - Mmad Op bindings ---------*- tablegen -*-===//
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
// One binding per capability key. UnitAttr absence is false. An absent bias
// operand stores a dummy f32 and L1; bias_present is false on that request.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYBINDING_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYBINDING_TD

include "bishengir/Dialect/HIVM/IR/HIVMDoc.td"
include "bishengir/Dialect/HIVM/IR/Capability/Cube/MmadMx/MmadMxCapability.td"

class OptionalElementTypeProjection<string fallbackExpr>
    : CapabilityProjection<"optional_element_type"> {
  string FallbackExpr = fallbackExpr;
}
class OptionalAddressSpaceProjection<string fallbackExpr>
    : CapabilityProjection<"optional_address_space"> {
  string FallbackExpr = fallbackExpr;
}
def MmadAttributePresence : CapabilityProjection<"attribute_presence">;
class IndexDivisibleProjection<int divisor>
    : CapabilityProjection<"index_divisible"> {
  int Divisor = divisor;
}
def MmadMxKMultipleOf64Projection : IndexDivisibleProjection<64>;
def MmadOptionalEnum : CapabilityProjection<"optional_enum">;
def MmadOptionalBiasType : OptionalElementTypeProjection<
                               "::mlir::Float32Type::get(op->getContext())">;
def MmadOptionalBiasSpace
    : OptionalAddressSpaceProjection<"::mlir::hivm::AddressSpace::L1">;

defvar MmadExtractors =
    [CapabilityExtractor<OperandBindingSource<"a">, ElementTypeProjection,
                         MmadAType>,
     CapabilityExtractor<OperandBindingSource<"a">, AddressSpaceProjection,
                         MmadASpace>,
     CapabilityExtractor<OperandBindingSource<"b">, ElementTypeProjection,
                         MmadBType>,
     CapabilityExtractor<OperandBindingSource<"b">, AddressSpaceProjection,
                         MmadBSpace>,
     CapabilityExtractor<OperandBindingSource<"c">, ElementTypeProjection,
                         MmadCType>,
     CapabilityExtractor<OperandBindingSource<"c">, AddressSpaceProjection,
                         MmadCSpace>,
     CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                         MmadOptionalBiasType, MmadBiasType>,
     CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                         MmadOptionalBiasSpace, MmadBiasSpace>,
     CapabilityExtractor<AttributeBindingSource<"a_transpose">,
                         MmadAttributePresence, MmadATranspose>,
     CapabilityExtractor<AttributeBindingSource<"b_transpose">,
                         MmadAttributePresence, MmadBTranspose>,
     CapabilityExtractor<AttributeBindingSource<"enable_HF32">,
                         MmadAttributePresence, MmadEnableHF32>,
     CapabilityExtractor<AttributeBindingSource<"enable_I4">,
                         MmadAttributePresence, MmadEnableI4>,
     CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                         PresenceProjection, MmadBiasPresent>];

def MmadCapabilityBinding : CapabilityBinding<Mmad, MmadL1Op, MmadExtractors>;
def BatchMmadCapabilityBinding
    : CapabilityBinding<BatchMmad, BatchMmadL1Op, MmadExtractors>;

def MmadMxCapabilityBinding
    : CapabilityBinding<
          MmadMx, MmadMxL1Op,
          [CapabilityExtractor<OperandBindingSource<"a">, ElementTypeProjection,
                               MmadAType>,
           CapabilityExtractor<OperandBindingSource<"a">,
                               AddressSpaceProjection, MmadASpace>,
           CapabilityExtractor<OperandBindingSource<"b">, ElementTypeProjection,
                               MmadBType>,
           CapabilityExtractor<OperandBindingSource<"b">,
                               AddressSpaceProjection, MmadBSpace>,
           CapabilityExtractor<OperandBindingSource<"scaleA">,
                               ElementTypeProjection, MmadMxScaleAType>,
           CapabilityExtractor<OperandBindingSource<"scaleA">,
                               AddressSpaceProjection, MmadMxScaleASpace>,
           CapabilityExtractor<OperandBindingSource<"scaleB">,
                               ElementTypeProjection, MmadMxScaleBType>,
           CapabilityExtractor<OperandBindingSource<"scaleB">,
                               AddressSpaceProjection, MmadMxScaleBSpace>,
           CapabilityExtractor<OperandBindingSource<"c">, ElementTypeProjection,
                               MmadCType>,
           CapabilityExtractor<OperandBindingSource<"c">,
                               AddressSpaceProjection, MmadCSpace>,
           CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                               MmadOptionalBiasType, MmadBiasType>,
           CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                               MmadOptionalBiasSpace, MmadBiasSpace>,
           CapabilityExtractor<AttributeBindingSource<"a_transpose">,
                               MmadAttributePresence, MmadATranspose>,
           CapabilityExtractor<AttributeBindingSource<"b_transpose">,
                               MmadAttributePresence, MmadBTranspose>,
           CapabilityExtractor<OperandBindingSource<"per_channel_bias">,
                               PresenceProjection, MmadBiasPresent>,
           CapabilityExtractor<AttributeBindingSource<"lhsFormat">,
                               MmadOptionalEnum, MmadMxLhsFormat>,
           CapabilityExtractor<AttributeBindingSource<"rhsFormat">,
                               MmadOptionalEnum, MmadMxRhsFormat>,
           CapabilityExtractor<AttributeBindingSource<"lhsFormat">,
                               MmadAttributePresence, MmadMxLhsFormatPresent>,
           CapabilityExtractor<AttributeBindingSource<"rhsFormat">,
                               MmadAttributePresence, MmadMxRhsFormatPresent>,
           CapabilityExtractor<OperandBindingSource<"real_k">,
                               MmadMxKMultipleOf64Projection,
                               MmadMxKMultipleOf64>]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADCAPABILITYBINDING_TD
