//===- FixpipeCapability.td - Fixpipe execution schema ----*- tablegen -*-===//
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
// Fixpipe execution-capability vocabulary. Architecture support rows live in
// Target/NPU/FixpipeCapabilityProfile.td. This file does not include HACC.
// The generic capability base does not include this file.

#ifndef BISHENGIR_DIALECT_HIVM_IR_FIXPIPECAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_FIXPIPECAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Target/Capability/CapabilityBase.td"

// ElementType is already a TableGen class, so the property record needs
// another name. The generated C++ and text names stay ElementType / element_type.
def FixpipeElementType
    : CapabilitySlotProperty<MLIRTypeDomain, "ElementType", "element_type">;

// Address space, layout, shape, and quant_scale presence are not fields.
// Those facts are not observable on this early execution request.
def Src : CapabilitySlot<[FixpipeElementType], "Src", "src">;
def Dst : CapabilitySlot<[FixpipeElementType], "Dst", "dst">;
def SrcType : SlotField<Src, FixpipeElementType>;
def DstType : SlotField<Dst, FixpipeElementType>;

// Parameters reuse the Fixpipe ODS attributes. Their domains are the enum
// cases already declared for the IR, not a second value list.
def DmaMode : EnumAttrCapabilityParameter<HIVM_FixpipeDMAModeAttr>;
def PreQuantMode : EnumAttrCapabilityParameter<HIVM_FixpipePreQuantModeAttr>;
def PreRelu : EnumAttrCapabilityParameter<HIVM_FixpipePreReluModeAttr>;
def ChannelSplit : BoolAttrCapabilityParameter<>;

def FixpipeSchema
    : RequestSchema<[Src, Dst], [DmaMode, PreQuantMode, PreRelu, ChannelSplit]>;
def Fixpipe : CapabilityKey<FixpipeSchema>;

// Pairs stay whole. A later row that allows f32 -> f16 does not allow f32 -> i8.
def FixpipeF32ToF32 : TypeSignature<[SrcType, DstType], [F32, F32]>;
def FixpipeI32ToI32 : TypeSignature<[SrcType, DstType], [I32, I32]>;
def FixpipeF32ToF16 : TypeSignature<[SrcType, DstType], [F32, F16]>;
def FixpipeF32ToBF16 : TypeSignature<[SrcType, DstType], [F32, BF16]>;
def FixpipeI32ToI8 : TypeSignature<[SrcType, DstType], [I32, I8]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_FIXPIPECAPABILITY_TD
