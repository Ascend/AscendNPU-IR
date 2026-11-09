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
// Target/NPU/DMA/Fixpipe/FixpipeCapabilityProfile.td. This file does not
// include HACC. The generic capability base does not include this file.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/Common.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Target/Capability/CapabilityBase.td"

// Layout and shape are not fields. Address space and quant_scale presence are.
// Parameters reuse the Fixpipe ODS attributes. Their domains are the enum
// cases already declared for the IR, not a second value list.
def DmaMode : EnumAttrCapabilityParameter<HIVM_FixpipeDMAModeAttr>;
def PreQuantMode : EnumAttrCapabilityParameter<HIVM_FixpipePreQuantModeAttr>;
def PreRelu : EnumAttrCapabilityParameter<HIVM_FixpipePreReluModeAttr>;
def ChannelSplit : BoolAttrCapabilityParameter<>;
// True when the optional quant_scale operand is present. This is not a
// numeric scale value.
def QuantScalePresent
    : BoolAttrCapabilityParameter<"QuantScalePresent", "quant_scale_present">;

def FixpipeSchema
    : RequestSchema<[Src, Dst],
                    [DmaMode, PreQuantMode, PreRelu, ChannelSplit,
                     QuantScalePresent]>;
def Fixpipe : CapabilityKey<FixpipeSchema> {
  let DocNote = "Layout and shape are outside this schema. "
                "A missing src or dst memory scope is Unknown until the query "
                "sets it.";
}

// Pairs stay whole. A later row that allows f32 -> f16 does not allow f32 ->
// i8.
def FixpipeF32ToF32 : TypeSignature<[SrcType, DstType], [F32, F32]>;
def FixpipeI32ToI32 : TypeSignature<[SrcType, DstType], [I32, I32]>;
def FixpipeF32ToF16 : TypeSignature<[SrcType, DstType], [F32, F16]>;
def FixpipeF32ToBF16 : TypeSignature<[SrcType, DstType], [F32, BF16]>;
def FixpipeI32ToI8 : TypeSignature<[SrcType, DstType], [I32, I8]>;
def FixpipeI32ToI16 : TypeSignature<[SrcType, DstType], [I32, I16]>;
def FixpipeI32ToF16 : TypeSignature<[SrcType, DstType], [I32, F16]>;
def FixpipeI32ToU8 : TypeSignature<[SrcType, DstType], [I32, UI8]>;
def FixpipeF32ToI8 : TypeSignature<[SrcType, DstType], [F32, I8]>;
def FixpipeF32ToU8 : TypeSignature<[SrcType, DstType], [F32, UI8]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITY_TD
