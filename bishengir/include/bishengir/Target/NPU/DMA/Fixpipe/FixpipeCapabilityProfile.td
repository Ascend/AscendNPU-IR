//===- FixpipeCapabilityProfile.td - Fixpipe profiles -----*- tablegen -*-===//
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
// Fixpipe rows taken from the cube-library symbols. C310 is the RegBase
// (__DAV_C310__) library. C220 is the mem-based Ascend910B library.
// A miss is Unknown, not Unsupported.
//
// src is L0C. C220 destination is GM only. C310 destinations are GM, L1, and
// UB. quant_scale presence is Native only on C310. A Partial miss is Unknown,
// not Unsupported. If src or dst memory scope is absent and the query does
// not set it, the result is Unknown.
//
// NORMAL_RELU pairs with the same confirmed DMA/pre-quant rows as NO_RELU.
// C310 channel_split=true is only f32->f32 NO_QUANT / QF322F32_PRE, and only
// L0C to L1. GM and UB with channel_split stay Unknown. C220 has no
// channel_split=true row.

#ifndef BISHENGIR_TARGET_NPU_DMA_FIXPIPE_FIXPIPECAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_FIXPIPE_FIXPIPECAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapability.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

defvar nz2nd = [HIVM_FIXPIPE_DMA_MODE_NZ2ND];
defvar normal = [HIVM_FIXPIPE_DMA_MODE_NZ2NZ];
defvar c310DmaModes = [HIVM_FIXPIPE_DMA_MODE_NZ2ND, HIVM_FIXPIPE_DMA_MODE_NZ2DN,
                       HIVM_FIXPIPE_DMA_MODE_NZ2NZ];
defvar c310Transformed = [HIVM_FIXPIPE_DMA_MODE_NZ2ND,
                          HIVM_FIXPIPE_DMA_MODE_NZ2DN];
defvar noReluAndNormalRelu = [HIVM_FIXPIPEPRERELU_NO_RELU,
                              HIVM_FIXPIPEPRERELU_NORMAL_RELU];
defvar c220DstSpaces = [HIVM_AddressSpace_GM];
defvar c310DstSpaces = [HIVM_AddressSpace_GM, HIVM_AddressSpace_L1,
                        HIVM_AddressSpace_UB];
defvar l1Only = [HIVM_AddressSpace_L1];

// quant_scale presence is false. C220 uses this class.
class FixpipeExecutionRow<TypeSignature signature,
                          list<I32EnumAttrCase> dmaModes,
                          I32EnumAttrCase preQuant,
                          list<I32EnumAttrCase> dstSpaces>
    : SupportCase<Fixpipe, [signature, EnumAnyOf<DmaMode, dmaModes>,
                            EnumEquals<PreQuantMode, preQuant>,
                            EnumAnyOf<PreRelu, noReluAndNormalRelu>,
                            BoolValueConstraint<ChannelSplit, 0>,
                            EnumEquals<SrcSpace, HIVM_AddressSpace_L0C>,
                            EnumAnyOf<DstSpace, dstSpaces>,
                            BoolValueConstraint<QuantScalePresent, 0>]>;

// Same execution row, quant_scale may be present or absent. C310 uses this.
class FixpipeExecutionRowAllowQuant<TypeSignature signature,
                                    list<I32EnumAttrCase> dmaModes,
                                    I32EnumAttrCase preQuant,
                                    list<I32EnumAttrCase> dstSpaces>
    : SupportCase<Fixpipe, [signature, EnumAnyOf<DmaMode, dmaModes>,
                            EnumEquals<PreQuantMode, preQuant>,
                            EnumAnyOf<PreRelu, noReluAndNormalRelu>,
                            BoolValueConstraint<ChannelSplit, 0>,
                            EnumEquals<SrcSpace, HIVM_AddressSpace_L0C>,
                            EnumAnyOf<DstSpace, dstSpaces>,
                            AnyValue<QuantScalePresent>]>;

// C310 f32->f32 with channel_split=true. Only L0C to L1.
class FixpipeChannelSplitRow<I32EnumAttrCase preQuant>
    : SupportCase<Fixpipe, [FixpipeF32ToF32, EnumAnyOf<DmaMode, c310DmaModes>,
                            EnumEquals<PreQuantMode, preQuant>,
                            EnumAnyOf<PreRelu, noReluAndNormalRelu>,
                            BoolValueConstraint<ChannelSplit, 1>,
                            EnumEquals<SrcSpace, HIVM_AddressSpace_L0C>,
                            EnumAnyOf<DstSpace, l1Only>,
                            AnyValue<QuantScalePresent>]>;

// Ascend910B. normal is 2D to 2D and only the two float conversions.
// nz2nd is 4D to 2D. i32 -> i16 and i32 -> f16 are in that library but are
// not type pairs in this schema.
def C220Nz2ndF32F32 : FixpipeExecutionRow<FixpipeF32ToF32, nz2nd,
                                          HIVM_FIXPIPEPREQUANT_NO_QUANT,
                                          c220DstSpaces>;
def C220Nz2ndI32I32 : FixpipeExecutionRow<FixpipeI32ToI32, nz2nd,
                                          HIVM_FIXPIPEPREQUANT_NO_QUANT,
                                          c220DstSpaces>;
def C220Nz2ndF32F16
    : FixpipeExecutionRow<FixpipeF32ToF16, nz2nd, HIVM_FIXPIPEPREQUANT_F322F16,
                          c220DstSpaces>;
def C220Nz2ndF32BF16 : FixpipeExecutionRow<FixpipeF32ToBF16, nz2nd,
                                           HIVM_FIXPIPEPREQUANT_F322BF16,
                                           c220DstSpaces>;
def C220Nz2ndI32I8
    : FixpipeExecutionRow<FixpipeI32ToI8, nz2nd, HIVM_FIXPIPEPREQUANT_S322I8,
                          c220DstSpaces>;
def C220Nz2ndI32I16
    : FixpipeExecutionRow<FixpipeI32ToI16, nz2nd, HIVM_FIXPIPEPREQUANT_NO_QUANT,
                          c220DstSpaces>;
def C220Nz2ndI32F16
    : FixpipeExecutionRow<FixpipeI32ToF16, nz2nd, HIVM_FIXPIPEPREQUANT_NO_QUANT,
                          c220DstSpaces>;
def C220NormalF32F16 : FixpipeExecutionRow<FixpipeF32ToF16, normal,
                                           HIVM_FIXPIPEPREQUANT_F322F16,
                                           c220DstSpaces>;
def C220NormalF32BF16 : FixpipeExecutionRow<FixpipeF32ToBF16, normal,
                                            HIVM_FIXPIPEPREQUANT_F322BF16,
                                            c220DstSpaces>;

// RegBase. i32 -> i8 normal is L1-only in the cube library, so it is not here.
def C310F32F32NoQuant
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToF32, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT,
                                    c310DstSpaces>;
def C310I32I32NoQuant
    : FixpipeExecutionRowAllowQuant<FixpipeI32ToI32, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT,
                                    c310DstSpaces>;
def C310F32F16
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToF16, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_F322F16, c310DstSpaces>;
def C310F32F16PreQuant
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToF16, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_QF322F16_PRE,
                                    c310DstSpaces>;
def C310F32BF16
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToBF16, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_F322BF16,
                                    c310DstSpaces>;
def C310I32I8
    : FixpipeExecutionRowAllowQuant<FixpipeI32ToI8, c310Transformed,
                                    HIVM_FIXPIPEPREQUANT_S322I8, c310DstSpaces>;
def C310F32F32PreQuant
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToF32, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_QF322F32_PRE,
                                    c310DstSpaces>;
def C310F32I8
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToI8, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT, c310DstSpaces>;
def C310F32U8
    : FixpipeExecutionRowAllowQuant<FixpipeF32ToU8, c310DmaModes,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT, c310DstSpaces>;
def C310I32U8
    : FixpipeExecutionRowAllowQuant<FixpipeI32ToU8, c310Transformed,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT, c310DstSpaces>;
def C310I32F16
    : FixpipeExecutionRowAllowQuant<FixpipeI32ToF16, c310Transformed,
                                    HIVM_FIXPIPEPREQUANT_NO_QUANT,
                                    c310DstSpaces>;
def C310F32F32NoQuantSplit
    : FixpipeChannelSplitRow<HIVM_FIXPIPEPREQUANT_NO_QUANT>;
def C310F32F32PreQuantSplit
    : FixpipeChannelSplitRow<HIVM_FIXPIPEPREQUANT_QF322F32_PRE>;

def C220FixpipeCoverage
    : ArchitectureCapabilityCoverage<
          Fixpipe,
          [C220Nz2ndF32F32, C220Nz2ndI32I32, C220Nz2ndF32F16, C220Nz2ndF32BF16,
           C220Nz2ndI32I8, C220Nz2ndI32I16, C220Nz2ndI32F16, C220NormalF32F16,
           C220NormalF32BF16],
          PartialProfile>;
def C310FixpipeCoverage
    : ArchitectureCapabilityCoverage<Fixpipe,
                                     [C310F32F32NoQuant, C310I32I32NoQuant,
                                      C310F32F16, C310F32F16PreQuant, C310F32BF16,
                                      C310I32I8,
                                      C310F32F32PreQuant, C310F32I8, C310F32U8,
                                      C310I32U8, C310I32F16,
                                      C310F32F32NoQuantSplit,
                                      C310F32F32PreQuantSplit],
                                     PartialProfile>;

// dav-920r1 accepts every well-formed Fixpipe request. AnyValue still
// requires each field to be present, so a missing memory scope stays Unknown.
def Dav920R1AnyFixpipe
    : SupportCase<Fixpipe, [AnyValue<SrcType>, AnyValue<SrcSpace>,
                            AnyValue<DstType>, AnyValue<DstSpace>,
                            AnyValue<DmaMode>, AnyValue<PreQuantMode>,
                            AnyValue<PreRelu>, AnyValue<ChannelSplit>,
                            AnyValue<QuantScalePresent>]>;
def Dav920R1FixpipeCoverage
    : ArchitectureCapabilityCoverage<Fixpipe, [Dav920R1AnyFixpipe],
                                     PartialProfile>;

def Dav_C220FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C220, [C220FixpipeCoverage]>;
def Dav_C310FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C310, [C310FixpipeCoverage]>;
def Dav_920R1FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_920R1, [Dav920R1FixpipeCoverage]>;

#endif // BISHENGIR_TARGET_NPU_DMA_FIXPIPE_FIXPIPECAPABILITYPROFILE_TD
