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
// Owner-confirmed Fixpipe execution rows for C220 and C310. Both coverages
// stay Partial: a miss is Unknown, not Unsupported. M300 is not derived from
// the execution model or from Cube/Vector features.

#ifndef BISHENGIR_TARGET_NPU_FIXPIPECAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_FIXPIPECAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/FixpipeCapability.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

defvar c220DmaModes = [HIVM_FIXPIPE_DMA_MODE_NZ2ND, HIVM_FIXPIPE_DMA_MODE_NZ2NZ];
defvar c310DmaModes = [HIVM_FIXPIPE_DMA_MODE_NZ2ND, HIVM_FIXPIPE_DMA_MODE_NZ2DN,
                       HIVM_FIXPIPE_DMA_MODE_NZ2NZ];

// pre_relu is the two confirmed modes. LEAKY_RELU and P_RELU stay unlisted.
class FixpipeExecutionRow<TypeSignature signature,
                          list<EnumAttrCaseInfo> dmaModes,
                          I32EnumAttrCase preQuant, bit channelSplit>
    : SupportCase<Fixpipe, [
          signature, EnumAnyOf<DmaMode, dmaModes>,
          EnumEquals<PreQuantMode, preQuant>,
          EnumAnyOf<PreRelu, [HIVM_FIXPIPEPRERELU_NO_RELU,
                              HIVM_FIXPIPEPRERELU_NORMAL_RELU]>,
          BoolValueConstraint<ChannelSplit, channelSplit>]>;

// channel_split=false rows. The type pair and pre-quant mode stay associated.
def C220F32F32NoQuant
    : FixpipeExecutionRow<FixpipeF32ToF32, c220DmaModes,
                          HIVM_FIXPIPEPREQUANT_NO_QUANT, 0>;
def C220I32I32NoQuant
    : FixpipeExecutionRow<FixpipeI32ToI32, c220DmaModes,
                          HIVM_FIXPIPEPREQUANT_NO_QUANT, 0>;
def C220F32F16
    : FixpipeExecutionRow<FixpipeF32ToF16, c220DmaModes,
                          HIVM_FIXPIPEPREQUANT_F322F16, 0>;
def C220F32BF16
    : FixpipeExecutionRow<FixpipeF32ToBF16, c220DmaModes,
                          HIVM_FIXPIPEPREQUANT_F322BF16, 0>;
def C220I32I8
    : FixpipeExecutionRow<FixpipeI32ToI8, c220DmaModes,
                          HIVM_FIXPIPEPREQUANT_S322I8, 0>;

def C310F32F32NoQuant
    : FixpipeExecutionRow<FixpipeF32ToF32, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_NO_QUANT, 0>;
def C310I32I32NoQuant
    : FixpipeExecutionRow<FixpipeI32ToI32, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_NO_QUANT, 0>;
def C310F32F16
    : FixpipeExecutionRow<FixpipeF32ToF16, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_F322F16, 0>;
def C310F32BF16
    : FixpipeExecutionRow<FixpipeF32ToBF16, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_F322BF16, 0>;
def C310I32I8
    : FixpipeExecutionRow<FixpipeI32ToI8, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_S322I8, 0>;
def C310F32F32PreQuant
    : FixpipeExecutionRow<FixpipeF32ToF32, c310DmaModes,
                          HIVM_FIXPIPEPREQUANT_QF322F32_PRE, 0>;

// C310 channel_split=true is only f32 -> f32, for both confirmed pre-quant
// modes. DMA is unrestricted among the enum. C220 has no true row.
def C310SplitF32NoQuant
    : SupportCase<Fixpipe, [
          FixpipeF32ToF32, AnyValue<DmaMode>,
          EnumEquals<PreQuantMode, HIVM_FIXPIPEPREQUANT_NO_QUANT>,
          EnumAnyOf<PreRelu, [HIVM_FIXPIPEPRERELU_NO_RELU,
                              HIVM_FIXPIPEPRERELU_NORMAL_RELU]>,
          BoolValueConstraint<ChannelSplit, 1>]>;
def C310SplitF32PreQuant
    : SupportCase<Fixpipe, [
          FixpipeF32ToF32, AnyValue<DmaMode>,
          EnumEquals<PreQuantMode, HIVM_FIXPIPEPREQUANT_QF322F32_PRE>,
          EnumAnyOf<PreRelu, [HIVM_FIXPIPEPRERELU_NO_RELU,
                              HIVM_FIXPIPEPRERELU_NORMAL_RELU]>,
          BoolValueConstraint<ChannelSplit, 1>]>;

def C220FixpipeCoverage : ArchitectureCapabilityCoverage<
    Fixpipe,
    [C220F32F32NoQuant, C220I32I32NoQuant, C220F32F16, C220F32BF16, C220I32I8],
    PartialProfile>;
def C310FixpipeCoverage : ArchitectureCapabilityCoverage<
    Fixpipe,
    [C310F32F32NoQuant, C310I32I32NoQuant, C310F32F16, C310F32BF16, C310I32I8,
     C310F32F32PreQuant, C310SplitF32NoQuant, C310SplitF32PreQuant],
    PartialProfile>;

def Dav_C220FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C220, [C220FixpipeCoverage]>;
def Dav_C310FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C310, [C310FixpipeCoverage]>;

#endif // BISHENGIR_TARGET_NPU_FIXPIPECAPABILITYPROFILE_TD
