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
// Destination memory, rank, and dual destination are not query fields. A row
// is listed only when that source/destination pair and DMA mode are registered
// for GM. L1-only symbols stay unlisted. NORMAL_RELU and channel_split=true
// are not confirmed, so those requests stay Unknown.

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

// One confirmed pre-quant mode, NO_RELU, and channel_split=false.
class FixpipeExecutionRow<TypeSignature signature,
                          list<I32EnumAttrCase> dmaModes,
                          I32EnumAttrCase preQuant>
    : SupportCase<Fixpipe, [signature, EnumAnyOf<DmaMode, dmaModes>,
                            EnumEquals<PreQuantMode, preQuant>,
                            EnumEquals<PreRelu, HIVM_FIXPIPEPRERELU_NO_RELU>,
                            BoolValueConstraint<ChannelSplit, 0>]>;

// Ascend910B. normal is 2D to 2D and only the two float conversions.
// nz2nd is 4D to 2D. i32 -> i16 and i32 -> f16 are in that library but are
// not type pairs in this schema.
def C220Nz2ndF32F32 : FixpipeExecutionRow<FixpipeF32ToF32, nz2nd,
                                          HIVM_FIXPIPEPREQUANT_NO_QUANT>;
def C220Nz2ndI32I32 : FixpipeExecutionRow<FixpipeI32ToI32, nz2nd,
                                          HIVM_FIXPIPEPREQUANT_NO_QUANT>;
def C220Nz2ndF32F16
    : FixpipeExecutionRow<FixpipeF32ToF16, nz2nd, HIVM_FIXPIPEPREQUANT_F322F16>;
def C220Nz2ndF32BF16 : FixpipeExecutionRow<FixpipeF32ToBF16, nz2nd,
                                           HIVM_FIXPIPEPREQUANT_F322BF16>;
def C220Nz2ndI32I8
    : FixpipeExecutionRow<FixpipeI32ToI8, nz2nd, HIVM_FIXPIPEPREQUANT_S322I8>;
def C220NormalF32F16 : FixpipeExecutionRow<FixpipeF32ToF16, normal,
                                           HIVM_FIXPIPEPREQUANT_F322F16>;
def C220NormalF32BF16 : FixpipeExecutionRow<FixpipeF32ToBF16, normal,
                                            HIVM_FIXPIPEPREQUANT_F322BF16>;

// RegBase. i32 -> i8 normal is L1-only, so it is not a GM row.
def C310F32F32NoQuant : FixpipeExecutionRow<FixpipeF32ToF32, c310DmaModes,
                                            HIVM_FIXPIPEPREQUANT_NO_QUANT>;
def C310I32I32NoQuant : FixpipeExecutionRow<FixpipeI32ToI32, c310DmaModes,
                                            HIVM_FIXPIPEPREQUANT_NO_QUANT>;
def C310F32F16 : FixpipeExecutionRow<FixpipeF32ToF16, c310DmaModes,
                                     HIVM_FIXPIPEPREQUANT_F322F16>;
def C310F32BF16 : FixpipeExecutionRow<FixpipeF32ToBF16, c310DmaModes,
                                      HIVM_FIXPIPEPREQUANT_F322BF16>;
def C310I32I8 : FixpipeExecutionRow<FixpipeI32ToI8, c310Transformed,
                                    HIVM_FIXPIPEPREQUANT_S322I8>;
def C310F32F32PreQuant : FixpipeExecutionRow<FixpipeF32ToF32, c310DmaModes,
                                             HIVM_FIXPIPEPREQUANT_QF322F32_PRE>;

def C220FixpipeCoverage
    : ArchitectureCapabilityCoverage<
          Fixpipe,
          [C220Nz2ndF32F32, C220Nz2ndI32I32, C220Nz2ndF32F16, C220Nz2ndF32BF16,
           C220Nz2ndI32I8, C220NormalF32F16, C220NormalF32BF16],
          PartialProfile>;
def C310FixpipeCoverage
    : ArchitectureCapabilityCoverage<Fixpipe,
                                     [C310F32F32NoQuant, C310I32I32NoQuant,
                                      C310F32F16, C310F32BF16, C310I32I8,
                                      C310F32F32PreQuant],
                                     PartialProfile>;

def Dav_C220FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C220, [C220FixpipeCoverage]>;
def Dav_C310FixpipeProfile
    : ArchitectureCapabilityProfile<Dav_C310, [C310FixpipeCoverage]>;

#endif // BISHENGIR_TARGET_NPU_DMA_FIXPIPE_FIXPIPECAPABILITYPROFILE_TD
