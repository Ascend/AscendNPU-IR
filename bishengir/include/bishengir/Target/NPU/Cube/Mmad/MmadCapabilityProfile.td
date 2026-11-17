//===- MmadCapabilityProfile.td - Mmad profiles ------------*- tablegen -*-===//
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
// Rows from the cube-library symbols.
// C220: bishengir/lib/Template/lib/Cube/LocalMmad.cpp
// C310: bishengir/lib/Template/lib/RegBase/Cube/LocalMmad.cpp
//       and LocalMmadMx.cpp
// A miss is Unknown. MmadL0 is not catalogued. Rank and unit flag are not
// fields. A and B are L1. C is L0C. Bias, when present, is L1.

#ifndef BISHENGIR_TARGET_NPU_CUBE_MMAD_MMADCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_CUBE_MMAD_MMADCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapability.td"
include "bishengir/Target/NPU/Cube/MmadMx/MmadMxCapabilityProfile.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

class MmadNoBias<CapabilityKey cap, TypeSignature sig, bit ta, bit tb, bit hf32,
                 bit i4>
    : SupportCase<cap, [sig, AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
                        BoolValueConstraint<MmadATranspose, ta>,
                        BoolValueConstraint<MmadBTranspose, tb>,
                        BoolValueConstraint<MmadEnableHF32, hf32>,
                        BoolValueConstraint<MmadEnableI4, i4>,
                        BoolValueConstraint<MmadBiasPresent, 0>,
                        EnumEquals<MmadASpace, HIVM_AddressSpace_L1>,
                        EnumEquals<MmadBSpace, HIVM_AddressSpace_L1>,
                        EnumEquals<MmadCSpace, HIVM_AddressSpace_L0C>]>;

class MmadBias<CapabilityKey cap, Type a, Type b, Type c, Type bias, bit ta,
               bit tb, bit hf32>
    : SupportCase<
          cap, [TypeSignature<[MmadAType, MmadBType, MmadCType, MmadBiasType],
                              [a, b, c, bias]>,
                EnumEquals<MmadBiasSpace, HIVM_AddressSpace_L1>,
                BoolValueConstraint<MmadATranspose, ta>,
                BoolValueConstraint<MmadBTranspose, tb>,
                BoolValueConstraint<MmadEnableHF32, hf32>,
                BoolValueConstraint<MmadEnableI4, 0>,
                BoolValueConstraint<MmadBiasPresent, 1>,
                EnumEquals<MmadASpace, HIVM_AddressSpace_L1>,
                EnumEquals<MmadBSpace, HIVM_AddressSpace_L1>,
                EnumEquals<MmadCSpace, HIVM_AddressSpace_L0C>]>;

multiclass MmadNoBias4<CapabilityKey cap, TypeSignature sig, bit hf32, bit i4> {
  def _NN : MmadNoBias<cap, sig, 0, 0, hf32, i4>;
  def _TN : MmadNoBias<cap, sig, 1, 0, hf32, i4>;
  def _NT : MmadNoBias<cap, sig, 0, 1, hf32, i4>;
  def _TT : MmadNoBias<cap, sig, 1, 1, hf32, i4>;
}

multiclass MmadBias4<CapabilityKey cap, Type a, Type b, Type c, Type bias,
                     bit hf32> {
  def _NN : MmadBias<cap, a, b, c, bias, 0, 0, hf32>;
  def _TN : MmadBias<cap, a, b, c, bias, 1, 0, hf32>;
  def _NT : MmadBias<cap, a, b, c, bias, 0, 1, hf32>;
  def _TT : MmadBias<cap, a, b, c, bias, 1, 1, hf32>;
}

defm MmadF16 : MmadNoBias4<Mmad, MmadF16ToF32, 0, 0>;
defm MmadBF16 : MmadNoBias4<Mmad, MmadBF16ToF32, 0, 0>;
defm MmadF32 : MmadNoBias4<Mmad, MmadF32ToF32, 0, 0>;
defm MmadI8 : MmadNoBias4<Mmad, MmadI8ToI32, 0, 0>;
defm MmadF32HF32 : MmadNoBias4<Mmad, MmadF32ToF32, 1, 0>;
defm MmadF8E4 : MmadNoBias4<Mmad, MmadF8E4ToF32, 0, 0>;
defm MmadF8E5 : MmadNoBias4<Mmad, MmadF8E5ToF32, 0, 0>;

def MmadI4_NN : MmadNoBias<Mmad, MmadI8ToI32, 0, 0, 0, 1>;
def MmadI4_NT : MmadNoBias<Mmad, MmadI8ToI32, 0, 1, 0, 1>;
def MmadI4HF32_NN : MmadNoBias<Mmad, MmadI8ToI32, 0, 0, 1, 1>;
def MmadI4HF32_NT : MmadNoBias<Mmad, MmadI8ToI32, 0, 1, 1, 1>;

defm MmadF16BiasF32 : MmadBias4<Mmad, F16, F16, F32, F32, 0>;
defm MmadBF16BiasF32 : MmadBias4<Mmad, BF16, BF16, F32, F32, 0>;
defm MmadF32BiasF32 : MmadBias4<Mmad, F32, F32, F32, F32, 0>;
defm MmadI8BiasI32 : MmadBias4<Mmad, I8, I8, I32, I32, 0>;
defm MmadF32BiasF32HF32 : MmadBias4<Mmad, F32, F32, F32, F32, 1>;
defm MmadF8E4BiasF32 : MmadBias4<Mmad, F8E4M3FN, F8E4M3FN, F32, F32, 0>;
defm MmadF8E5BiasF32 : MmadBias4<Mmad, F8E5M2, F8E5M2, F32, F32, 0>;

def MmadF16BiasF16_NN : MmadBias<Mmad, F16, F16, F32, F16, 0, 0, 0>;
def MmadF32BiasF16_NN : MmadBias<Mmad, F32, F32, F32, F16, 0, 0, 0>;
def MmadBF16BiasF16_NN : MmadBias<Mmad, BF16, BF16, F32, F16, 0, 0, 0>;
def MmadF16BiasF16_NT : MmadBias<Mmad, F16, F16, F32, F16, 0, 1, 0>;
def MmadF8E4BiasF16_NN : MmadBias<Mmad, F8E4M3FN, F8E4M3FN, F32, F16, 0, 0, 0>;
def MmadF8E5BiasF16_NN : MmadBias<Mmad, F8E5M2, F8E5M2, F32, F16, 0, 0, 0>;
def MmadF8E4BiasF16_TN : MmadBias<Mmad, F8E4M3FN, F8E4M3FN, F32, F16, 1, 0, 0>;
def MmadF8E5BiasF16_TN : MmadBias<Mmad, F8E5M2, F8E5M2, F32, F16, 1, 0, 0>;
def MmadF16BiasF16_TN : MmadBias<Mmad, F16, F16, F32, F16, 1, 0, 0>;
def MmadBF16BiasF16_TN : MmadBias<Mmad, BF16, BF16, F32, F16, 1, 0, 0>;
def MmadF8E4BiasF16_NT : MmadBias<Mmad, F8E4M3FN, F8E4M3FN, F32, F16, 0, 1, 0>;
def MmadF8E5BiasF16_NT : MmadBias<Mmad, F8E5M2, F8E5M2, F32, F16, 0, 1, 0>;
def MmadBF16BiasF16_NT : MmadBias<Mmad, BF16, BF16, F32, F16, 0, 1, 0>;
def MmadF32BiasF16_NT : MmadBias<Mmad, F32, F32, F32, F16, 0, 1, 0>;
def MmadF8E4BiasF16_TT : MmadBias<Mmad, F8E4M3FN, F8E4M3FN, F32, F16, 1, 1, 0>;
def MmadF8E5BiasF16_TT : MmadBias<Mmad, F8E5M2, F8E5M2, F32, F16, 1, 1, 0>;
def MmadF16BiasF16_TT : MmadBias<Mmad, F16, F16, F32, F16, 1, 1, 0>;
def MmadBF16BiasF16_TT : MmadBias<Mmad, BF16, BF16, F32, F16, 1, 1, 0>;
def MmadF32BiasF16_TT : MmadBias<Mmad, F32, F32, F32, F16, 1, 1, 0>;

defm BatchF16 : MmadNoBias4<BatchMmad, MmadF16ToF32, 0, 0>;
defm BatchBF16 : MmadNoBias4<BatchMmad, MmadBF16ToF32, 0, 0>;
defm BatchF32 : MmadNoBias4<BatchMmad, MmadF32ToF32, 0, 0>;
defm BatchF32HF32 : MmadNoBias4<BatchMmad, MmadF32ToF32, 1, 0>;

defvar mmadCommon = [MmadF16_NN, MmadF16_TN, MmadF16_NT, MmadF16_TT,
                     MmadBF16_NN, MmadBF16_TN, MmadBF16_NT, MmadBF16_TT,
                     MmadF32_NN, MmadF32_TN, MmadF32_NT, MmadF32_TT, MmadI8_NN,
                     MmadI8_TN, MmadI8_NT, MmadI8_TT, MmadF32HF32_NN,
                     MmadF32HF32_TN, MmadF32HF32_NT, MmadF32HF32_TT,
                     MmadF16BiasF32_NN, MmadF16BiasF32_TN, MmadF16BiasF32_NT,
                     MmadF16BiasF32_TT, MmadBF16BiasF32_NN, MmadBF16BiasF32_TN,
                     MmadBF16BiasF32_NT, MmadBF16BiasF32_TT, MmadF32BiasF32_NN,
                     MmadF32BiasF32_TN, MmadF32BiasF32_NT, MmadF32BiasF32_TT,
                     MmadI8BiasI32_NN, MmadI8BiasI32_TN, MmadI8BiasI32_NT,
                     MmadI8BiasI32_TT, MmadF32BiasF32HF32_NN,
                     MmadF32BiasF32HF32_TN, MmadF32BiasF32HF32_NT,
                     MmadF32BiasF32HF32_TT, MmadF16BiasF16_NN,
                     MmadF32BiasF16_NN, MmadBF16BiasF16_NN, MmadF16BiasF16_NT];

def C220MmadCoverage
    : ArchitectureCapabilityCoverage<
          Mmad,
          !listconcat(
              mmadCommon, [MmadI4_NN, MmadI4_NT, MmadI4HF32_NN, MmadI4HF32_NT]),
          PartialProfile>;

def C310MmadCoverage
    : ArchitectureCapabilityCoverage<
          Mmad,
          !listconcat(
              mmadCommon,
              [MmadF8E4_NN, MmadF8E4_TN, MmadF8E4_NT, MmadF8E4_TT, MmadF8E5_NN,
               MmadF8E5_TN, MmadF8E5_NT, MmadF8E5_TT, MmadF8E4BiasF32_NN,
               MmadF8E4BiasF32_TN, MmadF8E4BiasF32_NT, MmadF8E4BiasF32_TT,
               MmadF8E5BiasF32_NN, MmadF8E5BiasF32_TN, MmadF8E5BiasF32_NT,
               MmadF8E5BiasF32_TT, MmadF8E4BiasF16_NN, MmadF8E5BiasF16_NN,
               MmadF8E4BiasF16_TN, MmadF8E5BiasF16_TN, MmadF16BiasF16_TN,
               MmadBF16BiasF16_TN, MmadF8E4BiasF16_NT, MmadF8E5BiasF16_NT,
               MmadBF16BiasF16_NT, MmadF32BiasF16_NT, MmadF8E4BiasF16_TT,
               MmadF8E5BiasF16_TT, MmadF16BiasF16_TT, MmadBF16BiasF16_TT,
               MmadF32BiasF16_TT]),
          PartialProfile>;

def C310BatchMmadCoverage
    : ArchitectureCapabilityCoverage<
          BatchMmad,
          [BatchF16_NN, BatchF16_TN, BatchF16_NT, BatchF16_TT, BatchBF16_NN,
           BatchBF16_TN, BatchBF16_NT, BatchBF16_TT, BatchF32_NN, BatchF32_TN,
           BatchF32_NT, BatchF32_TT, BatchF32HF32_NN, BatchF32HF32_TN,
           BatchF32HF32_NT, BatchF32HF32_TT],
          PartialProfile>;

// Every row in C310MmadMxCoverage requires k_multiple_of_64.
def C310MmadMxCoverage
    : ArchitectureCapabilityCoverage<
          MmadMx,
          [MxE5E5_NN, MxE5E5_TN, MxE5E5_NT, MxE5E5_TT, MxE5E5_BNN, MxE5E5_BTN,
           MxE5E5_BNT, MxE5E5_BTT, MxE4E4_NN, MxE4E4_TN, MxE4E4_NT, MxE4E4_TT,
           MxE4E4_BNN, MxE4E4_BTN, MxE4E4_BNT, MxE4E4_BTT, MxE4E5_NN, MxE4E5_TN,
           MxE4E5_NT, MxE4E5_TT, MxE4E5_BNN, MxE4E5_BTN, MxE4E5_BNT, MxE4E5_BTT,
           MxE5E4_NN, MxE5E4_TN, MxE5E4_NT, MxE5E4_TT, MxE5E4_BNN, MxE5E4_BTN,
           MxE5E4_BNT, MxE5E4_BTT, MxFmtE5E5_NN, MxFmtE5E5_TN, MxFmtE5E5_NT,
           MxFmtE5E5_TT, MxFmtE5E5_BNN, MxFmtE5E5_BTN, MxFmtE5E5_BNT,
           MxFmtE5E5_BTT, MxFmtE4E4_NN, MxFmtE4E4_TN, MxFmtE4E4_NT,
           MxFmtE4E4_TT, MxFmtE4E4_BNN, MxFmtE4E4_BTN, MxFmtE4E4_BNT,
           MxFmtE4E4_BTT, MxFmtE4E5_NN, MxFmtE4E5_TN, MxFmtE4E5_NT,
           MxFmtE4E5_TT, MxFmtE4E5_BNN, MxFmtE4E5_BTN, MxFmtE4E5_BNT,
           MxFmtE4E5_BTT, MxFmtE5E4_NN, MxFmtE5E4_TN, MxFmtE5E4_NT,
           MxFmtE5E4_TT, MxFmtE5E4_BNN, MxFmtE5E4_BTN, MxFmtE5E4_BNT,
           MxFmtE5E4_BTT, MxFmtF4_NN, MxFmtF4_TN, MxFmtF4_NT, MxFmtF4_TT,
           MxFmtF4_BNN, MxFmtF4_BTN, MxFmtF4_BNT, MxFmtF4_BTT],
          PartialProfile>;

def Dav_C220MmadProfile
    : ArchitectureCapabilityProfile<Dav_C220, [C220MmadCoverage]>;
def Dav_C310MmadProfile
    : ArchitectureCapabilityProfile<Dav_C310, [C310MmadCoverage,
                                               C310BatchMmadCoverage,
                                               C310MmadMxCoverage]>;

// dav-920r1 compiles the same RegBase cube library. AnyValue still requires
// each field to be present, so a missing memory scope stays Unknown.
def Dav920R1AnyMmad
    : SupportCase<Mmad, [AnyValue<MmadAType>, AnyValue<MmadASpace>,
                         AnyValue<MmadBType>, AnyValue<MmadBSpace>,
                         AnyValue<MmadCType>, AnyValue<MmadCSpace>,
                         AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
                         AnyValue<MmadATranspose>, AnyValue<MmadBTranspose>,
                         AnyValue<MmadEnableHF32>, AnyValue<MmadEnableI4>,
                         AnyValue<MmadBiasPresent>]>;
def Dav920R1AnyBatchMmad
    : SupportCase<BatchMmad, [AnyValue<MmadAType>, AnyValue<MmadASpace>,
                              AnyValue<MmadBType>, AnyValue<MmadBSpace>,
                              AnyValue<MmadCType>, AnyValue<MmadCSpace>,
                              AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
                              AnyValue<MmadATranspose>,
                              AnyValue<MmadBTranspose>,
                              AnyValue<MmadEnableHF32>, AnyValue<MmadEnableI4>,
                              AnyValue<MmadBiasPresent>]>;
def Dav920R1AnyMmadMx
    : SupportCase<
          MmadMx, [AnyValue<MmadAType>, AnyValue<MmadASpace>,
                   AnyValue<MmadBType>, AnyValue<MmadBSpace>,
                   AnyValue<MmadMxScaleAType>, AnyValue<MmadMxScaleASpace>,
                   AnyValue<MmadMxScaleBType>, AnyValue<MmadMxScaleBSpace>,
                   AnyValue<MmadCType>, AnyValue<MmadCSpace>,
                   AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
                   AnyValue<MmadATranspose>, AnyValue<MmadBTranspose>,
                   AnyValue<MmadBiasPresent>, AnyValue<MmadMxLhsFormat>,
                   AnyValue<MmadMxRhsFormat>, AnyValue<MmadMxLhsFormatPresent>,
                   AnyValue<MmadMxRhsFormatPresent>,
                   AnyValue<MmadMxKMultipleOf64>]>;

def Dav920R1MmadCoverage
    : ArchitectureCapabilityCoverage<Mmad, [Dav920R1AnyMmad], PartialProfile>;
def Dav920R1BatchMmadCoverage
    : ArchitectureCapabilityCoverage<BatchMmad, [Dav920R1AnyBatchMmad],
                                     PartialProfile>;
def Dav920R1MmadMxCoverage
    : ArchitectureCapabilityCoverage<MmadMx, [Dav920R1AnyMmadMx],
                                     PartialProfile>;
def Dav_920R1MmadProfile
    : ArchitectureCapabilityProfile<Dav_920R1, [Dav920R1MmadCoverage,
                                                Dav920R1BatchMmadCoverage,
                                                Dav920R1MmadMxCoverage]>;

#endif // BISHENGIR_TARGET_NPU_CUBE_MMAD_MMADCAPABILITYPROFILE_TD
