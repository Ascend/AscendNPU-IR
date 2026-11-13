//===- MmadMxCapabilityProfile.td - MmadMx rows -----------*- tablegen -*-===//
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
// Rows from bishengir/lib/Template/lib/RegBase/Cube/LocalMmadMx.cpp.
// C220 has no mmadmx library. This file does not define an architecture
// profile; MmadCapabilityProfile.td attaches C310MmadMxCoverage.

#ifndef BISHENGIR_TARGET_NPU_CUBE_MMADMX_MMADMXCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_CUBE_MMADMX_MMADMXCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/Cube/MmadMx/MmadMxCapability.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

class MmadMxSpaces<list<CapabilityConstraint> extra> {
  list<CapabilityConstraint> Constraints =
      !listconcat(extra, [EnumEquals<MmadASpace, HIVM_AddressSpace_L1>,
                          EnumEquals<MmadBSpace, HIVM_AddressSpace_L1>,
                          EnumEquals<MmadMxScaleASpace, HIVM_AddressSpace_L1>,
                          EnumEquals<MmadMxScaleBSpace, HIVM_AddressSpace_L1>,
                          EnumEquals<MmadCSpace, HIVM_AddressSpace_L0C>,
                          BoolValueConstraint<MmadMxKMultipleOf64, 1>]);
}

class MmadMxNoBias<Type a, Type b, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [a, b, F32, I8, I8]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [a, b, F32, UI8, UI8]>,
        AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
        AnyValue<MmadMxLhsFormat>, AnyValue<MmadMxRhsFormat>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 0>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 0>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 0>]>.Constraints>;

class MmadMxF32Bias<Type a, Type b, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [a, b, F32, I8, I8, F32]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [a, b, F32, UI8, UI8, F32]>,
        EnumEquals<MmadBiasSpace, HIVM_AddressSpace_L1>,
        AnyValue<MmadMxLhsFormat>, AnyValue<MmadMxRhsFormat>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 0>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 0>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 1>]>.Constraints>;

class MmadMxFormatNoBias<I32EnumAttrCase lhs, I32EnumAttrCase rhs, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [I8, I8, F32, I8, I8]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [I8, I8, F32, UI8, UI8]>,
        AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
        EnumEquals<MmadMxLhsFormat, lhs>, EnumEquals<MmadMxRhsFormat, rhs>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 1>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 1>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 0>]>.Constraints>;

// fp8 element types with lhs/rhs format attrs (hfusion.matmul_mx from Triton).
class MmadMxFormatNoBiasFp8<Type a, Type b, I32EnumAttrCase lhs,
                             I32EnumAttrCase rhs, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [I8, I8, F32, I8, I8]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [I8, I8, F32, UI8, UI8]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [a, b, F32, I8, I8]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType], [a, b, F32, UI8, UI8]>,
        AnyValue<MmadBiasType>, AnyValue<MmadBiasSpace>,
        EnumEquals<MmadMxLhsFormat, lhs>, EnumEquals<MmadMxRhsFormat, rhs>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 1>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 1>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 0>]>.Constraints>;

class MmadMxFormatBias<I32EnumAttrCase lhs, I32EnumAttrCase rhs, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [I8, I8, F32, I8, I8, F32]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [I8, I8, F32, UI8, UI8, F32]>,
        EnumEquals<MmadBiasSpace, HIVM_AddressSpace_L1>,
        EnumEquals<MmadMxLhsFormat, lhs>, EnumEquals<MmadMxRhsFormat, rhs>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 1>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 1>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 1>]>.Constraints>;

class MmadMxFormatBiasFp8<Type a, Type b, I32EnumAttrCase lhs,
                          I32EnumAttrCase rhs, bit ta, bit tb>
    : SupportCase<MmadMx, MmadMxSpaces<[
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [I8, I8, F32, I8, I8, F32]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [I8, I8, F32, UI8, UI8, F32]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [a, b, F32, I8, I8, F32]>,
        TypeSignature<[MmadAType, MmadBType, MmadCType, MmadMxScaleAType,
                       MmadMxScaleBType, MmadBiasType],
                      [a, b, F32, UI8, UI8, F32]>,
        EnumEquals<MmadBiasSpace, HIVM_AddressSpace_L1>,
        EnumEquals<MmadMxLhsFormat, lhs>, EnumEquals<MmadMxRhsFormat, rhs>,
        BoolValueConstraint<MmadMxLhsFormatPresent, 1>,
        BoolValueConstraint<MmadMxRhsFormatPresent, 1>,
        BoolValueConstraint<MmadATranspose, ta>,
        BoolValueConstraint<MmadBTranspose, tb>,
        BoolValueConstraint<MmadBiasPresent, 1>]>.Constraints>;

multiclass MmadMxTypedPair<Type a, Type b> {
  def _NN : MmadMxNoBias<a, b, 0, 0>;
  def _TN : MmadMxNoBias<a, b, 1, 0>;
  def _NT : MmadMxNoBias<a, b, 0, 1>;
  def _TT : MmadMxNoBias<a, b, 1, 1>;
  def _BNN : MmadMxF32Bias<a, b, 0, 0>;
  def _BTN : MmadMxF32Bias<a, b, 1, 0>;
  def _BNT : MmadMxF32Bias<a, b, 0, 1>;
  def _BTT : MmadMxF32Bias<a, b, 1, 1>;
}

multiclass MmadMxFormatPair<I32EnumAttrCase lhs, I32EnumAttrCase rhs> {
  def _NN : MmadMxFormatNoBias<lhs, rhs, 0, 0>;
  def _TN : MmadMxFormatNoBias<lhs, rhs, 1, 0>;
  def _NT : MmadMxFormatNoBias<lhs, rhs, 0, 1>;
  def _TT : MmadMxFormatNoBias<lhs, rhs, 1, 1>;
  def _BNN : MmadMxFormatBias<lhs, rhs, 0, 0>;
  def _BTN : MmadMxFormatBias<lhs, rhs, 1, 0>;
  def _BNT : MmadMxFormatBias<lhs, rhs, 0, 1>;
  def _BTT : MmadMxFormatBias<lhs, rhs, 1, 1>;
}

multiclass MmadMxFormatPairFp8<Type a, Type b, I32EnumAttrCase lhs,
                               I32EnumAttrCase rhs> {
  def _NN : MmadMxFormatNoBiasFp8<a, b, lhs, rhs, 0, 0>;
  def _TN : MmadMxFormatNoBiasFp8<a, b, lhs, rhs, 1, 0>;
  def _NT : MmadMxFormatNoBiasFp8<a, b, lhs, rhs, 0, 1>;
  def _TT : MmadMxFormatNoBiasFp8<a, b, lhs, rhs, 1, 1>;
  def _BNN : MmadMxFormatBiasFp8<a, b, lhs, rhs, 0, 0>;
  def _BTN : MmadMxFormatBiasFp8<a, b, lhs, rhs, 1, 0>;
  def _BNT : MmadMxFormatBiasFp8<a, b, lhs, rhs, 0, 1>;
  def _BTT : MmadMxFormatBiasFp8<a, b, lhs, rhs, 1, 1>;
}

defm MxE5E5 : MmadMxTypedPair<F8E5M2, F8E5M2>;
defm MxE4E4 : MmadMxTypedPair<F8E4M3FN, F8E4M3FN>;
defm MxE4E5 : MmadMxTypedPair<F8E4M3FN, F8E5M2>;
defm MxE5E4 : MmadMxTypedPair<F8E5M2, F8E4M3FN>;
defm MxFmtE5E5 : MmadMxFormatPairFp8<F8E5M2, F8E5M2, HIVM_MatmulFormat_FP8E5M2,
                                     HIVM_MatmulFormat_FP8E5M2>;
defm MxFmtE4E4
    : MmadMxFormatPairFp8<F8E4M3FN, F8E4M3FN, HIVM_MatmulFormat_FP8E4M3,
                          HIVM_MatmulFormat_FP8E4M3>;
defm MxFmtE4E5
    : MmadMxFormatPairFp8<F8E4M3FN, F8E5M2, HIVM_MatmulFormat_FP8E4M3,
                          HIVM_MatmulFormat_FP8E5M2>;
defm MxFmtE5E4
    : MmadMxFormatPairFp8<F8E5M2, F8E4M3FN, HIVM_MatmulFormat_FP8E5M2,
                          HIVM_MatmulFormat_FP8E4M3>;
defm MxFmtF4
    : MmadMxFormatPair<HIVM_MatmulFormat_FP4E2M1, HIVM_MatmulFormat_FP4E2M1>;

#endif // BISHENGIR_TARGET_NPU_CUBE_MMADMX_MMADMXCAPABILITYPROFILE_TD
