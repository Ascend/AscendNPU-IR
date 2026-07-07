//===- Kinds.h --------------------------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_KINDS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_KINDS_H

namespace mlir {

enum class UnaryKind { Rec, Sqrt, Abs, Not, Exp, Ln, Log2, Floor };
enum class BinaryKind {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  ModUnsigned,
  And,
  Xor,
  Min,
  Max,
  Or,
  MinSigned,
  MaxSigned,
  MinUnsigned,
  MaxUnsigned,
  Powf
};
enum class ShiftKind { Left, RightSigned, RightUnsigned };
enum class TernaryKind { Select };
enum class CompareKind { EQ, NE, LT, GT, GE, LE };
enum class CastRoundKind {
  Default,
  Round,
  Floor,
  Ceil,
  RInt,
  Trunc,
  TruncEnableOverflow,
  TruncWithOverflow,
};
enum class CastSignKind { Preserve, Signed, Unsigned };
enum class CastUnsignedModeKind {
  Preserve,
  SignedToSigned,
  SignedToUnsigned,
  UnsignedToSigned,
  UnsignedToUnsigned,
};
enum class OverflowCastKind {
  Unsupported,
  F32ToI16,
  F32ToI8,
  F16ToI8,
  I64ToI32,
  I64ToI16,
  I64ToI8,
  I32ToI16,
  I32ToI8,
  I16ToI8,
};

enum class TaylerMode { SIN, ATAN };

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_KINDS_H
