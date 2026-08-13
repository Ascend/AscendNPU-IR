//===- HFusionImpl.cpp - Implementation of HFusion Dialect Ops --*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <variant>

#if BSPUB_DAVINCI_BISHENGIR
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// Full castTo implementation with all parameters.
// All other castTo overloads delegate to this one.
//===----------------------------------------------------------------------===//

Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::RoundMode roundMode,
                      std::optional<Value> dst,
                      bool enableOverflow, bool enableSaturate,
                      hfusion::TypeFn castIntegerType,
                      hfusion::UnsignedMode unsignedMode) {
  Location loc = src.getLoc();
  if (!isa<TensorType>(src.getType())) {
    assert(src.getType().isIntOrIndexOrFloat());
    bool isUnsignedCast = (hfusion::TypeFn::cast_unsigned == castIntegerType);
    return convertScalarToDtype(builder, loc, src, targetElemType,
                                isUnsignedCast);
  }

  Value targetTensor;
  if (dst.has_value()) {
    targetTensor = dst.value();
  } else {
    targetTensor = utils::createEmptyOpWithTargetElemType(builder, loc, src,
                                                          targetElemType);
  }

  auto roundingAttr = builder.getAttr<hfusion::RoundModeAttr>(roundMode);
  auto unsignedAttr = builder.getAttr<hfusion::UnsignedModeAttr>(unsignedMode);
  auto enableOverflowVal = builder.getBoolAttr(enableOverflow);
  auto enableSaturateVal = builder.getBoolAttr(enableSaturate);
  auto castAttr = builder.getAttr<hfusion::TypeFnAttr>(castIntegerType);
  auto vcastOp = builder.create<hfusion::CastOp>(
      loc, SmallVector<Type>{targetTensor.getType()}, src, targetTensor,
      roundingAttr, enableOverflowVal, enableSaturateVal, castAttr,
      unsignedAttr);
  // TODO: Temporarily disable default-valued attributes so the printed IR stays backward-compatible.
  if (!enableSaturate)
    vcastOp->removeAttr("enable_saturate");
  if (unsignedMode == hfusion::UnsignedMode::SI2SI)
    vcastOp->removeAttr("unsigned_mode");
  return vcastOp->getResult(0);
}

//===----------------------------------------------------------------------===//
// Convenience overloads that delegate to the full implementation above.
//===----------------------------------------------------------------------===//

// 7-param overload (jy original) — delegates to 9-param with defaults.
Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::RoundMode roundMode, std::optional<Value> dst,
                      bool enableOverflow, hfusion::TypeFn castIntegerType) {
  return hfusion::castTo(builder, src, targetElemType, roundMode, dst,
                         enableOverflow,
                         /*enableSaturate=*/false, castIntegerType,
                         hfusion::UnsignedMode::SI2SI);
}

// 5-param overload (TypeFn + UnsignedMode) — selects rounding automatically.
Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::TypeFn castIntegerType,
                      hfusion::UnsignedMode unsignedMode) {
  Type srcElemType = getElementTypeOrSelf(src.getType());
  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                       targetElemType);
  return hfusion::castTo(builder, src, targetElemType, rounding,
                         /*dst=*/std::nullopt,
                         /*enableOverflow=*/true,
                         /*enableSaturate=*/false,
                         castIntegerType, unsignedMode);
}

// 4-param overload (TypeFn only) — defaults unsignedMode to SI2SI.
Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::TypeFn castIntegerType) {
  hfusion::UnsignedMode unsignedMode = hfusion::UnsignedMode::SI2SI;
  return hfusion::castTo(builder, src, targetElemType,
                         castIntegerType, unsignedMode);
}

// 5-param overload (RoundMode + TypeFn) — jy convenience overload.
Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::RoundMode roundMode,
                      hfusion::TypeFn castIntegerType) {
  return hfusion::castTo(builder, src, targetElemType, roundMode, std::nullopt,
                         /*enableOverflow=*/true, castIntegerType);
}

// 5-param overload (bool enableOverflow + TypeFn) — jy convenience overload.
Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      bool enableOverflow, hfusion::TypeFn castIntegerType) {
  Type srcElemType = getElementTypeOrSelf(src.getType());
  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                       targetElemType);
  return hfusion::castTo(builder, src, targetElemType, rounding, std::nullopt,
                         /*enableOverflow=*/enableOverflow, castIntegerType);
}
