//===- HFusionImpl.cpp - Implementation of HFusion Dialect Ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#if BSPRIV_DAVINCI_BISHENGIR
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif

using namespace mlir;
using namespace mlir::hfusion;

Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::RoundMode roundMode,
                      std::optional<Value> dst,
                      bool enableOverflow, bool enableSaturate,
		      hfusion::TypeFn castIntegerType) {
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
  auto enableOverflowVal = builder.getBoolAttr(enableOverflow);
  auto enableSaturateVal = builder.getBoolAttr(enableSaturate);
  auto castAttr = builder.getAttr<hfusion::TypeFnAttr>(castIntegerType);
  auto vcastOp = builder.create<hfusion::CastOp>(
      loc, SmallVector<Type>{targetTensor.getType()}, src, targetTensor,
      roundingAttr, enableOverflowVal, enableSaturateVal, castAttr);
  return vcastOp->getResult(0);
}

Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::TypeFn castIntegerType) {
  Type srcElemType = getElementTypeOrSelf(src.getType());
  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                       targetElemType);
  return hfusion::castTo(builder, src, targetElemType, rounding, std::nullopt,
                         /*enableOverflow=*/true, false, castIntegerType);
}
