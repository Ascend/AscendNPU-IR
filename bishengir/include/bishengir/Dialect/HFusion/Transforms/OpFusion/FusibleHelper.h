//===- FusibleHelper.h --------------------------------------- --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Debug.h"

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEHELPER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEHELPER_H

namespace mlir {
namespace hfusion {

/// Return whether the input function can be classified to one of the
/// supported fusion kind.
LogicalResult canFuse(func::FuncOp func);

/// Infer the `hfusion::FusionKind` for the input function.
///
/// \note The least inclusive fusion kind is returned if multiple fusion kinds
///       are suitable.
FusionKind inferFuncFusionKind(func::FuncOp func);

namespace opfusion {

enum class OpPattern : uint8_t {
  kAuxiliary = 0,
  kBuffer = 1,
  kOpaque = 2,
  kZeroRankElemwise = 3,
  kReshape = 50,
  kLoadStore = 51,
  kInsertSlice = 52,
  kMidFusionAuxiliary = 54,

  // Put it >= 100 if you want it to be single outlinable
  kElementWise = 100,
  kLastAxisReduce = 101,
  kLastAxisBroadcast = 102,
  kOtherReduce = 103,
  kOtherBroadcast = 104,
  kMatmul = 105,
  kAllReduce = 106,
  kAllGather = 107,
  kReduceScatter = 108,
  kMidFusionImportantAux = 109,
  kInterleave = 110,
  kTranspose = 111,
  kExtractSlice = 112,
};

enum class TypePattern : uint8_t {
  kPureElementWise = 0,
  kPureMatmul = 1,
  kSuffixElementWise = 2,
  kOpaque = 3,
};

class FusibleHelper {
public:
  explicit FusibleHelper(FusionKind fusionKind, bool bufferToOut = true,
                         int32_t maxHorizontalFusionSize = -1);

  bool moveOutToParam() const;
  int32_t maxHorizontalFusion() const;
  static bool isPossibleCountingAux(Operation *defOp);
  static bool isBuffer(Operation *op);
  static bool isAuxiliary(Operation *op);
  static bool isZeroRankElemwise(Operation *op);
  bool isFusible(Operation *a, Operation *b) const;
  uint8_t obtainType(Operation *op) const;
  uint8_t adjustType(const uint8_t &typeA, const uint8_t &typeB,
                     bool isHorizontal) const;
  uint8_t adjustType(const uint8_t &typeA, const uint8_t &typeB) const;
  uint8_t adjustTypeHorizontal(const uint8_t &typeA,
                               const uint8_t &typeB) const;
  bool isRestrictedByNodeType(const uint8_t &typeA, const uint8_t &typeB,
                              bool isHorizontal) const;
  bool isRestrictedByNodeType(const uint8_t &typeA, const uint8_t &typeB) const;
  bool hasMatmulTypePattern(const uint8_t &typePattern) const;
  bool isRestrictedByDynamicShape(Operation *op, bool horizontal = false) const;
  int obtainLastReduceRank(Operation *op) const;
  int obtainReduceDim(Operation *op) const;
  bool isShapePivot(Operation *op) const;
  bool isRestrictedByReduceRank(const int &a, const int &b) const;
  bool isRestrictedByReduceDim(const int &a, const int &b) const;
  bool schedulable(Operation *op) const;
  static OpPattern getOpPattern(Operation *op);
  static bool isImportantPattern(Operation *op);
  static bool isSingleOutlinable(Operation *op);
  static FusionKind getSingleFusionKind(Operation *op);
  static bool isShallowFusion(FusionKind fusionKind);
  FusionKind getFusionKind() const;

private:
  FusionKind fusionKind_;
  bool moveOutToParam_;
  int32_t maxHorizontalFusion_;

  bool isFusible(const OpPattern &patternA, const OpPattern &patternB) const;
  bool isPureElemwiseFusible(const OpPattern &patternA,
                             const OpPattern &patternB) const;
  bool isLastAxisPBRFusible(const OpPattern &patternA,
                            const OpPattern &patternB) const;
  bool isAnyPBRFusible(const OpPattern &patternA,
                       const OpPattern &patternB) const;
  bool isShallowCVFusible(const OpPattern &patternA,
                          const OpPattern &patternB) const;
  bool isShallowVVFusible(const OpPattern &patternA,
                          const OpPattern &patternB) const;
  bool isMixCVFusible(const OpPattern &patternA,
                      const OpPattern &patternB) const;
  bool isMixC2Fusible(const OpPattern &patternA,
                      const OpPattern &patternB) const;
  bool isAnyPBFusible(const OpPattern &patternA,
                      const OpPattern &patternB) const;

  static size_t getMaxRank(const SmallVector<Value> &operands);
  static bool isImportantPattern(const OpPattern &pattern);
};
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEHELPER_H
