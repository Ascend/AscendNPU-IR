//===- Utils.h - Utilities to support the AVE dialect ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AVE_UTILS_UTILS_H
#define MLIR_DIALECT_AVE_UTILS_UTILS_H

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hivmave {


Value castToDstVectorType(Value src, VectorType dstTy, OpBuilder &b);

/// Get the indices that the given load/store operation is operating on.
ValueRange getIndices(mlir::Operation *op);

/// Check whether the memref with indices is aligned by 32B.
bool isLoadStoreIndexAligned(Value memrefVal,
                             mlir::Operation::operand_range indices);

/// Get lowest element number from PGE pattern.
uint32_t getNumfromPgePattern(VFPgeOp pge);

/// Get PGE/PSET pattern.
FailureOr<PgePatternAttr> getPgePatternAttr(PatternRewriter &rewriter,
                                            int64_t trueShape,
                                            int64_t resultShape);

/// Get store element count or size from mask
Value getElemSizeByStoreMask(Value mask, Type dElemType, Location loc,
                             PatternRewriter &rewriter, bool getCnt = false);

/// Create mask using PGE
Value createMaskByPGE(VectorType vecTy, PatternRewriter &rewriter, Location loc,
                     bool allTrue = true);

/// Find the reuseable mask for the masked op.
Value findReuseableMask(Operation *maskedOp, PatternRewriter &rewriter);

/// Find the reuseable mask for the masked op, if find failed, create one
/// PGE op as mask.
Value findReuseableMaskOrCreateOne(Operation *maskedOp, VectorType vecTy,
                                   PatternRewriter &rewriter);

/// Rewrite loop iter_arg to drop unit dims or to fixed hardware types
template <bool DropUnitDimOnly>
struct ForOpLegalization : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override;
  virtual ~ForOpLegalization() = default;
};

/// Safely extracts a constant integer value from a Value.
/// Supports arith.constant, llvm.constant, casts, and function arguments attributes.
std::optional<int64_t> getConstantIntValue(Value val);

/// Tags constant arguments on callee FuncOps for inter-procedural analysis.
void tagConstantArguments(ModuleOp module);

/// Create broadcast op
Operation *getBroadcastOp(Value scalar, VectorType tileType,
                          PatternRewriter &rewriter, const Location &loc);

/// Sparse vector by interleave
Value sparseByIntlv(Value src, RewriterBase &rewriter, const Location &loc,
                    Attribute attr = mlir::Attribute());

/// Dense vector by deinterleave
Value denseByDIntlv(Value src, RewriterBase &rewriter, const Location &loc,
                    Attribute attr = mlir::Attribute());

/// Checks if a Vector Store/Write operation accesses memory continuously.
/// Logic: LoopStep * MemRefStride == VectorLength
template <typename SourceOp>
bool checkVectorStoreContinuity(SourceOp op) {
  // 1. Locate the outer loop (scf::ForOp)
  auto forOp = op->template getParentOfType<scf::ForOp>();
  if (!forOp) {
    return false;
  }

  // 2. Resolve Loop Step
  auto stepValOpt = getConstantIntValue(forOp.getStep());
  if (!stepValOpt) {
    return false;
  }
  int64_t step = *stepValOpt;

  // 3. Identify Base MemRef
  Value currentMemRef;
  if constexpr (std::is_same_v<SourceOp, vector::TransferWriteOp>) {
    currentMemRef = op.getSource();
  } else {
    currentMemRef = op.getBase(); // Assumes op has getBase()
  }

  Value iv = forOp.getInductionVar();
  int64_t targetStride = -1;
  bool foundIV = false;

  // --- Trace Back Logic ---
  while (auto subview = currentMemRef.getDefiningOp<memref::SubViewOp>()) {
    auto offsets = subview.getMixedOffsets();
    int64_t ivDimIdx = -1;

    for (size_t i = 0; i < offsets.size(); ++i) {
      if (auto val = mlir::dyn_cast<Value>(offsets[i])) {
        // Direct IV usage
        if (val == iv) {
          ivDimIdx = static_cast<int64_t>(i);
          break;
        }
        // Usage via IndexCast
        if (auto cast = val.getDefiningOp<arith::IndexCastOp>()) {
          if (cast.getIn() == iv) {
            ivDimIdx = static_cast<int64_t>(i);
            break;
          }
        }
      }
    }

    if (ivDimIdx != -1) {
      auto srcType = mlir::cast<MemRefType>(subview.getSource().getType());
      SmallVector<int64_t> strides;
      int64_t offset;

      if (failed(getStridesAndOffset(srcType, strides, offset))) {
        return false;
      }

      if (ivDimIdx >= static_cast<int64_t>(strides.size())) {
        return false;
      }

      if (ShapedType::isDynamic(strides[ivDimIdx])) {
        return false;
      }

      targetStride = strides[ivDimIdx];
      foundIV = true;
      break;
    }

    currentMemRef = subview.getSource();
  }

  if (!foundIV) {
    return false;
  }

  // 4. Determine Vector Length
  int64_t vecLen = 1;
  if constexpr (std::is_same_v<SourceOp, vector::TransferWriteOp>) {
    vecLen = op.getVectorType().getNumElements();
  } else {
    Type valType = op.getValueToStore().getType();
    if (auto vecType = mlir::dyn_cast<VectorType>(valType))
      vecLen = vecType.getNumElements();
  }

  // 5. Check Continuity
  bool isContinuous = (step * targetStride) == vecLen;
  return isContinuous;
}

} // namespace hivmave
} // namespace mlir

#endif // MLIR_DIALECT_AVE_UTILS_UTILS_H
