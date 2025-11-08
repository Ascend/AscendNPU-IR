//===------------------ Util.h ---- utility of buffer alignment -----------===//
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_ALIGNBUFFER_UTIL_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_ALIGNBUFFER_UTIL_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace hivm {

struct AlignInfo {
  llvm::SmallVector<int32_t> alignDims;
  llvm::SmallVector<int32_t> alignBytes;

  AlignInfo(ArrayRef<int32_t> alignDims_, ArrayRef<int32_t> alignBytes_) {
    alignDims = SmallVector<int32_t>(alignDims_);
    alignBytes = SmallVector<int32_t>(alignBytes_);
  }

  AlignInfo(llvm::SmallVector<int32_t> alignDims_,
            llvm::SmallVector<int32_t> alignBytes_) {
    alignDims = alignDims_;
    alignBytes = alignBytes_;
  }

  AlignInfo(AlignInfo &&other) {
    alignDims = other.alignDims;
    alignBytes = other.alignBytes;
  }

  bool operator==(const AlignInfo &other);
  bool operator!=(const AlignInfo &other);

  void dump();
};

struct OperAlignInfo : public AlignInfo {
  Value operand;

  OperAlignInfo(Value operand_, ArrayRef<int32_t> alignDims_,
                ArrayRef<int32_t> alignBytes_)
      : AlignInfo(alignDims_, alignBytes_) {
    operand = operand_;
  }

  OperAlignInfo(Value operand_, llvm::SmallVector<int32_t> alignDims_,
                llvm::SmallVector<int32_t> alignBytes_)
      : AlignInfo(alignDims_, alignBytes_) {
    operand = operand_;
  }
};

/// Recursively update annotation mark backward through view-like memref ops
/// until allocation.
///
/// Handled memref ops: cast/subview/collapse_shape/expand_shape/reshape/view
///
/// Note: MemRef should not be returned by SCF op. In such case, the control
/// flow should be sinked.
///
/// Example:
///
/// clang-format off
/// ```
/// %c = scf.if %cond -> memref<15x15xf32> {
///   scf.yield %a : memref<15x15xf32>
/// } else {
///   scf.yield %b : memref<15x15xf32>
/// }
/// annotaiton.mark %c
/// hivm.hir.elementwise <...> ins(%c) ...
/// ```
/// clang-format on
///
/// This cannot be handled by hivm enable-storage-align pass.
/// Instead, the code should be sinked by another pass before
/// hivm-mark-stride-align pass like:
///
/// clang-format off
/// ```
/// scf.if %cond {
///   hivm.hir.elementwise <...> ins(%a) ...
///   scf.yield
/// } else {
///   hivm.hir.elementwise <...> ins(%b) ...
///   scf.yield
/// }
/// clang-format on
struct PropagateAlignUpToRootAllocationPattern
    : public OpRewritePattern<mlir::annotation::MarkOp> {
  PropagateAlignUpToRootAllocationPattern(MLIRContext *context,
                                          std::string alignDimAttrName,
                                          std::string alignBytesAttrName)
      : OpRewritePattern<mlir::annotation::MarkOp>(context) {
    alignDimAttrName_ = alignDimAttrName;
    alignBytesAttrName_ = alignBytesAttrName;
  }

  mlir::LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                      PatternRewriter &rewriter) const override;

private:
  std::string alignDimAttrName_;
  std::string alignBytesAttrName_;
};

struct PropagateAlignDownToLeafOperandsPattern
    : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override;
};

void populatePropagateAlignUpToRootAllocationPattern(
    RewritePatternSet &patterns, std::string alignDimAttrName,
    std::string alignBytesAttrName);

std::pair<llvm::SmallVector<int32_t>, llvm::SmallVector<int32_t>>
unionAlignInfo(const ArrayRef<int32_t> &alignDims,
               const ArrayRef<int32_t> &alignBytes,
               const ArrayRef<int32_t> &otherAlignDims,
               const ArrayRef<int32_t> &otherAlignBytes, bool isSorted = true);
std::vector<std::pair<int32_t, int32_t>>
sortAlignInfo(ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes);

// Create mark op with align annotations
std::optional<annotation::MarkOp> createAlignMarkOp(
    OpBuilder &builder, const Location loc, Value markedVal,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str());

OpFoldResult AlignUpOFR(OpBuilder &b, const Location loc, OpFoldResult lenOFR,
                        uint64_t alignUnit);

std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
calculateAlignedShape(OpBuilder &b, const Location loc,
                      const SmallVector<OpFoldResult> &shape,
                      const SmallVector<int> &alignUnits);

LogicalResult replaceAndPropagateMemRefType(RewriterBase &rewriter,
                                            const Location loc, Value from,
                                            Value to);
std::optional<int>
getLastNotUnitDim(const SmallVectorImpl<MemRefType> &memRefTypes,
                  ReassociationIndices reassociations);
std::optional<int32_t> adjustAlignDim(Operation *op, Value operand,
                                      std::optional<int32_t> alignDim);
std::pair<llvm::SmallVector<int32_t>, llvm::SmallVector<int32_t>>
adjustAlignInfo(Operation *op, Value operand,
                const ArrayRef<int32_t> &alignDims,
                const ArrayRef<int32_t> &alignBytes);

void dump(const ArrayRef<int32_t> &alignDims,
          const ArrayRef<int32_t> &alignBytes, StringRef debugType = "");
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_ALIGNBUFFER_UTIL_H
