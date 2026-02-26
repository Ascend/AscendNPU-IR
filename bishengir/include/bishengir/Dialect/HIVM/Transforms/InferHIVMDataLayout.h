//===- InferHIVMDataLayout.h ----Infer Data Layout for HIVM Ops -----------===//
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMDATALAYOUT_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMDATALAYOUT_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace hivm {

enum class LayoutConversionKind : uint32_t {
  INVALID = 0,
  DOT_ND_TO_zN,
  DOT_ND_TO_nZ,
  ND_TO_nZ,
  ND_TO_zN,
  nZ_TO_ND,
  zN_TO_ND
};

class DataLayoutInferAndPropagateHelper {
public:
  /// Structure to keep track of the layout information associated to a value.
  struct LayoutInfo {
    bool operator==(const LayoutInfo &other) const {
      return currentLayout == other.currentLayout &&
             targetLayout == other.currentLayout;
    }

    bool operator!=(const LayoutInfo &other) const {
      return !(this->operator==(other));
    }

    bool noLayoutConflict() const { // TODO: Might wanna check the fractal sizes?
      return currentLayout.getDataLayout() == targetLayout.getDataLayout(); }

    DataLayoutAttr currentLayout;
    DataLayoutAttr targetLayout;
  };

  DataLayoutInferAndPropagateHelper(func::FuncOp func) : func_(func) {}

  /// Find the anchor ops and record their current and target data layout.
  void initAnchorLayout();

  /// Recursively propagate current layout to all the users of the anchor ops
  /// until we reach a fix point.
  void propagateLayout();

  /// Try to resolve the data layout conflicts of the anchor ops.
  void resolveConflicts();

  /// Rewrite the IR for the full module.
  void rewrite();

private:
  /// Map the original value to the rewritten one and record the layout info.
  void map(Value oldValue, Value newValue, DataLayoutAttr newLayout);
  /// Return the mapped value for the given data layout. If there is no such
  /// value, create and return a ConvertLayoutOp.
  Value getValueAs(Value value, DataLayoutAttr layout);
  /// Get target and current layout for input value.
  DataLayoutAttr getTargetLayout(Value value);
  DataLayoutAttr getCurrentLayout(Value value);

  /// Propagate layout info the users of the input value.
  SmallVector<Value> propagateDataLayoutToUsers(Value val,
                                                const LayoutInfo &info);
  /// Update the layout info for all the values.
  void updateLayout(ValueRange values, const LayoutInfo &info,
                    SmallVector<Value> &changed);
  /// Update the layout info for the input value. Return true if the layout
  /// is different than before.
  bool updateLayoutIfChanged(Value value, const LayoutInfo &info);

  /// Check whether the layout conversion is supported.
  bool isConversionValid(const LayoutInfo &info);

  /// Compute new shape based on layout conversion pattern.
  FailureOr<SmallVector<Value>>
  computeTargetLayoutShape(SmallVector<Value> currentShape,
                           const LayoutInfo &info, OpBuilder &builder,
                           Location loc);

  /// Compute new offset based on layout conversion pattern.
  FailureOr<SmallVector<Value>>
  computeTargetLayoutOffset(SmallVector<Value> currentOffset,
                            const LayoutInfo &info, OpBuilder &builder,
                            Location loc);

  /// Offset conversion pattern from DOT_{A/B/C} to {zN}.
  FailureOr<SmallVector<Value>>
  computeDOTNDToFractalzNOffset(SmallVector<Value> currentOffset,
                                OpBuilder &builder, Location loc,
                                SmallVector<int64_t> kBlockSizes) const;
  /// Offset conversion pattern from DOT_{A/B/C} to {nZ}.
  FailureOr<SmallVector<Value>>
  computeDOTNDToFractalnZOffset(SmallVector<Value> currentOffset,
                                OpBuilder &builder, Location loc,
                                SmallVector<int64_t> kBlockSizes) const;

  /// Shape conversion pattern from DOT_{A/B/C} to {zN}.
  FailureOr<SmallVector<Value>>
  computeDOTNDToFractalzNShape(SmallVector<Value> currentShape,
                               OpBuilder &builder, Location loc,
                               SmallVector<int64_t> kBlockSizes) const;
  /// Shape conversion pattern from DOT_{A/B/C} to {nZ}.
  FailureOr<SmallVector<Value>>
  computeDOTNDToFractalnZShape(SmallVector<Value> currentShape,
                               OpBuilder &builder, Location loc,
                               SmallVector<int64_t> kBlockSizes) const;

  /// Create ConvertLayoutOp.
  FailureOr<ConvertLayoutOp> createLayoutConversion(Value currentValue,
                                                    const LayoutInfo &info,
                                                    OpBuilder &builder);

  /// Rewrite the IR for a region.
  void rewriteRegion(Region &region);
  /// Rewrite an op based on the layout info collected during propagation.
  Operation *rewriteOp(Operation *op);
  /// Rewrite implementation for different ops.
  Operation *rewriteAllocOp(memref::AllocOp op);
  Operation *rewriteForOp(scf::ForOp op);
  Operation *rewriteSubViewOp(memref::SubViewOp op);
  Operation *rewriteCollapseShapeOp(memref::CollapseShapeOp op);
  Operation *rewriteMemrefCastOp(memref::CastOp op);
  void rewriteCopyOp(mlir::Operation *op);
  /// Try to fold ConvertLayoutOp + CopyOp to HIVM Data Copy Ops with
  /// on-the-fly data layout conversion.
  LogicalResult tryFoldLayoutConversionIntoCopy(Value src, Value dst,
                                                Operation *originalOp,
                                                OpBuilder &builder);

private:
  func::FuncOp func_;
  llvm::SmallDenseSet<Operation *> anchor_ops_;
  /// Mapping from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layout_info_;
  /// Map of the values rewrite based on their layout.
  DenseMap<std::pair<Value, Attribute>, Value> rewriteMappingWithLayout_;
  SetVector<Operation *> opsToDelete_;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMDATALAYOUT_H
