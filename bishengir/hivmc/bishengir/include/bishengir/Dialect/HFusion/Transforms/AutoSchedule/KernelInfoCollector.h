//===- KernelInfoCollector.h - Def. for Kernel Info Collector ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFOCOLLECTOR_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFOCOLLECTOR_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"

namespace mlir {
struct AutoScheduleOptions;

namespace func {
class FuncOp;
} // namespace func

namespace hfusion {

//===----------------------------------------------------------------------===//
// KernelInfoCollector
//===----------------------------------------------------------------------===//

class KernelInfoCollector {
public:
  explicit KernelInfoCollector(KernelInfo *info) : info_(info) {}
  explicit KernelInfoCollector(KernelInfo *info, AutoScheduleOptions options)
      : info_(info), scheduleOptions_(options) {}
  virtual ~KernelInfoCollector() = default;

  /// Main entry to collect information.
  /// Will call \c visitFuncImpl and \c postVisitFuncImpl.
  LogicalResult run();

protected:
  /// Safe functions to get kernel info pointer.
  KernelInfo *getInfo();
  KernelInfo *getInfo() const;

  AutoScheduleOptions getScheduleOptions() const { return scheduleOptions_; }

  /// Helper function to get bit vector representing anchor axis for value
  BitVector getAnchorCommonAxis(Value value) const;

  /// Visit function by traversing the operations in pre-order, with callbacks
  /// to various types of operations.
  LogicalResult visitFuncImpl(func::FuncOp f);

  /// Implementation of post processing logic. Can be overwritten accordingly by
  /// derived classes.
  virtual LogicalResult postVisitFuncImpl(func::FuncOp f);

private:
  /// Various visitors call by \c visitFuncImpl. Should not be called directly.
  LogicalResult visitLinalgOp(Operation *op);
  LogicalResult visitTensorExtractOp(Operation *op);
  LogicalResult visitTensorPadOp(Operation *op);
  LogicalResult visitTensorConcatOp(Operation *op);
  LogicalResult visitTensorInsertSliceOp(Operation *op);
  LogicalResult visitTensorExtractSliceOp(Operation *op);
  LogicalResult visitDeinterleaveOp(Operation *op);
  LogicalResult visitInterleaveOp(Operation *op);

  /// Actual implementation of various visitors. Can be overwritten accordingly
  /// by derived classes.
  virtual LogicalResult visitLinalgOpImpl(Operation *op) { return success(); };
  virtual LogicalResult visitTensorExtractOpImpl(Operation *op) {
    return success();
  };
  virtual LogicalResult visitTensorPadOpImpl(Operation *op) {
    return success();
  };
  virtual LogicalResult visitTensorConcatOpImpl(Operation *op) {
    return success();
  };
  virtual LogicalResult visitTensorInsertSliceOpImpl(Operation *op) {
    return success();
  };
  virtual LogicalResult visitTensorExtractSliceOpImpl(Operation *op) {
    return success();
  };
  virtual LogicalResult visitDeinterleaveOpImpl(Operation *op) {
    return success();
  }
  virtual LogicalResult visitInterleaveOpImpl(Operation *op) {
    return success();
  }

  /// Analyze and count the maximum number of buffers that must co-exists on
  /// local memory at the same time. The number of max buffer is in terms of
  /// the tensor with the smallest type.
  ///
  /// The analysis is based on
  ///   a) Whether the operation support in-place reuse
  ///   b) Whether the operations' operand will enable multi-buffer optimization
  ///   c) Whether the operation requires additional buffers to store
  ///      intermediate results
  LogicalResult
  countMaxBuffer(const utils::BufferAnalysisOptions &options = {});

private:
  /// Pointer to kernel info.
  KernelInfo *info_{nullptr};
  /// Auto schedule options.
  AutoScheduleOptions scheduleOptions_;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFOCOLLECTOR_H
