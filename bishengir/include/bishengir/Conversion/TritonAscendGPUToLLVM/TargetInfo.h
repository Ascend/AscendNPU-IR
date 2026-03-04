//===--TargetInfo.h - TritonAscendGPU Target Info ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_TARGETINFO_H
#define TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_TARGETINFO_H

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ascend {

class TargetInfo : public mlir::triton::NVIDIA::TargetInfo {
public:
  TargetInfo() : NVIDIA::TargetInfo(70, 73) {}

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  int getSharedAddressSpace() const override {
    return (int)ascend_dpx::AscendDPXAddressSpace::SHARED_MEM;
  }

  bool isCuda() const override { return false; }

  bool supportMaximumMinimum() const override { return false; }

  Value ballot(RewriterBase &, Location, Type, Value) const override {
    llvm_unreachable("not implemented");
  }

  void barrier(Location, RewriterBase &, bool = false) const override;

  bool supportStMatrix() const override { return false; }
  bool supportLdMatrix() const override { return false; }

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;

  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type loadTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  Value shuffleXor(RewriterBase &, Location, Value, int) const override;

  Value shuffleUp(RewriterBase &, Location, Value, int) const override;

  Value shuffleIdx(RewriterBase &, Location, Value, int) const override;

  Value shuffleIdx(RewriterBase &, Location, Value, Value) const override;

  Value permute(RewriterBase &, Location, Value, Value, Value) const override {
    llvm_unreachable("not implemented");
  }

  Value programId(RewriterBase &, Location, ModuleOp,
                  ProgramIDDim) const override {
    llvm_unreachable("not implemented");
  }

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override {
    return false;
  }

  std::string getMulhiFuncName(Type) const override {
    llvm_unreachable("not implemented");
  }

  void printf(RewriterBase &, Value, int, ValueRange,
              ArrayRef<bool> = {}) const override {
    llvm_unreachable("not implemented");
  }

  void printf(RewriterBase &, StringRef, ValueRange,
              ArrayRef<bool> = {}) const override {}

  int getAddressSpace(Attribute) const override { return 0; }

  bool supportVectorizedAtomics() const override { return false; }
};

} // namespace mlir::triton::ascend

#endif
