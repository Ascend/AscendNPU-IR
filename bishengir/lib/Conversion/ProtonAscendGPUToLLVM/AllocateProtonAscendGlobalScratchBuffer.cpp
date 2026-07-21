//===- AllocateProtonAscendGlobalScratchBuffer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ascend-specific version of proton global scratch buffer allocation.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ProtonAscendGPUToLLVM/ProtonAscendGPUToLLVM.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace bishengir::triton::proton::gpu {

#define GEN_PASS_DEF_ALLOCATEPROTONASCENDGLOBALSCRATCHBUFFERPASS
#include "bishengir/Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

struct AllocateProtonAscendGlobalScratchBufferPass
    : public impl::AllocateProtonAscendGlobalScratchBufferPassBase<
          AllocateProtonAscendGlobalScratchBufferPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // If TritonAscendGPUToLLVM hasn't run yet (tt::FuncOp still present),
    // skip this pass.
    bool hasTTFunc = false;
    mod.walk([&](mlir::triton::FuncOp op) { hasTTFunc = true; });
    if (hasTTFunc)
      return;

    int numFuncOps = 0;
    FunctionOpInterface func;
    mod.walk([&](FunctionOpInterface op) {
      // Use NVVM kernel attribute to identify kernel functions.
      // Some kernel names also contain "__", so the upstream name-based
      // heuristic doesn't work for Ascend.
      if (op->hasAttr(NVVM::NVVMDialect::getKernelFuncAttrName())) {
        numFuncOps += 1;
        func = op;
      }
    });

    assert(numFuncOps == 1);

    int32_t cumulativeMemorySize = 0; // bytes
    std::vector<uint32_t> alignments;

    func.walk([&](mlir::triton::proton::gpu::GlobalScratchAllocOp op) {
      int offset = llvm::alignTo(cumulativeMemorySize,
                                 mlir::triton::proton::gpu::getBytesPerClockEntry());
      op->setAttr("offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      cumulativeMemorySize += op.getNbytes();
      alignments.push_back(op.getAlignment());
    });
    if (alignments.empty())
      return;

    bool allAlignmentsEqual = std::equal(alignments.begin() + 1,
                                         alignments.end(), alignments.begin());
    assert(allAlignmentsEqual &&
           "all global scratch buffer alignment values must be the same");
    mod->setAttr("ttg.profile_scratch_memory_size",
                 builder.getI32IntegerAttr(cumulativeMemorySize));
    mod->setAttr("ttg.profile_scratch_memory_alignment",
                 builder.getI32IntegerAttr(alignments.front()));
  }
};

std::unique_ptr<mlir::Pass> createAllocateProtonAscendGlobalScratchBufferPass() {
  return std::make_unique<AllocateProtonAscendGlobalScratchBufferPass>();
}

} // namespace bishengir::triton::proton::gpu
