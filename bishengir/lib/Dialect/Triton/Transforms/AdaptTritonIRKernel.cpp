//===- AdaptTritonIRKernel.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to adapt TritonIR Kernel for BiShengIR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/IR/TritonExtension.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_ADAPTTRITONIRKERNEL
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

constexpr uint64_t kGridSize = 3;

class AdaptTritonIRKernelPass
    : public impl::AdaptTritonIRKernelBase<AdaptTritonIRKernelPass> {
public:
  using impl::AdaptTritonIRKernelBase<
      AdaptTritonIRKernelPass>::AdaptTritonIRKernelBase;
  void runOnOperation() override {
    mlir::triton::FuncOp ttFunc = getOperation();
    if (!ttFunc.isPublic())
      return;

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    ModuleOp moduleOp = ttFunc->getParentOfType<ModuleOp>();
    if (moduleOp && superBlockBarrier) {
      moduleOp->setAttr(bishengir::AttrSuperBlockBarrier,
                        builder.getBoolAttr(superBlockBarrier));
    }

    Location loc = UnknownLoc::get(ctx);
    NamedAttribute divisibility(builder.getStringAttr("tt.divisibility"),
                                builder.getI32IntegerAttr(1));
    for (uint64_t gridDim = 0; gridDim < kGridSize; gridDim++) {
      std::optional<mlir::gpu::MappingId> maybeMappingId =
          mlir::gpu::symbolizeMappingId(gridDim);
      if (!maybeMappingId.has_value())
        break;

      ttFunc.insertArgument(
          ttFunc.getNumArguments(), builder.getI32Type(),
          DictionaryAttr::get(
              ctx,
              SmallVector<NamedAttribute>{
                  divisibility,
                  NamedAttribute(builder.getStringAttr(
                                     mlir::gpu::GPUBlockMappingAttr::name),
                                 mlir::gpu::GPUBlockMappingAttr::get(
                                     ctx, *maybeMappingId))}),
          loc);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
createAdaptTritonIRKernelPass(const AdaptTritonIRKernelOptions &options) {
  return std::make_unique<AdaptTritonIRKernelPass>(options);
}

} // namespace triton
} // namespace bishengir