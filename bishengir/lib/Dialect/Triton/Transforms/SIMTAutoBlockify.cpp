//===- SIMTAutoBlockify.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass blockify Triton SIMT kernels at TTIR level.
//
//   logical:         [0 ................................ logicalGridSize)
//   physical launch: [0 .... physicalCoreNum)
//
//   chunk = ceildiv(logicalGridSize, physicalCoreNum)
//   for linear in [hw_block_id * chunk,
//                  min((hw_block_id + 1) * chunk, logicalGridSize)):
//     (pidX, pidY, pidZ) = unflatten(linear, gridX, gridY)
//
// The hardware (linear) block id is sourced from `gpu.linear_block_id`, which
// lowers to `ascend_dpx::BlockIdxOp` and ultimately to the linear hardware
// block-id intrinsic. `tt.get_program_id x/y/z` are 3D indices and would NOT
// give the linear core id.
//
// The pass rewrites all `tt.get_program_id x/y/z` ops in the kernel.
// Grid metadata (`tt.get_num_programs`) stays intact for later lowering.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_SIMTAUTOBLOCKIFY
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace bishengir {
namespace triton {

namespace {

struct GridArgs {
  BlockArgument x;
  BlockArgument y;
  BlockArgument z;
};

struct LogicalProgramIds {
  Value x;
  Value y;
  Value z;
};

static FailureOr<BlockArgument> getGridArg(mlir::triton::FuncOp funcOp,
                                           gpu::MappingId mapping) {
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    auto dictAttr = funcOp.getArgAttrDict(idx);
    if (!dictAttr)
      continue;
    auto blockAttr = dyn_cast_or_null<gpu::GPUBlockMappingAttr>(
        dictAttr.get(gpu::GPUBlockMappingAttr::name));
    if (blockAttr && blockAttr.getBlock() == mapping)
      return arg;
  }
  return failure();
}

static FailureOr<int64_t> getPhysicalBlockNum(mlir::triton::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
  if (!maybeSpecInterface.has_value())
    return failure();

  auto specInterface = maybeSpecInterface.value();
  auto attr = specInterface.getSpecForIdentifierEnum(
      hacc::DeviceSpec::VECTOR_CORE_COUNT);
  auto intAttr = dyn_cast<IntegerAttr>(attr.getValue());
  if (!intAttr)
    return failure();
  return intAttr.getValue().getSExtValue();
}

static LogicalProgramIds buildLogicalProgramIds(OpBuilder &builder,
                                                Location loc,
                                                Value linearProgramId,
                                                GridArgs grid) {
  Value divX = builder.create<arith::DivUIOp>(loc, linearProgramId, grid.x);
  return {
      builder.create<arith::RemUIOp>(loc, linearProgramId, grid.x),
      builder.create<arith::RemUIOp>(loc, divX, grid.y),
      builder.create<arith::DivUIOp>(loc, divX, grid.y),
  };
}

struct SIMTAutoBlockifyPass
    : public impl::SIMTAutoBlockifyBase<SIMTAutoBlockifyPass> {
  void runOnOperation() override {
    mlir::triton::FuncOp ttFunc = getOperation();
    // FIXME: Canonicalize the checking on entry functions.
    // Skip non-kernel functions.
    if (!ttFunc.isPublic() || !ttFunc.getResultTypes().empty())
      return;

    // Collect `tt.program_id`.
    SmallVector<mlir::triton::GetProgramIdOp> programIdOps;
    ttFunc.walk([&programIdOps](Operation *op) {
      if (auto getProgramId = dyn_cast<mlir::triton::GetProgramIdOp>(op))
        programIdOps.push_back(getProgramId);
    });
    // Skip if no program id is used.
    if (programIdOps.empty())
      return;

    // Collect the underlying hardware capability.
    auto maybePhysicalBlockNum = getPhysicalBlockNum(ttFunc);
    if (failed(maybePhysicalBlockNum)) {
      ttFunc.emitError(
          "failed to infer physical vector core count for SIMT auto blockify");
      return signalPassFailure();
    }

    // Collect grid dims.
    auto gridX = getGridArg(ttFunc, gpu::MappingId::DimX);
    auto gridY = getGridArg(ttFunc, gpu::MappingId::DimY);
    auto gridZ = getGridArg(ttFunc, gpu::MappingId::DimZ);
    if (failed(gridX) || failed(gridY) || failed(gridZ)) {
      ttFunc.emitError("failed to get grid args for SIMT auto blockify");
      return signalPassFailure();
    }
    GridArgs grid = {gridX.value(), gridY.value(), gridZ.value()};

    bool originalFuncHasOneBlock = ttFunc.getBody().hasOneBlock();

    Block *entryBlock = &ttFunc.getBody().front();
    // Split the entry block so that
    // - 'entryBlock' is the new entry block where the prologue code is
    //   populated, and
    // - 'bodyBlock' is the beginning of the original function body.
    auto *bodyBlock = entryBlock->splitBlock(entryBlock->begin());

    // Generate the prologue to prepare the loop over blocks.
    Location loc = ttFunc.getLoc();
    OpBuilder builder(entryBlock, entryBlock->begin());

    Value yz = builder.create<arith::MulIOp>(loc, grid.y, grid.z);
    Value logicalBlockNums = builder.create<arith::MulIOp>(loc, grid.x, yz);
    Value linearHwBlockIdx = builder.create<gpu::LinearBlockIdOp>(loc);
    Value blockIdx = builder.create<arith::IndexCastOp>(
        loc, builder.getI32Type(), linearHwBlockIdx);
    Value physicalBlockNum =
        builder.create<arith::ConstantIntOp>(loc, *maybePhysicalBlockNum, 32);
    Value chunk = builder.create<arith::CeilDivUIOp>(loc, logicalBlockNums,
                                                     physicalBlockNum);
    Value lowerBound = builder.create<arith::MulIOp>(loc, blockIdx, chunk);
    Value end = builder.create<arith::AddIOp>(loc, lowerBound, chunk);
    Value upperBound =
        builder.create<arith::MinUIOp>(loc, end, logicalBlockNums);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

    // Create the loop over blocks.
    // scf.for LinearIdx = blockIdx * chunk
    //          to min((blockIdx + 1) * chunk, logicalBlocks), step 1
    auto forOp = builder.create<scf::ForOp>(loc, lowerBound, upperBound, one);
    // Create the final `tt.return` following that loop.
    builder.create<mlir::triton::ReturnOp>(loc);

    if (originalFuncHasOneBlock) {
      assert(std::next(Region::iterator(bodyBlock)) == ttFunc.getBody().end() &&
             "If there's only one block, the original block must be the last "
             "one!");
      // Now transfer original operations into the new body block.
      auto *newBodyBlock = forOp.getBody();
      newBodyBlock->getOperations().splice(newBodyBlock->begin(),
                                           bodyBlock->getOperations());
      bodyBlock->erase();
    } else {
      // Create 'scf.execute_region' op to hold the function body with multiple
      // blocks.
      OpBuilder b(forOp.getBody(), forOp.getBody()->begin());
      auto execRegionOp =
          b.create<scf::ExecuteRegionOp>(forOp.getLoc(), TypeRange{});
      auto &newBodyRegion = execRegionOp.getRegion();
      // Now transfer original blocks into the new body region.
      auto &bodyRegion = ttFunc.getBody();
      newBodyRegion.getBlocks().splice(
          newBodyRegion.begin(), bodyRegion.getBlocks(),
          Region::iterator(bodyBlock), bodyRegion.end());
    }

    // Start the replacement of `tt.program_id` and `tt.return` in that loop
    // body.

    builder.setInsertionPointToStart(forOp.getBody());
    Value linearProgramId = forOp.getInductionVar();

    builder.setInsertionPointAfterValue(linearProgramId);
    LogicalProgramIds logicalPids =
        buildLogicalProgramIds(builder, loc, linearProgramId, grid);

    auto replaceProgramId = [&logicalPids](mlir::triton::GetProgramIdOp op) {
      Value replacement = logicalPids.x;
      switch (op.getAxis()) {
      case mlir::triton::ProgramIDDim::X:
        replacement = logicalPids.x;
        break;
      case mlir::triton::ProgramIDDim::Y:
        replacement = logicalPids.y;
        break;
      case mlir::triton::ProgramIDDim::Z:
        replacement = logicalPids.z;
        break;
      }
      op.replaceAllUsesWith(replacement);
      op.erase();
    };
    for (auto op : programIdOps)
      replaceProgramId(op);

    // Collect all `tt.return`s within that loop.
    SmallVector<mlir::triton::ReturnOp> retOps;
    forOp.walk([&](Operation *op) {
      if (auto retOp = dyn_cast<mlir::triton::ReturnOp>(op))
        retOps.push_back(retOp);
    });
    // Erase all 'tt.return's and replace them with `scf.yield` if the function
    // body has multiple blocks.
    for (auto retOp : retOps) {
      if (!originalFuncHasOneBlock) {
        OpBuilder b(retOp);
        b.create<scf::YieldOp>(retOp.getLoc(), retOp.getOperands());
      }
      retOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIMTAutoBlockifyPass() {
  return std::make_unique<SIMTAutoBlockifyPass>();
}

} // namespace triton
} // namespace bishengir
