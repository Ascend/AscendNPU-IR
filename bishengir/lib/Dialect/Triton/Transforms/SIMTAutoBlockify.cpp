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
//   for linear in [GET.BLOCK.IDX * chunk,
//                  min((GET.BLOCK.IDX + 1) * chunk, logicalGridSize)):
//     (pidX, pidY, pidZ) = unflatten(linear, gridX, gridY)
//
// The pass rewrites only `tt.get_program_id x/y/z`.  Grid metadata
// (`tt.get_num_programs`) stays intact for later lowering.
// NOTE: Currently, GET.BLOCK.IDX is represented via tt.get_program_id x. It
// will be replaced with the actual HIVM intrinsic during AdaptGPUKernel.
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
    if (!ttFunc.isPublic())
      return;
    if (!ttFunc.getBody().hasOneBlock())
      return signalPassFailure();

    Block &entryBlock = ttFunc.getBody().front();
    auto returnOp =
        dyn_cast<mlir::triton::ReturnOp>(entryBlock.getTerminator());
    if (!returnOp)
      return signalPassFailure();

    SmallVector<Operation *> originalBodyOps;
    for (auto &op : entryBlock) {
      if (&op != returnOp.getOperation())
        originalBodyOps.push_back(&op);
    }

    SmallVector<mlir::triton::GetProgramIdOp> programIdOps;
    ttFunc.walk([&programIdOps](Operation *op) {
      if (auto getProgramId = dyn_cast<mlir::triton::GetProgramIdOp>(op))
        programIdOps.push_back(getProgramId);
    });

    if (programIdOps.empty())
      return;

    auto maybePhysicalBlockNum = getPhysicalBlockNum(ttFunc);
    if (failed(maybePhysicalBlockNum)) {
      ttFunc.emitError(
          "failed to infer physical vector core count for SIMT auto blockify");
      return signalPassFailure();
    }

    auto gridX = getGridArg(ttFunc, gpu::MappingId::DimX);
    auto gridY = getGridArg(ttFunc, gpu::MappingId::DimY);
    auto gridZ = getGridArg(ttFunc, gpu::MappingId::DimZ);

    if (failed(gridX) || failed(gridY) || failed(gridZ)) {
      ttFunc.emitError("failed to get grid args for SIMT auto blockify");
      return signalPassFailure();
    }

    GridArgs grid = {gridX.value(), gridY.value(), gridZ.value()};

    OpBuilder builder(ttFunc);
    builder.setInsertionPointToStart(&entryBlock);
    Location loc = ttFunc.getLoc();

    Value yz = builder.create<arith::MulIOp>(loc, grid.y, grid.z);
    Value logicalBlockNums = builder.create<arith::MulIOp>(loc, grid.x, yz);
    Value blockIdx = builder.create<mlir::triton::GetProgramIdOp>(
        loc, static_cast<int32_t>(mlir::triton::ProgramIDDim::X));
    Value physicalBlockNum = builder.create<arith::ConstantIntOp>(
        loc, maybePhysicalBlockNum.value(), 32);
    Value chunk = builder.create<arith::CeilDivUIOp>(loc, logicalBlockNums,
                                                     physicalBlockNum);
    Value lowerBound = builder.create<arith::MulIOp>(loc, blockIdx, chunk);
    Value end = builder.create<arith::AddIOp>(loc, lowerBound, chunk);
    Value upperBound =
        builder.create<arith::MinUIOp>(loc, end, logicalBlockNums);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    // scf.for LinearIdx = blockIdx * chunk
    //          to min((blockIdx + 1) * chunk, logicalBlocks), step 1
    auto forOp = builder.create<scf::ForOp>(loc, lowerBound, upperBound, one);
    builder.setInsertionPointToStart(forOp.getBody());
    Value linearProgramId = forOp.getInductionVar();

    Block *loopBody = forOp.getBody();
    Operation *yieldOp = loopBody->getTerminator();
    for (Operation *op : originalBodyOps)
      op->moveBefore(yieldOp);

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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIMTAutoBlockifyPass() {
  return std::make_unique<SIMTAutoBlockifyPass>();
}

} // namespace triton
} // namespace bishengir
