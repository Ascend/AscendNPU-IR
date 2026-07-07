//===-------------------- InsertCVDataMovement.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts tight-coupled buffers for CV operations:
//   Pattern 1: UB buffer between FixpipeOp and Vector/VF consumers
//   Pattern 2: L1 buffer between Vector/VF producers and MmadL1Op
//
// NOTE: This pass assumes MmadL1 operands are already in 4D/5D NZ/zN layout.
//       No layout transformation is performed.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "insert-cv-data-movement"

namespace mlir {
#define GEN_PASS_DEF_INSERTCVDATAMOVEMENT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Returns true if the operation is a vector or VF operation.
static bool isVectorOrVFOp(Operation *op) {
  if (!op)
    return false;
  if (isa<HIVMStructuredOp>(op))
    return true;
  if (auto callOp = dyn_cast<func::CallOp>(op))
    return isVFCall(callOp);
  return false;
}

/// Returns the source value if the op is a "pass-through" that preserves
/// data identity (reshapes, casts, slices, transposes, etc.)
/// Returns nullptr if not a pass-through op.
Value getPassThroughSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
         // Tensor ops
         .Case([](tensor::ReshapeOp op) { return op.getSource(); })
         .Case([](tensor::CollapseShapeOp op) { return op.getSrc(); })
         .Case([](tensor::ExpandShapeOp op) { return op.getSrc(); })
         .Case([](tensor::ExtractSliceOp op) { return op.getSource(); })
         .Case([](tensor::InsertSliceOp op) { return op.getSource(); })
         // Bufferization ops
         .Case([](bufferization::ToTensorOp op) { return op.getMemref(); })
         .Case([](bufferization::ToMemrefOp op) { return op.getTensor(); })
         // Memref ops
         .Case([](memref::CastOp op) { return op.getSource(); })
         .Case([](memref::MemorySpaceCastOp op) { return op.getSource(); })
         .Case([](memref::SubViewOp op) { return op.getSource(); })
         .Case([](memref::CollapseShapeOp op) { return op.getViewSource(); })
         .Case([](memref::ExpandShapeOp op) { return op.getViewSource(); })
         .Case([](memref::ReinterpretCastOp op) { return op.getSource(); })
         .Case([](memref::ViewOp op) { return op.getSource(); })
         .Case([](memref::ReshapeOp op) { return op.getSource(); })
         // Linalg ops
         .Default([](Operation *) { return Value(); });
}

/// Returns a string description of the producer chain for debugging.
static std::string getProducerChainDescription(Value value) {
  std::string result;
  llvm::raw_string_ostream os(result);

  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 8> worklist{value};

  os << "Producer chain: ";
  bool first = true;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (!first)
      os << " -> ";
    first = false;
    os << defOp->getName().getStringRef();

    if (Value source = getPassThroughSource(defOp))
      worklist.push_back(source);
  }

  return result;
}

/// Traces through pass-through ops to check if value originates from vector/VF op.
static bool isProducedByVectorOrVF(Value value) {
  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 8> worklist{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    if (!current || !visited.insert(current).second)
      continue;

    // Handle block arguments (e.g., loop iter_args)
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()))
        if (auto *init = forOp.getTiedLoopInit(blockArg))
          worklist.push_back(init->get());
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    // Found vector/VF op
    if (isVectorOrVFOp(defOp))
      return true;

    // Handle scf.for results - trace through yielded values
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned idx = cast<OpResult>(current).getResultNumber();
      worklist.push_back(forOp.getYieldedValues()[idx]);
      continue;
    }

    // Handle scf.if results - check both branches
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      unsigned idx = cast<OpResult>(current).getResultNumber();
      auto &thenBlock = ifOp.getThenRegion().front();
      worklist.push_back(thenBlock.getTerminator()->getOperand(idx));
      if (!ifOp.getElseRegion().empty()) {
        auto &elseBlock = ifOp.getElseRegion().front();
        worklist.push_back(elseBlock.getTerminator()->getOperand(idx));
      }
      continue;
    }

    // Trace through pass-through ops
    if (Value source = getPassThroughSource(defOp)) {
      LLVM_DEBUG(llvm::dbgs() << "  Tracing through pass-through op: "
          << defOp->getName() << "\n");
      worklist.push_back(source);
    }
  }

  return false;
}

/// Returns true if the given value is directly produced by data movement ops.
/// Used to detect if L1 buffer has already been inserted.
bool isDirectlyProducedByDataMovement(Value value) {
  if (auto defOp = value.getDefiningOp())
    return isa<CopyOp, ND2NZOp>(defOp);
  return false;
}

//===----------------------------------------------------------------------===//
// Memory Allocation Helpers
//===----------------------------------------------------------------------===//

/// Holds allocated memref and its plain (no address space) cast.
struct AllocationResult {
  Value spacedMemref; // Memref with address space attribute
  Value plainMemref; // Memref without address space (after cast)
};

/// Creates a memref allocation with the specified address space.
AllocationResult createAddressSpaceAllocation(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape,
    Type elemType, AddressSpace addrSpace) {
  MLIRContext *ctx = rewriter.getContext();

  auto spaceAttr = AddressSpaceAttr::get(ctx, addrSpace);
  auto spacedType = MemRefType::get(shape, elemType, nullptr, spaceAttr);
  auto plainType = MemRefType::get(shape, elemType);

  Value alloc = rewriter.create<memref::AllocOp>(loc, spacedType);
  Value cast = rewriter.create<
    memref::MemorySpaceCastOp>(loc, plainType, alloc);

  return {alloc, cast};
}

/// Creates a UB (Unified Buffer) allocation.
AllocationResult createUBAllocation(PatternRewriter &rewriter,
                                    Location loc,
                                    RankedTensorType tensorType) {
  return createAddressSpaceAllocation(rewriter, loc, tensorType.getShape(),
                                      tensorType.getElementType(),
                                      AddressSpace::UB);
}

/// Creates an L1 allocation.
AllocationResult createL1Allocation(PatternRewriter &rewriter,
                                    Location loc,
                                    RankedTensorType tensorType) {
  return createAddressSpaceAllocation(rewriter, loc, tensorType.getShape(),
                                      tensorType.getElementType(),
                                      AddressSpace::L1);
}

//===----------------------------------------------------------------------===//
// Pattern 1: Insert UB Buffer After FixpipeOp
//===----------------------------------------------------------------------===//
//
// Transforms:
//   %result = hivm.hir.fixpipe ins(...) outs(...) -> tensor<...>
//   %user = hivm.hir.vector_op ins(%result, ...) ...
//
// Into:
//   %alloc = memref.alloc : memref<..., #hivm.address_space<ub>>
//   %cast = memref.memory_space_cast %alloc : ... to memref<...>
//   hivm.hir.fixpipe ins(...) outs(%alloc : memref<...>)
//   %tensor = bufferization.to_tensor %cast restrict writable : ...
//   %user = hivm.hir.vector_op ins(%tensor, ...) ...
//===----------------------------------------------------------------------===//

class InsertUBAfterFixpipePattern : public OpRewritePattern<FixpipeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FixpipeOp fixpipeOp,
                                PatternRewriter &rewriter) const override {
    // Only transform if fixpipe has vector/VF users
    bool hasVectorUser = llvm::any_of(
        fixpipeOp->getUsers(),
        [](Operation *user) { return isVectorOrVFOp(user); });

    if (!hasVectorUser) {
      return rewriter.notifyMatchFailure(fixpipeOp,
                                         "no vector/VF users found for fixpipe result");
    }

    auto resultType = fixpipeOp.getResultTensor().getType();

    Location loc = fixpipeOp.getLoc();

    // Create UB allocation
    auto [ubMemref, plainMemref] =
        createUBAllocation(rewriter, loc, resultType);

    // Create new fixpipe writing to UB memref (returns void)
    SmallVector<Value> oprs({fixpipeOp.getSrc(), ubMemref});
    if (auto quantScale = fixpipeOp.getQuantScale())
      oprs.push_back(quantScale);
    rewriter.create<FixpipeOp>(loc, TypeRange{}, oprs, fixpipeOp->getAttrs());

    // Convert memref back to tensor for users
    auto toTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, plainMemref,
        /*restrict=*/true, /*writable=*/true);

    rewriter.replaceOp(fixpipeOp, toTensor.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Insert L1 Buffer Before MmadL1Op
//===----------------------------------------------------------------------===//
//
// This pattern assumes MmadL1 operands are already in 4D/5D NZ/zN layout.
// It only inserts L1/cbuf buffer allocation and copy, without layout
// transformation.
//
// Transforms:
//   %vec_result = hivm.hir.vcast ins(...) outs(...) -> tensor<1x8x16x16xf16>
//   %mmad = hivm.hir.mmadL1 ins(%vec_result, ...) ...
//
// Into:
//   %vec_result = hivm.hir.vcast ins(...) outs(...) -> tensor<1x8x16x16xf16>
//   %alloc = memref.alloc : memref<1x8x16x16xf16, #hivm.address_space<cbuf>>
//   %cast = memref.memory_space_cast %alloc : ... to memref<1x8x16x16xf16>
//   %dst = bufferization.to_tensor %cast restrict writable : ...
//   %copy = hivm.hir.copy ins(%vec_result) outs(%dst) -> tensor<1x8x16x16xf16>
//   %mmad = hivm.hir.mmadL1 ins(%copy, ...) ...
//===----------------------------------------------------------------------===//

class InsertL1BeforeMmadPattern : public OpRewritePattern<MmadL1Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MmadL1Op mmadOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<std::string, 4> failureReasons;

    for (OpOperand *operand : mmadOp.getDpsInputOperands()) {
      std::string reason;
      if (succeeded(transformOperandToL1(mmadOp, *operand, rewriter, reason))) {
        changed = true;
      } else if (!reason.empty()) {
        failureReasons.push_back(reason);
      }
    }

    if (!changed) {
      std::string combinedReason = "no operands transformed: ";
      for (size_t i = 0; i < failureReasons.size(); ++i) {
        if (i > 0)
          combinedReason += "; ";
        combinedReason += failureReasons[i];
      }
      return rewriter.notifyMatchFailure(mmadOp, combinedReason);
    }

    return success();
  }

private:
  static LogicalResult transformOperandToL1(MmadL1Op mmadOp, OpOperand &operand,
                                            PatternRewriter &rewriter,
                                            std::string &failureReason) {
    Value inputTensor = operand.get();
    unsigned operandIdx = operand.getOperandNumber();

    auto tensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
    if (!tensorType) {
      failureReason = "operand " + std::to_string(operandIdx) +
                      ": not a ranked tensor type";
      return failure();
    }

    // Skip if L1 buffer already inserted (immediate producer is CopyOp)
    if (isDirectlyProducedByDataMovement(inputTensor)) {
      failureReason = "operand " + std::to_string(operandIdx) +
                      ": already has L1 buffer (produced by CopyOp)";
      return failure();
    }

    // Only transform tensor operands produced by vector/VF ops
    if (!isProducedByVectorOrVF(inputTensor)) {
      // Provide detailed debugging info
      std::string producerInfo;
      if (Operation *defOp = inputTensor.getDefiningOp()) {
        producerInfo = defOp->getName().getStringRef().str();
        producerInfo += " (" + getProducerChainDescription(inputTensor) + ")";
      } else {
        producerInfo = "block argument";
      }
      failureReason = "operand " + std::to_string(operandIdx) +
                      ": not produced by vector/VF op, producer is: " +
                      producerInfo;
      LLVM_DEBUG(
          llvm::dbgs() << "InsertL1BeforeMmad: " << failureReason << "\n");
      return failure();
    }

    Location loc = mmadOp.getLoc();
    rewriter.setInsertionPoint(mmadOp);

    LLVM_DEBUG(llvm::dbgs() << "InsertL1BeforeMmad: transforming operand "
        << operandIdx << " of " << mmadOp->getName() << "\n");

    // Create L1 allocation (tensor is already in correct fractal layout)
    auto [l1Memref, plainMemref] =
        createL1Allocation(rewriter, loc, tensorType);

    // Convert to tensor for destination
    auto dstTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, plainMemref,
        /*restrict=*/true, /*writable=*/true);

    // Create copy to L1 buffer
    auto copyOp = rewriter.create<CopyOp>(
        loc, tensorType, inputTensor, dstTensor.getResult());

    // Update operand to use the L1-buffered tensor
    rewriter.modifyOpInPlace(mmadOp, [&]() {
      operand.set(copyOp.getResult(0));
    });

    failureReason.clear();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct InsertCVDataMovementPass
    : public impl::InsertCVDataMovementBase<
      InsertCVDataMovementPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<InsertUBAfterFixpipePattern, InsertL1BeforeMmadPattern>(ctx);

    GreedyRewriteConfig config;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::hivm::createInsertCVDataMovementPass() {
  return std::make_unique<InsertCVDataMovementPass>();
}