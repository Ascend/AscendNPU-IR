//===- AdaptTritonIRKernel.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that transforms
// slice-based concatenation (tensor.extract_slice + tensor.insert_slice chains)
// into select-based multiplexing (arith.select with precomputed masks).
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define DEBUG_TYPE "fix-fused-cat"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace bishengir::triton {
#define GEN_PASS_DEF_FIXFUSEDCAT
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

// example of horizontal concat of 2D tensors
// ---------------
// BEFORE Pattern:
// ---------------
//   %load_0 = tt.load %ptr0[indices] : tensor<RxCxf32>
//   // where C is the size of result tensor
//   ...
//   
//   %slice_i = tensor.extract_slice %load0[0, 0] [R, Size_i] [1, 1]
//   %result_i = tensor.insert_slice %slice0 into %init[0, Offset_i] [R, Size_i] [1, 1]
//   // where Offset_i = Sum_{k<i} Size_k
//
// AFTER Pattern:
// --------------
//   // offset the load in the oppsite direction to compensate
//   %offset_i = arith.constant dense<-Offset_i> : tensor<RxCxi32>
//   %adj_indices_i = arith.addi %base_indices, %offset_i
//   ...
//   
//   // Load with multiple offset patterns
//   %load_i = tt.load %ptr_i[indices_i + offset_i]
//   ...
//   
//   // Generate masks based on column index boundaries
//   %mask_i = arith.cmpi slt, %col_idx, offset_i
//   ...
//   
//   // Cascade of selects to multiplex sources
//   %seli = arith.select %mask_i, %load_i, %load_i+1
static LogicalResult rewriteStore(triton::StoreOp op,
                                  PatternRewriter &rewriter) {
  LDBG("rewrite store " << op);

  // The stored value must be produced by tensor.insert_slice.
  auto rootInsert =
      op.getValue().getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!rootInsert)
    return failure();
  
  // We expect a chain of insert_slice operations, all ultimately
  // inserting extracted slices into a common base tensor.
  llvm::SmallVector<std::tuple<
    triton::LoadOp,
    mlir::tensor::ExtractSliceOp,
    mlir::tensor::InsertSliceOp>> chain;

  LDBG("walking insert chain");

  // Walk backwards: rootInsert.dest() -> previous insert, etc.
  auto currInsertOp = rootInsert;
  while (true) {
    LDBG(currInsertOp);

    auto extractOp = 
      currInsertOp.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    LDBG(extractOp);
        
    auto loadOp = extractOp.getSource().getDefiningOp<triton::LoadOp>();
    if (!loadOp)
      return failure();
    LDBG(loadOp);

    chain.push_back({loadOp, extractOp, currInsertOp});

    auto dest = currInsertOp.getDest();
    if (auto prevInsertOp =
             dest.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      currInsertOp = prevInsertOp;
      continue;
    }

    // curr is the last insert in the chain
    // The base must be some "init" constant tensor.
    if (!dest.getDefiningOp()) {
      // block argument is not expected for these patterns
      return failure();
    }

    if (!mlir::isa<mlir::arith::ConstantOp>(dest.getDefiningOp()))
      return failure();

    break;
  }

  auto shape = rootInsert.getResultType().getShape();
  // currently work with only 2D tensors
  if (shape.size() != 2)
    return failure();

  int64_t ySize = shape[0];
  int64_t xSize = shape[1];

  // currently only work with horizontal concats of 2D tensors
  llvm::SmallVector<std::tuple<triton::LoadOp, int32_t, int32_t>> loads;
  int32_t expectedOffset = 0;

  // walk the chain backward to accumulate offsets
  for (auto &[loadOp, extractOp, insertOp] : llvm::reverse(chain)) {
    // Extract the offset and size of *this* slice.
    auto offsetAttr = mlir::getConstantIntValue(insertOp.getMixedOffsets()[1]);
    auto sizeAttr   = mlir::getConstantIntValue(insertOp.getMixedSizes()[1]);

    if (!offsetAttr || !sizeAttr)
      return failure();  // only handle static

    int64_t offset = *offsetAttr;
    int64_t size   = *sizeAttr;

    // Check that the current slice is inserted exactly at the expected offset.
    if (offset != expectedOffset)
      return failure();

    LDBG(size << " " << offset);

    loads.push_back({loadOp, size, offset});

    // Update the expected next offset.
    expectedOffset += size;
  }

  SmallVector<mlir::Value> tensors;
  tensors.reserve(loads.size());
  Value Columns;
  auto loc = rewriter.getUnknownLoc();

  for (auto [loadOp, size, offset] : loads) {
    LDBG("transforming " << loadOp);
    LDBG(size << " " << offset);

    auto addPtr = dyn_cast<triton::AddPtrOp>(loadOp.getPtr().getDefiningOp());
    if (!addPtr || !mlir::isa<RankedTensorType>(addPtr.getType()))
      return failure();
    
    auto base = addPtr.getOperand(0);
    auto ids  = addPtr.getOperand(1);
    
    if (!base.getDefiningOp<triton::SplatOp>())
      return failure();
    
    auto addI = dyn_cast<arith::AddIOp>(ids.getDefiningOp());
    if (!addI)
      return failure();
    
    auto columns = addI.getOperand(0);
    
    if (!Columns)
      Columns = columns;
    else if (Columns != columns)
      return failure();

    rewriter.setInsertionPointAfter(columns.getDefiningOp());

    auto offsetC = rewriter.create<arith::ConstantIntOp>(loc, -offset, /*bitwidth=*/32).getResult();

    LDBG(offsetC);

    auto columnOffsets = rewriter.create<triton::SplatOp>(
      loc,
      RankedTensorType::get({ySize, xSize}, rewriter.getI32Type()),
      offsetC);

    LDBG(columnOffsets);
    
    auto newColumns = rewriter.create<arith::AddIOp>(
      loc,
      RankedTensorType::get({ySize, xSize}, rewriter.getI32Type()),
      columns, columnOffsets.getResult()
    );

    LDBG(newColumns);
    
    rewriter.setInsertionPointAfter(loadOp);

    auto newAddI = rewriter.create<arith::AddIOp>(
      loc,
      RankedTensorType::get({ySize, xSize}, rewriter.getI32Type()),
      newColumns.getResult(), addI.getOperand(1)
    );

    LDBG(newAddI);

    auto newAddPtr = rewriter.create<triton::AddPtrOp>(
      loc,
      addPtr.getType(),
      base, newAddI.getResult()
    );

    LDBG(newAddPtr);

    auto newLoadOp = rewriter.create<triton::LoadOp>(
      loc,
      loadOp.getType(),
      newAddPtr.getResult(),
      loadOp.getMask(),
      loadOp.getOther(),
      loadOp.getBoundaryCheckAttr(),
      loadOp.getPaddingAttr(),
      loadOp.getCache(),
      loadOp.getEvict(),
      loadOp.getIsVolatileAttr().getValue()
    );

    LDBG(newLoadOp);

    tensors.push_back(newLoadOp);
  }

  if (!Columns)
    return failure();

  LDBG(Columns);

  mlir::TypedValue<mlir::RankedTensorType> columns;
  if (auto op = Columns.getDefiningOp<triton::BroadcastOp>())
    columns = op.getSrc();
  else if (auto op = Columns.getDefiningOp<triton::ExpandDimsOp>())
    columns = op.getSrc();
  else
    return failure();

  SmallVector<mlir::Value> masks;
  masks.reserve(loads.size());

  rewriter.setInsertionPointAfter(rootInsert);

  for (auto [loadOp, size, offset] : loads) {
    mlir::RankedTensorType tensorType;
    if (columns.getType().getRank() == 2)
      tensorType = RankedTensorType::get({1, xSize}, rewriter.getI32Type());
    else
      tensorType = RankedTensorType::get({xSize}, rewriter.getI32Type());
    auto splatAttr = SplatElementsAttr::get(
      tensorType,
      rewriter.getI32IntegerAttr(offset)
    );
    auto offsetC = rewriter.create<arith::ConstantOp>(
      loc,
      tensorType,
      splatAttr
    );

    LDBG(offsetC);

    auto cmpI = rewriter.create<arith::CmpIOp>(
      loc,
      arith::CmpIPredicate::slt,
      columns,
      offsetC.getResult()
    );

    LDBG(cmpI);

    masks.push_back(cmpI);
  }

  // foldr
  auto finalTensor = tensors.back();
  for (int i = static_cast<int>(tensors.size() - 2); i >= 0; i--) {
    auto tensor = tensors[i];
    auto mask = masks[i+1];

    LDBG(mask);

    if (columns.getType().getRank() == 2)
      mask = rewriter.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get({ySize, xSize}, rewriter.getI1Type()),
        mask
      );
    else
      mask = rewriter.create<triton::ExpandDimsOp>(
        loc,
        RankedTensorType::get({1, xSize}, rewriter.getI1Type()),
        mask,
        /*axis=*/0
      );

    LDBG(mask);

    finalTensor = rewriter.create<arith::SelectOp>(
      loc,
      mask,
      tensor,
      finalTensor
    );

    LDBG(finalTensor);
  }

  op.setOperand(1, finalTensor);

  // erase backward
  for (auto [loadOp, extractOp, insertOp] : chain) {
    rewriter.eraseOp(insertOp);
    rewriter.eraseOp(extractOp);
  }

  // If we reach here, we matched the fused-concatenation pattern.
  return success();
}

class FixFusedCatPass : public impl::FixFusedCatBase<FixFusedCatPass> {
public:
  using FixFusedCatBase::FixFusedCatBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool rewritten = false;

    mod.walk([&](triton::StoreOp op) {
      if (rewritten)
        return;

      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);

      if (succeeded(rewriteStore(op, rewriter))) {
        rewritten = true;
      }
    });

    // this pass should not fail:
    // either rewrite succeeds or the pattern is not matched
    if (rewritten)
      LDBG("Fix success");
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createFixFusedCatPass() {
  return std::make_unique<FixFusedCatPass>();
}

} // namespace bishengir::triton