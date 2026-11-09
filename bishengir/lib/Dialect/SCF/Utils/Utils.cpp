//===- Utils.cpp -- SCF Utils -----------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-scf-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace scf {
namespace utils {

LogicalResult splitForOpByTripCount(OpBuilder &builder, scf::ForOp forOp,
                                    int64_t lb, int64_t ub, int64_t step,
                                    uint64_t maxTripCount) {
  if (step <= 0 || maxTripCount == 0)
    return failure();

  llvm::APInt lower(128, lb, true), upper(128, ub, true);
  llvm::APInt stride(128, step, true);
  if (upper.sle(lower))
    return success();

  const llvm::APInt tripWide = (upper - lower + stride - 1).udiv(stride);
  if (tripWide.getActiveBits() > 64)
    return failure();
  const uint64_t trip = tripWide.getZExtValue();
  if (trip <= maxTripCount)
    return success();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  scf::YieldOp origYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  Type boundType = forOp.getLowerBound().getType();
  auto createBound = [&](int64_t value) -> Value {
    if (isa<IndexType>(boundType))
      return builder.create<arith::ConstantIndexOp>(loc, value);
    return builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(boundType, value));
  };

  SmallVector<std::pair<int64_t, int64_t>> bounds;
  for (uint64_t loIt = 0; loIt < trip;) {
    const uint64_t hiIt = loIt + std::min(maxTripCount, trip - loIt);
    const int64_t lo = (lower + llvm::APInt(128, loIt) * stride).getSExtValue();
    const int64_t hi =
        hiIt == trip ? ub
                     : (lower + llvm::APInt(128, hiIt) * stride).getSExtValue();
    bounds.emplace_back(lo, hi);
    loIt = hiIt;
  }

  SmallVector<Value> inits(forOp.getInitArgs());
  scf::ForOp lastChunk = nullptr;
  Value constantStep = createBound(step);
  for (auto [lo, hi] : bounds) {
    Value loValue = createBound(lo);
    Value hiValue = createBound(hi);
    scf::ForOp chunk = builder.create<scf::ForOp>(
        loc, loValue, hiValue, constantStep, inits,
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          IRMapping mapping;
          mapping.map(forOp.getStep(), constantStep);
          mapping.map(forOp.getInductionVar(), iv);
          for (auto [oldArg, newArg] :
               llvm::zip(forOp.getRegionIterArgs(), iterArgs))
            mapping.map(oldArg, newArg);
          for (Operation &op : forOp.getBody()->without_terminator())
            b.clone(op, mapping);
          SmallVector<Value> mappedYields;
          for (Value value : origYield.getOperands())
            mappedYields.push_back(mapping.lookupOrDefault(value));
          b.create<scf::YieldOp>(bodyLoc, mappedYields);
        });
    for (NamedAttribute attr : forOp->getAttrs())
      chunk->setAttr(attr.getName(), attr.getValue());
    lastChunk = chunk;
    inits = SmallVector<Value>(chunk.getResults());
  }

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), lastChunk.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  forOp.erase();
  return success();
}

DiagnosedSilenceableFailure verifyMapForToForallCondition(
    scf::ForOp forOp, DenseSet<Operation *> &insertSliceOps, Location loc) {
  // Verify that we're only yielding tensor.insert_slices.
  // Otherwise we cannot map it to tensor.in_parallel.insert_slices.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!llvm::all_of(yieldOp->getOperands(), [&insertSliceOps](Value v) {
        if (!isa<BlockArgument>(v) &&
            isa<tensor::InsertSliceOp>(v.getDefiningOp())) {
          insertSliceOps.insert(v.getDefiningOp());
          return true;
        }
        return false;
      }))
    return emitDefiniteFailure(loc)
           << "the target loop can only yield tensor.insert_slices!";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
verifyMapForToForallDeviceMapping(Attribute maybeDeviceMappingAttr,
                                  Location loc) {
  if (isa<DeviceMappingAttrInterface>(maybeDeviceMappingAttr))
    return DiagnosedSilenceableFailure::success();
  return emitDefiniteFailure(loc) << "attribute is not a device mapping!";
}

DiagnosedSilenceableFailure
mapForToForallImpl(OpBuilder &builder, scf::ForOp forOp,
                   std::optional<ArrayAttr> deviceMappings,
                   scf::ForallOp &forallOp) {
  DenseSet<Operation *> insertSliceOps;
  DiagnosedSilenceableFailure diag =
      verifyMapForToForallCondition(forOp, insertSliceOps, forOp->getLoc());
  if (diag.isDefiniteFailure())
    return diag;

  if (deviceMappings.has_value()) {
    if (!llvm::hasSingleElement(deviceMappings.value()))
      return emitDefiniteFailure(forOp->getLoc())
             << "requires exactly one mapping attr!";

    diag = verifyMapForToForallDeviceMapping(
        (*deviceMappings).getValue().front(), forOp->getLoc());
    if (diag.isDefiniteFailure())
      return diag;
  }

  builder.setInsertionPoint(forOp);
  auto regionIterArg = forOp.getBody()->getArguments();
  forallOp = builder.create<scf::ForallOp>(
      forOp.getLoc(),
      /*lbs=*/SmallVector<OpFoldResult>{forOp.getLowerBound()},
      /*ubs=*/SmallVector<OpFoldResult>{forOp.getUpperBound()},
      /*steps=*/SmallVector<OpFoldResult>{forOp.getStep()},
      /*outputs=*/ValueRange{forOp.getInitArgs()},
      /*mapping=*/deviceMappings,
      /*bodyBuilderFn=*/
      [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        IRMapping mapping;
        assert(regionIterArg.size() == regionArgs.size() &&
               "expect same region args");
        mapping.map(regionIterArg, regionArgs);

        Block *loopBlock = forOp.getBody();
        auto newTerminator = builder.create<scf::InParallelOp>(loc);
        builder.setInsertionPointToStart(newTerminator->getBlock());

        for (auto &nestedOp : loopBlock->without_terminator()) {
          if (insertSliceOps.contains(&nestedOp)) {
            auto insertSlice = cast<tensor::InsertSliceOp>(&nestedOp);
            Value sourceVal = mapping.lookup(insertSlice.getSource());
            Value destVal = mapping.lookup(insertSlice.getDest());
            SmallVector<OpFoldResult> offsets;
            for (OpFoldResult offset : insertSlice.getMixedOffsets()) {
              if (auto valueOffset = dyn_cast<Value>(offset))
                offsets.push_back(mapping.lookupOrDefault(valueOffset));
              else
                offsets.push_back(offset);
            }
            SmallVector<OpFoldResult> sizes;
            for (OpFoldResult size : insertSlice.getMixedSizes()) {
              if (auto valueSize = dyn_cast<Value>(size))
                sizes.push_back(mapping.lookupOrDefault(valueSize));
              else
                sizes.push_back(size);
            }
            SmallVector<OpFoldResult> strides;
            for (OpFoldResult stride : insertSlice.getMixedStrides()) {
              if (auto valueStride = dyn_cast<Value>(stride))
                strides.push_back(mapping.lookupOrDefault(valueStride));
              else
                strides.push_back(stride);
            }
            assert(offsets.size() == sizes.size());
            assert(offsets.size() == strides.size());
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(newTerminator.getBody());
            builder.create<tensor::ParallelInsertSliceOp>(
                loc, sourceVal, destVal, offsets, sizes, strides);
            continue;
          }
          Operation *clone = builder.clone(nestedOp, mapping);
          mapping.map(nestedOp.getResults(), clone->getResults());
        }
      });
  return DiagnosedSilenceableFailure::success();
}

bool isNormalized(LoopLikeOpInterface forOp) {
  auto allEqual = [](std::optional<ArrayRef<OpFoldResult>> results,
                     int64_t val) {
    if (!results.has_value()) {
      return false;
    }
    return llvm::all_of(results.value(), [val](OpFoldResult ofr) {
      auto intValue = getConstantIntValue(ofr);
      return intValue.has_value() && intValue == val;
    });
  };
  return allEqual(forOp.getLoopLowerBounds(), 0) &&
         allEqual(forOp.getLoopSteps(), 1);
}

} // namespace utils
} // namespace scf
} // namespace mlir
