

#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-scf-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace scf {
namespace utils {
static constexpr llvm::StringLiteral kMappingAttrName = "mapping";

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
verifySingleMapForToForallDeviceMapping(std::optional<ArrayAttr> deviceMapping,
                                        Location loc) {
  if (deviceMapping.has_value()) {
    if (!llvm::hasSingleElement(deviceMapping.value()))
      return emitDefiniteFailure(loc) << "requires exactly one mapping attr!";

    DiagnosedSilenceableFailure diag = verifyMapForToForallDeviceMapping(
        (*deviceMapping).getValue().front(), loc);
    if (diag.isDefiniteFailure())
      return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

/// Collect the perfectly nested `scf.for` loops rooted at `forOp`.
/// The root itself is not included in `forOps`, only its perfectly nested
/// inner loops.
static void findPerfectlyNestedScfFors(scf::ForOp forOp,
                                       SmallVectorImpl<scf::ForOp> &forOps) {
  scf::ForOp current = forOp;

  while (true) {
    Block &body = current.getRegion().front();

    // A perfect loop nest requires exactly one nested forOp and a yield.
    if (body.getOperations().size() != 2)
      break;

    auto innerFor = dyn_cast<scf::ForOp>(&body.front());
    auto yieldOp = dyn_cast<scf::YieldOp>(&body.back());
    if (!innerFor || !yieldOp)
      break;

    forOps.push_back(innerFor);
    current = innerFor;
  }
}

static ArrayAttr getOrCreateDeviceMappingAttr(scf::ForOp forOp,
                                              OpBuilder &builder) {
  if (auto mapping = forOp->getAttrOfType<ArrayAttr>(kMappingAttrName)) {
    // Use existing mapping attribute.
    return mapping;
  }
  // Otherwise, provide a default one.
  return builder.getArrayAttr(
      {hivm::HIVMBlockMappingAttr::get(forOp.getContext())});
}

static void buildForallBody(OpBuilder &builder, Location loc, scf::ForOp forOp,
                            ValueRange regionArgs,
                            ArrayRef<BlockArgument> regionIterArgs,
                            DenseSet<Operation *> &insertSliceOps) {
  IRMapping mapping;
  assert(regionIterArgs.size() == regionArgs.size() &&
         "expect same region args");
  mapping.map(regionIterArgs, regionArgs);
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
      builder.create<tensor::ParallelInsertSliceOp>(loc, sourceVal, destVal,
                                                    offsets, sizes, strides);
      continue;
    }
    Operation *clone = builder.clone(nestedOp, mapping);
    mapping.map(nestedOp.getResults(), clone->getResults());
  }
}

DiagnosedSilenceableFailure
mapForToForallImpl(OpBuilder &builder, scf::ForOp forOp,
                   std::optional<ArrayAttr> deviceMapping,
                   scf::ForallOp &forallOp, bool CheckMapForToForall) {
  scf::ForOp outerforOp = forOp;
  SmallVector<scf::ForOp> forOps;
  forOps.push_back(forOp);
  findPerfectlyNestedScfFors(forOp, forOps);

  SmallVector<OpFoldResult> lbs;
  SmallVector<OpFoldResult> ubs;
  SmallVector<OpFoldResult> steps;
  llvm::SmallVector<mlir::BlockArgument> regionIterArgs;
  llvm::SmallVector<mlir::Attribute, 3> repeated;

  for (unsigned i = 0; i < forOps.size(); i++) {
    scf::ForOp currentforOp = forOps[i];
    // check for map_for_to_forall
    if (CheckMapForToForall && !currentforOp->hasAttrOfType<UnitAttr>(
                                   ::mlir::utils::kMapForToForallAttrName))
      return DiagnosedSilenceableFailure::definiteFailure();

    lbs.push_back(currentforOp.getLowerBound());
    ubs.push_back(currentforOp.getUpperBound());
    steps.push_back(currentforOp.getStep());
    regionIterArgs.push_back(currentforOp.getBody()->getArguments().front());
    // check for device mapping
    ArrayAttr deviceMappingAttrs =
        getOrCreateDeviceMappingAttr(currentforOp, builder);
    DiagnosedSilenceableFailure diag = verifySingleMapForToForallDeviceMapping(
        deviceMappingAttrs, currentforOp->getLoc());
    if (diag.isDefiniteFailure())
      return diag;
    repeated.push_back(deviceMappingAttrs[0]);

    // Handle the inner most loop
    if (i != forOps.size() - 1)
      continue;
    forOp = currentforOp;
    DenseSet<Operation *> insertSliceOps;
    diag =
        verifyMapForToForallCondition(forOp, insertSliceOps, forOp->getLoc());
    if (diag.isDefiniteFailure())
      return diag;

    builder.setInsertionPoint(outerforOp);
    regionIterArgs.append(forOp.getBody()->getArguments().begin() + 1,
                          forOp.getBody()->getArguments().end());
    forallOp = builder.create<scf::ForallOp>(
        outerforOp.getLoc(),
        /*lbs=*/lbs,
        /*ubs=*/ubs,
        /*steps=*/steps,
        /*outputs=*/ValueRange{outerforOp.getInitArgs()},
        /*mapping=*/builder.getArrayAttr(repeated),
        /*bodyBuilderFn=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          buildForallBody(builder, loc, forOp, regionArgs, regionIterArgs,
                          insertSliceOps);
        });
  }
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
