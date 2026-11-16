//===- CustomOpFlatten.cpp - Flatten interface for HIVM custom ops --------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//
// `hivm.hir.custom` and `hivm.hir.custom_macro` wrap an implementation the
// compiler cannot inspect, so flattening them is driven by what the author
// declared on the op:
//
//   * Every `iterator_types` entry must be `parallel`. A rank alone cannot
//     identify the new position of an axis-dependent computation after a
//     partial collapse, and these attributes are not passed to the kernel.
//   * `indexing_map`, when present, must stay the identity: a permuting or
//     projecting map cannot be replayed on the collapsed iteration space.
//   * `flatten_ranks` / `flatten_symbols` name the implementation to call for
//     each rank the operands may be collapsed to. The kernel behind `symbol`
//     was built against memref descriptors of the original rank, so a collapse
//     only happens when the author supplied a kernel for the resulting rank,
//     and `symbol` is then switched to it. Declaring an entry promises that
//     the implementation is equivalent for every supported reassociation and
//     layout that selects it; `parallel` alone does not establish equivalence.
//
// A missing or unsupported declaration degrades to the identity reassociation,
// which makes `HIVMFlattenOps` leave the op untouched.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "flatten-custom-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// Rank of the iteration space a custom op is defined over. Temp buffers are
/// scratch memory outside that space, so they are excluded exactly like the
/// rest of the flatten pipeline does (`getHIVMOperands(false)`).
///
/// Returns `std::nullopt` when the operands do not agree on a single rank, or
/// when one carries a shaped type the pass will not rewrite: only top-level
/// `MemRefType` operands are collapsed, so a `tensor` operand would keep its
/// rank while its siblings -- and the rebuilt `iterator_types` -- drop to the
/// collapsed one.
template <typename CustomOpT>
std::optional<int64_t> getIterationSpaceRank(CustomOpT op) {
  std::optional<int64_t> rank;
  auto structuredOp = cast<HIVMStructuredOp>(op.getOperation());
  for (OpOperand *operand : structuredOp.getHIVMOperands(false)) {
    Type type = operand->get().getType();
    // Scalar operands do not take part in the iteration space.
    if (!isa<ShapedType>(type))
      continue;
    auto memrefType = dyn_cast<MemRefType>(type);
    if (!memrefType)
      return std::nullopt;

#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    bool hasStridedLayout = isStrided(memrefType);
#else
    bool hasStridedLayout = memrefType.isStrided();
#endif
    // The collapse pipeline builds the collapsed type before anything verifies
    // the operand, and `computeCollapsedType` dereferences the layout's stride
    // interface. A layout with no stride form crashes there, so refuse it here.
    if (!hasStridedLayout) {
      LDBG("custom op operand layout has no strided form");
      return std::nullopt;
    }
    if (rank && *rank != memrefType.getRank())
      return std::nullopt;
    rank = memrefType.getRank();
  }
  return rank;
}

/// Memref operands that take part in the iteration space.
template <typename CustomOpT>
SmallVector<MemRefType> getIterationSpaceOperandTypes(CustomOpT op) {
  SmallVector<MemRefType> types;
  auto structuredOp = cast<HIVMStructuredOp>(op.getOperation());
  for (OpOperand *operand : structuredOp.getHIVMOperands(false))
    if (auto memrefType = dyn_cast<MemRefType>(operand->get().getType()))
      types.push_back(memrefType);
  return types;
}

/// With an identity `indexing_map` every operand walks the same iteration
/// domain, so a loop dimension must have one extent. Nothing verifies that, and
/// a collapse would paper the inconsistency over: `memref<2x6xf32>` and
/// `memref<3x4xf32>` both flatten to `memref<12xf32>`.
///
/// Two dynamic extents are accepted, since they may carry the same value at run
/// time. Skipping dynamic extents instead would let `memref<?x6xf32>` pass
/// against `memref<3x?xf32>`, which is the very mismatch this rejects.
template <typename CustomOpT> bool hasConsistentIterationDomain(CustomOpT op) {
  SmallVector<MemRefType> types = getIterationSpaceOperandTypes(op);
  if (types.empty())
    return true;
  ArrayRef<int64_t> pivot = types.front().getShape();
  return llvm::all_of(llvm::drop_begin(types), [pivot](MemRefType type) {
    return type.getShape() == pivot;
  });
}

/// A temp buffer also passed as an input or output operand cannot be flattened
/// consistently: the rewrite maps values, not operand positions, so collapsing
/// the input/output occurrence rewrites the `tmps` occurrence too.
///
/// The operand segments are walked directly because `getHIVMOperands(false)`
/// filters by value and could never observe the reuse.
template <typename CustomOpT>
bool hasTempBufferUsedAsInputOrOutput(CustomOpT op) {
  DenseSet<Value> tempBuffers(op.getTempBuffers().begin(),
                              op.getTempBuffers().end());
  if (tempBuffers.empty())
    return false;

  auto aliasesTempBuffer = [&tempBuffers](ValueRange operands) {
    return llvm::any_of(operands, [&tempBuffers](Value operand) {
      return tempBuffers.contains(operand);
    });
  };
  if (aliasesTempBuffer(op.getInputs()) || aliasesTempBuffer(op.getOutputs())) {
    LDBG("custom op reuses a temp buffer as an input or output operand");
    return true;
  }
  return false;
}

/// Reads `iterator_types` without trusting its contents: no verifier rejects a
/// malformed one today, and the structured-op accessor hard-casts every
/// element. Returns `std::nullopt` when the attribute is absent or malformed,
/// since neither states what the loops mean.
template <typename CustomOpT>
std::optional<SmallVector<IteratorType>> readIteratorTypes(CustomOpT op) {
  Attribute attr = op->getAttr(CustomOpT::kIteratorTypesName);
  if (!attr)
    return std::nullopt;
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr) {
    LDBG("custom op has a malformed " << CustomOpT::kIteratorTypesName);
    return std::nullopt;
  }
  SmallVector<IteratorType> iteratorTypes;
  for (Attribute element : arrayAttr) {
    auto iteratorTypeAttr = dyn_cast<IteratorTypeAttr>(element);
    if (!iteratorTypeAttr) {
      LDBG("custom op has a malformed " << CustomOpT::kIteratorTypesName);
      return std::nullopt;
    }
    iteratorTypes.push_back(iteratorTypeAttr.getValue());
  }
  return iteratorTypes;
}

/// An `indexing_map` that is not the plain identity describes a permutation, a
/// projection or a broadcast that cannot be replayed on the collapsed space. An
/// absent attribute means "no remapping" and is accepted.
///
/// Like `iterator_types` this attribute has no verifier, and the structured-op
/// accessor turns a malformed value into an empty list that would pass as "no
/// remapping", so it is read raw.
template <typename CustomOpT>
bool hasFlattenableIndexingMaps(CustomOpT op, int64_t rank) {
  Attribute attr = op->getAttr(CustomOpT::kIndexingMapName);
  if (!attr)
    return true;
  auto maps = dyn_cast<ArrayAttr>(attr);
  if (!maps) {
    LDBG("custom op has a malformed " << CustomOpT::kIndexingMapName);
    return false;
  }
  // One map per structured operand. Consumers index this list by operand
  // number without a bounds check, so a mismatched count must not be carried
  // into the collapsed op.
  if (maps.size() != op.getInputs().size() + op.getOutputs().size()) {
    LDBG("custom op has " << maps.size() << " indexing maps for "
                          << (op.getInputs().size() + op.getOutputs().size())
                          << " structured operands");
    return false;
  }
  return llvm::all_of(maps, [rank](Attribute element) {
    auto mapAttr = dyn_cast<AffineMapAttr>(element);
    if (!mapAttr)
      return false;
    AffineMap map = mapAttr.getAffineMap();
    return map.getNumDims() == static_cast<unsigned>(rank) && map.isIdentity();
  });
}

/// Symbol declared in `flatten_symbols` for `rank`, null when the table is
/// absent or has no entry for `rank`. The table is verified by
/// `verifyCustomOpFlattenSymbols`.
template <typename CustomOpT>
StringAttr lookupFlattenSymbol(CustomOpT op, int64_t rank) {
  auto ranks =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kFlattenRanksAttrName);
  auto symbols =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kFlattenSymbolsAttrName);
  if (!ranks || !symbols)
    return nullptr;
  for (auto [rankAttr, symbolAttr] : llvm::zip(ranks, symbols))
    if (cast<IntegerAttr>(rankAttr).getInt() == rank)
      return cast<StringAttr>(symbolAttr);
  return nullptr;
}

/// Whether the op declares any implementation for a collapsed rank. Without one
/// there is no rank the pass could legally produce.
template <typename CustomOpT> bool hasFlattenSymbols(CustomOpT op) {
  auto symbols =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kFlattenSymbolsAttrName);
  return symbols && !symbols.empty();
}

/// `align_dim` is carried per operand inside `arg_attrs`, not on the op itself.
template <typename CustomOpT> bool hasAlignDimRequirement(CustomOpT op) {
  auto argAttrs =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kArgAttrsName);
  if (!argAttrs)
    return false;
  return llvm::any_of(argAttrs, [](Attribute attr) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    return dict && dict.contains(CustomOpT::kAlignDimName);
  });
}

/// Whether the op's declaration and operand layouts permit flattening.
template <typename CustomOpT> bool isFlattenableCustomOp(CustomOpT op) {
  // Builtins are lowered to a compiler-owned template whose signature is
  // pinned, so their rank must not change.
  if (op.isBuiltin()) {
    LDBG("builtin custom op is not flattened: " << op.getName());
    return false;
  }

  // The kernel behind `symbol` is compiled against descriptors of the original
  // rank. Without an implementation declared for a lower rank there is nothing
  // a collapsed op could be lowered to.
  if (!hasFlattenSymbols(op)) {
    LDBG("custom op declares no " << CustomOpT::kFlattenSymbolsAttrName
                                  << ", not flattened");
    return false;
  }

  // `align_dim` anchors an alignment requirement on one dimension; merging that
  // dimension into a group would drop the requirement it expresses.
  if (hasAlignDimRequirement(op)) {
    LDBG("custom op with " << CustomOpT::kAlignDimName << " is not flattened");
    return false;
  }

  // Only operands are rewritten, so a shaped result would be left behind at the
  // original rank.
  if (llvm::any_of(op->getResultTypes(), llvm::IsaPred<ShapedType>)) {
    LDBG("custom op with a shaped result is not flattened");
    return false;
  }

  // A value used both as an input or output operand and as a temp buffer
  // cannot be collapsed for one role only.
  if (hasTempBufferUsedAsInputOrOutput(op))
    return false;

  std::optional<int64_t> iterationRank = getIterationSpaceRank(op);
  if (!iterationRank || *iterationRank < 2)
    return false;
  int64_t rank = *iterationRank;

  // Without `iterator_types` the loop semantics are unknown, and guessing them
  // would risk merging a reduction into a parallel dimension.
  std::optional<SmallVector<IteratorType>> iteratorTypes =
      readIteratorTypes(op);
  if (!iteratorTypes)
    return false;
  if (static_cast<int64_t>(iteratorTypes->size()) != rank) {
    LDBG("custom op without matching iterator_types is not flattened");
    return false;
  }

  // Different partial collapses can produce identical descriptors while
  // moving a reduction, cumulative or other distinguished axis to different
  // positions. The rank-to-symbol table cannot distinguish those calls.
  if (llvm::any_of(*iteratorTypes, [](IteratorType type) {
        return type != IteratorType::kParallel;
      })) {
    LDBG("custom op with non-parallel iterator types is not flattened");
    return false;
  }

  if (!hasConsistentIterationDomain(op))
    return false;

  return hasFlattenableIndexingMaps(op, rank);
}

/// Every dimension index of the iteration space, i.e. "nothing may be merged".
/// Used as the limited-axes answer for ops that did not opt in, like
/// `VArangeOp` does. The widest ranked operand is used because the iteration
/// space rank is unavailable precisely when the operands disagree on it.
template <typename CustomOpT> SmallVector<int64_t> getAllAxes(CustomOpT op) {
  int64_t maxRank = 0;
  auto structuredOp = cast<HIVMStructuredOp>(op.getOperation());
  for (OpOperand *operand : structuredOp.getHIVMOperands(false))
    if (auto shapedType = dyn_cast<ShapedType>(operand->get().getType());
        shapedType && shapedType.hasRank())
      maxRank = std::max(maxRank, shapedType.getRank());
  SmallVector<int64_t> axes(maxRank);
  std::iota(axes.begin(), axes.end(), 0);
  return axes;
}

template <typename CustomOpT>
SmallVector<int64_t> getCustomOpLimitedAxes(CustomOpT op) {
  if (!isFlattenableCustomOp(op))
    return getAllAxes(op);
  return {};
}

template <typename CustomOpT>
FailureOr<FlattenResult> getCustomOpFlattened(CustomOpT op,
                                              FlattenOptions &options) {
  auto identityResult = [&op]() {
    FlattenResult identity(op.getOperation());
    identity.fillWithIdentity();
    return identity;
  };

  if (!isFlattenableCustomOp(op))
    return identityResult();

  auto structuredOp = cast<HIVMStructuredOp>(op.getOperation());
  FailureOr<FlattenResult> result =
      mlir::hivm::detail::collapseUniformReassociationPipeline(structuredOp,
                                                               options, {});
  if (failed(result))
    return result;

  // The collapse is only legal when the author supplied an implementation for
  // the resulting rank. Otherwise the op keeps its rank and its `symbol`.
  if (!result->isIdentityCollapse() &&
      !lookupFlattenSymbol(op, result->getRankAfterFlatten())) {
    LDBG("custom op declares no implementation for rank "
         << result->getRankAfterFlatten() << ", not flattened");
    return identityResult();
  }
  return result;
}

/// Rewrites the dimension-indexed attributes of a custom op so that they
/// describe the collapsed iteration space.
///
/// Every collapsed group consists entirely of `parallel` dimensions.
template <typename CustomOpT>
void adjustCustomOpTargetDimensions(CustomOpT op, OpBuilder &builder,
                                    const FlattenResult &result) {
  MLIRContext *context = op.getContext();
  int64_t newRank = result.getRankAfterFlatten();

  // The operands now carry descriptors of the collapsed rank, so the call must
  // go to the implementation declared for that rank. `getFlattened` only
  // produced this result after checking the entry exists.
  StringAttr symbol = lookupFlattenSymbol(op, newRank);
  assert(symbol && "flattened to a rank with no declared implementation");
  op->setAttr(CustomOpT::kSymbolAttrName, symbol);

  SmallVector<Attribute> newIteratorTypes(
      newRank, IteratorTypeAttr::get(context, IteratorType::kParallel));
  op->setAttr(CustomOpT::kIteratorTypesName,
              ArrayAttr::get(context, newIteratorTypes));

  // Only identity maps reach this point (see hasFlattenableIndexingMaps), so
  // the collapsed maps are the identity of the new rank.
  auto mapsAttr =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kIndexingMapName);
  if (mapsAttr && !mapsAttr.empty()) {
    SmallVector<Attribute> newMaps(
        mapsAttr.size(),
        AffineMapAttr::get(builder.getMultiDimIdentityMap(newRank)));
    op->setAttr(CustomOpT::kIndexingMapName, ArrayAttr::get(context, newMaps));
  }
}

} // namespace

SmallVector<int64_t> CustomOp::getLimitedAxes() {
  return getCustomOpLimitedAxes(*this);
}

SmallVector<int64_t> CustomMacroOp::getLimitedAxes() {
  return getCustomOpLimitedAxes(*this);
}

FailureOr<FlattenResult> CustomOp::getFlattened(FlattenOptions flattenOptions) {
  return getCustomOpFlattened(*this, flattenOptions);
}

FailureOr<FlattenResult>
CustomMacroOp::getFlattened(FlattenOptions flattenOptions) {
  return getCustomOpFlattened(*this, flattenOptions);
}

void CustomOp::adjustTargetDimensions(OpBuilder &builder,
                                      const FlattenResult &result) {
  adjustCustomOpTargetDimensions(*this, builder, result);
}

void CustomMacroOp::adjustTargetDimensions(OpBuilder &builder,
                                           const FlattenResult &result) {
  adjustCustomOpTargetDimensions(*this, builder, result);
}
