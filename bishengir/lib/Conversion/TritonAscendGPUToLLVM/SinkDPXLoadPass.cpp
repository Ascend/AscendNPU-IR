//===- SinkDPXLoadPass.cpp - Sink ascend_dpx.load to reduce reg pressure --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After ConvertTritonAscendGPUToLLVM, tensor operations are fully unrolled into
// per-thread scalar operations. The lowering packs all per-element results into
// LLVM structs via insertvalue chains and unpacks them via extractvalue.
// This creates an artificial sequential dependency through the struct that
// prevents interleaving of independent load-compute-store chains.
//
// This pass has two phases:
//   1. SROA-like struct bypass: replace extractvalue(insertvalue_chain, idx)
//      with the value that was inserted at that index, eliminating the struct
//      as a dependency bottleneck.
//   2. Bottom-up scheduling: reorder operations so each store's dependency
//      tree (load, compute, store) is emitted contiguously.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton::ascend {

#define GEN_PASS_DEF_SINKDPXLOAD
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

namespace {

// A struct to hold the result of tracing a pointer value back to its underlying
// base object, along with much more data to help track the pointer's origins
struct TracedPointer {
  Value baseObject; // base object
  int64_t
      startByteOffset = 0;   // constant byte offset from the base object, if known
  int64_t endByteOffset = 0; // startByteOffset + size of the accessed memory region
  llvm::DenseMap<Value, int64_t>
      valueOffsets; // dynamic/runtime offsets from the base object, keyed by
                    // the Value they depend on (e.g. {%4: 3} means baseObject +
                    // 3 * %4)

  Value
      dynGEPOpBase; // the last GEP op with untrackable dynamic indices, if any
  llvm::DenseMap<int64_t, Value>
      dynGEPOpIdxVals; // the dynamic index values of that GEP, keyed by the
                       // index position (e.g. {0: %4} means the first index is
                       // dynamic and has value %4)
  llvm::DenseMap<int64_t, int64_t>
      dynGEPOpIdxConsts; // the constant index values of that GEP, keyed by the
                         // index position (e.g. {1: 16} means the second index
                         // is constant with value 16)

  bool hasDynamicIndices =
      false; // tracks if the pointer has any dynamic indices in its GEP chain,
             // which prevents us from knowing exact byte offsets

  Value extractFrom; // if this pointer is traced from an extractelement or
                     // extractvalue, the value being extracted from (vector or
                     // struct)
  SmallVector<int64_t>
      indices; // the indices of the extractelement/extractvalue, if applicable
               // (e.g. for %5 = extractvalue %s[2][3], indices would be [2, 3])
  int64_t extrStartOffset = 0; // the constant byte offset from base object at
                               // time of earliest extraction
  llvm::DenseMap<Value, int64_t>
      extrValueOffsets; // the known dynamic byte offsets from base object at
                        // time of earliest extraction
  int64_t extrEndOffset = 0; // extrStartOffset + size of accessed memory region
  bool knownExtrOffsets = true; // tracks if the extraction offsets are known
                                // constant or dynamically changed later on

  bool fromDynExtrElemOp = false; // tracks if this pointer is traced from an
                                  // extractelement with dynamic index
  Value dynExtrElemIdx; // the dynamic index value for the extractelement, if
                        // applicable
};
//===----------------------------------------------------------------------===//
// Phase 1: SROA - Scalar Replacement of Aggregates
//===----------------------------------------------------------------------===//

/// Given the final value of an insertvalue chain, trace back and build a map
/// from field index to the value inserted at that index.
/// For example, for the chain:
///   %s0 = llvm.undef : !llvm.struct<(i64, i64)>
///   %s1 = llvm.insertvalue %a, %s0[0]
///   %s2 = llvm.insertvalue %b, %s1[1]
/// Returns {0: %a, 1: %b}.
static void traceInsertValueChain(Value structVal,
                                  llvm::DenseMap<int64_t, Value> &fieldMap) {
  while (auto insertOp = structVal.getDefiningOp<LLVM::InsertValueOp>()) {
    auto position = insertOp.getPosition();
    // Only handle single-level struct indexing (no nested structs).
    if (position.size() != 1)
      return;
    int64_t idx = position[0];
    // Only record the first (outermost) insertion for each index,
    // since later insertions overwrite earlier ones.
    if (!fieldMap.contains(idx))
      fieldMap[idx] = insertOp.getValue();
    structVal = insertOp.getContainer();
  }
}

/// Perform SROA on extractvalue operations: if an extractvalue reads from
/// an insertvalue chain, replace it with the value that was directly inserted
/// at that index.
static bool sroaStructs(Block &block) {
  bool changed = false;
  llvm::SmallVector<LLVM::ExtractValueOp> extracts;

  for (auto &op : block) {
    if (auto extractOp = dyn_cast<LLVM::ExtractValueOp>(&op))
      extracts.push_back(extractOp);
  }

  for (auto extractOp : extracts) {
    auto position = extractOp.getPosition();
    if (position.size() != 1)
      continue;
    int64_t idx = position[0];

    llvm::DenseMap<int64_t, Value> fieldMap;
    traceInsertValueChain(extractOp.getContainer(), fieldMap);

    auto it = fieldMap.find(idx);
    if (it == fieldMap.end())
      continue;

    extractOp.getResult().replaceAllUsesWith(it->second);
    changed = true;
  }

  return changed;
}

/// Remove dead operations (ops with no users and no side effects).
static void removeDeadOps(Block &block) {
  // Walk in reverse to handle chains of dead ops.
  llvm::SmallVector<Operation *> toDelete;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation *op = &*it;
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (op->use_empty() && isMemoryEffectFree(op))
      toDelete.push_back(op);
  }
  for (auto *op : toDelete)
    op->erase();
}

//===----------------------------------------------------------------------===//
// Phase 2: Bottom-up store-rooted scheduling
//===----------------------------------------------------------------------===//

/// Check whether any operation in the range [begin, end) produces or consumes
/// a vector<2xf16> value.  This type is the hallmark of f16 element packing
/// introduced by ConvertTritonAscendGPUToLLVM.  Scheduling segments that
/// contain these packed operations triggers a miscompile in the downstream
/// backend (hivmc), so we conservatively skip them.
static bool containsF16VectorPacking(Block::iterator begin,
                                     Block::iterator end) {
  for (auto it = begin; it != end; ++it) {
    for (auto result : it->getResults()) {
      if (auto vecTy = dyn_cast<VectorType>(result.getType())) {
        if (vecTy.getElementType().isF16() && vecTy.getNumElements() == 2)
          return true;
      }
    }
    for (auto operand : it->getOperands()) {
      if (auto vecTy = dyn_cast<VectorType>(operand.getType())) {
        if (vecTy.getElementType().isF16() && vecTy.getNumElements() == 2)
          return true;
      }
    }
  }
  return false;
}

// Recursively trace an index value through add and sub operations, accumulating
// a numeric offset and a map of dynamic value indices with the number of times
// those values are to be added to the current numeric offset
// (e.g. for %idx3 = %idx1 + %idx2 - 4, numericOffset would be -4
// and valueIndices would be {%idx1: 1, %idx2: 1}).
void traceAddSub(int64_t &numericOffset, DenseMap<Value, int64_t> &valueIndices,
                 Value &toTrace, bool flip) {
  auto defOp = toTrace.getDefiningOp();
  auto constOp = dyn_cast_or_null<LLVM::ConstantOp>(defOp);
  auto addOp = dyn_cast_or_null<LLVM::AddOp>(defOp);
  auto subOp = dyn_cast_or_null<LLVM::SubOp>(defOp);
  Value leftOperand;
  Value rightOperand;
  if (addOp || subOp) {
    leftOperand = defOp->getOperand(0);
    rightOperand = defOp->getOperand(1);
  }
  int64_t indexValue = 0;
  if (constOp) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      indexValue = intAttr.getInt();
      if (flip) {
        numericOffset -= indexValue;
      } else {
        numericOffset += indexValue;
      }
    }
  } else if (addOp) {
    traceAddSub(numericOffset, valueIndices, leftOperand, flip);
    traceAddSub(numericOffset, valueIndices, rightOperand, flip);
  } else if (subOp) {
    traceAddSub(numericOffset, valueIndices, leftOperand, flip);
    traceAddSub(numericOffset, valueIndices, rightOperand, !flip);
  } else {
    valueIndices[toTrace] += flip ? -1 : 1;
  }
}

// Trace a pointer value back to its underlying base object,
// while also tracking constant and dynamic byte offsets along the way.
TracedPointer traceToUnderlyingObject(Value value,
                                      const DataLayout &dataLayout) {
  TracedPointer result = TracedPointer();
  result.baseObject = value;

  // Traceback through blocks not currently supported, end loop if we hit a
  // block argument. Currently support traceback through GEP, addrspacecast,
  // bitcast, extractelement, and extractvalue operations.
  while (auto definingOp = result.baseObject.getDefiningOp()) {
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(definingOp)) {
      result.baseObject = gepOp.getBase();

      // skip through if dynamic indices were found later down the chain,
      // since we won't be able to track offsets accurately
      if (result.hasDynamicIndices) {
        continue;
      }

      Type currentType = gepOp.getElemTypeAttr().getValue();

      int curInd = -1;
      for (auto gepArg : gepOp.getIndices()) {
        ++curInd;
        // extract constant index value if available, otherwise treat as dynamic
        // index
        int64_t indexValue = 0;
        bool isConstant = false;

        if (auto valindex = dyn_cast<Value>(gepArg)) {
          if (auto constOp = valindex.getDefiningOp<LLVM::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              indexValue = intAttr.getInt();
              isConstant = true;
            }
          }
        } else if (auto intAttr = dyn_cast<IntegerAttr>(gepArg)) {
          indexValue = intAttr.getInt();
          isConstant = true;
        }

        // if result was found to have dynamic indices just now, start
        // tracking them and record the current GEP's base and indices
        if (result.hasDynamicIndices) {
          if (!isConstant) {
            result.dynGEPOpIdxVals[curInd] = dyn_cast<Value>(gepArg);
          } else {
            result.dynGEPOpIdxConsts[curInd] = indexValue;
          }
          continue;
        }

        // if this index is dynamic and isn't the first index, we cannot
        // calculate offsets correctly at all. Start tracking indices for the
        // rest of the GEP indices If this index is dynamic but is the first
        // index, we can still calculate offsets as a multiple of dynamic values
        // and correctly calculate offsets for subsequent constant indices,
        // since they will be at a fixed offset from the dynamic index.
        if (!isConstant) {
          if (curInd != 0) {
            result.hasDynamicIndices = true;
            result.dynGEPOpBase = result.baseObject;
            result.dynGEPOpIdxVals[curInd] = dyn_cast<Value>(gepArg);
          } else {
            int64_t numericOffset = 0;
            DenseMap<Value, int64_t> valueIndices;
            auto valIndex = dyn_cast<Value>(gepArg);
            traceAddSub(numericOffset, valueIndices, valIndex, false);
            for (auto &[key, value] : valueIndices) {
              result.valueOffsets[key] +=
                  value *
                  static_cast<int64_t>(dataLayout.getTypeSize(currentType));
            }
            result.startByteOffset +=
                numericOffset *
                static_cast<int64_t>(dataLayout.getTypeSize(currentType));
          }
          continue;
        }

        // if this is the first index and is constant, we can calculate a fixed
        // byte offset and continue
        if (curInd == 0) {
          int64_t typeSize =
              static_cast<int64_t>(dataLayout.getTypeSize(currentType));
          result.startByteOffset += indexValue * typeSize;
          continue;
        }

        // for subsequent constant indices, we can calculate the byte offset
        // based on the current type
        if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
          int64_t elementSize = static_cast<int64_t>(
              dataLayout.getTypeSize(arrayType.getElementType()));
          result.startByteOffset += indexValue * elementSize;
          currentType = arrayType.getElementType();
        } else if (auto structType =
                       dyn_cast<LLVM::LLVMStructType>(currentType)) {
          int64_t currentOffset = 0;
          for (int64_t i = 0; i < indexValue; ++i) {
            Type fieldType = structType.getBody()[i];
            int64_t alignment =
                static_cast<int64_t>(dataLayout.getTypeABIAlignment(fieldType));
            if (!structType.isPacked() && currentOffset % alignment != 0) {
              currentOffset += alignment - (currentOffset % alignment);
            }
            currentOffset +=
                static_cast<int64_t>(dataLayout.getTypeSize(fieldType));
          }
          Type targetFieldType = structType.getBody()[indexValue];
          int64_t targetAlignment = static_cast<int64_t>(
              dataLayout.getTypeABIAlignment(targetFieldType));
          if (!structType.isPacked() && currentOffset % targetAlignment != 0) {
            currentOffset +=
                targetAlignment - (currentOffset % targetAlignment);
          }
          result.startByteOffset += currentOffset;
          currentType = targetFieldType;
        } else if (auto vecType = dyn_cast<VectorType>(currentType)) {
          int64_t elementSize = static_cast<int64_t>(
              dataLayout.getTypeSize(vecType.getElementType()));
          result.startByteOffset += indexValue * elementSize;
          currentType = vecType.getElementType();
        } else {
          // fallback for int/float types
          int64_t typeSize =
              static_cast<int64_t>(dataLayout.getTypeSize(currentType));
          result.startByteOffset += indexValue * typeSize;
        }
      }
      continue;
    }

    // addrspacecast and bitcast they only change the underlying pointer value
    if (auto addrCastOp = dyn_cast<LLVM::AddrSpaceCastOp>(definingOp)) {
      result.baseObject = addrCastOp.getOperand();
      continue;
    }

    if (auto bitCastOp = dyn_cast<LLVM::BitcastOp>(definingOp)) {
      result.baseObject = bitCastOp.getOperand();
      continue;
    }

    // trace the pointer through extractelement op if possible,
    // if dynamic index found then remember it and end trace
    if (auto extractElemOp = dyn_cast<LLVM::ExtractElementOp>(definingOp)) {
      auto vec = extractElemOp.getVector();
      auto idx = extractElemOp.getPosition();
      int64_t indexValue = 0;
      bool isConstant = false;

      if (auto valindex = dyn_cast<Value>(idx)) {
        if (auto constOp = valindex.getDefiningOp<LLVM::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            indexValue = intAttr.getInt();
            isConstant = true;
          }
        }
      }

      result.extractFrom = vec;
      result.extrStartOffset = result.startByteOffset;
      result.extrValueOffsets = result.valueOffsets;
      result.knownExtrOffsets = !result.hasDynamicIndices;
      if (!isConstant) {
        result.fromDynExtrElemOp = true;
        result.dynExtrElemIdx = dyn_cast<Value>(idx);
        break;
      }

      result.indices = SmallVector<int64_t>{indexValue};

      auto vecDefOp = vec.getDefiningOp();
      bool found = false;
      if (!vecDefOp)
        break;

      while (auto insertElemOp = dyn_cast<LLVM::InsertElementOp>(vecDefOp)) {
        auto insIdx = insertElemOp.getPosition();
        int64_t insIdxVal = 0;
        isConstant = false;
        if (auto valindex = dyn_cast<Value>(insIdx)) {
          if (auto constOp = valindex.getDefiningOp<LLVM::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              insIdxVal = intAttr.getInt();
              isConstant = true;
            }
          }
        }

        if (!isConstant && dyn_cast<Value>(insIdx) != dyn_cast<Value>(idx)) {
          break;
        }

        if ((!isConstant &&
              dyn_cast<Value>(insIdx) == dyn_cast<Value>(idx)) ||
            (isConstant && insIdxVal == indexValue)) {
          result.baseObject = insertElemOp.getValue();
          found = true;
          break;
        } else {
          vecDefOp = insertElemOp.getVector().getDefiningOp();
          if (!vecDefOp)
            break;
        }
      }
      if (found)
        continue;
    }

    // similarly, trace the pointer through extractvalue op if possible,
    // if dynamic index found then remember all relevant indices and end trace.
    if (auto extractValOp = dyn_cast<LLVM::ExtractValueOp>(definingOp)) {
      auto container = extractValOp.getContainer();
      auto indices = extractValOp.getPosition();
      auto contDefOp = container.getDefiningOp();
      bool found = false;

      result.extractFrom = container;
      result.extrStartOffset = result.startByteOffset;
      result.extrValueOffsets = result.valueOffsets;
      result.indices = SmallVector<int64_t>(indices.begin(), indices.end());
      result.knownExtrOffsets = !result.hasDynamicIndices;
      if (!contDefOp)
        break;

      size_t numMatches = 0;
      while (auto insertValOp = dyn_cast<LLVM::InsertValueOp>(contDefOp)) {
        auto insIdxs = insertValOp.getPosition();
        int curMatchCount = 0;

        if (insIdxs.size() + numMatches > indices.size())
          break;

        for (size_t i = 0; i < insIdxs.size(); ++i) {
          if (indices[numMatches + i] == insIdxs[i]) {
            ++curMatchCount;
          } else {
            curMatchCount = 0;
            break;
          }
        }

        if (!curMatchCount) {
          container = insertValOp.getContainer();
          contDefOp = container.getDefiningOp();
          continue;
        }

        numMatches += static_cast<size_t>(curMatchCount);
        if (numMatches == indices.size()) {
          result.baseObject = insertValOp.getValue();
          found = true;
          break;
        } else {
          container = insertValOp.getValue();
          contDefOp = container.getDefiningOp();
          if (!contDefOp)
            break;
          result.extractFrom = container;
          result.indices = SmallVector<int64_t>(indices.begin() + numMatches,
                                                indices.end());
        }
      }
      if (found)
        continue;
    }

    break;
  }
  return result;
}

// function to decide based on pointer traces if they may reference the same
// memory or not
bool mayRefSameMem(TracedPointer &loadTrace, TracedPointer &storeTrace) {
  // If the base object of the pointers is different, we can:
  // 1. check if they are extracted from the same indices of the same container
  //    and if the offsets beyond this extraction are known, then make
  //    inferrences based on the constant/dynamic offsets at the time of
  //    extraction
  // 2. check if the two are from different top-level function arguments,
  //    which are guaranteed to not overlap in memory
  // 3. check if they are from different allocations, which are guaranteed to be
  //    different objects and hence not overlap in memory
  if (loadTrace.baseObject != storeTrace.baseObject) {
    if (loadTrace.knownExtrOffsets && storeTrace.knownExtrOffsets &&
        loadTrace.extractFrom == storeTrace.extractFrom &&
        loadTrace.extrValueOffsets == storeTrace.extrValueOffsets &&
        ((!(loadTrace.fromDynExtrElemOp || storeTrace.fromDynExtrElemOp ||
            loadTrace.indices.empty()) &&
          loadTrace.indices == storeTrace.indices) ||
         (loadTrace.fromDynExtrElemOp && storeTrace.fromDynExtrElemOp &&
          loadTrace.dynExtrElemIdx == storeTrace.dynExtrElemIdx))) {
      if ((loadTrace.extrStartOffset <= storeTrace.extrStartOffset &&
           storeTrace.extrStartOffset < loadTrace.extrEndOffset) ||
          (storeTrace.extrStartOffset <= loadTrace.extrStartOffset &&
           loadTrace.extrStartOffset < storeTrace.extrEndOffset)) {
        return true;
      }
      return false;
    }

    auto isKernelArg = [](BlockArgument &arg) {
      auto bb = arg.getOwner();
      auto bbParent = bb->getParentOp();
      return bb->isEntryBlock() && isa<LLVM::LLVMFuncOp>(bbParent) &&
             bbParent->getParentOp() && isa<ModuleOp>(bbParent->getParentOp()) &&
             !bbParent->getParentOp()->getParentOp();
    };

    auto arg1 = dyn_cast<BlockArgument>(loadTrace.baseObject);
    auto arg2 = dyn_cast<BlockArgument>(storeTrace.baseObject);
    if (arg1 && arg2 &&
        arg1.getOwner() == arg2.getOwner() &&
        isKernelArg(arg1)) {
      return false;
    }

    auto defOp1 = loadTrace.baseObject.getDefiningOp();
    auto defOp2 = storeTrace.baseObject.getDefiningOp();
    if (defOp1 && defOp2 && defOp1 != defOp2) {
      auto memEffect1 = dyn_cast<MemoryEffectOpInterface>(defOp1);
      auto memEffect2 = dyn_cast<MemoryEffectOpInterface>(defOp2);
      if (memEffect1 && memEffect2 &&
          memEffect1.hasEffect<MemoryEffects::Allocate>() &&
          memEffect2.hasEffect<MemoryEffects::Allocate>()) {
        return false;
      }
    }

    return true;
  }

  // If the base object of the two is the same, then we can check if the two
  // are from known memory offsets or if they are from the same dynamic index
  // access from the same pointer, then make inferrences based on the constant
  // offsets
  if ((!loadTrace.hasDynamicIndices && !storeTrace.hasDynamicIndices &&
       loadTrace.valueOffsets == storeTrace.valueOffsets) ||
      (loadTrace.hasDynamicIndices && storeTrace.hasDynamicIndices &&
       loadTrace.valueOffsets == storeTrace.valueOffsets &&
       loadTrace.dynGEPOpBase == storeTrace.dynGEPOpBase &&
       loadTrace.dynGEPOpIdxVals == storeTrace.dynGEPOpIdxVals &&
       loadTrace.dynGEPOpIdxConsts == storeTrace.dynGEPOpIdxConsts)) {
    if ((loadTrace.startByteOffset <= storeTrace.startByteOffset &&
         storeTrace.startByteOffset < loadTrace.endByteOffset) ||
        (storeTrace.startByteOffset <= loadTrace.startByteOffset &&
         loadTrace.startByteOffset < storeTrace.endByteOffset)) {
      return true;
    }
    return false;
  }

  return true;
}

/// Schedule a range of operations [segBegin, segEnd) using bottom-up
/// store-rooted scheduling. Only operations within this segment are reordered;
/// dependency-tree walks stop at the segment boundary.
static void scheduleSegment(Block &block, Block::iterator segBegin,
                            Block::iterator segEnd,
                            const DataLayout &dataLayout) {
  // Skip scheduling for segments with f16 vector packing — the downstream
  // backend is sensitive to instruction ordering for these patterns.
  if (containsF16VectorPacking(segBegin, segEnd))
    return;

  // Collect loads/stores in this segment.
  llvm::SmallVector<ascend_dpx::StoreOp> stores;
  llvm::SmallVector<ascend_dpx::LoadOp> loads;

  // Also collect the insert point (first overlapping store, if any) for each
  // load and the tracedpointer for each load
  llvm::DenseMap<ascend_dpx::LoadOp, Operation *> loadMap;
  llvm::DenseMap<ascend_dpx::LoadOp, TracedPointer> loadMemRefs;

  for (auto it = segBegin; it != segEnd; ++it) {
    if (auto storeOp = dyn_cast<ascend_dpx::StoreOp>(&*it)) {
      stores.push_back(storeOp);

      auto storeMemRef = storeOp.getPtr();
      auto storeTrace = traceToUnderlyingObject(storeMemRef, dataLayout);
      auto stv = storeOp.getValue();

      int64_t totalMemOffset = 0;
      if (auto vectorType = dyn_cast<VectorType>(stv.getType())) {
        int64_t elementSize = static_cast<int64_t>(
            dataLayout.getTypeSize(vectorType.getElementType()));
        totalMemOffset = elementSize * vectorType.getNumElements();
      } else {
        totalMemOffset =
            static_cast<int64_t>(dataLayout.getTypeSize(stv.getType()));
      }
      // add the total memory offset to the constant start/extraction offsets to
      // get the end offsets
      storeTrace.endByteOffset = storeTrace.startByteOffset + totalMemOffset;
      storeTrace.extrEndOffset = storeTrace.extrStartOffset + totalMemOffset;

      // for every load that doesn't already have an insert point,
      // if the load/store may reference the same memory, add the
      // current store as the insertion point for that load
      for (auto loadOp : loads) {
        if (!loadMap.contains(loadOp)) {
          auto loadTrace = loadMemRefs[loadOp];
          if (mayRefSameMem(loadTrace, storeTrace)) {
            loadMap[loadOp] = &*it;
            continue;
          }
        }
      }
    } else if (auto loadOp = dyn_cast<ascend_dpx::LoadOp>(&*it)) {
      loads.push_back(loadOp);

      auto loadMemRef = loadOp.getPtr();
      auto loadTrace = traceToUnderlyingObject(loadMemRef, dataLayout);
      auto ltv = loadOp.getResult();

      int64_t totalMemOffset = 0;
      if (auto vectorType = dyn_cast<VectorType>(ltv.getType())) {
        int64_t elementSize = static_cast<int64_t>(
            dataLayout.getTypeSize(vectorType.getElementType()));
        totalMemOffset = elementSize * vectorType.getNumElements();
      } else {
        totalMemOffset =
            static_cast<int64_t>(dataLayout.getTypeSize(ltv.getType()));
      }
      loadTrace.endByteOffset = loadTrace.startByteOffset + totalMemOffset;
      loadTrace.extrEndOffset = loadTrace.extrStartOffset + totalMemOffset;

      // store the traced pointer of each load
      loadMemRefs[loadOp] = loadTrace;
    }
  }

  if (stores.empty())
    return;

  // Build the set of operations that belong to this segment so we can
  // confine dependency walks to within the segment boundary.
  llvm::DenseSet<Operation *> segOps;
  for (auto it = segBegin; it != segEnd; ++it)
    segOps.insert(&*it);

  // `insertPtStores` is the operation at segEnd (a barrier or the block
  // terminator). All scheduled ops will be placed just before it.
  Operation *insertPtStores = &*segEnd;
  llvm::DenseSet<Operation *> scheduledLoads;
  llvm::DenseSet<Operation *> scheduledStores;
  llvm::SmallVector<Operation *>
      scheduleLine; // to track the scheduled operations in order
  scheduledStores.insert(insertPtStores);
  bool loadLine = false;

  // keep track oft he current load and store being scheduled
  Operation *curStore;
  ascend_dpx::LoadOp curLoad;

  // Check if the given load op should be scheduled as part of the current store
  // or not; if the current store appear before the insertion point, then yes,
  // otherwise no
  std::function<bool(const Operation *)> scheduleLoadOp = [&](const Operation *insertPt) {
    for (auto op : stores) {
      if (op.getOperation() == curStore) {
        return true;
      } else if (op.getOperation() == insertPt) {
        return false;
      }
    }
    return true;
  };

  // check if given already-scheduled operation should be re-scheduled as a part
  // of the current load or not; if the operation has already been scheduled
  // before the load's insertion point, then no, otherwise yes
  std::function<bool(const Operation *, const Operation *)> scheduleLoadLineOp =
      [&](const Operation *loadLineOp, const Operation *insertPt) {
        for (auto op : scheduleLine) {
          if (op == insertPt) {
            return true;
          } else if (op == loadLineOp) {
            return false;
          }
        }
        return true;
      };

  // Recursively schedule an operation and all its in-segment dependencies,
  // placing them just before `insertPt`.
  std::function<void(Operation *, Operation *)> scheduleOp =
      [&](Operation *op, Operation *insertPt) {
        if (!segOps.contains(op))
          return;

        // if we're currently scheduling a store and encounter an "unsafe" load,
        // check if we should schedule it
        auto loadOp = dyn_cast<ascend_dpx::LoadOp>(op);
        if (!loadLine && loadOp && loadMap.contains(loadOp) &&
            !scheduleLoadOp(loadMap[loadOp])) {
          return;
        }

        // if an operation has already been scheduled, don't reschedule it
        // unless we are currently scheduling a load and must reschedule to
        // avoid dependency issues
        if (scheduledLoads.contains(op) || scheduledStores.contains(op)) {
          if (loadLine && (scheduledLoads.contains(op) ||
                           !scheduleLoadLineOp(op, loadMap[curLoad]))) {
            return;
          } else if (!loadLine) {
            return;
          }
        }

        // mark the operation as scheduled
        if (loadLine) {
          scheduledLoads.insert(op);
        } else {
          scheduledStores.insert(op);
        }

        // schedule the defining operation for each operand first to avoid
        // dependency issues
        for (auto operand : op->getOperands()) {
          auto *defOp = operand.getDefiningOp();
          if (!defOp || defOp->getBlock() != &block)
            continue;
          scheduleOp(defOp, insertPt);
        }

        // schedule the operation if all its precursors/parents have been
        // scheduled and add it to the scheduleLine to maintain scheduling order
        op->moveBefore(insertPt);
        if (!loadLine)
          scheduleLine.push_back(op);
      };

  // Schedule each store's dependency tree contiguously.
  for (auto storeOp : stores) {
    curStore = storeOp.getOperation();
    scheduleOp(curStore, insertPtStores);
  }

  // update loadLine and schedule each "unsafe" load, using the insert point
  // calculated previously
  loadLine = true;
  for (auto loadOp : loads) {
    curLoad = loadOp;
    if (loadMap.contains(loadOp)) {
      scheduleOp(curLoad.getOperation(), loadMap[loadOp]);
    }
  }

  // Move any remaining unscheduled segment ops before the store trees.
  // We iterate the entire block (not segBegin..segEnd) because scheduling
  // may have moved segBegin, invalidating that iterator range.
  llvm::SmallVector<Operation *> remaining;
  for (auto &op : block) {
    if (segOps.contains(&op) && !scheduledStores.contains(&op) &&
        !scheduledLoads.contains(&op))
      remaining.push_back(&op);
  }
  for (auto *op : remaining)
    op->moveBefore(insertPtStores);
}

/// Process a single basic block using bottom-up store-rooted scheduling.
///
/// After SROA, the struct intermediaries are gone and each store's dependency
/// tree is independent (loads → extractelement → div → insertelement → store).
/// We reorder operations so each store's full dependency tree is contiguous.
///
/// Scheduling respects barrier boundaries (e.g. ascend_dpx.sync_threads):
/// operations are never reordered across a barrier. The block is split into
/// segments at each barrier, and each segment is scheduled independently.
static void scheduleBlock(Block &block, const DataLayout &dataLayout) {
  if (block.empty())
    return;

  // Split the block into segments at barrier operations and schedule each
  // segment independently. This ensures barriers maintain their relative
  // ordering with respect to loads and stores.
  Operation *terminator = block.getTerminator();
  Block::iterator termIt = terminator->getIterator();
  Block::iterator segBegin = block.begin();
  for (auto it = block.begin(); it != termIt; ++it) {
    if (isa<ascend_dpx::SyncThreadsOp>(&*it)) {
      // Schedule the segment [segBegin, it) — everything before this barrier.
      if (segBegin != it)
        scheduleSegment(block, segBegin, it, dataLayout);
      // Skip past the barrier; next segment starts after it.
      segBegin = std::next(it);
    }
  }
  // Schedule the final segment (after the last barrier, up to the terminator).
  if (segBegin != termIt)
    scheduleSegment(block, segBegin, termIt, dataLayout);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SinkDPXLoadPass : public impl::SinkDPXLoadBase<SinkDPXLoadPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    DataLayout dataLayout(mod);
    mod->walk([&](LLVM::LLVMFuncOp func) {
      for (auto &block : func.getBody()) {
        // Phase 1: Break struct pack/unpack patterns.
        sroaStructs(block);
        removeDeadOps(block);

        // Phase 2: Reorder for minimal register pressure.
        scheduleBlock(block, dataLayout);
      }
    });
  }
};

} // namespace

} // namespace mlir::triton::ascend
