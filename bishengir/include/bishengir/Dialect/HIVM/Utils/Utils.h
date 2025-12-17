//===- Utils.h - Utilities to support the HIVM dialect -----------*- C++-*-===//
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

#ifndef MLIR_DIALECT_HIVM_UTILS_UTILS_H
#define MLIR_DIALECT_HIVM_UTILS_UTILS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <cstdint>
#include <queue>
#include <set>
#include <type_traits>

namespace mlir {
namespace utils {
// Value comparator for std::map
inline bool isLessValue(const Value &a, const Value &b) {
  return a.getImpl() < b.getImpl();
}

struct ValueComparator {
  bool operator()(const Value &a, const Value &b) const {
    return isLessValue(a, b);
  }
};
} // namespace utils

namespace hivm {

// TODO : put it into platform info
#define MASTK_MODE_CTROL_BIT 56

static constexpr llvm::StringLiteral kMappingAttrName = "mapping";
static constexpr llvm::StringLiteral kMapForToForallAttrName =
    "map_for_to_forall";

/// TODO: add into hivm attrs
static constexpr llvm::StringLiteral kBufferSizeInByteAttr =
    "buffer_size_in_byte";

static constexpr llvm::StringLiteral kLogicalBlockNumAttr = "logical_block_num";

const std::string Ascend910BCubeTriple = "ascend_910b_cube-unknown-cce-unknown";
const std::string Ascend910BDataLayout =
    "e-i1:8:32-i8:8:32-i16:16:32-i64:64-f16:16:32-v16:16-v32:32-n64-S64";

// The amount of data processed by the VBITSORT instruction in one repeat.
constexpr const int VBITSORT_NUM_PER_REPEAT = 32;

const std::set<hivm::AddressSpace> LocalBufferSpace{
    hivm::AddressSpace::UB, hivm::AddressSpace::L1, hivm::AddressSpace::L0C};

const std::map<TFuncCoreType, TCoreType> kTFuncCoreType2TCoreType = {
    {TFuncCoreType::AIC, TCoreType::CUBE},
    {TFuncCoreType::AIV, TCoreType::VECTOR},
    {TFuncCoreType::MIX, TCoreType::CUBE_OR_VECTOR},
};

/// Set the input type's memory scope to the input HIVM Address Space.
void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope);

/// Get the root MemRef AllocOp for the input operand, return failure if there
/// is unsupported Ops on the search path or if the defining op is not a MemRef
/// AllocOp.
FailureOr<memref::AllocOp> getMemRefAlloc(Value operand);

SmallVector<Value>
getValueListFromMixedTypeLists(SmallVector<Value> dynamicValues,
                               ArrayRef<int64_t> staticValues, Location loc,
                               OpBuilder &builder);

// Get value's shape as operands.
FailureOr<SmallVector<Value>> getValueFromShape(Value currentValue,
                                                OpBuilder &builder);

bool IsAscend910B(Attribute triple);

/// Returns the result of MLIR's alignUp operation on constants. The RHS is
/// expected to be non-zero.
uint64_t AlignUp(uint64_t lhs, uint64_t rhs);

/// Obtain the current number of supported pipe flow types.
constexpr unsigned int getPipeNum() {
  return static_cast<unsigned int>(hivm::PIPE::PIPE_NUM);
}

/// Determine value as buffer type.
std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand);

/// Get operation all touch buffer.
SmallVector<Value> getOpTouchBuffer(Operation *op);

/// Determine whether there is a Local Buffer in the current operation.
bool isOpTouchLocalBuffer(Operation *op);

/// Determine whether there is in ub buffer.
bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr);

/// Determine whether there is a global Buffer in the current operation.
bool isOpTouchGlobalBuffer(Operation *op);

/// Utilities for Map Forall To HIVMBlocks pass and transform op
struct ForallRewriteResult {
  Value mappingId;
};

/// Eliminates scf.forall ops, move their bodies to their current location, and
/// replace uses of the index variable with delinearized hivm blk idx, via
/// affine.delinearize_index.
///
/// Requires forallOp to be the top level forall of a nest, and all forall's be
/// normalized. Dynamic upper bounds are ok.
DiagnosedSilenceableFailure mapForallToBlocksImpl(
    RewriterBase &rewriter, scf::ForallOp forallOp, ForallRewriteResult &result,
    std::optional<transform::TransformOpInterface> transformOp = std::nullopt);

/// Remove attr from markOp, and remove markOp if no attr left.
void removeMarkOpAttr(annotation::MarkOp markOp, ::llvm::StringLiteral attrName,
                      bool removeOp = true);

// Remove attr from markOp, but use rewriter
void removeMarkOpAttr(annotation::MarkOp markOp, StringRef attrName,
                      RewriterBase &rewriter, bool removeOp = true);

// Check whether current for loop is subblock binded.
bool isSubBlockBindedFor(scf::ForOp op);

// Find containing subblock loop of current op.
FailureOr<scf::ForOp> findContainingSubblockLoop(Operation *op);

/// Get parent loop of val.
/// If val is yielded by the parent loop, need to get parent of parent loop.
LoopLikeOpInterface getParentLoop(Value val);

/// Flatten ptrCastOp's parent and ancestor loops into one dimension and then
/// modulo modular.
/// In the position of ptrCastOp, affineApply and indexCastOp would be
/// created.
///
/// \return IndexCastOp of affineApply
Value createNestedIndexModular(OpBuilder &builder, Operation *op,
                               int modular = 2);

Value createNestedIndexForOp(OpBuilder &builder, Operation *operation);

/// Create nested loops by choosing `loopDims` of `target`.
/// For example:
///  `target` = memref<Ax16xBxf32>
///  `loopDims` = {0, 1}
/// The generated loops are:
///   scf.for 0 ... A
///     scf.for 0 ... 16
/// The optional `lowBound` can be used to specify the lower bound.
/// The optional `forInitArgs` can be used to specify the iter arg's initial
/// value.
/// For example:
///   scf.for lowBound ... A iter_arg(%iter1 = forInitArgs[0])
///     scf.for lowBound ... 16 iter_arg(%iter2 = forInitArgs[1])
template <typename Func>
std::vector<scf::ForOp> createNestedLoops(
    OpBuilder &rewriter, Location loc, Value target, std::set<int> loopDims,
    Func buildLoopBody, int lowBound = 0,
    std::optional<SmallVector<Value>> forInitArgs = std::nullopt) {
  std::vector<scf::ForOp> nestedFor;
  llvm::SmallVector<Value> indexes;
  if (forInitArgs.has_value())
    assert(loopDims.size() == 1 &&
           "Only support non-nested loop to use iterator arg");

  auto index = [&rewriter, &loc](int i) {
    return rewriter.create<arith::ConstantIndexOp>(loc, i);
  };
  ShapedType dstType = dyn_cast<ShapedType>(target.getType());
  assert(dstType != nullptr);
  for (int dim = 0; dim < dstType.getRank(); dim++) {
    if (!loopDims.count(dim))
      continue;
    Value upperBound;
    if (dstType.isDynamicDim(dim)) {
      upperBound = rewriter.create<memref::DimOp>(loc, target, dim);
    } else {
      upperBound = index(dstType.getDimSize(dim));
    }
    scf::ForOp forOp =
        forInitArgs.has_value()
            ? rewriter.create<scf::ForOp>(loc, index(lowBound), upperBound,
                                          index(1), forInitArgs.value())
            : rewriter.create<scf::ForOp>(loc, index(lowBound), upperBound,
                                          index(1));
    nestedFor.push_back(forOp);
    indexes.push_back(forOp.getInductionVar());
    rewriter.setInsertionPointToStart(forOp.getBody());
  }
  if constexpr (std::is_invocable_v<Func, SmallVector<Value>>)
    buildLoopBody(indexes);
  else
    buildLoopBody(indexes, nestedFor[0].getRegionIterArgs());

  return nestedFor;
}

// Util func `traceForPotentialMatrixC` aims to judge whether current operand
// value of store-related operation could come from matmul result MatrixC.
//
// And it should be used with fixpipe optimization.
FailureOr<SmallVector<Operation *>> traceForPotentialMatrixC(Value v,
                                                             Block *storeBlock);

bool isMarkedAsHIVMElementwiseOp(Operation *op);

bool isMixModule(ModuleOp mod);

bool isAICModule(ModuleOp mod);

bool isAIVModule(ModuleOp mod);

/// Getter setter of the hivm.module_core_type attribute.
TModuleCoreTypeAttr getModuleCoreTypeAttr(ModuleOp mod);
void setModuleCoreTypeAttr(ModuleOp mod, TModuleCoreType coreType);
void removeModuleCoreTypeAttr(ModuleOp mod);

/// Get user op of the 'op'
/// Constraints: Skip tensor::CollapseShapeOp/ExpandShapeOp
/// Constraints: Skip memref::CollapseShapeOp/ExpandShapeOp
/// Constraints: Skip memref::SubViewOp/ViewOp/ReinterpretCastOp
/// Constraints: Skip bufferization::ToMemrefOp
void getOpUsers(Operation *op, SmallVector<Operation *, 8> &userOps);

// Create local workspace of current block
Value createAllocLocalWorkSpace(OpBuilder &builder, Location loc,
                                SmallVector<int64_t> shape, Type elementType);

Value getLocalWorkSpaceTensor(PatternRewriter &rewriter, Location loc,
                              ArrayRef<int64_t> targetShapes, Type elementType);

// Create local lock var
hivm::CreateSyncBlockLockOp createSyncBlockLockVar(OpBuilder &builder,
                                                   Location loc);

/// get Operation alias pair.
std::vector<std::pair<Value, Value>> getOperationAliasInfo(Operation *op);

/// Get buffer static size.
std::optional<uint32_t> GetBufferSize(Value buffer);

// get is operation aligned according to the broadcast/reduce dim and rank
AlignKind isBrcOpAligned(VBrcOp vbrcOp, int dim, int rank);

// set bind sub block attr
void setSubBlockMapping(RewriterBase &rewriter, Operation *loop);

/// find vector ops between store and targetOp
template <typename OpType>
LogicalResult traceHIVMOpUntil(RewriterBase &rewriter, Operation *op,
                               SmallVector<Operation *> &tracedOps) {
  std::queue<Operation *> q;
  q.push(op);
  auto parentOp = op->getParentOp();

  while (!q.empty()) {
    Operation *curOp = q.front();
    q.pop();

    if (parentOp != curOp->getParentOp())
      return failure();

    if (isa<OpType>(curOp)) {
      assert(tracedOps.size() >= 1 && "there should be vector ops");
      tracedOps.push_back(curOp);
      return success();
    }

    for (const Value &src : curOp->getOperands()) {
      Operation *defOp = src.getDefiningOp();
      if (defOp != nullptr)
        q.push(defOp);
    }

    if (curOp->getDialect()->getNamespace() ==
        HIVMDialect::getDialectNamespace()) {
      tracedOps.push_back(curOp);
    }
  }

  return failure();
}

namespace util {
enum class BitWidth : uint32_t {
  B1 = 1,
  B4 = 4,
  B8 = 8,
  B16 = 16,
  B32 = 32,
  B64 = 64,
};
  
constexpr static unsigned int VL = 256;
constexpr static unsigned int BL = VL / 8;
const static int vectorBlockSizeBit = 256;
const static int srcNumPerRepeatOfVBRCBIntrin = 8;

constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;
constexpr static unsigned int VNCHWCONV_INTR_BYTES_PER_REPEAT = 512;

// Returns if the given source MemRef type is collapsible with the specified
// reassociation indices. This function works as a strict extension based
// on `memref::CollapseShapeOp::isGuaranteedCollapsible`, which has weak
// constraints on the strides of trailing one-size dimensions.
bool isGuaranteedCollapsibleStrictly(
    MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);

/// Return the MemRefTypes
SmallVector<MemRefType> getMemRefTypes(TypeRange types);

/// Judge if all MemRefTypes has same rank value
bool isAllSameRank(const SmallVectorImpl<MemRefType> &memrefTypes);

inline int64_t ceilFactor(int64_t x, int64_t y) { return (x + y - 1) / y * y; }

bool isLastDimContiguous(Value operand);

/// Check if the operation is hivm::PointerCastOp with GM space
/// Used to check if it is lowered from triton::IntToPtrOp
bool isGMPointerCastOp(Operation *op);

bool isArgminOrArgmax(ReduceOperation op);

} // namespace util
} // namespace hivm
} // namespace mlir

// TODO : move to platform file
const std::set<std::string> HWSupportedCast{
    "bfloat16_t_to_float_rintmode",   "bfloat16_t_to_int32_t_roundmode",
    "bfloat16_t_to_int32_t_ceilmode", "bfloat16_t_to_int32_t_floormode",
    "bfloat16_t_to_int32_t_rintmode", "bfloat16_t_to_int32_t_truncmode",
    "half_to_float_roundmode",        "half_to_float_floormode",
    "half_to_float_rintmode",         "half_to_int16_t_roundmode",
    "half_to_int16_t_ceilmode",       "half_to_int16_t_floormode",
    "half_to_int16_t_rintmode",       "half_to_int16_t_truncmode",
    "half_to_int32_t_roundmode",      "half_to_int32_t_ceilmode",
    "half_to_int32_t_floormode",      "half_to_int32_t_rintmode",
    "half_to_int32_t_truncmode",      "half_to_int4_t_roundmode",
    "half_to_int4_t_ceilmode",        "half_to_int4_t_floormode",
    "half_to_int4_t_rintmode",        "half_to_int4_t_truncmode",
    "half_to_int8_t_roundmode",       "half_to_int8_t_ceilmode",
    "half_to_int8_t_floormode",       "half_to_int8_t_rintmode",
    "half_to_int8_t_truncmode",       "half_to_uint8_t_roundmode",
    "half_to_uint8_t_ceilmode",       "half_to_uint8_t_floormode",
    "half_to_uint8_t_rintmode",       "half_to_uint8_t_truncmode",
    "float_to_bfloat16_t_roundmode",  "float_to_bfloat16_t_ceilmode",
    "float_to_bfloat16_t_floormode",  "float_to_bfloat16_t_rintmode",
    "float_to_bfloat16_t_truncmode",  "float_to_half_roundmode",
    "float_to_half_ceilmode",         "float_to_half_floormode",
    "float_to_half_oddmode",          "float_to_half_rintmode",
    "float_to_half_truncmode",        "float_to_float_roundmode",
    "float_to_float_ceilmode",        "float_to_float_floormode",
    "float_to_float_rintmode",        "float_to_float_truncmode",
    "float_to_int16_t_roundmode",     "float_to_int16_t_ceilmode",
    "float_to_int16_t_floormode",     "float_to_int16_t_rintmode",
    "float_to_int16_t_truncmode",     "float_to_int32_t_roundmode",
    "float_to_int32_t_ceilmode",      "float_to_int32_t_floormode",
    "float_to_int32_t_rintmode",      "float_to_int32_t_truncmode",
    "float_to_int64_t_roundmode",     "float_to_int64_t_ceilmode",
    "float_to_int64_t_floormode",     "float_to_int64_t_rintmode",
    "float_to_int64_t_truncmode",     "int16_t_to_half_roundmode",
    "int16_t_to_half_ceilmode",       "int16_t_to_half_floormode",
    "int16_t_to_half_rintmode",       "int16_t_to_half_truncmode",
    "int16_t_to_float_rintmode",      "int16_t_to_float_truncmode",
    "int32_t_to_float_roundmode",     "int32_t_to_float_ceilmode",
    "int32_t_to_float_floormode",     "int32_t_to_float_rintmode",
    "int32_t_to_float_truncmode",     "int32_t_to_int16_t_rintmode",
    "int32_t_to_int64_t_rintmode",    "int4_t_to_half_rintmode",
    "int64_t_to_float_roundmode",     "int64_t_to_float_ceilmode",
    "int64_t_to_float_floormode",     "int64_t_to_float_rintmode",
    "int64_t_to_float_truncmode",     "int64_t_to_int32_t_rintmode",
    "int8_t_to_half_rintmode",        "int8_t_to_half_truncmode",
    "uint8_t_to_half_rintmode",       "half_to_int32_t_rintmode",
    "half_to_float_truncmode",        "bfloat16_t_to_float_roundmode"};

#endif // MLIR_DIALECT_HIVM_UTILS_UTILS_H
