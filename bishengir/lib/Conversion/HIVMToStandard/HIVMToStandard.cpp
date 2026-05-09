//===- HIVMToStandard.cpp - conversion from HIVM to Standard dialect ------===//
//
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

#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

static MemRefType makeStridedLayoutAndShapeDynamic(MemRefType type) {
  return MemRefType::Builder(type)
      .setLayout(StridedLayoutAttr::get(
          type.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)))
      .setShape(SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));
}

static SmallVector<Type, 4>
extractOperandTypes(const SmallVectorImpl<Value> &operands) {
  SmallVector<Type, 4> result;
  result.reserve(operands.size());
  for (const auto &operand : operands) {
    auto type = operand.getType();
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = dyn_cast<MemRefType>(type))
      result.push_back(makeStridedLayoutAndShapeDynamic(memrefType));
    else
      result.push_back(type);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/LinalgToStandard/LinalgToStandard.cpp
//===----------------------------------------------------------------------===//

static SmallVector<Value, 4>
createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc,
                                      ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = dyn_cast<MemRefType>(op.getType());
    if (!memrefType) {
      res.push_back(op);
      continue;
    }
    Value cast = b.create<memref::CastOp>(
        loc, makeStridedLayoutAndShapeDynamic(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

static func::CallOp createLibCall(PatternRewriter &rewriter, Operation *op,
                                  ModuleOp mod, const std::string &libCallName,
                                  const SmallVector<Value> &inputOperands,
                                  TypeRange resultTypes) {
  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  FlatSymbolRefAttr fnNameAttr = SymbolRefAttr::get(ctx, libCallName);
  if (!mod.lookupSymbol(fnNameAttr.getAttr())) {
    auto libFnType = rewriter.getFunctionType(
        extractOperandTypes(inputOperands), resultTypes);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));

    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        mlir::FileLineColLoc::get(ctx, "internal", 0, 0), fnNameAttr.getValue(),
        libFnType);
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));

    auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
    funcOp->setAttr(haccAlwaysInlineAttr, rewriter.getUnitAttr());

    funcOp.setPrivate();

    // label a func core type attribute to the lib call decl, based on either
    // special cases such as debug helper functions or the core
    // type of original op.
    if (llvm::StringRef(libCallName).starts_with("_mlir_ciface_init_debug") ||
        llvm::StringRef(libCallName).starts_with("_mlir_ciface_finish_debug")) {
      funcOp->setAttr(
          mlir::hivm::TFuncCoreTypeAttr::name,
          hivm::TFuncCoreTypeAttr::get(funcOp->getContext(),
                                       hivm::TFuncCoreType::AIC_OR_AIV));
    } else if (auto infer = dyn_cast<CoreTypeInterface>(op)) {
      auto coreTypeMaybe = infer.getCoreType();
      if (coreTypeMaybe) {
        hivm::TFuncCoreType fc{};
        switch (coreTypeMaybe.value()) {
        case hivm::TCoreType::CUBE:
          fc = hivm::TFuncCoreType::AIC;
          break;
        case hivm::TCoreType::VECTOR:
          fc = hivm::TFuncCoreType::AIV;
          break;
        case hivm::TCoreType::CUBE_OR_VECTOR:
        case hivm::TCoreType::CUBE_AND_VECTOR:
          llvm::report_fatal_error(
              "standard library call shouldn't have mix core type!");
          break;
        }

        funcOp->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
                        hivm::TFuncCoreTypeAttr::get(op->getContext(), fc));
      }
    }
  }

  return rewriter.create<func::CallOp>(
      loc, fnNameAttr.getValue(), resultTypes,
      createTypeCanonicalizedMemRefOperands(rewriter, loc, inputOperands));
}

template <typename OpType>
static void replaceWithLibCall(PatternRewriter &rewriter, OpType op,
                               const std::string &libCallName,
                               const SmallVector<Value> &inputOperands,
                               TypeRange resultTypes) {
  ModuleOp mod = op->template getParentOfType<ModuleOp>();
  func::CallOp call =
      createLibCall(rewriter, op, mod, libCallName, inputOperands, resultTypes);

  rewriter.replaceOp(op, call);
}

static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                       MemRefType sourceRankedTensorType,
                                       ArrayRef<int64_t> offsets,
                                       ArrayRef<int64_t> sizes,
                                       ArrayRef<int64_t> strides,
                                       const std::set<int> &dims) {
  auto inferredType = llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
      sourceRankedTensorType, offsets, sizes, strides));
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  // Compute the layout and result type.
  auto inferredLayout = llvm::cast<StridedLayoutAttr>(inferredType.getLayout());
  SmallVector<int64_t> rankReducedStrides;
  rankReducedStrides.reserve(resultShape.size());
  for (auto [idx, value] : llvm::enumerate(inferredLayout.getStrides())) {
    if (dims.find(idx) == dims.end())
      rankReducedStrides.push_back(value);
  }
  return MemRefType::get(resultShape, inferredType.getElementType(),
                         StridedLayoutAttr::get(inferredLayout.getContext(),
                                                inferredLayout.getOffset(),
                                                rankReducedStrides),
                         inferredType.getMemorySpace());
}

static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                       MemRefType sourceRankedTensorType,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides,
                                       const std::set<int> &dims) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return inferRankReducedResultType(resultShape, sourceRankedTensorType,
                                    staticOffsets, staticSizes, staticStrides,
                                    dims);
}

/// Using reducedAxes to specify arbitrary axis to create forOp.
/// \param reducedAxes reducedAxes to create forOp
/// \return SubView of memrefValsMaybe
static SmallVector<Value>
reduceMemrefsToNestedForUsingAxes(PatternRewriter &rewriter, Location loc,
                                  ValueRange memrefValsMaybe,
                                  std::set<int> reducedAxes) {
  Value memrefVal;
  for (auto val : memrefValsMaybe) {
    if (isa<MemRefType>(val.getType())) {
      memrefVal = val;
      break;
    }
  }
  if (!memrefVal) {
    // nothing needs to be reduced
    return memrefValsMaybe;
  }

  MemRefType vecType = cast<MemRefType>(memrefVal.getType());
  int64_t rank = vecType.getRank();

  for (auto val : memrefValsMaybe) {
    if (isa<MemRefType>(val.getType())) {
      if (dyn_cast<MemRefType>(val.getType()).getRank() != rank) {
        // do nothing for invalid ranks
        return memrefValsMaybe;
      }
    }
  }

  std::vector<scf::ForOp> nestedFor;
  OpBuilder::InsertionGuard guard(rewriter);

  SmallVector<Value> reducedVals;
  auto buildLoopBody =
      [&rewriter, &loc, &memrefValsMaybe, &reducedAxes, &rank,
       &reducedVals](llvm::SmallVector<Value> indexes) -> void {
    for (auto val : memrefValsMaybe) {
      MemRefType vecType = dyn_cast<MemRefType>(val.getType());
      if (!vecType) {
        reducedVals.push_back(val);
        continue;
      }

      SmallVector<OpFoldResult> viewOffset, viewSize,
          viewStride(rank, rewriter.getIndexAttr(1));
      std::vector<int64_t> reducedSize;

      int nestedForIdx = 0;
      for (int dim = 0; dim < rank; dim++) {
        if (reducedAxes.count(dim)) {
          viewOffset.push_back(indexes[nestedForIdx++]);
          viewSize.push_back(rewriter.getIndexAttr(1));
        } else {
          viewOffset.push_back(rewriter.getIndexAttr(0));
          viewSize.push_back(memref::getMixedSize(rewriter, loc, val, dim));

          reducedSize.push_back(vecType.getDimSize(dim));
        }
      }

      MemRefType reducedType = cast<MemRefType>(inferRankReducedResultType(
          reducedSize, vecType, viewOffset, viewSize, viewStride, reducedAxes));

      reducedVals.push_back(rewriter.create<memref::SubViewOp>(
          loc, reducedType, val, viewOffset, viewSize, viewStride));
    }
  };
  createNestedLoops(rewriter, loc, memrefVal, reducedAxes, buildLoopBody);

  return reducedVals;
}

// reduce the rank of the values(with type MemRefType) inside memrefValsMaybe,
// by wrapping them inside nested for loops. the scalar values inside will be
// returned intact.
//
// the indices of the reducing axis are in the range of [forRangeStart,
// forRangeEnd). these axis must have the same size in each reduced dimension.
//
// every reduced value will be converted to a subview of the original value,
// residing in the innest loop. all value reduced are aggregated in a collection
// and return. the sequence of the returned collection are guaranteed to obey
// the lexicial sequence in the block.
//
// Example:
//   memrefValsMaybe = {v1 : memref<1x8x16x32x64x128xi16>,
//                      v2 : memref<10x8x16x32x64x128xi16>,
//                      v3 : i16}
//   forRangeStart = 1
//   forRangeEnd = 4
//
//   will generate:
//     for i in [0, 8]
//       for j in [0, 16]
//         for k in [0, 32]
//           v1' = subview of v1 : memref<1x64x128xi16>
//           v2' = subview of v2 : memref<10x64x128xi16>
//     return {v1', v2', v3}
static SmallVector<Value> reduceMemrefsToNestedFor(PatternRewriter &rewriter,
                                                   Location loc,
                                                   ValueRange memrefValsMaybe,
                                                   int forRangeStart,
                                                   int forRangeEnd) {
  std::set<int> indices;
  for (int i = forRangeStart; i < forRangeEnd; ++i) {
    indices.insert(i);
  }

  return reduceMemrefsToNestedForUsingAxes(rewriter, loc, memrefValsMaybe,
                                           indices);
}

//===----------------------------------------------------------------------===//
// Patterns to convert a HIVMOp to func.call @external library
// implementation.
//===----------------------------------------------------------------------===//
namespace mlir::hivm {
class MmadL1OpToLibraryCallPattern : public OpRewritePattern<hivm::MmadL1Op> {
public:
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(MmadL1Op op,
                                PatternRewriter &rewriter) const final {
    // inputs
    SmallVector<Value> libParams =
        op.getInputOperands(/*includeSyncRelatedArgs=*/false);

    // outputs
    libParams.push_back(op.getC());

    // additional sync arguments
    SmallVector<Value> additionalArgs;
    genAdditionalFunctionArgs(op, additionalArgs, rewriter);
    libParams.append(additionalArgs.begin(), additionalArgs.end());

    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }

private:
  void genAdditionalFunctionArgs(MmadL1Op op,
                                 SmallVector<Value> &additionalArgs,
                                 PatternRewriter &rewriter) const {
    if (op.getSyncRelatedArgs().empty()) {
      auto negOneDefaultValue = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(-1));
      op.getSyncRelatedArgsMutable().assign(ValueRange(
          SmallVector<Value>(op.getNumSyncRelatedArgs(), negOneDefaultValue)));
    }

    auto syncRelatedArgs = op.getSyncRelatedArgs();
    std::copy(syncRelatedArgs.begin(), syncRelatedArgs.end(),
              std::back_inserter(additionalArgs));

    additionalArgs.push_back(op.getUnitFlagModeLibValue(rewriter));
  }
};

class Conv1DL1OpToLibraryCallPattern
    : public OpRewritePattern<hivm::Conv1DL1Op> {
public:
  using OpRewritePattern<hivm::Conv1DL1Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::Conv1DL1Op op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value> libParams = op.getLibraryCallOperands(rewriter);

    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }
};

class Conv2DL1OpToLibraryCallPattern
    : public OpRewritePattern<hivm::Conv2DL1Op> {
public:
  using OpRewritePattern<hivm::Conv2DL1Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::Conv2DL1Op op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value> libParams = op.getLibraryCallOperands(rewriter);

    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }
};

class ND2NZOpToLibraryCallPattern : public OpRewritePattern<hivm::ND2NZOp> {
public:
  using OpRewritePattern<hivm::ND2NZOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ND2NZOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.getDstContinuous()) {
      op->emitError(
          "ND2NZOp's library function implementation requires continuous dst!");
      return failure();
    }

    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       op->getOperands(), {});
    return success();
  }
};

class NZ2NDOpToLibraryCallPattern : public OpRewritePattern<hivm::NZ2NDOp> {
  using OpRewritePattern<hivm::NZ2NDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::NZ2NDOp op,
                                PatternRewriter &rewriter) const final {
    // TODO: merge this with ND2NZOpToLibraryCallPattern
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       op->getOperands(), {});
    return success();
  }
};

class FixpipeOpToLibraryCallPattern : public OpRewritePattern<hivm::FixpipeOp> {
public:
  using OpRewritePattern<hivm::FixpipeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Value> additionalArgs;
    genAdditionalFunctionArgs(op, additionalArgs, rewriter);

    SmallVector<Value> libCallOperands = op->getOperands();
    libCallOperands.resize(2); // src, dst

    libCallOperands.append(additionalArgs);

    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libCallOperands,
                       /*resultTypes=*/{});
    return success();
  }

private:
  static void genPreQuant(FixpipeOp op, PatternRewriter &rewriter,
                          Value &preQuant) {
    // TODO, after inlinefipipe finish, delete this func
    if (static_cast<uint32_t>(op.getPreQuant()) != 0) {
      return;
    }

    Type srcElemType = getElementTypeOrSelf(op.getSrcOperandType());
    Type dstElemType = getElementTypeOrSelf(op.getDstOperandType());
    if (srcElemType.isF32() && dstElemType.isF16()) {
      const int32_t preQuantVal =
          static_cast<int32_t>(FixpipePreQuantMode::F322F16);
      preQuant =
          rewriter.create<arith::ConstantIntOp>(op->getLoc(), preQuantVal, 64);
      return;
    }
    if (srcElemType.isF32() && dstElemType.isBF16()) {
      const int32_t preQuantVal =
          static_cast<int32_t>(FixpipePreQuantMode::F322BF16);
      preQuant =
          rewriter.create<arith::ConstantIntOp>(op->getLoc(), preQuantVal, 64);
      return;
    }
    if (srcElemType.isInteger(32) && // special case source of 32 bits
        dstElemType.isInteger(8)) {  // special case target of 8 bits
      const int32_t preQuantVal =
          static_cast<int32_t>(FixpipePreQuantMode::S322I8);
      preQuant =
          rewriter.create<arith::ConstantIntOp>(op->getLoc(), preQuantVal, 64);
      return;
    }
  }

  void genAdditionalFunctionArgs(FixpipeOp op,
                                 SmallVector<Value> &additionalArgs,
                                 PatternRewriter &rewriter) const {
    const uint32_t preQuantVal = static_cast<uint32_t>(op.getPreQuant());
    Value preQuant =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), preQuantVal, 64);
    const uint32_t preReluVal = static_cast<uint32_t>(op.getPreRelu());
    Value preRelu =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), preReluVal, 64);
    Value channelSplit = rewriter.create<arith::ConstantIntOp>(
        op->getLoc(), op.getChannelSplit(), 1);
    Value unitFlagMode = op.getUnitFlagModeLibValue(rewriter);

    genPreQuant(op, rewriter, preQuant);
    additionalArgs.push_back(preQuant);
    additionalArgs.push_back(preRelu);
    additionalArgs.push_back(channelSplit);
    additionalArgs.push_back(unitFlagMode);
  }
};

//===----------------------------------------------------------------------===//
// Multi-dimensional Op Translation
//===----------------------------------------------------------------------===//

template <typename SourceOp>
class MultiDimOpToLibraryCallPattern : public OpRewritePattern<SourceOp> {
public:
  explicit MultiDimOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<SourceOp>(context, benefit, generatedNames) {}
  virtual ~MultiDimOpToLibraryCallPattern() = default;

protected:
  /// Creates a constant index type.
  Value constantIndex(PatternRewriter &rewriter, Location lc, int64_t i) const {
    return rewriter.create<arith::ConstantIndexOp>(lc, i);
  }

  /// Creates a constant I32 type.
  Value constantI32(PatternRewriter &rewriter, Location loc, int64_t i) const {
    return rewriter.create<arith::ConstantIntOp>(loc, i,
                                                 32); // for const 32 type
  }

  Value constant(PatternRewriter &rewriter, Location loc, int64_t i,
                 Type t) const {
    return rewriter.create<arith::ConstantIntOp>(loc, i, t);
  }

  virtual SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const {
    SmallVector<Value> hivmOperands;
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    for (OpOperand *operand :
         structuredOp.getHIVMOperands(includeExtraBuffer)) {
      Value value = operand->get();
      hivmOperands.push_back(value);
    }
    return hivmOperands;
  }
};

template <typename CumOp>
class CumOpToLibraryCallPattern : public MultiDimOpToLibraryCallPattern<CumOp> {
public:
  explicit CumOpToLibraryCallPattern(MLIRContext *context,
                                     PatternBenefit benefit = 1,
                                     ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<CumOp>(context, benefit,
                                              generatedNames) {}

  LogicalResult matchAndRewrite(CumOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    auto src = op.getSrc();
    auto dst = op.getDst();
    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    MemRefType srcVecType = cast<MemRefType>(op.getSrc().getType());
    int rank = srcVecType.getRank();
    llvm::ArrayRef<int64_t> cumDims = op.getCumDims();
    assert(!cumDims.empty() && "cumDims shouldn't be empty.");
    if (cumDims.size() > 1) {
      return op.emitError("cum dimensions array is not decomposed yet");
    }
    int64_t cumDim = cumDims[0];
    if (cumDim == rank - 1) {
      return op.emitError("cum dimension with last dimension should be "
                          "decomposed to scalar operation");
    }

    // For loop axes would be used to create forOp.
    auto axes = getForLoopAxes(cumDim, rank);
    SmallVector<Value> operands = reduceMemrefsToNestedForUsingAxes(
        rewriter, op.getLoc(), {src, dst}, axes);

    rewriter.setInsertionPointAfter(
        operands[operands.size() - 1].getDefiningOp());

    auto libCallName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    createLibCall(rewriter, op, mod, libCallName, operands, {});
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// The for loop axes would be used to create forOp.
  std::set<int> getForLoopAxes(int64_t cumDim, int maxRank) const {
    std::set<int> res;
    /// in ra condition, the rank of memref after convert to standard is 2,
    // and
    // the cumsum dim should be the first xis, and the 'a' dimension is the
    // last axis
    std::set<int> keepAxes{static_cast<int>(cumDim), maxRank - 1};
    if (cumDim > 0) {
      keepAxes.insert(cumDim - 1);
    }
    for (int idx = 0; idx < maxRank; ++idx) {
      if (keepAxes.count(idx) == 0) {
        res.insert(idx);
      }
    }
    return res;
  }
};

/// Adds dimension identifier for different multi-dimensional ops.
std::string addDimName(int64_t rank, int64_t maxOpRank) {
  assert(rank > 0 && "Invalid operand rank.");
  assert(maxOpRank > 0 && "Invalid max operand rank.");
  return std::to_string(std::min(rank, maxOpRank)) + "d";
}

class MatmulOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::MatmulOp> {
public:
  explicit MatmulOpToLibraryCallPattern(MLIRContext *context,
                                        PatternBenefit benefit = 1,
                                        ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::MatmulOp>(context, benefit,
                                                       generatedNames) {}

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    SmallVector<Value> libParams = {op.getA(), op.getB(), op.getC()};
    if (op.getBias()) {
      libParams.push_back(op.getBias());
    }
    if (op.getDescale() && op.getDescaleMode() &&
        op.getDescaleMode().value() != hivm::DescaleMode::DescaleNull) {
      libParams.push_back(op.getDescale());
    }
    if (op.getTilingParams()) {
      libParams.push_back(op.getTilingParams());
    }
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }
};

class MixMatmulOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::MixMatmulOp> {
public:
  explicit MixMatmulOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::MixMatmulOp>(context, benefit,
                                                          generatedNames) {}

  LogicalResult matchAndRewrite(MixMatmulOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    SmallVector<Value> libParams = {op.getA(), op.getB(), op.getC()};
    if (op.getBias()) {
      libParams.push_back(op.getBias());
    }
    if (op.getDescale() && op.getDescaleMode() &&
        op.getDescaleMode().value() != hivm::DescaleMode::DescaleNull) {
      libParams.push_back(op.getDescale());
    }
    for (auto vecIn : op.getPostVecFuncIns()) {
      libParams.push_back(vecIn);
    }
    for (auto workIn : op.getWorkspaceIns()) {
      libParams.push_back(workIn);
    }
    if (op.getTilingParams()) {
      libParams.push_back(op.getTilingParams());
    }
    if (op.getCommParams()) {
      libParams.push_back(op.getCommParams());
    }
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }
};

class MixGroupMatmulOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::MixGroupMatmulOp> {
public:
  explicit MixGroupMatmulOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::MixGroupMatmulOp>(context, benefit,
                                                               generatedNames) {
  }

  LogicalResult matchAndRewrite(MixGroupMatmulOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    SmallVector<Value> libParams = {op.getB(), op.getA(), op.getC()};
    if (op.getBias()) {
      libParams.push_back(op.getBias());
    }
    if (op.getDescale() && op.getDescaleMode() &&
        op.getDescaleMode().value() != hivm::DescaleMode::DescaleNull) {
      libParams.push_back(op.getDescale());
    }
    for (auto vecOut : op.getPostVecFuncOuts()) {
      libParams.push_back(vecOut);
    }
    for (auto vecIn : op.getPostVecFuncIns()) {
      libParams.push_back(vecIn);
    }
    libParams.push_back(op.getTokensPerExpert());
    for (auto workIn : op.getWorkspaceIns()) {
      libParams.push_back(workIn);
    }
    if (op.getTilingParams()) {
      libParams.push_back(op.getTilingParams());
    }
    if (op.getCommParams()) {
      libParams.push_back(op.getCommParams());
    }
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       libParams, {});
    return success();
  }
};

template <typename CopyOpType>
class CopyOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<CopyOpType> {
public:
  explicit CopyOpToLibraryCallPattern(MLIRContext *context,
                                      PatternBenefit benefit = 1,
                                      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<CopyOpType>(context, benefit,
                                                   generatedNames) {}

  LogicalResult matchAndRewrite(CopyOpType op,
                                PatternRewriter &rewriter) const final {
    auto opWithLibCall = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    MemRefType srcType = dyn_cast<MemRefType>(op.getSrcOperandType());

    int64_t rank = srcType.getRank();
    std::string fnName =
        opWithLibCall.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);

    int maxOpRank = opWithLibCall.getOpLibraryMaxRank().value();
    if (rank <= maxOpRank) {
      // Directly create library calls when doing 1d/2d/3d copy.
      replaceWithLibCall(rewriter, op, fnName,
                         getLibraryCallOperands(rewriter, op), {});
    } else {
      SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
          rewriter, op.getLoc(), {op.getSrc(), op.getDst()}, 0,
          rank - maxOpRank);

      rewriter.setInsertionPointAfter(
          reducedVals[reducedVals.size() - 1].getDefiningOp());

      appendExtraOperands(rewriter, op, reducedVals);

      ModuleOp mod = op->template getParentOfType<ModuleOp>();
      createLibCall(rewriter, op, mod, fnName, reducedVals, {});

      rewriter.eraseOp(op);
    }

    return success();
  }

private:
  bool isCopyToCbuf(CopyOpType op) const {
    auto dstMemSpaceAttr =
        cast<MemRefType>(op.getDst().getType()).getMemorySpace();
    auto dstAddrSpace =
        dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr).getAddressSpace();
    if (dstAddrSpace == AddressSpace::L1) {
      // Load GM->L1, only support 1D with continuous stride
      // TODO: support 1,2,3d load and relax the limitation
      MemRefType mem = dyn_cast<MemRefType>(op.getSrc().getType());
      assert((mem.getRank() == 1) &&
             "when Load GM->L1, only support 1D with continuous stride");
      return true;
    }
    return false;
  }

  void calculatePaddingNum(PatternRewriter &rewriter, hivm::LoadOp op,
                           SmallVector<Value> &inputOperands) const {
    auto srcMemSpaceAttr =
        cast<MemRefType>(op.getSrc().getType()).getMemorySpace();
    auto dstMemSpaceAttr =
        cast<MemRefType>(op.getDst().getType()).getMemorySpace();
    auto srcAddrSpace =
        dyn_cast<AddressSpaceAttr>(srcMemSpaceAttr).getAddressSpace();
    auto dstAddrSpace =
        dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr).getAddressSpace();
    if (srcAddrSpace != AddressSpace::GM || dstAddrSpace != AddressSpace::UB) {
      return;
    }

    // calculate left padding num
    Value leftPaddingNum =
        op.getLeftPaddingNum()
            ? op.getLeftPaddingNum()
            : rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0)
                  .getResult();
    inputOperands.push_back(leftPaddingNum);
  }

  void appendExtraOperands(PatternRewriter &rewriter, CopyOpType op,
                           SmallVector<Value> &inputOperands) const {
    if constexpr (std::is_same_v<CopyOpType, hivm::CopyOp> ||
                  std::is_same_v<CopyOpType, hivm::LoadOp>) {
      Value padMode;
      if (op.getPadMode()) {
        padMode = this->constantI32(
            rewriter, op->getLoc(),
            static_cast<uint32_t>(op.getPadModeAttr().getPadmode()));
      } else {
        padMode = this->constantI32(rewriter, op->getLoc(), 0);
      }
      inputOperands.push_back(padMode);

      if (isCopyToCbuf(op))
        return;

      Value padValue;
      if (op.getPadValue()) {
        padValue = op.getPadValue();
      } else {
        Type elemType = getElementTypeOrSelf(op.getSrc().getType());
        padValue =
            llvm::TypeSwitch<Type, Value>(elemType)
                .Case([&](IntegerType intType) {
                  // TODO: fix to use int type after support uint op
                  auto width = cast<mlir::IntegerType>(intType).getWidth();
                  return this->constant(rewriter, op->getLoc(), 0,
                                        rewriter.getIntegerType(width));
                })
                .Case([&](FloatType floatType) {
                  return rewriter.create<arith::ConstantOp>(
                      op.getLoc(), rewriter.getFloatAttr(floatType, 0.0));
                })
                .Default([](Type) {
                  llvm::report_fatal_error("Unsupported type of pad value!");
                  return Value{};
                });
      }
      inputOperands.push_back(padValue);

      if constexpr (std::is_same_v<CopyOpType, hivm::LoadOp>) {
        calculatePaddingNum(rewriter, op, inputOperands);
      }
    }
    if constexpr (std::is_same_v<CopyOpType, hivm::StoreOp>) {
      // adding atomic kind attr
      // choose 0 to represent that
      // the atomickind args is null
      // As the hivm.copy not only comes from hfusion.store
      // we need to check if hivm.copy has attr too
      Value atomicKind = this->constantI32(rewriter, op->getLoc(),
                                           (int64_t)hivm::AtomicKind::NONE);
      if (op.getAtomicKind()) {
        atomicKind = this->constantI32(
            rewriter, op->getLoc(),
            static_cast<uint32_t>(op.getAtomicKind().value()));
      }
      inputOperands.push_back(atomicKind);
    }
  }

  /// Copyop need special treatment due to the need to convert PadMode attribute
  /// into a operand
  SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const override {
    CopyOpType copyOp = cast<CopyOpType>(op);
    SmallVector<Value> inputOperands;
    inputOperands = {copyOp.getSrc(), copyOp.getDst()};
    appendExtraOperands(rewriter, copyOp, inputOperands);
    return inputOperands;
  }
};

void createDummyBuffer(PatternRewriter &rewriter, Operation *op,
                       SmallVector<Value> &inputOperands) {
  MemRefType srcVecType = cast<MemRefType>(op->getOpOperand(0).get().getType());
  Type elemType = srcVecType.getElementType();
  SmallVector<int64_t> shape = {0};
  StridedLayoutAttr layout = {};
  Type zeroMemrefType =
      MemRefType::get(shape, elemType, layout, srcVecType.getMemorySpace());
  Value zeroConstant = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getI64IntegerAttr(0));
  Value tmpBuf = rewriter.create<hivm::PointerCastOp>(
      op->getLoc(), zeroMemrefType, zeroConstant);
  inputOperands.push_back(tmpBuf);
}

template <typename HIVMVectorOp>
class VectorOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<HIVMVectorOp> {
public:
  explicit VectorOpToLibraryCallPattern(MLIRContext *context,
                                        PatternBenefit benefit = 1,
                                        ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<HIVMVectorOp>(context, benefit,
                                                     generatedNames) {}

  LogicalResult matchAndRewrite(HIVMVectorOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    auto opWithLibCall = cast<OpWithLibraryFunction>(op.getOperation());
    int64_t rank = op.getNumLoops();
    int maxOpRank = opWithLibCall.getOpLibraryMaxRank().value();
    std::string fnName =
        opWithLibCall.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    if (rank <= maxOpRank) {
      // Directly create library calls when doing 1d/2d/3d/4d vector ops.
      replaceWithLibCall(rewriter, op, fnName,
                         this->getLibraryCallOperands(rewriter, op), {});
      return success();
    }
    // All subsequent ops need to be reconstructed to first complete the
    // external throw and then call getLibraryCallOperands.
    SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
        rewriter, op.getLoc(),
        this->getLibraryCallOperands(rewriter, op,
                                     /*includeExtraBuffer=*/false),
        0, rank - maxOpRank);

    // the 'dst' of an elementwise op must be a memref value
    rewriter.setInsertionPointAfter(
        reducedVals[reducedVals.size() - 1].getDefiningOp());

    if (auto opWithExtraBuffer =
            dyn_cast<ExtraBufferOpInterface>(op.getOperation())) {
      auto extraBuffers = opWithExtraBuffer.getExtraBuffers();
      reducedVals.insert(reducedVals.end(), extraBuffers.begin(),
                         extraBuffers.end());
      if (extraBuffers.size() == 0) {
        if constexpr (isElementwiseOp()) {
          // tmp_buf is required in the elementwise scene op library.
          createDummyBuffer(rewriter, op, reducedVals);
        }
      }
    }

    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    createLibCall(rewriter, op, mod, fnName, reducedVals, {});
    rewriter.eraseOp(op);
    return success();
  }

private:
  // TODO: Whitelist needs to be unified and rectified
  static constexpr bool isElementwiseOp() {
    return static_cast<bool>(
        std::is_same<HIVMVectorOp, hivm::VAddOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VMulOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VSubOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VDivOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VMaxOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VMinOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VOrOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VAndOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VXorOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VExpOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VAbsOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VLnOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VReluOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VRsqrtOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VSqrtOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VRecOp>::value ||
        std::is_same<HIVMVectorOp, hivm::VNotOp>::value);
  }

  SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const override {
    SmallVector<Value> inputOperands =
        MultiDimOpToLibraryCallPattern<HIVMVectorOp>::getLibraryCallOperands(
            rewriter, op, includeExtraBuffer);
    if (!includeExtraBuffer) {
      return inputOperands;
    }

    if constexpr (isElementwiseOp()) {
      auto hivmVectorOp = cast<HIVMVectorOp>(op);
      std::optional<int64_t> bufSizeMaybe = hivmVectorOp.getExtraBufferSize();
      if (!bufSizeMaybe) {
        // tmp_buf is required in the elementwise scene op library.
        createDummyBuffer(rewriter, op, inputOperands);
      }
    }
    return inputOperands;
  }
};

class VArangeOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VArangeOp> {
public:
  explicit VArangeOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VArangeOp>(context, benefit,
                                                        generatedNames) {}

  LogicalResult matchAndRewrite(hivm::VArangeOp op,
                                PatternRewriter &rewriter) const final {
    auto opWithLibCall = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    int64_t rank = op.getNumLoops();
    int maxOpRank = opWithLibCall.getOpLibraryMaxRank().value();
    std::string fnName =
        opWithLibCall.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    if (rank <= maxOpRank) {
      auto operandVec = this->getLibraryCallOperands(rewriter, op);
      if (!op.getOffset()) {
        Value arangeOffset =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        operandVec.insert(operandVec.begin() + 1, arangeOffset);
      }
      // Directly create library calls when doing 1d/2d/3d/4d vector ops.
      replaceWithLibCall(rewriter, op, fnName, operandVec, {});
    } else {
      SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
          rewriter, op.getLoc(),
          this->getLibraryCallOperands(rewriter, op,
                                       /*includeExtraBuffer=*/false),
          0, rank - maxOpRank);

      memref::SubViewOp subViewOp =
          llvm::dyn_cast<memref::SubViewOp>(reducedVals[0].getDefiningOp());
      // Only 1 subview for VArangeOp
      assert(subViewOp != nullptr && "SubViewOp not found!");
      rewriter.setInsertionPointAfter(subViewOp);

      // subviewOp's offset = offset + arg1 * stride1 + arg2 * stride2 + ...
      Value subViewArrangeOffset;
      if (op.getOffset())
        subViewArrangeOffset = reducedVals[1];
      else
        subViewArrangeOffset =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

      auto subviewOffsets = subViewOp.getOffsets();
      ValueRange arangeOpStrides = op.getStrides();

      // calculate the offset for subviewOp
      int numOfReducedRank = 0;
      for (const auto &subviewOffset : subviewOffsets) {
        auto step = rewriter.create<arith::MulIOp>(
            op.getLoc(), subviewOffset, arangeOpStrides[numOfReducedRank]);
        subViewArrangeOffset = rewriter.create<arith::AddIOp>(
            op.getLoc(), subViewArrangeOffset, step);
        numOfReducedRank++;
      }

      // insert or change the offset of subviewOp
      if (op.getOffset())
        reducedVals[1] = subViewArrangeOffset;
      else
        reducedVals.insert(reducedVals.begin() + 1, subViewArrangeOffset);

      // remove unnecessary strides
      for (; numOfReducedRank > 0; numOfReducedRank--)
        reducedVals.erase(reducedVals.begin() +
                          2); // the first 2 elements of strides are unused

      ModuleOp mod = op->template getParentOfType<ModuleOp>();
      createLibCall(rewriter, op, mod, fnName, reducedVals, {});

      rewriter.eraseOp(op);
    }

    return success();
  }
};

class VCastOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VCastOp> {
public:
  explicit VCastOpToLibraryCallPattern(MLIRContext *context,
                                       PatternBenefit benefit = 1,
                                       ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VCastOp>(context, benefit,
                                                      generatedNames) {}

  LogicalResult matchAndRewrite(hivm::VCastOp op,
                                PatternRewriter &rewriter) const final {
    auto opWithLibCall = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    MemRefType srcType = cast<MemRefType>(op.getSingleSrc().getType());
    int64_t rank = srcType.getRank();
    int maxOpRank = opWithLibCall.getOpLibraryMaxRank().value();
    std::string fnName =
        opWithLibCall.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);

    if (rank <= maxOpRank) {
      // Directly create library calls when doing 1d/2d/3d/4d cast ops.
      replaceWithLibCall(rewriter, op, fnName,
                         this->getLibraryCallOperands(rewriter, op), {});
    } else {
      SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
          rewriter, op.getLoc(), {op.getSingleSrc(), op.getSingleDst()}, 0,
          rank - maxOpRank);
      rewriter.setInsertionPointAfter(
          reducedVals[reducedVals.size() - 1].getDefiningOp());

      auto tempBuffer = op.getTempBuffer();
      if (tempBuffer) {
        reducedVals.push_back(tempBuffer);
      }

      appendExtraOperands(rewriter, op, reducedVals);

      ModuleOp mod = op->template getParentOfType<ModuleOp>();
      createLibCall(rewriter, op, mod, fnName, reducedVals, {});

      rewriter.eraseOp(op);
    }

    return success();
  }

private:
  void appendExtraOperands(PatternRewriter &rewriter, hivm::VCastOp op,
                           SmallVector<Value> &inputOperands) const {
    Value roundNum = constantI32(rewriter, op->getLoc(),
                                 static_cast<uint32_t>(op.getRoundMode()));
    inputOperands.push_back(roundNum);
  }

  // Castop need special treatment due to the need to convert PadMode attribute
  // into a operand passed into func. Also update inputOperands with operands.
  SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const override {
    auto vCastOp = cast<hivm::VCastOp>(op);
    SmallVector<Value> inputOperands =
        MultiDimOpToLibraryCallPattern::getLibraryCallOperands(rewriter,
                                                               vCastOp);
    appendExtraOperands(rewriter, vCastOp, inputOperands);
    return inputOperands;
  }
};

class BrcOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VBrcOp> {
public:
  explicit BrcOpToLibraryCallPattern(MLIRContext *context,
                                     bool isAligned = false,
                                     PatternBenefit benefit = 1,
                                     ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VBrcOp>(context, benefit,
                                                     generatedNames) {
    isOpsAligned_ = isAligned;
  }

  LogicalResult matchAndRewrite(hivm::VBrcOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    Type srcType = op.getSrc().getType();
    MemRefType dstVecType = cast<MemRefType>(op.getDst().getType());
    int rank = dstVecType.getRank();
    int maxLibraryRank = libraryOp.inferOpLibraryMaxRank();
    int exceedRank = rank - maxLibraryRank;
    int rankRangeShift = 0;
    int rankRangeEnd = exceedRank;
    int broadcastDim = 0;
    bool isMidAxisExceed = false;

    // Update the variable if broadcast source is high-dim memref
    if (!isScalarLike(srcType)) {
      assert(op.getBroadcastDims().size() == 1 &&
             "broadcast dimensions array is not decomposed yet.");
      broadcastDim = op.getBroadcastDims()[0];
      AxisKind axisKind = utils::getAxisKind(broadcastDim, rank);
      if (axisKind == AxisKind::FIRST)
        // Not to outline to first axis
        rankRangeShift = 1;
      if (axisKind == AxisKind::MIDDLE && exceedRank >= broadcastDim) {
        // Inner loop creation is needed
        isMidAxisExceed = true;
        rankRangeEnd = broadcastDim;
      }
    }

    // TODO: Unify the logic with deduceAlignmentForDPSInitOperand
    AlignKind alignKind;
    if (isScalarLike(srcType)) {
      alignKind = AlignKind::ALIGN;
    } else {
      alignKind = isBrcOpAligned(op, broadcastDim, dstVecType.getRank());
    }
    std::string libFnName = libraryOp.getOpLibraryCallName(
        isOpsAligned_ && alignKind == AlignKind::ALIGN);

    // replace without nested loop creation
    if (exceedRank <= 0) {
      replaceWithLibCall(rewriter, op, libFnName,
                         getLibraryCallOperands(rewriter, op), {});
      return success();
    }

    // replace with nested loop creation
    SmallVector<Value> reducedVals;
    reducedVals = reduceMemrefsToNestedFor(
        rewriter, op.getLoc(), {op.getSrc(), op.getDst()}, rankRangeShift,
        rankRangeEnd + rankRangeShift);
    if (isMidAxisExceed) {
      // replace with second (inner) nested loop creation. It happens when the
      // broadcasted middle axis dimension exceeds the highest dimension
      // supported by the library.
      rewriter.setInsertionPointAfter(reducedVals.back().getDefiningOp());
      reducedVals = reduceMemrefsToNestedFor(rewriter, op.getLoc(), reducedVals,
                                             1, exceedRank - rankRangeEnd + 1);
    }

    rewriter.setInsertionPointAfter(reducedVals.back().getDefiningOp());
    auto tempBuffer = op.getTempBuffer();
    if (tempBuffer) {
      reducedVals.push_back(tempBuffer);
    }

    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    createLibCall(rewriter, op, mod, libFnName, reducedVals, {});
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool isOpsAligned_{false};
};

class ReduceOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VReduceOp> {
public:
  explicit ReduceOpToLibraryCallPattern(MLIRContext *context,
                                        PatternBenefit benefit = 1,
                                        ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VReduceOp>(context, benefit,
                                                        generatedNames) {}

  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    MemRefType srcVecType = cast<MemRefType>(op.getSrc().getType());
    int rank = srcVecType.getRank();

    llvm::ArrayRef<int64_t> reduceDims = op.getReduceDims();
    if (reduceDims.size() > 1) {
      return op.emitError("reduce dimensions array is not decomposed yet");
    }

    int reduceIdx = reduceDims[0];
    bool firstAxis = (reduceIdx == 0);
    bool lastAxis = (reduceIdx == rank - 1);
    bool midAxis = !firstAxis && !lastAxis;

    int reducedRank = rank;
    SmallVector<Value> convertedVals;
    SmallVector<Value> memrefValsMaybe = {op.getSrc()};
    if (op.getIndices()) {
      memrefValsMaybe.push_back(op.getIndices());
    }
    memrefValsMaybe.push_back(op.getDstValue());
    auto arith = op.getArithAttr();
    if (VReduceOp::isArgminOrArgmax(arith.getReduceOp())) {
      memrefValsMaybe.push_back(op.getDstIndex());
    }

    if (midAxis) {
      // TODO: enhance inferOpLibraryMaxRank for VReduceOp
      int originalRank = rank;

      std::set<int> loopIndices;
      for (int loopIndice = 0; loopIndice < reduceIdx; loopIndice++) {
        loopIndices.insert(loopIndice);
      }

      int curReducedRank =
          originalRank - static_cast<int>(loopIndices.size());

      if (curReducedRank > 3) {
        int additionalLoopNum = curReducedRank - 3;

        for (int loopIndice = reduceIdx + 1;
             loopIndice < originalRank && additionalLoopNum > 0; loopIndice++) {
          if (loopIndices.count(loopIndice) == 0) {
            loopIndices.insert(loopIndice);
            additionalLoopNum--;
          }
        }
      }

      convertedVals = reduceMemrefsToNestedForUsingAxes(
          rewriter, op.getLoc(), memrefValsMaybe, loopIndices);

      reducedRank = originalRank - static_cast<int>(loopIndices.size());
    } else if (firstAxis && rank > 3) {
      convertedVals = reduceMemrefsToNestedFor(rewriter, op.getLoc(),
                                               memrefValsMaybe, 1, rank - 2);
      reducedRank = 3;
    } else if (lastAxis && rank > 2) {
      convertedVals = reduceMemrefsToNestedFor(rewriter, op.getLoc(),
                                               memrefValsMaybe, 0, rank - 2);
      reducedRank = 2;
    } else if (firstAxis && (rank > 2) &&
               VReduceOp::isArgminOrArgmax(arith.getReduceOp())) {
      convertedVals = reduceMemrefsToNestedFor(rewriter, op.getLoc(),
                                               memrefValsMaybe, 1, rank - 1);
      reducedRank = 2;
    }

    if (reducedRank == rank) {
      std::string libFnName =
          libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
      SmallVector<Value> operands = getLibraryCallOperands(rewriter, op);
      replaceWithLibCall(rewriter, op, libFnName, operands, {});
      return success();
    }

    rewriter.setInsertionPointAfter(
        convertedVals[convertedVals.size() - 1].getDefiningOp());
    std::string libFnName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);

    auto tempBuffer = op.getTempBuffer();
    if (tempBuffer) {
      convertedVals.push_back(tempBuffer);
    } else {
      // tmp_buf is required in the reduce scene op library.
      createDummyBuffer(rewriter, op, convertedVals);
    }

    if (failed(appendInitValue(rewriter, op, convertedVals)))
      return failure();

    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    createLibCall(rewriter, op, mod, libFnName, convertedVals, {});
    rewriter.eraseOp(op);

    return success();
  }

private:
  LogicalResult appendInitValue(PatternRewriter &rewriter, hivm::VReduceOp op,
                                SmallVector<Value> &v) const {
    Attribute init = op.getInit();
    if (!init) {
      return op.emitError("failed to calculate init value");
    }

    if (auto initInt = dyn_cast_or_null<IntegerAttr>(init)) {
      v.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), initInt));
    } else if (auto initFloat = dyn_cast_or_null<FloatAttr>(init)) {
      v.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), initFloat));
    } else {
      return op.emitError(
          "invalid init value type. this is an implementation error.");
    }

    return success();
  }

  // Reduceop need special treatment due to the op library requires
  // initvalue to
  // initialize the tmp buffer.
  SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const override {
    hivm::VReduceOp reduceOp = cast<hivm::VReduceOp>(op);
    SmallVector<Value> inputOperands =
        MultiDimOpToLibraryCallPattern::getLibraryCallOperands(rewriter,
                                                               reduceOp);
    if (reduceOp.getIndices()) {
      auto indices = std::move(inputOperands.back());
      inputOperands.pop_back();
      auto *it = inputOperands.begin() + 1;
      inputOperands.insert(it, indices);
    }
    auto bufSizeMaybe = getExtraBufferSizeForReduceOp(
        op, mlir::hivm::util::BufferSizeUnit::ELEMENT);
    if (!bufSizeMaybe) {
      // tmp_buf is required in the reduce scene op library.
      createDummyBuffer(rewriter, op, inputOperands);
    }
    if (failed(appendInitValue(rewriter, reduceOp, inputOperands)))
      return {};

    return inputOperands;
  }
};

class TransposeOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VTransposeOp> {
public:
  explicit TransposeOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VTransposeOp>(context, benefit,
                                                           generatedNames) {}

  LogicalResult matchAndRewrite(hivm::VTransposeOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    auto src = op.getSrc();
    auto dst = op.getDst();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto libCallName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    auto perm = op.getPermutation();
    auto dim = libraryOp.inferOpLibraryMaxRank();
    if (static_cast<int>(perm.size()) == dim) {
      SmallVector<Value> operands = getLibraryCallOperands(rewriter, op);
      createLibCall(rewriter, op, mod, libCallName, operands, {});
      rewriter.eraseOp(op);
      return success();
    }

    // Reduced axes would be used to create forOp.
    auto axes = getReducedAxes(perm, dim);
    SmallVector<Value> operands = reduceMemrefsToNestedForUsingAxes(
        rewriter, op.getLoc(), {src, dst}, axes);

    rewriter.setInsertionPointAfter(
        operands[operands.size() - 1].getDefiningOp());

    auto tempBuffer = op.getTempBuffer();
    if (tempBuffer) {
      operands.push_back(tempBuffer);
    }

    createLibCall(rewriter, op, mod, libCallName, operands, {});
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// The reduces axes would be used to create forOp.
  std::set<int> getReducedAxes(ArrayRef<int64_t> permutation,
                               size_t maxRank) const {
    std::set<int> res;

    const auto rank = static_cast<int>(permutation.size());
    for (int idx : permutation) {
      if (idx == permutation[idx] && idx != rank - 1) {
        res.insert(idx);
      }
    }
    SmallVector<int64_t> transposeAxes =
        hivm::util::getTransposeAxes(permutation);
    assert(!transposeAxes.empty() && "transposeAxes shouldn't be empty.");
    assert(transposeAxes.size() == 2 &&
           "Vtranspose only support two axes transpose.");

    if (hivm::util::isTransposeWithLastAxis(permutation) &&
        !hivm::util::isTransposeAdjacentAxes(transposeAxes) &&
        permutation.size() > maxRank) {
      // if permutation = [0, 4, 2, 3, 1], we need to throw out axes 0, 2,
      // and
      // call transpose 3d lib
      const int lastSecondAxis = static_cast<int>(permutation.size()) - 2;
      res.erase(lastSecondAxis);
    }
    return res;
  }
};

class VInterleaveOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VInterleaveOp> {
public:
  explicit VInterleaveOpToLibraryCallPattern(
      MLIRContext *context, PatternBenefit benefit = 1,
      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VInterleaveOp>(context, benefit,
                                                            generatedNames) {}
  LogicalResult matchAndRewrite(hivm::VInterleaveOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");
    int64_t rank = op.getNumLoops();
    int maxOpRank = libraryOp.getOpLibraryMaxRank().value();
    std::string fnName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    if (rank <= maxOpRank) {
      // Directly create library calls when doing 1d interleave op.
      replaceWithLibCall(rewriter, op, fnName,
                         this->getLibraryCallOperands(rewriter, op), {});
    } else {
      // TODO: currently the interleave dim can only be last dimension and
      // support any dim after add interleave dim for interleave op
      SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
          rewriter, op.getLoc(), {op.getSrc()[0], op.getSrc()[1], op.getDst()},
          0, rank - maxOpRank);

      rewriter.setInsertionPointAfter(
          reducedVals[reducedVals.size() - 1].getDefiningOp());

      auto tempBuffer = op.getTempBuffer();
      if (tempBuffer) {
        reducedVals.push_back(tempBuffer);
      }

      ModuleOp mod = op->template getParentOfType<ModuleOp>();
      createLibCall(rewriter, op, mod, fnName, reducedVals, {});
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DebugOpToLibraryCallPattern : public OpRewritePattern<hivm::DebugOp> {
  using OpRewritePattern<hivm::DebugOp>::OpRewritePattern;

  static constexpr llvm::StringLiteral HIVMDebugTypePrint = "print";
  static constexpr llvm::StringLiteral HIVMDebugTypeAssert = "assert";
  inline static int prefixNumber = 0;

  LogicalResult matchAndRewrite(hivm::DebugOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    SmallVector<Value> funcOperands;

    // Convert the prefix attribute to two arguments of the called func
    StringAttr prefixAttr = op.getPrefixAttr();
    auto prefixStrName = "_debug_prefix_" + std::to_string(prefixNumber++);
    auto prefixValue =
        LLVM::createGlobalString(op.getLoc(), rewriter, prefixStrName,
                                 prefixAttr.strref(), LLVM::Linkage::Private);
    auto ctx = op->getContext();
    auto prefixLenType = IntegerType::get(ctx, 64);
    auto prefixLenValue = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), prefixLenType,
        rewriter.getI64IntegerAttr(prefixAttr.size()));
    funcOperands.push_back(prefixValue);
    funcOperands.push_back(prefixLenValue);

    // Add DebugOp's (non-attr) arguments
    auto operands = op->getOperands();
    for (auto val : operands) {
      funcOperands.push_back(val);
    }

    // dispatch to different lib calls for assert/print
    std::string libCallName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    if (op.getDebugtype() == HIVMDebugTypePrint) {
      // Convert the hex attr to an argument of the print lib call
      auto hexBool = op.getHex();
      Value hexInt = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getI8IntegerAttr(hexBool));
      funcOperands.push_back(hexInt);
    }

    // create the library call and modify the corresponding funcOp's
    // TFuncCoreType
    ModuleOp moduleOp = op->template getParentOfType<ModuleOp>();
    func::CallOp callOp =
        createLibCall(rewriter, op, moduleOp, libCallName, funcOperands, {});
    func::FuncOp funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    /// Note: in tensor/memref case, this attribute will be copied
    /// to the outer wrapper generated by
    /// mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
    funcOp->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
                    hivm::TFuncCoreTypeAttr::get(
                        funcOp->getContext(), hivm::TFuncCoreType::AIC_OR_AIV));

    // finally replace the debug op with lib call
    rewriter.replaceOp(op, callOp);

    return success();
  }
};

class CustomOpToLibraryCallPattern : public OpRewritePattern<hivm::CustomOp> {
  using OpRewritePattern<hivm::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CustomOp op,
                                PatternRewriter &rewriter) const final {
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       op.getOperands(), op.getResultTypes());
    return success();
  }
};

class SortOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VSortOp> {
public:
  explicit SortOpToLibraryCallPattern(MLIRContext *context,
                                      PatternBenefit benefit = 1,
                                      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VSortOp>(context, benefit,
                                                      generatedNames) {}
  LogicalResult matchAndRewrite(hivm::VSortOp op,
                                PatternRewriter &rewriter) const final {
    auto libraryOp = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    MemRefType srcVecType = cast<MemRefType>(op.getSrc().getType());
    int64_t rank = srcVecType.getRank();
    int maxOpRank = libraryOp.getOpLibraryMaxRank().value();
    std::string fnName =
        libraryOp.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);

    if (rank <= maxOpRank) {
      // Directly create library calls when doing 1d sort ops.
      replaceWithLibCall(rewriter, op, fnName,
                         this->getLibraryCallOperands(rewriter, op), {});
    } else {
      SmallVector<Value> memrefValsMaybe = {op.getSrc(), op.getDstValue()};
      if (op.getDst().size() == 2) {
        memrefValsMaybe.push_back(op.getDstIndex());
      }
      SmallVector<Value> reducedVals = reduceMemrefsToNestedFor(
          rewriter, op.getLoc(), memrefValsMaybe, 0, rank - maxOpRank);
      rewriter.setInsertionPointAfter(
          reducedVals[reducedVals.size() - 1].getDefiningOp());

      auto tempBuffer = op.getTempBuffer();
      if (tempBuffer) {
        reducedVals.push_back(tempBuffer);
      }
      appendExtraOperands(rewriter, op, reducedVals);

      ModuleOp mod = op->template getParentOfType<ModuleOp>();
      createLibCall(rewriter, op, mod, fnName, reducedVals, {});
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  void appendExtraOperands(PatternRewriter &rewriter, hivm::VSortOp op,
                           SmallVector<Value> &inputOperands) const {
    auto descending = op.getDescending();
    Value descendingInt = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getI8IntegerAttr(descending));
    inputOperands.push_back(descendingInt);
  }

  // Sortop need special treatment due to the op library requires initvalue
  // to
  // initialize the tmp buffer.
  SmallVector<Value>
  getLibraryCallOperands(PatternRewriter &rewriter, Operation *op,
                         bool includeExtraBuffer = true) const override {
    auto vsortOp = cast<hivm::VSortOp>(op);
    SmallVector<Value> inputOperands =
        MultiDimOpToLibraryCallPattern::getLibraryCallOperands(rewriter,
                                                               vsortOp);
    appendExtraOperands(rewriter, vsortOp, inputOperands);

    return inputOperands;
  }
};

class FlipOpToLibraryCallPattern
    : public MultiDimOpToLibraryCallPattern<hivm::VFlipOp> {
public:
  explicit FlipOpToLibraryCallPattern(MLIRContext *context,
                                      PatternBenefit benefit = 1,
                                      ArrayRef<StringRef> generatedNames = {})
      : MultiDimOpToLibraryCallPattern<hivm::VFlipOp>(context, benefit,
                                                      generatedNames) {}
  LogicalResult matchAndRewrite(hivm::VFlipOp op,
                                PatternRewriter &rewriter) const final {
    auto opWithLibCall = cast<OpWithLibraryFunction>(op.getOperation());
    assert(op.hasPureBufferSemantics() &&
           "Operating on tensor, please bufferize.");

    MemRefType srcVecType = cast<MemRefType>(op.getSrc().getType());
    uint64_t rank = static_cast<uint64_t>(srcVecType.getRank());
    std::string fnName =
        opWithLibCall.getOpLibraryCallName(/*isOpsAligned=*/std::nullopt);
    SmallVector<Value> memrefValsMaybe = {op.getSrc(), op.getDst()};
    std::set<int> reducedAxes;
    for (uint64_t curAxis = 0; curAxis < rank; curAxis++) {
      if (curAxis != op.getFlipAxis()) {
        reducedAxes.insert(curAxis);
      }
    }

    if (!reducedAxes.empty()) {
      memrefValsMaybe = reduceMemrefsToNestedForUsingAxes(
          rewriter, op.getLoc(), memrefValsMaybe, reducedAxes);
      rewriter.setInsertionPointAfter(memrefValsMaybe.back().getDefiningOp());
    }

    replaceWithLibCall(rewriter, op, fnName, memrefValsMaybe, {});

    return success();
  }
};

template <typename T>
class PlainOpToLibraryCallPattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    replaceWithLibCall(rewriter, op,
                       cast<OpWithLibraryFunction>(op.getOperation())
                           .getOpLibraryCallName(/*isOpsAligned=*/std::nullopt),
                       /*inputOperands=*/{}, {});
    return success();
  }
};

template <typename SyncBlockOp>
class SyncBlockOpToLibraryCallPattern : public OpRewritePattern<SyncBlockOp> {
public:
  using OpRewritePattern<SyncBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SyncBlockOp op,
                                PatternRewriter &rewriter) const final {
    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    std::string libCallName = op.getOpName().str();
    if (op->hasAttr(SyncBlockLockWithSubblockAttr::name))
      libCallName += "_with_subblock";
    createLibCall(rewriter, op, mod, libCallName, op->getOperands(), {});
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace mlir::hivm

void mlir::hivm::populateHIVMToStandardConversionPatterns(
    RewritePatternSet &patterns, bool isOpsAligned) {
  // clang-format off
  patterns.add<
               MmadL1OpToLibraryCallPattern,
               Conv1DL1OpToLibraryCallPattern,
               Conv2DL1OpToLibraryCallPattern,
               ND2NZOpToLibraryCallPattern,
               NZ2NDOpToLibraryCallPattern,
               FixpipeOpToLibraryCallPattern,
               MatmulOpToLibraryCallPattern,
               MixMatmulOpToLibraryCallPattern,
               MixGroupMatmulOpToLibraryCallPattern,
               CopyOpToLibraryCallPattern<hivm::CopyOp>,
               CopyOpToLibraryCallPattern<hivm::LoadOp>,
               CopyOpToLibraryCallPattern<hivm::StoreOp>,
               VectorOpToLibraryCallPattern<hivm::VAddOp>,
               VectorOpToLibraryCallPattern<hivm::VMulOp>,
               VectorOpToLibraryCallPattern<hivm::VSubOp>,
               VectorOpToLibraryCallPattern<hivm::VDivOp>,
               VectorOpToLibraryCallPattern<hivm::VMaxOp>,
               VectorOpToLibraryCallPattern<hivm::VMinOp>,
               VectorOpToLibraryCallPattern<hivm::VOrOp>,
               VectorOpToLibraryCallPattern<hivm::VAndOp>,
               VectorOpToLibraryCallPattern<hivm::VXorOp>,
               VectorOpToLibraryCallPattern<hivm::VExpOp>,
               VectorOpToLibraryCallPattern<hivm::VAbsOp>,
               VectorOpToLibraryCallPattern<hivm::VLnOp>,
               VectorOpToLibraryCallPattern<hivm::VReluOp>,
               VectorOpToLibraryCallPattern<hivm::VRsqrtOp>,
               VectorOpToLibraryCallPattern<hivm::VSqrtOp>,
               VectorOpToLibraryCallPattern<hivm::VRecOp>,
               VectorOpToLibraryCallPattern<hivm::VNotOp>,
               VectorOpToLibraryCallPattern<hivm::VCmpOp>,
               VectorOpToLibraryCallPattern<hivm::VSelOp>,
               VectorOpToLibraryCallPattern<hivm::VShLOp>,
               VectorOpToLibraryCallPattern<hivm::VShROp>,
               VectorOpToLibraryCallPattern<hivm::VDeinterleaveOp>,
               VectorOpToLibraryCallPattern<hivm::VMulextendedOp>,
               VectorOpToLibraryCallPattern<hivm::VPowOp>,
               VectorOpToLibraryCallPattern<hivm::VGatherOp>,
               VectorOpToLibraryCallPattern<hivm::VGatherMaskOp>,
               VArangeOpToLibraryCallPattern,
               VCastOpToLibraryCallPattern,
               ReduceOpToLibraryCallPattern,
               CumOpToLibraryCallPattern<hivm::VCumsumOp>,
               CumOpToLibraryCallPattern<hivm::VCumprodOp>,
               TransposeOpToLibraryCallPattern,
               DebugOpToLibraryCallPattern,
               CustomOpToLibraryCallPattern,
               VInterleaveOpToLibraryCallPattern,
               PlainOpToLibraryCallPattern<hivm::InitDebugOp>,
               PlainOpToLibraryCallPattern<hivm::FinishDebugOp>,
               SyncBlockOpToLibraryCallPattern<hivm::SyncBlockLockOp>,
               SyncBlockOpToLibraryCallPattern<hivm::SyncBlockUnlockOp>,
               SyncBlockOpToLibraryCallPattern<hivm::FreeLockVarOp>,
               SortOpToLibraryCallPattern,
               FlipOpToLibraryCallPattern
               >
               (patterns.getContext());
  // clang-format on
  patterns.add<BrcOpToLibraryCallPattern>(patterns.getContext(), isOpsAligned);
}

namespace {
struct ConvertHIVMToStandardPass
    : public impl::ConvertHIVMToStandardBase<ConvertHIVMToStandardPass> {
  explicit ConvertHIVMToStandardPass(
      const ConvertHIVMToStandardOptions &options)
      : ConvertHIVMToStandardBase(options) {}

  void runOnOperation() override;
};
} // namespace

void ConvertHIVMToStandardPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, memref::MemRefDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         LLVM::LLVMDialect>();
  target.addLegalOp<hivm::PointerCastOp>();
  // Abstract Intrinsic Ops must be converted.
  // clang-format off
  target.addIllegalOp<hivm::MmadL1Op,
                      hivm::Conv1DL1Op,
                      hivm::Conv2DL1Op,
                      hivm::ND2NZOp,
                      hivm::NZ2NDOp,
                      hivm::FixpipeOp,
                      hivm::MatmulOp,
                      hivm::CopyOp,
                      hivm::CustomOp,
                      hivm::LoadOp,
                      hivm::StoreOp,
                      hivm::VAddOp,
                      hivm::VMulOp,
                      hivm::VSubOp,
                      hivm::VDivOp,
                      hivm::VMaxOp,
                      hivm::VMinOp,
                      hivm::VOrOp,
                      hivm::VAndOp,
                      hivm::VXorOp,
                      hivm::VExpOp,
                      hivm::VAbsOp,
                      hivm::VLnOp,
                      hivm::VReluOp,
                      hivm::VRsqrtOp,
                      hivm::VSqrtOp,
                      hivm::VRecOp,
                      hivm::VNotOp,
                      hivm::VCastOp,
                      hivm::VCmpOp,
                      hivm::VSelOp,
                      hivm::VArangeOp,
                      hivm::VBrcOp,
                      hivm::VReduceOp,
                      hivm::VCumsumOp,
                      hivm::VCumprodOp,
                      hivm::VTransposeOp,
                      hivm::VShLOp,
                      hivm::VShROp,
                      hivm::VInterleaveOp,
                      hivm::VDeinterleaveOp,
                      hivm::VFlipOp,
                      hivm::VMulextendedOp,
                      hivm::VPadOp,
                      hivm::VPowOp,
                      hivm::VGatherOp,
                      hivm::VGatherMaskOp,
                      hivm::DebugOp,
                      hivm::SyncBlockLockOp,
                      hivm::SyncBlockUnlockOp,
                      hivm::FreeLockVarOp,
                      hivm::VSortOp
                      >();

  // clang-format on

  RewritePatternSet patterns(&getContext());
  SmallVector<func::FuncOp> deviceFuncs;
  module->walk([&](func::FuncOp func) {
    if (hacc::utils::isDevice(func)) {
      deviceFuncs.push_back(func);
    }
  });

  for (auto deviceFunc : deviceFuncs) {
    patterns.clear();
    populateHIVMToStandardConversionPatterns(
        patterns,
        /*isOpsAligned=*/deviceFunc->hasAttr(hivm::StrideAlignDimsAttr::name) ||
            this->isOpsAligned);
    if (failed(
            applyPartialConversion(deviceFunc, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  // Add canonicalization patterns for subviewOp and forOp.
  RewritePatternSet canonicalPatterns(module->getContext());
  memref::SubViewOp::getCanonicalizationPatterns(
      canonicalPatterns, canonicalPatterns.getContext());
  scf::ForOp::getCanonicalizationPatterns(canonicalPatterns,
                                          canonicalPatterns.getContext());
  if (failed(applyPatternsGreedily(module, std::move(canonicalPatterns)))) {
    return signalPassFailure();
  }

  // mark DebugOp's global strings as AIC_OR_AIV
  // this cannot be done after LLVM::createGlobalString because that function
  // doesn't return the LLVM::GlobalOp itself (getDefiningOp() doesn't refer to
  // that Op either)
  getOperation()->walk([](LLVM::GlobalOp globalOp) {
    if (globalOp.getSymName().starts_with("_debug_prefix_")) {
      globalOp->setAttr(
          mlir::hivm::TFuncCoreTypeAttr::name,
          hivm::TFuncCoreTypeAttr::get(globalOp->getContext(),
                                       hivm::TFuncCoreType::AIC_OR_AIV));
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertHIVMToStandardPass(
    const ConvertHIVMToStandardOptions &options) {
  return std::make_unique<ConvertHIVMToStandardPass>(options);
}
