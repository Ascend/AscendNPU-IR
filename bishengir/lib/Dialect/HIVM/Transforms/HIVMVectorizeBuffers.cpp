//===- HIVMVectorizeBuffers.cpp - Late buffer data movement ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/HIVMVectorizeBuffers.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/TransformOps/Outline.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeLinalg.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeResult.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
static bool supportsScalarBroadcast(Type element) {
  // A one-element store needs a masked data store after AVE lowering.
  if (auto integer = dyn_cast<IntegerType>(element))
    return integer.isSignless() &&
           llvm::is_contained(ArrayRef<unsigned>{8, 16, 32},
                              integer.getWidth());
  return isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
             Float8E5M2Type>(element);
}

// A generic scalar store would be cloned onto both cores of a MIX kernel.
// When the AIV already has the scalar and exclusively consumes a fresh
// singleton UB, a vector broadcast writes the entire allocation on its owner.
static void promotePrivateVectorInput(memref::StoreOp store) {
  auto function = store->getParentOfType<func::FuncOp>();
  if (!function || queryFuncCoreType(function) != TFuncCoreType::MIX ||
      store->hasAttr(TCoreTypeAttr::name))
    return;
  Value dest = store.getMemRef();
  auto alloc = dest.getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != store->getBlock() ||
      !llvm::hasNItems(dest.getUses(), 2))
    return;
  auto type = alloc.getType();
  if (type.getShape() != ArrayRef<int64_t>{1} ||
      !type.getLayout().isIdentity() ||
      getOptionalHIVMAddressSpace(type) != AddressSpace::UB ||
      !supportsScalarBroadcast(type.getElementType()) ||
      !matchPattern(store.getIndices().front(), m_Zero()))
    return;
  auto isLocalVF = [&](Operation *user) {
    return user->getBlock() == store->getBlock() && isVFCall(user);
  };
  Value scalar = store.getValue();
  auto arg = dyn_cast<BlockArgument>(scalar);
  bool available = (arg && arg.getOwner() == &function.front()) ||
                   scalar.getDefiningOp<arith::ConstantOp>() ||
                   llvm::any_of(scalar.getUsers(), isLocalVF);
  // The call can publish the buffer address. Initialize it before the call
  // and in the allocation's block so the address is fresh on every iteration.
  auto initializedByStore = [&](Operation *user) {
    return isLocalVF(user) && store->isBeforeInBlock(user);
  };
  if (!available || !llvm::any_of(dest.getUsers(), initializedByStore))
    return;
  for (Operation *parent = store->getParentOp(); parent != function;
       parent = parent->getParentOp()) {
    auto owner = getCoreType(parent);
    if (failed(owner) || *owner != TCoreType::CUBE_OR_VECTOR)
      return;
  }
  IRRewriter rewriter(store);
  rewriter.replaceOpWithNewOp<VBrcOp>(store, TypeRange{}, scalar, dest);
}

bool isBufferMovement(Operation &operation) {
  auto structured = dyn_cast<HIVMStructuredOp>(operation);
  if (!structured || !structured.hasPureBufferSemantics() ||
      !isa<VBrcOp, VTransposeOp, VInterleaveOp, VDeinterleaveOp>(operation))
    return false;
  return llvm::all_of(operation.getOperandTypes(), [](Type type) {
    auto buffer = dyn_cast<MemRefType>(type);
    if (!buffer)
      return true;
    auto space = dyn_cast_or_null<AddressSpaceAttr>(buffer.getMemorySpace());
    return !space || space.getAddressSpace() == AddressSpace::UB;
  });
}

SmallVector<Value> getBufferMovementInputs(Operation &operation) {
  if (auto interleave = dyn_cast<VInterleaveOp>(operation))
    return llvm::to_vector(interleave.getSrc());
  if (auto deinterleave = dyn_cast<VDeinterleaveOp>(operation))
    return {deinterleave.getSrc()};
  if (auto broadcast = dyn_cast<VBrcOp>(operation))
    return {broadcast.getSrc()};
  return {cast<VTransposeOp>(operation).getSrc()};
}

LogicalResult validateChannelMovement(Operation &operation) {
  auto structured = cast<HIVMStructuredOp>(operation);
  auto inputs = getBufferMovementInputs(operation);
  auto outputs = structured.getDpsInits();
  auto deinterleave = dyn_cast<VDeinterleaveOp>(operation);
  auto interleave = dyn_cast<VInterleaveOp>(operation);
  if ((interleave && interleave.getInterleaveChannelNums() != 2) ||
      (deinterleave && deinterleave.getChannelNum() != 2))
    return operation.emitError(
        "channel lowering requires exactly two channels");
  auto expectedInputs = interleave ? 2u : 1u;
  auto expectedOutputs = deinterleave && deinterleave.getIndexMode() ==
                                             DeinterleaveMode::ALL_CHANNELS
                             ? 2u
                             : 1u;
  if (inputs.size() != expectedInputs || outputs.size() != expectedOutputs)
    return operation.emitError(
        "channel lowering has incompatible operand counts");
  SmallVector<Value> buffers(inputs.begin(), inputs.end());
  llvm::append_range(buffers, outputs);
  auto elementType = getElementTypeOrSelf(inputs.front().getType());
  for (auto buffer : buffers) {
    auto type = dyn_cast<MemRefType>(buffer.getType());
    if (!type || type.getRank() != 1 || !type.hasStaticShape())
      return operation.emitError(
          "channel lowering requires statically shaped rank-one buffers");
    if (type.getElementType() != elementType)
      return operation.emitError("channel buffer element types must match");
  }
  auto extent = [](Value buffer) {
    return cast<MemRefType>(buffer.getType()).getDimSize(0);
  };
  auto channelSize = extent(interleave ? inputs.front() : outputs.front());
  auto pairedSize = extent(interleave ? outputs.front() : inputs.front());
  if (pairedSize % 2 || pairedSize / 2 != channelSize)
    return operation.emitError(
        "paired buffer extent must be twice the channel extent");
  ValueRange channels = interleave ? ValueRange(inputs) : ValueRange(outputs);
  if (llvm::any_of(channels, [&](Value channel) {
        return extent(channel) != channelSize;
      }))
    return operation.emitError("channel buffer extents must match");
  return success();
}

bool compatibleExtent(int64_t lhs, int64_t rhs) {
  return lhs == rhs || ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs);
}

LogicalResult validateMappedMovement(Operation &operation) {
  auto structured = cast<HIVMStructuredOp>(operation);
  auto inputs = getBufferMovementInputs(operation);
  auto outputs = structured.getDpsInits();
  if (inputs.size() != 1 || outputs.size() != 1)
    return operation.emitError(
        "buffer movement requires one input and one output");
  auto outputType = dyn_cast<MemRefType>(outputs.front().getType());
  if (!outputType || getElementTypeOrSelf(inputs.front().getType()) !=
                         outputType.getElementType())
    return operation.emitError(
        "buffer movement requires matching element types");
  auto inputType = dyn_cast<MemRefType>(inputs.front().getType());
  if (!inputType) {
    if (!isa<VBrcOp>(operation) || isa<ShapedType>(inputs.front().getType()))
      return operation.emitError("only broadcast accepts a scalar source");
    return success();
  }
  if (inputType.getRank() != outputType.getRank())
    return operation.emitError(
        "buffer movement input and output ranks must match");
  if (auto transpose = dyn_cast<VTransposeOp>(operation)) {
    auto permutation = transpose.getPermutation();
    if (permutation.size() != static_cast<size_t>(inputType.getRank()))
      return operation.emitError(
          "transpose permutation must match the buffer rank");
    SmallVector<bool> seen(inputType.getRank(), false);
    for (auto [dim, axis] : llvm::enumerate(permutation)) {
      if (axis < 0 || axis >= inputType.getRank() || seen[axis])
        return operation.emitError("transpose requires a valid permutation");
      seen[axis] = true;
      if (!compatibleExtent(inputType.getDimSize(axis),
                            outputType.getDimSize(dim)))
        return operation.emitError("transpose buffer shapes do not match");
    }
    return success();
  }
  auto broadcast = cast<VBrcOp>(operation);
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (llvm::is_contained(broadcast.getBroadcastDims(), dim)) {
      if (inputType.getDimSize(dim) != 1)
        return operation.emitError("broadcast source dimension must be one");
    } else if (!compatibleExtent(inputType.getDimSize(dim),
                                 outputType.getDimSize(dim))) {
      return operation.emitError("non-broadcast buffer dimensions must match");
    }
  }
  return success();
}

LogicalResult validateBufferMovement(Operation &operation) {
  if (isa<VInterleaveOp, VDeinterleaveOp>(operation))
    return validateChannelMovement(operation);
  return validateMappedMovement(operation);
}

void copyChannelElement(IRRewriter &rewriter, Location loc, Value source,
                        Value sourceIndex, Value destination,
                        Value destinationIndex) {
  auto element =
      rewriter.create<memref::LoadOp>(loc, source, ValueRange{sourceIndex});
  rewriter.create<memref::StoreOp>(loc, element, destination,
                                   ValueRange{destinationIndex});
}

void emitChannelElements(Operation &operation, IRRewriter &rewriter,
                         Value index) {
  auto loc = operation.getLoc();
  auto two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
  auto pairIndex = rewriter.create<arith::MulIOp>(loc, index, two);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto oddIndex = rewriter.create<arith::AddIOp>(loc, pairIndex, one);
  if (auto interleave = dyn_cast<VInterleaveOp>(operation)) {
    copyChannelElement(rewriter, loc, interleave.getSrc()[0], index,
                       interleave.getDst(), pairIndex);
    copyChannelElement(rewriter, loc, interleave.getSrc()[1], index,
                       interleave.getDst(), oddIndex);
    return;
  }
  auto deinterleave = cast<VDeinterleaveOp>(operation);
  if (deinterleave.getIndexMode() == DeinterleaveMode::ALL_CHANNELS) {
    copyChannelElement(rewriter, loc, deinterleave.getSrc(), pairIndex,
                       deinterleave.getDst()[0], index);
    copyChannelElement(rewriter, loc, deinterleave.getSrc(), oddIndex,
                       deinterleave.getDst()[1], index);
    return;
  }
  auto sourceIndex = deinterleave.getIndexMode() == DeinterleaveMode::CHANNEL_0
                         ? Value(pairIndex)
                         : Value(oddIndex);
  copyChannelElement(rewriter, loc, deinterleave.getSrc(), sourceIndex,
                     deinterleave.getDst().front(), index);
}

LogicalResult liftChannelMovement(Operation &operation, IRRewriter &rewriter) {
  auto structured = cast<HIVMStructuredOp>(operation);
  auto inputs = getBufferMovementInputs(operation);
  auto outputs = structured.getDpsInits();
  auto interleave = dyn_cast<VInterleaveOp>(operation);
  auto iterationBuffer = interleave ? inputs.front() : outputs.front();
  rewriter.setInsertionPoint(&operation);
  auto loc = operation.getLoc();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto end = rewriter.create<memref::DimOp>(loc, iterationBuffer, 0);
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = rewriter.create<scf::ForOp>(loc, zero, end, step);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    emitChannelElements(operation, rewriter, loop.getInductionVar());
  }
  rewriter.eraseOp(&operation);
  return success();
}

LogicalResult liftBufferMovement(Operation &operation, IRRewriter &rewriter) {
  auto structured = cast<HIVMStructuredOp>(operation);
  if (isa<VInterleaveOp, VDeinterleaveOp>(operation))
    return liftChannelMovement(operation, rewriter);
  if (!isa<VBrcOp, VTransposeOp>(operation))
    return operation.emitError("buffer movement has no vector lowering");
  auto outputs = structured.getDpsInits();
  auto inputs = getBufferMovementInputs(operation);
  auto outputType = cast<MemRefType>(outputs.front().getType());
  auto rank = outputType.getRank();
  auto identity = rewriter.getMultiDimIdentityMap(rank);
  AffineMap inputMap;
  if (!isa<ShapedType>(inputs.front().getType())) {
    inputMap = AffineMap::get(rank, 0, rewriter.getContext());
  } else if (auto transpose = dyn_cast<VTransposeOp>(operation)) {
    inputMap = inversePermutation(AffineMap::getPermutationMap(
        llvm::to_vector(transpose.getPermutation()), rewriter.getContext()));
  } else {
    auto broadcast = cast<VBrcOp>(operation);
    SmallVector<AffineExpr> indices;
    for (int64_t dim = 0; dim < rank; ++dim)
      indices.push_back(llvm::is_contained(broadcast.getBroadcastDims(), dim)
                            ? rewriter.getAffineConstantExpr(0)
                            : rewriter.getAffineDimExpr(dim));
    inputMap = AffineMap::get(rank, 0, indices, rewriter.getContext());
  }
  rewriter.setInsertionPoint(&operation);
  rewriter.create<linalg::GenericOp>(
      operation.getLoc(), TypeRange{}, inputs, outputs,
      ArrayRef<AffineMap>{inputMap, identity},
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [](OpBuilder &builder, Location loc, ValueRange args) {
        builder.create<linalg::YieldOp>(loc, args.front());
      });
  rewriter.eraseOp(&operation);
  return success();
}

void annotateBufferFunction(func::FuncOp function) {
  auto unit = UnitAttr::get(function.getContext());
  function->setAttr("hivm.vector_function", unit);
  function->setAttr("no_inline", unit);
  if (function->hasAttr(TFuncCoreTypeAttr::name)) {
    function->setAttr(
        TFuncCoreTypeAttr::name,
        TFuncCoreTypeAttr::get(function.getContext(), TFuncCoreType::AIV));
    function->removeAttr(TCoreRatioAttr::name);
    function->setAttr("mix_mode",
                      StringAttr::get(function.getContext(), "aiv"));
  }
  function->getParentOfType<ModuleOp>().walk([&](func::CallOp call) {
    if (call.getCallee() == function.getSymName()) {
      call->setAttr("hivm.vector_function", unit);
      call->setAttr("no_inline", unit);
    }
  });
}

LogicalResult vectorizeBufferFunctions(ModuleOp module) {
  if (hacc::utils::isAscend960(module))
    module.walk(promotePrivateVectorInput);
  Groups groups;
  auto checked = module.walk([&](Operation *operation) {
    if (!isBufferMovement(*operation))
      return WalkResult::advance();
    if (failed(validateBufferMovement(*operation)))
      return WalkResult::interrupt();
    groups.push_back({operation});
    return WalkResult::advance();
  });
  if (checked.wasInterrupted())
    return failure();
  if (groups.empty())
    return success();
  std::string error;
  auto functions = outlineGroups(module, groups, error);
  if (failed(functions))
    return module.emitError(error);
  IRRewriter rewriter(module.getContext());
  for (auto function : *functions) {
    annotateBufferFunction(function);
    SmallVector<Operation *> targets;
    function.walk([&](Operation *operation) {
      if (isBufferMovement(*operation))
        targets.push_back(operation);
    });
    for (auto *operation : targets)
      if (failed(liftBufferMovement(*operation, rewriter)))
        return failure();
  }
  if (failed(vectorizeFunctions(*functions, VectorizeOptions{}, error)) ||
      failed(cleanupFunctions(*functions, error)))
    return module.emitError(error);
  if (failed(checkVectorizedFunctions(*functions, error)))
    return module.emitError(error);
  return verify(module);
}

struct HIVMVectorizeBuffersPass
    : PassWrapper<HIVMVectorizeBuffersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HIVMVectorizeBuffersPass)

  StringRef getArgument() const final { return "hivm-vectorize-buffers"; }
  StringRef getDescription() const final {
    return "Vectorize buffer data movement introduced after tensor "
           "vectorization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, transform::TransformDialect,
                    linalg::LinalgDialect, vector::VectorDialect>();
    linalg::registerTransformDialectExtension(registry);
    linalg::registerTilingInterfaceExternalModels(registry);
    scf::registerTransformDialectExtension(registry);
    tensor::registerTransformDialectExtension(registry);
  }

  void runOnOperation() override {
    if (failed(vectorizeBufferFunctions(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createHIVMVectorizeBuffersPass() {
  return std::make_unique<HIVMVectorizeBuffersPass>();
}

void mlir::hivm::registerHIVMVectorizeBuffersPass() {
  PassRegistration<HIVMVectorizeBuffersPass>();
}
