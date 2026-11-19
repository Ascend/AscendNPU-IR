//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/BufferizableOpInterfaceImpl.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace hivm;
using namespace mlir::bufferization;

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult bufferizeDestinationStyleOpInterface(
    RewriterBase &rewriter, DestinationStyleOpInterface op,
    const BufferizationOptions &options,
    bool supportMixedTensorBufferMode = false) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics()) {
    return success();
  }

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics() && !supportMixedTensorBufferMode) {
    return op->emitError() << "op does not have tensor semantics";
  }

  // New operands for the cloned op.
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (!isa<TensorType>(opOperand.get().getType())) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand.get(), options);
    if (failed(buffer)) {
      return failure();
    }
    newOperands.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer)) {
      return failure();
    }
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands.
  clone(rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

struct MmadL0OpInterface
    : public DstBufferizableOpInterfaceExternalModel<MmadL0OpInterface,
                                                     hivm::MmadL0Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &state) const {
    auto matmul = cast<MmadL0Op>(op);
    auto dps = cast<DestinationStyleOpInterface>(op);
    return dps.isDpsInput(&operand) ||
           (dps.isDpsInit(&operand) && matmul.readsAccumulator());
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Physical L0/BT storage is supplied by the caller. Do not introduce
    // tensor copies or allocations with inferred layouts/address spaces.
    if (!cast<DestinationStyleOpInterface>(op).hasPureBufferSemantics())
      return op->emitOpError("standalone mmadL0 requires explicit memrefs");
    return success();
  }
};

struct MmadL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<MmadL1OpInterface,
                                                     hivm::MmadL1Op> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct BatchMmadL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          BatchMmadL1OpInterface, hivm::BatchMmadL1Op> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct Conv1DL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<Conv1DL1OpInterface,
                                                     hivm::Conv1DL1Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct Conv2DL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<Conv2DL1OpInterface,
                                                     hivm::Conv2DL1Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct Conv3DL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<Conv3DL1OpInterface,
                                                     hivm::Conv3DL1Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct FixpipeOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FixpipeOpInterface,
                                                     hivm::FixpipeOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (dpsOp.hasPureTensorSemantics()) {
      return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
    }
    // We only handle the case where fixpipe op's input is a tensor from
    // mmad and fixpipe op's output is a memref type.
    auto srcOp = dpsOp.getDpsInputOperand(0);
    auto dstOp = dpsOp.getDpsInitOperand(0);
    if (!isa<TensorType>(srcOp->get().getType()) ||
        !isa<MemRefType>(dstOp->get().getType())) {
      return op->emitError() << "src and dst op should have tensor and memref "
                                "type, respectively";
    }
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
    if (failed(buffer)) {
      return failure();
    }
    // Set insertion point now that potential alloc/dealloc are introduced.
    rewriter.setInsertionPoint(op);
    // Clone the op, but use the new operands.
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (hacc::utils::isRegBasedArch(moduleOp)) {
      // regbase: FixpipeOp may carry an extra quantScale operand.
      auto fixPipeOp = static_cast<FixpipeOp>(op);
      SmallVector<Value> opr{*buffer, dstOp->get()};
      if (fixPipeOp.getQuantScale())
        opr.push_back(fixPipeOp.getQuantScale());
      auto newOp = cast<DestinationStyleOpInterface>(
          clone(rewriter, op, /*newResultTypes=*/TypeRange{}, opr));
      rewriter.replaceOp(op, newOp);
    } else {
      // membase: original behavior with only src + dst operands.
      auto newOp = cast<DestinationStyleOpInterface>(clone(
          rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
    // We need to manually replace the old op because it has memory effects
    // and won't be deleted automatically.
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }
};

template <typename OpType>
struct NDNZConversionOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          NDNZConversionOpInterface<OpType>, OpType> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

template <typename CustomOpT>
static std::optional<int64_t> getDpsInitIndex(CustomOpT op,
                                              OpOperand &opOperand) {
  if (!op.isDpsInit(&opOperand))
    return std::nullopt;
  return opOperand.getOperandNumber() - op.getDpsInits().getBeginOperandIndex();
}

template <typename CustomOpT>
static bool isInplaceOutputOperand(CustomOpT op, OpOperand &opOperand) {
  auto outputIdx = getDpsInitIndex(op, opOperand);
  return outputIdx && op.getInplaceInputIndexForOutput(*outputIdx).has_value();
}

static std::optional<std::pair<unsigned, unsigned>>
getRequiredByteAlias(Operation *op) {
  auto custom = dyn_cast<CustomOp>(op);
  if (!custom || custom.getNumResults() == 0)
    return std::nullopt;
  for (unsigned output = 0; output < custom.getOutputs().size(); ++output) {
    auto input = custom.getInplaceInputIndexForOutput(output);
    if (input && custom.getInputs()[*input].getType() !=
                     custom.getOutputs()[output].getType())
      return std::make_pair(*input, output);
  }
  return std::nullopt;
}

static LogicalResult
getByteAliasBufferOperands(CustomOp op, std::pair<unsigned, unsigned> alias,
                           SmallVectorImpl<Value> &operands,
                           RewriterBase &rewriter,
                           const BufferizationOptions &options) {
  unsigned aliasedOutput = op.getInputs().size() + alias.second;
  operands.assign(op->operand_begin(), op->operand_end());
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (index == alias.first || index == aliasedOutput ||
        !isa<TensorType>(operand.getType()))
      continue;
    auto other = getBuffer(rewriter, operand, options);
    if (failed(other))
      return failure();
    operands[index] = *other;
  }
  return success();
}

static std::pair<Value, Value> replaceByteAliasAlloc(memref::AllocOp alloc,
                                                     Location loc,
                                                     RewriterBase &rewriter) {
  rewriter.setInsertionPoint(alloc);
  auto type = alloc.getType();
  int64_t bytes = type.getNumElements() * type.getElementTypeBitWidth() / 8;
  auto byteType = MemRefType::get({bytes}, rewriter.getI8Type(), AffineMap{},
                                  type.getMemorySpace());
  auto raw = rewriter.create<memref::AllocOp>(loc, byteType);
  raw->setAttrs(alloc->getAttrs());
  if (!raw.getAlignmentAttr() || raw.getAlignmentAttr().getInt() < 32)
    raw.setAlignmentAttr(rewriter.getI64IntegerAttr(32));
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto dense =
      rewriter.create<memref::ViewOp>(loc, type, raw, zero, ValueRange{});
  rewriter.replaceOp(alloc, dense.getResult());
  return {raw.getResult(), dense.getResult()};
}

static LogicalResult bufferizeByteAliasOutputs(
    CustomOp op, std::pair<unsigned, unsigned> alias, Value root, Value zero,
    SmallVectorImpl<Value> &operands, RewriterBase &rewriter,
    const BufferizationOptions &options) {
  auto scoreType = cast<MemRefType>(operands[alias.first].getType());
  auto outputType =
      bufferization::getBufferType(op->getResult(alias.second), options);
  if (failed(outputType))
    return failure();
  MemRefType pType = MemRefType::Builder(cast<MemRefType>(*outputType))
                         .setMemorySpace(scoreType.getMemorySpace());
  auto rootType = cast<MemRefType>(root.getType());
  int64_t flatElements = rootType.getNumElements() *
                         rootType.getElementTypeBitWidth() /
                         pType.getElementTypeBitWidth();
  auto flatType = MemRefType::get({flatElements}, pType.getElementType(),
                                  AffineMap{}, pType.getMemorySpace());
  auto flat = rewriter.create<memref::ViewOp>(op.getLoc(), flatType, root, zero,
                                              ValueRange{});
  auto [strides, offset] = utils::getStridesAndOffsetCompat(pType);
  auto p = rewriter.create<memref::ReinterpretCastOp>(
      op.getLoc(), pType, flat, offset, pType.getShape(), strides);
  unsigned inputCount = op.getInputs().size();
  operands[inputCount + alias.second] = p.getResult();
  Operation *pInit = op.getOutputs()[alias.second].getDefiningOp();
  auto bufferOp = clone(rewriter, op, /*newResultTypes=*/TypeRange{}, operands);
  auto attrs =
      llvm::to_vector(op->getAttrOfType<ArrayAttr>(CustomOp::kArgAttrsName));
  NamedAttrList inputAttrs(cast<DictionaryAttr>(attrs[alias.first]));
  inputAttrs.erase(CustomOp::kInplaceOutsAttrName);
  attrs[alias.first] = inputAttrs.getDictionary(op.getContext());
  bufferOp->setAttr(CustomOp::kArgAttrsName, rewriter.getArrayAttr(attrs));
  replaceOpWithBufferizedValues(
      rewriter, op,
      ArrayRef<Value>(operands).slice(inputCount, op.getOutputs().size()));
  if (pInit && isOpTriviallyDead(pInit))
    rewriter.eraseOp(pInit);
  return success();
}

static Value castBufferMemorySpace(RewriterBase &rewriter, Location loc,
                                   Value buffer, Attribute memorySpace) {
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getMemorySpace() == memorySpace)
    return buffer;
  MemRefType target = MemRefType::Builder(type).setMemorySpace(memorySpace);
  return rewriter.create<memref::MemorySpaceCastOp>(loc, target, buffer);
}

// Resolve a whole-buffer selection before constructing its byte views.
static Operation *getBufferSelector(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (isa<scf::IfOp, arith::SelectOp>(def))
      return def;
    if (!isa<memref::MemorySpaceCastOp, memref::CastOp>(def))
      break;
    value = def->getOperand(0);
  }
  return nullptr;
}

static LogicalResult
bufferizeSelectedByteAlias(CustomOp op, std::pair<unsigned, unsigned> alias,
                           Value score, Operation *selector,
                           RewriterBase &rewriter,
                           const BufferizationOptions &options) {
  auto reject = [&op]() {
    return op.emitOpError(
        "inplace_outs requires a selection of whole local input allocations");
  };
  auto ifSelector = dyn_cast<scf::IfOp>(selector);
  auto selectSelector = dyn_cast<arith::SelectOp>(selector);
  Value selectorResult = selector->getResult(0);
  Value selectorCondition =
      ifSelector ? ifSelector.getCondition() : selectSelector.getCondition();
  auto scoreType = dyn_cast<MemRefType>(score.getType());
  if (!scoreType || selector->getNumResults() != 1 ||
      selector->getBlock() != op->getBlock())
    return reject();
  auto selectedType = dyn_cast<MemRefType>(selectorResult.getType());
  auto isUBOrGeneric = [](MemRefType type) {
    return type && (!type.getMemorySpace() ||
                    getOptionalHIVMAddressSpace(type) == AddressSpace::UB);
  };
  if (!isUBOrGeneric(scoreType) || !isUBOrGeneric(selectedType))
    return reject();

  SmallVector<memref::AllocOp> allocations;
  SmallVector<std::pair<Value, Block *>> selectorInputs;
  if (ifSelector) {
    for (Block *block : {ifSelector.thenBlock(), ifSelector.elseBlock()}) {
      auto yield = cast<scf::YieldOp>(block->getTerminator());
      selectorInputs.push_back({yield.getOperand(0), block});
    }
  } else {
    selectorInputs.push_back({selectSelector.getTrueValue(), nullptr});
    selectorInputs.push_back({selectSelector.getFalseValue(), nullptr});
  }
  for (auto [rootValue, block] : selectorInputs) {
    Value value = rootValue;
    // Each address-preserving step has one source.
    while (Operation *def = value.getDefiningOp()) {
      if (block ? def->getBlock() != block : isa<memref::AllocOp>(def))
        break;
      if (!isa<memref::MemorySpaceCastOp, memref::CastOp,
               bufferization::ToMemrefOp, bufferization::ToTensorOp>(def))
        return reject();
      value = def->getOperand(0);
    }
    auto alloc = value.getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType().getShape() != scoreType.getShape() ||
        alloc.getType().getElementType() != scoreType.getElementType() ||
        !alloc.getType().getLayout().isIdentity() ||
        !isUBOrGeneric(alloc.getType()))
      return reject();
    allocations.push_back(alloc);
  }
  if (allocations[0].getType().getMemorySpace() !=
      allocations[1].getType().getMemorySpace())
    return reject();
  // Acquire the remaining operands before replacing allocations.
  SmallVector<Value> operands;
  if (failed(
          getByteAliasBufferOperands(op, alias, operands, rewriter, options)))
    return failure();
  bool sameAllocation = allocations[0] == allocations[1];
  Value trueRoot =
      replaceByteAliasAlloc(allocations[0], op.getLoc(), rewriter).first;
  Value falseRoot =
      sameAllocation
          ? trueRoot
          : replaceByteAliasAlloc(allocations[1], op.getLoc(), rewriter).first;
  // Select the same allocation for both views. The original branch keeps
  // its waits and other effects until ordinary canonicalization drops its
  // now-unused result.
  rewriter.setInsertionPoint(selector);
  auto selected = rewriter.create<arith::SelectOp>(
      selector->getLoc(), selectorCondition, trueRoot, falseRoot);
  if (selectSelector)
    selected->setAttrs(selector->getAttrs());
  auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value root = castBufferMemorySpace(
      rewriter, op.getLoc(), selected.getResult(), scoreType.getMemorySpace());
  MemRefType denseType = MemRefType::Builder(scoreType).setLayout({});
  Value dense = rewriter.create<memref::ViewOp>(op.getLoc(), denseType, root,
                                                zero, ValueRange{});
  if (dense.getType() != scoreType)
    dense = rewriter.create<memref::CastOp>(op.getLoc(), scoreType, dense);
  operands[alias.first] = dense;

  rewriter.setInsertionPoint(op);
  return bufferizeByteAliasOutputs(op, alias, root, zero, operands, rewriter,
                                   options);
}

static LogicalResult
bufferizeRequiredByteAlias(CustomOp op, std::pair<unsigned, unsigned> alias,
                           RewriterBase &rewriter,
                           const BufferizationOptions &options) {
  if (options.copyBeforeWrite)
    return op.emitOpError("inplace_outs cannot use copyBeforeWrite");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  FailureOr<Value> buffer =
      getBuffer(rewriter, op.getInputs()[alias.first], options);
  if (failed(buffer))
    return failure();
  Value score = *buffer;
  auto scoreType = dyn_cast<MemRefType>(score.getType());
  if (auto selected = getBufferSelector(score))
    return bufferizeSelectedByteAlias(op, alias, score, selected, rewriter,
                                      options);
  if (!scoreType || !scoreType.getLayout().isIdentity())
    return op.emitOpError(
        "inplace_outs requires a dense identity input buffer");

  Value root = score;
  if (auto cast = root.getDefiningOp<memref::MemorySpaceCastOp>())
    root = cast.getSource();
  if (auto view = root.getDefiningOp<memref::ViewOp>()) {
    if (!matchPattern(view.getByteShift(), m_Zero()))
      return op.emitOpError("inplace_outs requires a zero-offset input view");
    root = view.getSource();
  }
  auto alloc = root.getDefiningOp<memref::AllocOp>();
  int64_t bytes =
      scoreType.getNumElements() * scoreType.getElementTypeBitWidth() / 8;
  auto rootType = dyn_cast<MemRefType>(root.getType());
  bool byteRoot =
      rootType && rootType.getRank() == 1 &&
      rootType.getElementType().isInteger(8) && rootType.hasStaticShape() &&
      rootType.getDimSize(0) == bytes && rootType.getLayout().isIdentity();
  bool typedRoot = rootType && rootType.getShape() == scoreType.getShape() &&
                   rootType.getElementType() == scoreType.getElementType() &&
                   rootType.getLayout().isIdentity();
  if (!alloc || (!byteRoot && !typedRoot))
    return op.emitOpError(
        "inplace_outs requires a whole local input allocation");

  // Resolve other buffers before changing allocations; the aliased output
  // initializer requires neither a buffer nor a copy.
  SmallVector<Value> operands;
  if (failed(
          getByteAliasBufferOperands(op, alias, operands, rewriter, options)))
    return failure();
  if (!byteRoot) {
    bool directScore = score == root;
    auto [raw, dense] = replaceByteAliasAlloc(alloc, op.getLoc(), rewriter);
    root = raw;
    if (directScore)
      score = dense;
  }
  if (byteRoot &&
      (!alloc.getAlignmentAttr() || alloc.getAlignmentAttr().getInt() < 32))
    rewriter.modifyOpInPlace(alloc, [&alloc, &rewriter] {
      alloc.setAlignmentAttr(rewriter.getI64IntegerAttr(32));
    });
  operands[alias.first] = score;

  rewriter.setInsertionPoint(op);
  root = castBufferMemorySpace(rewriter, op.getLoc(), root,
                               scoreType.getMemorySpace());
  auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  return bufferizeByteAliasOutputs(op, alias, root, zero, operands, rewriter,
                                   options);
}

template <typename CustomOpT>
struct HIVMCustomOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMCustomOpInterface<CustomOpT>, CustomOpT> {
  using Base =
      DstBufferizableOpInterfaceExternalModel<HIVMCustomOpInterface<CustomOpT>,
                                              CustomOpT>;

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand,
                               const AnalysisState &state) const {
    if (auto alias = getRequiredByteAlias(op)) {
      auto custom = cast<CustomOp>(op);
      return operand.getOperandNumber() == alias->first ||
             (custom.isDpsInit(&operand) &&
              operand.getOperandNumber() !=
                  custom.getInputs().size() + alias->second);
    }
    return Base::bufferizesToMemoryWrite(op, operand, state);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                      const AnalysisState &state) const {
    if (auto alias = getRequiredByteAlias(op)) {
      if (operand.getOperandNumber() == alias->first)
        return {{op->getResult(alias->second), BufferRelation::Unknown}};
      if (operand.getOperandNumber() ==
          cast<CustomOp>(op).getInputs().size() + alias->second)
        return {};
    }
    return Base::getAliasingValues(op, operand, state);
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    if (auto alias = getRequiredByteAlias(op);
        alias && value == op->getResult(alias->second)) {
      auto custom = cast<CustomOp>(op);
      auto inputType = bufferization::getBufferType(
          custom.getInputs()[alias->first], options, invocationStack);
      if (failed(inputType))
        return failure();
      auto output = cast<RankedTensorType>(value.getType());
      auto type = MemRefType::get(output.getShape(), output.getElementType(),
                                  AffineMap{}, inputType->getMemorySpace());
      auto attrs = cast<DictionaryAttr>(custom->getAttrOfType<ArrayAttr>(
          CustomOp::kArgAttrsName)[alias->first]);
      auto binding = attrs.getAs<ArrayAttr>(CustomOp::kInplaceOutsAttrName);
      if (!binding)
        return type;
      auto map = cast<AffineMapAttr>(binding[1]).getValue();
      if (map.getNumDims() != output.getRank() ||
          map.getNumResults() != output.getRank() || map.getNumSymbols())
        return custom.emitOpError("invalid inplace_outs output view map");
      auto offsetExpr = makeCanonicalStridedLayoutExpr(
          output.getShape(), map.getResults(), custom.getContext());
      auto layout = AffineMap::get(output.getRank(), 0, offsetExpr);
      type = MemRefType::Builder(type).setLayout(AffineMapAttr::get(layout));
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(utils::getStridesAndOffsetCompat(type, strides, offset)) ||
          offset != 0 ||
          llvm::any_of(strides, [](int64_t stride) { return stride < 0; }))
        return custom.emitOpError(
            "inplace_outs requires a static zero-offset output byte view");
      int64_t span = 1;
      for (auto [size, stride] : llvm::zip(type.getShape(), strides))
        span += (size - 1) * stride;
      auto input =
          cast<RankedTensorType>(custom.getInputs()[alias->first].getType());
      if (span * output.getElementTypeBitWidth() >
          input.getNumElements() * input.getElementTypeBitWidth())
        return custom.emitOpError(
            "inplace_outs output byte view exceeds the input capacity");
      type = MemRefType::Builder(type).setLayout(
          StridedLayoutAttr::get(custom.getContext(), offset, strides));
      return type;
    }
    return bufferization::detail::defaultGetBufferType(value, options,
                                                       invocationStack);
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto customOp = cast<CustomOpT>(op);
    if (isInplaceOutputOperand(customOp, opOperand)) {
      return false;
    }
    return true;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    auto customOp = cast<CustomOpT>(op);
    if (auto alias = getRequiredByteAlias(op)) {
      if (opOperand.getOperandNumber() == alias->first)
        return true;
      if (getDpsInitIndex(customOp, opOperand) == alias->second)
        return false;
    }
    return isInplaceOutputOperand(customOp, opOperand);
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                        const AnalysisState &state) const {
    if (!uRead || !uWrite || uRead->getOwner() != op ||
        uWrite->getOwner() != op || uRead->get() != uWrite->get())
      return false;

    auto customOp = cast<CustomOpT>(op);
    auto outputIdx = getDpsInitIndex(customOp, *uWrite);
    if (!outputIdx)
      return false;

    auto inputIdx = customOp.getInplaceInputIndexForOutput(*outputIdx);
    return inputIdx &&
           uRead->getOperandNumber() == static_cast<unsigned>(*inputIdx);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    if (auto alias = getRequiredByteAlias(op))
      return bufferizeRequiredByteAlias(cast<CustomOp>(op), *alias, rewriter,
                                        options);
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /* supportMixedTensorBufferMode */ true);
  }
};

struct HIVMLoadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMLoadOpInterface,
                                                     hivm::LoadOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto loadOp = cast<hivm::LoadOp>(op);
    auto dst = loadOp.getDst();
    if (dst.getDefiningOp<tensor::EmptyOp>())
      return getMemRefTypeWithStaticIdentityLayout(
          cast<TensorType>(dst.getType()));
    auto dstMemrefType =
        bufferization::getBufferType(loadOp.getDst(), options, invocationStack);
    return dstMemrefType;
  }
};

template <typename CopyOrStoreOpT>
struct HIVMCopyOrStoreOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMCopyOrStoreOpInterface<CopyOrStoreOpT>, CopyOrStoreOpT> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    // Clean up copyOps inserted by NormalizeToTensorOp pass
    if (dpsOp->hasAttr("to_be_replaced")) {
      rewriter.replaceAllUsesWith(dpsOp->getResult(0),
                                  dpsOp.getDpsInputOperand(0)->get());
      rewriter.eraseOp(dpsOp);
      return success();
    }
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (dpsOp.hasPureTensorSemantics()) {
      return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
    }
    // We only handle the case where fixpipe op's input is a tensor from
    // mmad and fixpipe op's output is a memref type.
    auto srcOp = dpsOp.getDpsInputOperand(0);
    auto dstOp = dpsOp.getDpsInitOperand(0);
    if (!isa<TensorType>(srcOp->get().getType()) ||
        !isa<MemRefType>(dstOp->get().getType())) {
      return op->emitError() << "src and dst op should have tensor and memref "
                                "type, respectively";
    }
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
    if (failed(buffer)) {
      return failure();
    }
    // Set insertion point now that potential alloc/dealloc are introduced.
    rewriter.setInsertionPoint(op);
    // Clone the op, but use the new operands.
    auto newOp = cast<DestinationStyleOpInterface>(clone(
        rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
    // We need to manually replace the old op because it has memory effects
    // and won't be deleted automatically.
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct HIVMMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMMatmulOpInterface,
                                                     hivm::MatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

struct HIVMMixMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMMixMatmulOpInterface,
                                                     hivm::MixMatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // The `tilingParams` operand might be already bufferized.
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

struct HIVMMixGroupMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMMixGroupMatmulOpInterface, hivm::MixGroupMatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // The `tilingParams` operand might be already bufferized.
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

template <typename OpTy>
struct VectorOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VectorOpInterface<OpTy>,
                                                     OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it is not an input/init.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    // Src0 and dst of elemwiseOp are not conflicting if the op bufferizes
    // to element-wise access.
    auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
    return hivmOp && hivmOp.isElemwiseNaryOp();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct DebugOpInterface
    : public BufferizableOpInterface::ExternalModel<DebugOpInterface,
                                                    hivm::DebugOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto debugOp = cast<hivm::DebugOp>(op);

    auto debugtype = debugOp.getDebugtype();
    auto prefix = debugOp.getPrefix();
    auto hex = debugOp.getHex();

    Value newArg;
    const auto &arg = debugOp.getArg();
    Value value = arg;
    if (isa<TensorType>(value.getType())) {
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);
      if (failed(maybeBuffer))
        return failure();
      Value buffer = *maybeBuffer;
      newArg = buffer;
    } else {
      newArg = value;
    }

    auto tcoretype = debugOp.getTcoretypeAttr();
    auto memscope = debugOp.getMemscopeAttr();
    replaceOpWithNewBufferizedOp<hivm::DebugOp>(
        rewriter, op, debugtype, prefix, hex, newArg, tcoretype, memscope);

    return success();
  }
};

struct VPadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VPadOpInterface,
                                                     hivm::VPadOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // TODO
    return failure();
  }
};

struct VConcatOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VConcatOpInterface,
                                                     hivm::VConcatOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

/// Helper structure that iterates over all VectorOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops> struct VectorOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<VectorOpInterface<Ops>>(*ctx), ...);
  }
};

struct BitcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BitcastOpInterface,
                                                    hivm::BitcastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto bitcastOp = dyn_cast<hivm::BitcastOp>(op);
    auto resultTensorType = dyn_cast<TensorType>(bitcastOp.getType());
    if (!resultTensorType)
      return success();

    FailureOr<Value> source = getBuffer(rewriter, bitcastOp.getSrc(), options);
    if (failed(source))
      return failure();
    auto sourceType = cast<BaseMemRefType>(source->getType());

    // Result type should have same layout and address space as the source type.
    BaseMemRefType resultType;
    if (auto rankedMemRefType = dyn_cast<MemRefType>(sourceType)) {
      resultType = MemRefType::get(
          rankedMemRefType.getShape(), resultTensorType.getElementType(),
          rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
    } else {
      auto unrankedMemrefType = cast<UnrankedMemRefType>(sourceType);
      resultType = UnrankedMemRefType::get(resultTensorType.getElementType(),
                                           unrankedMemrefType.getMemorySpace());
    }

    replaceOpWithNewBufferizedOp<hivm::BitcastOp>(rewriter, op, resultType,
                                                  *source);
    return success();
  }
};

struct IndirectStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<IndirectStoreOpInterface,
                                                    hivm::IndirectStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() > 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indirectStoreOp = cast<hivm::IndirectStoreOp>(op);

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, indirectStoreOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    FailureOr<Value> offsetBuffer =
        getBuffer(rewriter, indirectStoreOp.getOffsets(), options);
    if (failed(offsetBuffer))
      return failure();

    auto dstBuffer = indirectStoreOp.getDst();

    FailureOr<Value> maskBuffer = failure();
    if (indirectStoreOp.getMask()) {
      maskBuffer = getBuffer(rewriter, indirectStoreOp.getMask(), options);
      if (failed(maskBuffer))
        return failure();
    }

    auto mask = indirectStoreOp.getMask();
    hivm::IndirectStoreOp newOp;
    if (mask) {
      newOp = rewriter.create<hivm::IndirectStoreOp>(
          indirectStoreOp.getLoc(),
                                               /*operands*/
          dstBuffer, *offsetBuffer, *srcBuffer, *maskBuffer);
    } else {
      newOp = rewriter.create<hivm::IndirectStoreOp>(
          indirectStoreOp.getLoc(),
                                               /*operands*/
          dstBuffer, *offsetBuffer, *srcBuffer, mask);
    }

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct EmbeddingGatherOpInterface
    : public BufferizableOpInterface::ExternalModel<EmbeddingGatherOpInterface,
                                                    hivm::EmbeddingGatherOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() < 2;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto gatherOp = cast<EmbeddingGatherOp>(op);
    AliasingValueList result;

    if (opOperand.getOperandNumber() == 2) { // $dst
      // dst is alias of the result
      result.addAlias(
          {AliasingValue(gatherOp->getResult(0), BufferRelation::Equivalent,
                         /*isMustAlias=*/true)});
    }

    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<hivm::EmbeddingGatherOp>(op);

    auto srcBuffer = gatherOp.getSrc();

    FailureOr<Value> indexBuffer =
        getBuffer(rewriter, gatherOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();

    FailureOr<Value> dstBuffer =
        getBuffer(rewriter, gatherOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    Value bound = gatherOp.getBound();
    auto offsets = gatherOp.getOffsets();
    auto numels = gatherOp.getNumels();

    rewriter.create<EmbeddingGatherOp>(gatherOp.getLoc(),
                                       /*resultType*/ TypeRange{},
                                       /*operands*/
                                       srcBuffer, *indexBuffer, *dstBuffer,
                                       bound, offsets, numels);

    if (gatherOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }

    return success();
  }
};

struct IndirectLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<IndirectLoadOpInterface,
                                                    hivm::IndirectLoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() < 2 ||
           opOperand.getOperandNumber() == 3 ||
           opOperand.getOperandNumber() == 4;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto indirectLoadOp = cast<IndirectLoadOp>(op);
    AliasingValueList result;

    if (opOperand.getOperandNumber() == 2) { // $dst
      result.addAlias({AliasingValue(indirectLoadOp->getResult(0),
                                     BufferRelation::Equivalent,
                                     /*isMustAlias=*/true)});
    }

    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indirectLoadOp = cast<hivm::IndirectLoadOp>(op);

    auto srcBuffer = indirectLoadOp.getSrc();

    FailureOr<Value> offsetBuffer =
        getBuffer(rewriter, indirectLoadOp.getOffsets(), options);
    if (failed(offsetBuffer))
      return failure();

    FailureOr<Value> dstBuffer =
        getBuffer(rewriter, indirectLoadOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    FailureOr<Value> maskBuffer = failure();
    if (indirectLoadOp.getMask()) {
      maskBuffer = getBuffer(rewriter, indirectLoadOp.getMask(), options);
      if (failed(maskBuffer))
        return failure();
    }

    FailureOr<Value> otherBuffer = failure();
    if (indirectLoadOp.getOther()) {
      otherBuffer = getBuffer(rewriter, indirectLoadOp.getOther(), options);
      if (failed(otherBuffer))
        return failure();
    }

    auto mask = indirectLoadOp.getMask();
    auto other = indirectLoadOp.getOther();

    if (indirectLoadOp.getMask()) {
      if (indirectLoadOp.getOther()) {
        rewriter.create<IndirectLoadOp>(indirectLoadOp.getLoc(),
                                        /*resultType*/ TypeRange{},
                                        /*operands*/
                                        srcBuffer, *offsetBuffer, *dstBuffer,
                                        *maskBuffer, *otherBuffer,
                                        indirectLoadOp.getIsVolatileAttr());
      } else {
        rewriter.create<IndirectLoadOp>(indirectLoadOp.getLoc(),
                                        /*resultType*/ TypeRange{},
                                        /*operands*/
                                        srcBuffer, *offsetBuffer, *dstBuffer,
                                        *maskBuffer, other,
                                        indirectLoadOp.getIsVolatileAttr());
      }
    } else {
      rewriter.create<IndirectLoadOp>(indirectLoadOp.getLoc(),
                                      /*resultType*/ TypeRange{},
                                      /*operands*/
                                      srcBuffer, *offsetBuffer, *dstBuffer,
                                      mask, other,
                                      indirectLoadOp.getIsVolatileAttr());
    }

    if (indirectLoadOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }

    return success();
  }
};

struct StrideLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<StrideLoadOpInterface,
                                                    hivm::StrideLoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $src
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto strideLoadOp = cast<hivm::StrideLoadOp>(op);
    return &opOperand == &strideLoadOp.getDstMutable(); // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto strideLoadOp = cast<StrideLoadOp>(op);
    AliasingValueList result;
    if (&opOperand == &strideLoadOp.getDstMutable()) { // $dst
      result.addAlias(
          {AliasingValue(strideLoadOp->getResult(0), BufferRelation::Equivalent,
                         /*isMustAlias=*/true)});
    }
    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto strideLoadOp = cast<hivm::StrideLoadOp>(op);

    FailureOr<Value> dstBuffer =
        getBuffer(rewriter, strideLoadOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    rewriter.create<StrideLoadOp>(
        strideLoadOp.getLoc(),
        /*resultType*/ TypeRange{}, strideLoadOp.getSrc(), *dstBuffer,
        strideLoadOp.getOffset(), strideLoadOp.getOther(),
        strideLoadOp.getStride(), strideLoadOp.getNumel());

    if (strideLoadOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }

    return success();
  }
};

struct StrideStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<StrideStoreOpInterface,
                                                    hivm::StrideStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto strideStoreOp = cast<hivm::StrideStoreOp>(op);
    return &opOperand == &strideStoreOp.getSrcMutable(); // $src
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto strideStoreOp = cast<hivm::StrideStoreOp>(op);

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, strideStoreOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    rewriter.create<StrideStoreOp>(
        strideStoreOp.getLoc(), strideStoreOp.getDst(), *srcBuffer,
        strideStoreOp.getOffset(), strideStoreOp.getStride(),
        strideStoreOp.getNumel());
    rewriter.eraseOp(op);

    return success();
  }
};

struct GatherTOpInterface
    : public BufferizableOpInterface::ExternalModel<GatherTOpInterface,
                                                    hivm::GatherTOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0 ||
           opOperand.getOperandNumber() == 1;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto gatherOp = cast<GatherTOp>(op);
    AliasingValueList result;

    if (opOperand.getOperandNumber() == 2) { // $dst
      result.addAlias(
          {AliasingValue(gatherOp->getResult(0), BufferRelation::Equivalent,
                         /*isMustAlias=*/true)});
    }

    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<hivm::GatherTOp>(op);

    auto srcBuffer = gatherOp.getSrc();

    FailureOr<Value> indexBuffer =
        getBuffer(rewriter, gatherOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();

    FailureOr<Value> dstBuffer =
        getBuffer(rewriter, gatherOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    Value dim = gatherOp.getDim();
    Value bound = gatherOp.getBound();
    auto src_stride = gatherOp.getSrcStride();
    auto index_shape = gatherOp.getIndexShape();
    auto offsets = gatherOp.getOffsets();

    rewriter.create<GatherTOp>(gatherOp.getLoc(),
                               /*resultType*/ TypeRange{},
                               /*operands*/
                               srcBuffer, *indexBuffer, *dstBuffer, bound, dim,
                               src_stride, index_shape, offsets);

    if (gatherOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }

    return success();
  }
};

struct IndexPutOpInterface
    : public BufferizableOpInterface::ExternalModel<IndexPutOpInterface,
                                                    hivm::IndexPutOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1 ||
           opOperand.getOperandNumber() == 2;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indexPutOp = cast<hivm::IndexPutOp>(op);

    auto dstBuffer = indexPutOp.getDst();

    FailureOr<Value> indexBuffer =
        getBuffer(rewriter, indexPutOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();

    FailureOr<Value> valueBuffer =
        getBuffer(rewriter, indexPutOp.getValue(), options);
    if (failed(valueBuffer))
      return failure();

    Value scatter_dim = indexPutOp.getScatterDim();
    Value bound = indexPutOp.getBound();
    auto end_offset = indexPutOp.getEndOffset();
    auto start_offset = indexPutOp.getStartOffset();
    auto dst_stride = indexPutOp.getDstStride();

    IndexPutOp newOp = rewriter.create<IndexPutOp>(
        indexPutOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        dstBuffer, *indexBuffer, *valueBuffer, scatter_dim, bound, end_offset,
        start_offset, dst_stride);

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ScatterTOpInterface
    : public BufferizableOpInterface::ExternalModel<ScatterTOpInterface,
                                                    hivm::ScatterTOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1 ||
           opOperand.getOperandNumber() == 2;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto scatterTOp = cast<hivm::ScatterTOp>(op);

    FailureOr<Value> valueBuffer =
        getBuffer(rewriter, scatterTOp.getValue(), options);
    if (failed(valueBuffer))
      return failure();

    FailureOr<Value> indexTileBuffer =
        getBuffer(rewriter, scatterTOp.getIndexTile(), options);
    if (failed(indexTileBuffer))
      return failure();

    auto ret = rewriter.create<ScatterTOp>(
        scatterTOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        scatterTOp.getDst(), *valueBuffer, *indexTileBuffer,
        scatterTOp.getIndexBoundary(), scatterTOp.getDim(),
        scatterTOp.getDstStride(), scatterTOp.getIndexShape(),
        scatterTOp.getOffsets());

    rewriter.replaceOp(op, ret);

    return success();
  }
};

struct HIVMMatmulMxOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMMatmulMxOpInterface,
                                                     hivm::MmadMxL1Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // The `tilingParams` operand might be already bufferized.
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

} // namespace

void mlir::hivm::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, hivm::HIVMDialect *dialect) {
    FixpipeOp::attachInterface<FixpipeOpInterface>(*ctx);
    MmadL1Op::attachInterface<MmadL1OpInterface>(*ctx);
    MmadL0Op::attachInterface<MmadL0OpInterface>(*ctx);
    BatchMmadL1Op::attachInterface<BatchMmadL1OpInterface>(*ctx);
    Conv1DL1Op::attachInterface<Conv1DL1OpInterface>(*ctx);
    Conv2DL1Op::attachInterface<Conv2DL1OpInterface>(*ctx);
    Conv3DL1Op::attachInterface<Conv3DL1OpInterface>(*ctx);
    ND2NZOp::attachInterface<NDNZConversionOpInterface<ND2NZOp>>(*ctx);
    NZ2NDOp::attachInterface<NDNZConversionOpInterface<NZ2NDOp>>(*ctx);
    L12UBOp::attachInterface<HIVMCopyOrStoreOpInterface<L12UBOp>>(*ctx);
    LoadMXScaleOp::attachInterface<HIVMCopyOrStoreOpInterface<LoadMXScaleOp>>(
        *ctx);
    CopyOp::attachInterface<HIVMCopyOrStoreOpInterface<hivm::CopyOp>>(*ctx);
    CustomOp::attachInterface<HIVMCustomOpInterface<CustomOp>>(*ctx);
    CustomMacroOp::attachInterface<HIVMCustomOpInterface<CustomMacroOp>>(*ctx);
    LoadOp::attachInterface<HIVMLoadOpInterface>(*ctx);
    StoreOp::attachInterface<HIVMCopyOrStoreOpInterface<hivm::StoreOp>>(*ctx);
    MatmulOp::attachInterface<HIVMMatmulOpInterface>(*ctx);
    MixMatmulOp::attachInterface<HIVMMixMatmulOpInterface>(*ctx);
    MixGroupMatmulOp::attachInterface<HIVMMixGroupMatmulOpInterface>(*ctx);
    DebugOp::attachInterface<DebugOpInterface>(*ctx);
    VConcatOp::attachInterface<VConcatOpInterface>(*ctx);
    BitcastOp::attachInterface<BitcastOpInterface>(*ctx);
    EmbeddingGatherOp::attachInterface<EmbeddingGatherOpInterface>(*ctx);
    IndirectLoadOp::attachInterface<IndirectLoadOpInterface>(*ctx);
    StrideLoadOp::attachInterface<StrideLoadOpInterface>(*ctx);
    StrideStoreOp::attachInterface<StrideStoreOpInterface>(*ctx);
    IndirectStoreOp::attachInterface<IndirectStoreOpInterface>(*ctx);
    GatherTOp::attachInterface<GatherTOpInterface>(*ctx);
    IndexPutOp::attachInterface<IndexPutOpInterface>(*ctx);
    ScatterTOp::attachInterface<ScatterTOpInterface>(*ctx);
    MmadMxL1Op::attachInterface<HIVMMatmulMxOpInterface>(*ctx);
    // Register all HIVM Vector Ops
    VectorOpInterfaceHelper<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
