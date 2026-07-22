//===- CustomOpBase.cpp - HIVM custom op core implementation --------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-custom-op"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// CustomOp / CustomMacroOp - pipe and VF mode
//===----------------------------------------------------------------------===//

void CustomOp::setPipe(PIPE pipe) {
  getOperation()->setAttr(PipeAttr::name, PipeAttr::get(getContext(), pipe));
  getMaxRank();
}

PIPE CustomOp::getPipe() {
  if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(PipeAttr::name))
    return pipAttr.getPipe();

  return PIPE::PIPE_UNASSIGNED;
}

const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{
    {"__builtin_histogram",
     CustomOp::BuiltinInfo{TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT}},
};

void CustomMacroOp::setInPipe(PIPE pipe) {
  getOperation()->setAttr(inPipeName, PipeAttr::get(getContext(), pipe));
}

PIPE CustomMacroOp::getInPipe() {
  if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(inPipeName))
    return pipAttr.getPipe();

  return PIPE::PIPE_UNASSIGNED;
}

void CustomMacroOp::setOutPipe(PIPE pipe) {
  getOperation()->setAttr(outPipeName, PipeAttr::get(getContext(), pipe));
}

PIPE CustomMacroOp::getOutPipe() {
  if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(outPipeName))
    return pipAttr.getPipe();

  return PIPE::PIPE_UNASSIGNED;
}

namespace {
template <typename CustomOpT> bool customOpRequiresVFMode(CustomOpT op) {
  const auto coreType = op.getCoreType();
  if (coreType && *coreType == TCoreType::CUBE)
    return false;
  auto moduleOp = op.getOperation()->template getParentOfType<mlir::ModuleOp>();
  return hacc::utils::isAscend910_95(moduleOp);
}
} // namespace

bool CustomOp::requiresVFMode() { return customOpRequiresVFMode(*this); }

bool CustomMacroOp::requiresVFMode() { return customOpRequiresVFMode(*this); }

const DenseMap<StringRef, CustomMacroOp::BuiltinInfo>
    CustomMacroOp::kBuiltins{};

//===----------------------------------------------------------------------===//
// CustomMacroOp - sync slots and library call operands
//===----------------------------------------------------------------------===//

int CustomMacroOp::getNumSyncRelatedArgs() {
  return static_cast<int>(getSyncEventSlots().size());
}

SmallVector<SyncEventSlotAttr> CustomMacroOp::getUserSyncEventSlots() const {
  SmallVector<SyncEventSlotAttr> slots;
  auto slotsAttr =
      static_cast<Operation *>(*this)->template getAttrOfType<ArrayAttr>(
          kSyncEventSlotsName);
  if (!slotsAttr)
    return slots;
  for (auto attr : slotsAttr) {
    if (auto slotAttr = llvm::dyn_cast<SyncEventSlotAttr>(attr))
      slots.push_back(slotAttr);
  }
  return slots;
}

SmallVector<SyncEventSlotAttr> CustomMacroOp::getSyncEventSlots() {
  return getUserSyncEventSlots();
}

std::optional<int64_t>
CustomMacroOp::getUserPinnedEventId(PIPE setPipe, PIPE waitPipe) const {
  for (auto slot : getUserSyncEventSlots()) {
    if (!slot.getSetPipe())
      continue;
    if (slot.getSetPipe().getPipe() == setPipe &&
        slot.getWaitPipe().getPipe() == waitPipe && slot.getEvent())
      return static_cast<int64_t>(slot.getEvent().getEvent());
  }
  return std::nullopt;
}

void CustomMacroOp::ensureSyncRelatedArgsFilled(PatternRewriter &rewriter) {
  if (!getSyncRelatedArgs().empty())
    return;

  auto negOneDefaultValue = rewriter.create<arith::ConstantOp>(
      getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(-1));
  getSyncRelatedArgsMutable().assign(ValueRange(
      SmallVector<Value>(getNumSyncRelatedArgs(), negOneDefaultValue)));
}

SmallVector<Value>
CustomMacroOp::getLibraryCallOperands(PatternRewriter &rewriter) {
  SmallVector<Value> libParams;
  libParams.append(getInputs().begin(), getInputs().end());
  libParams.append(getOutputs().begin(), getOutputs().end());
  libParams.append(getTempBuffers().begin(), getTempBuffers().end());

  ensureSyncRelatedArgsFilled(rewriter);
  auto syncRelatedArgs = getSyncRelatedArgs();
  libParams.append(syncRelatedArgs.begin(), syncRelatedArgs.end());
  return libParams;
}

//===----------------------------------------------------------------------===//
// CustomOp / CustomMacroOp - parse / print
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseForCustomOps(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }

  SMLoc attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  { // Parse name
    std::string name{};
    if (parser.parseString(&name))
      return failure();

    result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  }

  { // Parse variadic args
    SmallVector<int32_t, 3> variadicArgsSizes;
    auto parseVariadicArgs = [&parser, &result,
                              &variadicArgsSizes](const std::string &nameHint) {
      SMLoc loc;
      SmallVector<Type, 1> types;
      SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

      if (succeeded(parser.parseOptionalKeyword(nameHint))) {
        loc = parser.getCurrentLocation();
        if (parser.parseLParen() || parser.parseOperandList(operands) ||
            parser.parseColonTypeList(types) || parser.parseRParen())
          return failure();
      }

      if (parser.resolveOperands(operands, types, loc, result.operands)) {
        return failure();
      }

      variadicArgsSizes.push_back(static_cast<int32_t>(operands.size()));
      return success();
    };

    if (failed(parseVariadicArgs("ins")) || failed(parseVariadicArgs("outs")) ||
        failed(parseVariadicArgs("tmps"))) {
      return failure();
    }

    const auto operandSegmentSizesAttr =
        parser.getBuilder().getDenseI32ArrayAttr(variadicArgsSizes);
    if (result.propertiesAttr) {
      NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
      attrs.append("operandSegmentSizes", operandSegmentSizesAttr);
      result.propertiesAttr = attrs.getDictionary(parser.getContext());
    } else {
      result.addAttribute("operandSegmentSizes", operandSegmentSizesAttr);
      std::optional<RegisteredOperationName> info =
          result.name.getRegisteredInfo();
      if (info) {
        if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
              return parser.emitError(attrsLoc)
                     << "'" << result.name.getStringRef() << "' op ";
            })))
          return failure();
      }
    }
  }

  { // Parse result types
    SmallVector<Type, 1> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
      return failure();
    }
    result.addTypes(resultTypes);
  }

  return success();
}

ParseResult parseForCustomMacroOps(OpAsmParser &parser,
                                   OperationState &result) {
  if (failed(parseForCustomOps(parser, result)))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> syncOperands;
  SmallVector<Type, 1> syncTypes;
  if (succeeded(parser.parseOptionalKeyword("sync_related_args"))) {
    if (parser.parseLParen() || parser.parseOperandList(syncOperands) ||
        parser.parseColonTypeList(syncTypes) || parser.parseRParen())
      return failure();
    if (parser.resolveOperands(syncOperands, syncTypes,
                               parser.getCurrentLocation(), result.operands))
      return failure();
  }

  auto segmentSizesAttr = result.attributes.get("operandSegmentSizes");
  if (!segmentSizesAttr)
    return failure();
  SmallVector<int32_t> sizes(
      cast<DenseI32ArrayAttr>(segmentSizesAttr).asArrayRef());
  if (sizes.size() != 3)
    return failure();
  sizes.push_back(static_cast<int32_t>(syncOperands.size()));
  result.attributes.set("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(sizes));

  return success();
}

template <typename CustomOpT>
void printForCustomOps(CustomOpT op, OpAsmPrinter &p) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "name"});

  p << " ";
  p.printString(op.getName());

  auto printVariadicArgs = [&p](const auto &args, const std::string &nameHint) {
    if (!args.empty())
      p << " " << nameHint << "(" << args << " : " << args.getTypes() << ")";
  };

  printVariadicArgs(op.getInputs(), "ins");
  printVariadicArgs(op.getOutputs(), "outs");
  printVariadicArgs(op.getTempBuffers(), "tmps");

  if (!op.getResults().empty())
    p.printOptionalArrowTypeList(op.getResultTypes());
}

template <typename CustomOpT> LogicalResult verifyBuiltins(CustomOpT op) {
  const auto &builtinInfo = CustomOpT::kBuiltins.at(op.getName());

  const auto &coreType = op.getCoreType();
  if (coreType && *coreType != builtinInfo.coreType)
    return op.emitOpError() << "Specified core type conflict with "
                            << op.getName() << "'s core type.";

  const auto &vfMode = op.getVFMode();
  if (vfMode && *vfMode != builtinInfo.vfMode)
    return op.emitOpError() << "Specified vf mode conflict with "
                            << op.getName() << "'s vf mode.";

  if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
    const auto &pipe = op.getPipe();
    if (pipe != PIPE::PIPE_UNASSIGNED && pipe != builtinInfo.pipe)
      return op.emitOpError()
             << "Specified pipe conflict with " << op.getName() << "'s pipe.";
  } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
    const auto &inPipe = op.getInPipe();
    if (inPipe != PIPE::PIPE_UNASSIGNED && inPipe != builtinInfo.inPipe)
      return op.emitOpError() << "Specified inPipe conflict with "
                              << op.getName() << "'s inPipe.";

    const auto &outPipe = op.getOutPipe();
    if (outPipe != PIPE::PIPE_UNASSIGNED && outPipe != builtinInfo.outPipe)
      return op.emitOpError() << "Specified outPipe conflict with "
                              << op.getName() << "'s outPipe.";
  }

  return success();
}

template <typename CustomOpT> LogicalResult verifyCustomOp(CustomOpT op) {
  if (op.isBuiltin())
    return verifyBuiltins(op);

  const auto coreType = op.getCoreType();
  if (!coreType)
    return op.emitOpError() << "Missing core type information";

  if (op.requiresVFMode() && !op.getVFMode())
    return op.emitOpError() << "Missing vf mode information";

  if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
    if (op.getPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing pipe information";
  } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
    if (op.getInPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing input pipe information";

    if (op.getOutPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing output pipe information";
  }

  const auto typeArrayAttr =
      op.getOperation()->template getAttrOfType<ArrayAttr>(
          op.kExtraBuffersTypesName);
  const auto sizesArrayAttr =
      op.getOperation()->template getAttrOfType<ArrayAttr>(
          op.kExtraBuffersSizesName);
  if (typeArrayAttr && sizesArrayAttr) {
    if (typeArrayAttr.size() != sizesArrayAttr.size())
      return op.emitOpError() << "Extra buffers' types and sizes mismatch";
  } else if (!typeArrayAttr && !sizesArrayAttr) {
    // No extra buffers attributes specified, ok
  } else {
    return op.emitOpError() << "Either extra buffers' types or sizes missing";
  }

  if (op.getSymbol().empty())
    return op.emitOpError() << "Missing implementation function name";

  return success();
}

template <typename CustomOpT> LogicalResult verifySyncEventSlots(CustomOpT op) {
  return success();
}

template <>
LogicalResult verifySyncEventSlots<CustomMacroOp>(CustomMacroOp op) {
  auto userSlots = op.getUserSyncEventSlots();
  auto syncRelatedArgs = op.getSyncRelatedArgs();
  const int numSlots = op.getNumSyncRelatedArgs();

  if (!syncRelatedArgs.empty() &&
      syncRelatedArgs.size() != static_cast<size_t>(numSlots)) {
    if (numSlots == 0)
      return op.emitOpError() << "sync_related_args should be empty";
    return op.emitOpError()
           << "sync_related_args should be of size " << numSlots;
  }

  llvm::DenseSet<std::tuple<hivm::PIPE, hivm::PIPE, int64_t>> reservedEvents;
  for (auto slot : userSlots) {
    switch (slot.getMacroSync()) {
    case hivm::SyncEventSlotMacroSync::internal:
      if (slot.getEvent()) {
        auto key = std::make_tuple(
            hivm::PIPE::PIPE_UNASSIGNED, hivm::PIPE::PIPE_UNASSIGNED,
            static_cast<int64_t>(slot.getEvent().getEvent()));
        if (!reservedEvents.insert(key).second)
          return op.emitOpError()
                 << "duplicate user-specified sync event id on internal slot";
      }
      break;
    case hivm::SyncEventSlotMacroSync::wait:
    case hivm::SyncEventSlotMacroSync::set: {
      if (!slot.getSetPipe() || !slot.getWaitPipe())
        return op.emitOpError()
               << "sync_event_slot requires set_pipe and wait_pipe for `wait` "
                  "and `set` macro_sync";

      auto setPipe = slot.getSetPipe().getPipe();
      auto waitPipe = slot.getWaitPipe().getPipe();

      if (!slot.getEvent())
        break;
      auto key = std::make_tuple(
          setPipe, waitPipe, static_cast<int64_t>(slot.getEvent().getEvent()));
      if (!reservedEvents.insert(key).second)
        return op.emitOpError() << "duplicate user-specified sync event id on "
                                   "the same pipe pair";
      break;
    }
    }
  }

  return success();
}

template <typename CustomOpT>
SmallVector<std::pair<Type, int64_t>>
getCustomOpExtraBuffersInfo(CustomOpT op) {
  SmallVector<std::pair<Type, int64_t>> res;
  const auto typeArrayAttr =
      static_cast<Operation *>(op)->template getAttrOfType<ArrayAttr>(
          CustomOpT::kExtraBuffersTypesName);
  const auto sizesArrayAttr =
      static_cast<Operation *>(op)->template getAttrOfType<ArrayAttr>(
          CustomOpT::kExtraBuffersSizesName);
  if (!typeArrayAttr) {
    LLVM_DEBUG(assert(!sizesArrayAttr && "custom op verification failed?"));
    return res;
  }

  LLVM_DEBUG(assert(typeArrayAttr.size() == sizesArrayAttr.size() &&
                    "custom op verification failed?"));

  for (auto [typeAttr, sizeAttr] : llvm::zip(typeArrayAttr, sizesArrayAttr)) {
    const mlir::Type type = cast<mlir::TypeAttr>(typeAttr).getValue();
    const int64_t size = cast<mlir::IntegerAttr>(sizeAttr).getInt();
    res.push_back(std::make_pair(type, size));
  }

  return res;
}

template <typename CustomOpT>
void getEffectsForCustomOps(
    CustomOpT op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!op.getNoSideEffect()) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Read::get(),
                         SideEffects::DefaultResource::get());
  }
}
} // namespace

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomOps(parser, result);
}

ParseResult CustomMacroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomMacroOps(parser, result);
}

void CustomOp::print(OpAsmPrinter &p) { printForCustomOps(*this, p); }

void CustomMacroOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "name"});
  p << " ";
  p.printString(getName());

  auto printVariadicArgs = [&p](const auto &args, const std::string &nameHint) {
    if (!args.empty())
      p << " " << nameHint << "(" << args << " : " << args.getTypes() << ")";
  };

  printVariadicArgs(getInputs(), "ins");
  printVariadicArgs(getOutputs(), "outs");
  printVariadicArgs(getTempBuffers(), "tmps");

  if (!getResults().empty())
    p.printOptionalArrowTypeList(getResultTypes());

  auto syncArgs = getSyncRelatedArgs();
  if (!syncArgs.empty())
    p << " sync_related_args(" << syncArgs << " : " << syncArgs.getTypes()
      << ")";
}

LogicalResult CustomOp::verify() { return verifyCustomOp(*this); }

LogicalResult CustomMacroOp::verify() {
  if (failed(verifyCustomOp(*this)))
    return failure();
  return verifySyncEventSlots(*this);
}

SmallVector<std::pair<Type, int64_t>> CustomOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}

SmallVector<std::pair<Type, int64_t>>
CustomMacroOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}

void CustomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsForCustomOps(*this, effects);
}

void CustomMacroOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsForCustomOps(*this, effects);
}
