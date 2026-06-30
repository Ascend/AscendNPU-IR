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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <sstream>

#define DEBUG_TYPE "hivm-custom-op"

using namespace mlir;
using namespace mlir::hivm;

namespace {
template <typename CustomOpT> bool customOpRequiresVFMode(CustomOpT op) {
  const auto coreType = op.getCoreType();
  if (coreType && *coreType == TCoreType::CUBE)
    return false;
  auto moduleOp = (op)->template getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return false;
  return hacc::utils::isRegBasedArch(moduleOp);
}
} // namespace

bool CustomOp::requiresVFMode() const { return customOpRequiresVFMode(*this); }

bool CustomMacroOp::requiresVFMode() const {
  return customOpRequiresVFMode(*this);
}

namespace {
template <typename CustomOpT>
static std::string getGatherLoadLibraryCallName(CustomOpT op) {
  const auto srcType = op->getOperand(0).getType();
  const std::string srcTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(srcType));

  const auto idxType = cast<ShapedType>(op->getOperand(1).getType());
  const auto rank = idxType.getRank();
  const std::string libCallDim = std::to_string(rank) + "d";
  const std::string idxTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(idxType));

  return "gather_out_to_ub_" + libCallDim + "_" + srcTypeStr + "_" + idxTypeStr;
}

template <typename CustomOpT>
std::string getIndexSelectLibraryCallName(CustomOpT op) {
  auto idxType = cast<ShapedType>(op->getOperand(1).getType());
  int idxRank = idxType.getRank();
  const std::string idxDim = std::to_string(idxRank) + "d";

  auto srcRank = op.getExtraAttr().getValue().str();
  const std::string srcDim = srcRank.substr(srcRank.length() - 1, 1) + "d";

  Type srcType = op->getOperand(0).getType();
  const std::string srcTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(srcType));
  const std::string idxTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(idxType));
  return "index_select_" + srcDim + "_" + srcTypeStr + "_" + idxDim + "_" +
         idxTypeStr;
}

template <typename CustomOpT>
static std::string getIndirectAtomicLibraryCallName(CustomOpT op) {
  std::string opName = "unknown";
  bool isBlockScope = false;
  if (auto extraAttr = op.getExtraAttr()) {
    SmallVector<StringRef> entries;
    extraAttr.getValue().split(entries, ',');
    for (StringRef entry : entries) {
      StringRef key;
      StringRef value;
      std::tie(key, value) = entry.split('=');
      key = key.trim();
      value = value.trim();
      if (key == "operate" && !value.empty()) {
        opName = value.str();
      } else if (key == "scope") {
        isBlockScope = value == "cta";
      }
    }
  }

  ValueRange inputs = op.getInputs();
  const bool hasMask = inputs.size() > 3;

  Type valueType = inputs[2].getType();
  const std::string valueTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(valueType));

  Type offsetsType = inputs[1].getType();
  const std::string offsetsTypeStr = hivm::detail::getTypeName(
      op->getLoc(), getElementTypeOrSelf(offsetsType));

  const bool isSoftwareAcceleratedOp =
      opName == "or" || opName == "and" || opName == "xor";
  std::string scopePrefix = "";
  if (isSoftwareAcceleratedOp)
    scopePrefix = isBlockScope ? "block_" : "soft_";

  return "indirect_atomic_" + scopePrefix + opName +
         (hasMask ? "" : "_no_mask") + "_" + valueTypeStr + "_" +
         offsetsTypeStr;
}

template <typename CustomOpT>
std::string getHistogramLibraryCallName(CustomOpT op) {
  ShapedType srcTy = cast<ShapedType>(op.getInputs()[0].getType());
  Type elemType = srcTy.getElementType();

  std::stringstream ss;
  ss << "histogram";
  if (srcTy.getRank() == 1) {
    ss << "_1d";
  }
  if (op.getInputs().size() > 2) {
    ss << "_masked";
  }
  ss << "_" << hivm::detail::getTypeName(op->getLoc(), elemType);
  return ss.str();
}

template <typename CustomOpT>
static std::string getCustomOpsLibraryCallName(CustomOpT op) {
  if (op.isBuiltin())
    return CustomOpT::kBuiltins.at(op.getName()).getOpLibraryCallName(op);

  return op.getSymbol().value();
}

ParseResult parseForCustomOps(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }

  SMLoc attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  {
    std::string name{};
    if (parser.parseString(&name))
      return failure();
    result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  }

  {
    SmallVector<int32_t, 3> variadicArgsSizes;
    SmallVector<Attribute> inputAttrs;
    bool hasInlineInputAttrs = false;
    auto parseVariadicArgs =
        [&parser, &result, &variadicArgsSizes, &inputAttrs,
         &hasInlineInputAttrs](const std::string &nameHint) {
          SMLoc loc;
          SmallVector<Type, 1> types;
          SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

          if (succeeded(parser.parseOptionalKeyword(nameHint))) {
            loc = parser.getCurrentLocation();
            if (parser.parseLParen())
              return failure();

            do {
              if (succeeded(parser.parseOptionalRParen()))
                break;

              if (parser.parseOperand(operands.emplace_back()))
                return failure();
            } while (succeeded(parser.parseOptionalComma()));

            if (parser.parseColon())
              return failure();

            for (size_t i = 0, e = operands.size(); i < e; ++i) {
              if (i != 0 && parser.parseComma())
                return failure();

              if (parser.parseType(types.emplace_back()))
                return failure();

              NamedAttrList operandAttrs;
              if (parser.parseOptionalAttrDict(operandAttrs))
                return failure();
              if (nameHint == "ins" && !operandAttrs.empty()) {
                hasInlineInputAttrs = true;
                inputAttrs.push_back(
                    operandAttrs.getDictionary(parser.getContext()));
              } else if (nameHint == "ins") {
                inputAttrs.push_back(DictionaryAttr::get(parser.getContext()));
              }
            }

            if (parser.parseRParen())
              return failure();
          }

          if (parser.resolveOperands(operands, types, loc, result.operands))
            return failure();

          variadicArgsSizes.push_back(static_cast<int32_t>(operands.size()));
          return success();
        };

    if (failed(parseVariadicArgs("ins")) || failed(parseVariadicArgs("outs")) ||
        failed(parseVariadicArgs("tmps"))) {
      return failure();
    }

    auto existingArgAttrs = dyn_cast_or_null<ArrayAttr>(
        result.attributes.get(CustomOp::kArgAttrsName));
    if (hasInlineInputAttrs || existingArgAttrs) {
      auto newArgAttrs = hasInlineInputAttrs
                             ? parser.getBuilder().getArrayAttr(inputAttrs)
                             : existingArgAttrs;
      if (hasInlineInputAttrs && existingArgAttrs) {
        SmallVector<Attribute> merged(inputAttrs);
        const size_t copySize =
            std::min(existingArgAttrs.size(), merged.size());
        for (size_t i = 0; i < copySize; ++i) {
          auto existingDict = dyn_cast<DictionaryAttr>(existingArgAttrs[i]);
          auto parsedDict = dyn_cast<DictionaryAttr>(merged[i]);
          if (!existingDict || !parsedDict || existingDict.empty())
            continue;

          NamedAttrList attrs(existingDict);
          for (NamedAttribute attr : parsedDict)
            attrs.set(attr.getName(), attr.getValue());
          merged[i] = attrs.getDictionary(parser.getContext());
        }
        newArgAttrs = parser.getBuilder().getArrayAttr(merged);
      }
      result.attributes.set(CustomOp::kArgAttrsName, newArgAttrs);
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

  {
    SmallVector<Type, 1> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes))
      return failure();
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
  ArrayAttr argAttrs =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kArgAttrsName);
  bool printInputAttrsInline = false;
  if (argAttrs) {
    for (Attribute attr : argAttrs) {
      auto dict = dyn_cast<DictionaryAttr>(attr);
      if (dict && dict.contains(CustomOpT::kInplaceOutsAttrName)) {
        printInputAttrsInline = true;
        break;
      }
    }
  }

  SmallVector<StringRef> elidedAttrs = {"operandSegmentSizes", "name"};
  if (printInputAttrsInline)
    elidedAttrs.push_back(CustomOpT::kArgAttrsName);
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  p << " ";
  p.printString(op.getName());

  auto printVariadicArgs = [&](const auto &args, const std::string &nameHint) {
    if (args.empty())
      return;

    p << " " << nameHint << "(" << args << " : ";
    llvm::interleaveComma(
        llvm::enumerate(args.getTypes()), p, [&](auto indexedType) {
          p << indexedType.value();
          if (!printInputAttrsInline || nameHint != "ins" || !argAttrs ||
              indexedType.index() >= argAttrs.size())
            return;

          auto dict = dyn_cast<DictionaryAttr>(argAttrs[indexedType.index()]);
          if (!dict || dict.empty())
            return;

          p << " ";
          p.printAttributeWithoutType(dict);
        });
    p << ")";
  };

  printVariadicArgs(op.getInputs(), "ins");
  printVariadicArgs(op.getOutputs(), "outs");
  printVariadicArgs(op.getTempBuffers(), "tmps");

  if (!op.getResults().empty())
    p.printOptionalArrowTypeList(op.getResultTypes());
}

template <typename CustomOpT>
static LogicalResult verifyCustomOpExtraBufferAttrs(CustomOpT op) {
  const auto typesAttr =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kExtraBuffersTypesName);
  const auto sizesAttr =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kExtraBuffersSizesName);
  if (!typesAttr && !sizesAttr)
    return success();
  if (!typesAttr || !sizesAttr)
    return op.emitOpError() << "Either extra buffers' types or sizes missing";
  if (typesAttr.size() != sizesAttr.size())
    return op.emitOpError() << "Extra buffers' types and sizes mismatch";

  for (Attribute typeAttr : typesAttr) {
    if (!isa<TypeAttr>(typeAttr))
      return op.emitOpError()
             << "Extra buffers' types must be an array of type attributes";
  }
  for (Attribute sizeAttr : sizesAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(sizeAttr);
    if (!intAttr)
      return op.emitOpError()
             << "Extra buffers' sizes must be an array of integer attributes";
    if (intAttr.getInt() < 0)
      return op.emitOpError() << "Extra buffer size must be non-negative";
  }

  return success();
}

template <typename CustomOpT>
static LogicalResult verifyCustomOpInplaceOperandsAttr(CustomOpT op) {
  auto argAttrs =
      op->template getAttrOfType<ArrayAttr>(CustomOpT::kArgAttrsName);
  if (!argAttrs)
    return success();

  llvm::DenseSet<int32_t> usedOutputs;
  const int64_t numInputs = op.getInputs().size();
  const int64_t numOutputs = op.getOutputs().size();
  for (auto [inputIdx, inputAttr] : llvm::enumerate(argAttrs)) {
    if (static_cast<int64_t>(inputIdx) >= numInputs)
      break;

    auto dict = dyn_cast<DictionaryAttr>(inputAttr);
    if (!dict)
      continue;

    auto inplaceOutAttr = dyn_cast_or_null<IntegerAttr>(
        dict.get(CustomOpT::kInplaceOutsAttrName));
    if (!inplaceOutAttr) {
      if (dict.contains(CustomOpT::kInplaceOutsAttrName))
        return op.emitOpError() << CustomOpT::kInplaceOutsAttrName
                                << " must be an integer attribute";
      continue;
    }

    int32_t outputIdx = inplaceOutAttr.getInt();
    if (outputIdx < 0 || outputIdx >= numOutputs) {
      return op.emitOpError() << CustomOpT::kInplaceOutsAttrName
                              << " must be a valid outs operand index";
    }
    if (!usedOutputs.insert(outputIdx).second) {
      return op.emitOpError()
             << CustomOpT::kInplaceOutsAttrName
             << " cannot map multiple ins operands to the same outs operand";
    }

    Type inputType = op.getInputs()[inputIdx].getType();
    Type outputType = op.getOutputs()[outputIdx].getType();
    if (!isa<RankedTensorType, MemRefType>(inputType) ||
        !isa<RankedTensorType, MemRefType>(outputType)) {
      return op.emitOpError()
             << CustomOpT::kInplaceOutsAttrName
             << " requires mapped ins and outs operands to be ranked tensors "
                "or memrefs";
    }
    if (inputType != outputType) {
      return op.emitOpError()
             << CustomOpT::kInplaceOutsAttrName
             << " requires mapped ins and outs operands to have identical "
                "ranked tensor or memref types";
    }
    if (op.getInputs()[inputIdx] != op.getOutputs()[outputIdx]) {
      return op.emitOpError()
             << CustomOpT::kInplaceOutsAttrName
             << " requires each mapped ins operand to be the same SSA value "
                "as its mapped outs operand";
    }
  }

  return success();
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
  if (failed(verifyCustomOpInplaceOperandsAttr(op)))
    return failure();

  if (op.isBuiltin())
    return verifyBuiltins(op);

  if (failed(verifyCustomOpExtraBufferAttrs(op)))
    return failure();

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

  if (!op.getSymbol().has_value() || op.getSymbol()->empty())
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
  const auto typesAttr = op.getOperation()->template getAttrOfType<ArrayAttr>(
      CustomOpT::kExtraBuffersTypesName);
  const auto sizesAttr = op.getOperation()->template getAttrOfType<ArrayAttr>(
      CustomOpT::kExtraBuffersSizesName);
  if (!typesAttr)
    return res;

  for (auto [typeAttr, sizeAttr] : llvm::zip(typesAttr, sizesAttr)) {
    const Type type = cast<TypeAttr>(typeAttr).getValue();
    const int64_t size = cast<IntegerAttr>(sizeAttr).getInt();
    res.push_back(std::make_pair(type, size));
  }
  return res;
}
} // namespace

const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{
    {kBuiltinGatherLoadName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getGatherLoadLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinIndexSelectName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getIndexSelectLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinIndirectAtomicName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getIndirectAtomicLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinHistogramName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getHistogramLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {})}};

const DenseMap<StringRef, CustomMacroOp::BuiltinInfo> CustomMacroOp::kBuiltins{
    {kBuiltinGatherLoadName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_MTE2, PIPE::PIPE_V, VFMode::SIMT,
                 getGatherLoadLibraryCallName<CustomMacroOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinIndexSelectName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_MTE2, PIPE::PIPE_V, VFMode::SIMT,
                 getIndexSelectLibraryCallName<CustomMacroOp>,
                 /* GM Addr Args Indices */ {0})}};

int CustomMacroOp::getNumSyncRelatedArgs() const {
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

SmallVector<SyncEventSlotAttr> CustomMacroOp::getSyncEventSlots() const {
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

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomOps(parser, result);
}

ParseResult CustomMacroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomMacroOps(parser, result);
}

void CustomOp::print(OpAsmPrinter &p) { printForCustomOps(*this, p); }

void CustomMacroOp::print(OpAsmPrinter &p) {
  printForCustomOps(*this, p);

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

std::string CustomOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getCustomOpsLibraryCallName<CustomOp>(*this);
}

std::string
CustomMacroOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getCustomOpsLibraryCallName<CustomMacroOp>(*this);
}

SmallVector<std::pair<Type, int64_t>> CustomOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}

SmallVector<std::pair<Type, int64_t>>
CustomMacroOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}
