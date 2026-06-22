//===- HIVMOps.cpp - HIVM ops implementation ------------------------------===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// For function inliner support
#include "mlir/Transforms/InliningUtils.h"

#include <numeric>

#define DEBUG_TYPE "hivm-ops"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

int64_t AddressSpaceAttr::getMappingId() const {
  return static_cast<int64_t>(getAddressSpace());
}

bool AddressSpaceAttr::isLinearMapping() const {
  llvm::report_fatal_error("AddressSpaceAttr does not support linear mapping");
}

int64_t AddressSpaceAttr::getRelativeIndex() const {
  llvm::report_fatal_error("AddressSpaceAttr does not support relative index");
}

//===----------------------------------------------------------------------===//
// DataLayoutAttr
//===----------------------------------------------------------------------===//

LogicalResult
DataLayoutAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                       hivm::DataLayout data_layout, BoolAttr transpose,
                       DenseI64ArrayAttr fractalSizes) {
  // ND is transpose agnostic
  if (data_layout == hivm::DataLayout::ND)
    return success();

  // Transpose option should and must be set for DOTA_ND and DOTB_ND layout.
  if (data_layout == hivm::DataLayout::DOTA_ND ||
      data_layout == hivm::DataLayout::DOTB_ND) {
    if (transpose == nullptr)
      return emitError() << "'transpose' must be set if data layout is "
                            "DOTA_ND or DOTB_ND";
    return success();
  }

  if (transpose != nullptr)
    return emitError() << "'transpose' is only valid if data layout is "
                          "DOTA_ND or DOTB_ND or ND like";
  return success();
}

//===----------------------------------------------------------------------===//
// HIVM Device Mapping Attributes
//===----------------------------------------------------------------------===//

int64_t HIVMBlockMappingAttr::getMappingId() const {
  // Currently only has a single mapping id
  return static_cast<int64_t>(MappingId::DimX);
}

bool HIVMBlockMappingAttr::isLinearMapping() const {
  // Since there's only one mapping id, the mapping is linear.
  return true;
}

int64_t HIVMBlockMappingAttr::getRelativeIndex() const {
  return getOrder().value_or(0);
}

//===----------------------------------------------------------------------===//
// HIVM Device Sub Block Mapping Attributes
//===----------------------------------------------------------------------===//

int64_t HIVMSubBlockMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getSubBlock());
}

bool HIVMSubBlockMappingAttr::isLinearMapping() const {
  llvm::report_fatal_error("HIVMSubBlockMappingAttr does not support linear mapping");
}

int64_t HIVMSubBlockMappingAttr::getRelativeIndex() const {
  llvm::report_fatal_error("HIVMSubBlockMappingAttr does not support relative index");
}

void hivm::populateHIVMAddressSpaceAttributeConversions(
    TypeConverter &typeConverter) {
  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType type, hivm::AddressSpaceAttr addressSpaceAttr) {
        return IntegerAttr::get(
            IntegerType::get(addressSpaceAttr.getContext(), 64),
            addressSpaceAttr.getMappingId());
      });
}

AddressSpaceAttr mlir::hivm::getHIVMAddressSpaceAttr(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  assert(memRefType && "input type must be a memref type");
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  assert(scopeAttr && "memory scope should be a hivm address scope");
  return scopeAttr;
}

hivm::AddressSpace mlir::hivm::getHIVMAddressSpace(Type type) {
  auto scopeAttr = getHIVMAddressSpaceAttr(type);
  return scopeAttr.getAddressSpace();
}

std::optional<AddressSpace> mlir::hivm::getOptionalHIVMAddressSpace(Type type) {
  auto memRefType = dyn_cast_if_present<BaseMemRefType>(type);
  if (!memRefType)
    return std::nullopt;

  if (!memRefType.getMemorySpace())
    return std::nullopt;

  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr)
    return std::nullopt;

  return scopeAttr.getAddressSpace();
}

//===----------------------------------------------------------------------===//
// PointerCastOp
//===----------------------------------------------------------------------===//

void PointerCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type result, Value addr) {
  build(odsBuilder, odsState, result, ValueRange({addr}), {});
}

void PointerCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type result, ValueRange addrs) {
  build(odsBuilder, odsState, result, addrs, {});
}

void PointerCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type result, Value addr, ValueRange dynamicSizes) {
  build(odsBuilder, odsState, result, ValueRange({addr}), dynamicSizes);
}

void PointerCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypeRange resultTypes, Value addr,
                          ValueRange dynamicSizes) {
  build(odsBuilder, odsState, resultTypes, ValueRange({addr}), dynamicSizes);
}

TypedValue<IntegerType> PointerCastOp::getSingleAddr() {
  return cast<TypedValue<IntegerType>>(getAddrs()[0]);
}

LogicalResult PointerCastOp::verify() {
  auto addrs = getAddrs();
  if (addrs.empty()) {
    return emitOpError("addrs of PointerCastOp should not be empty!");
  }
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  const int64_t expectedDynamicDims =
      static_cast<int64_t>(getResult().getType().getNumDynamicDims());
  if (static_cast<int64_t>(getDynamicSizes().size()) != expectedDynamicDims) {
    return emitOpError() << "expected " << expectedDynamicDims
                         << " dynamic size operands, but found "
                         << getDynamicSizes().size();
  }
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
  return success();
}

//===----------------------------------------------------------------------===//
// LoadScalarOp
//===----------------------------------------------------------------------===//

LogicalResult LoadScalarOp::verify() {
  auto ptrTy = cast<LLVM::LLVMPointerType>(getAddr().getType());
  if (ptrTy.getAddressSpace() != 1)
    return emitOpError("expecting GM address");
  return success();
}

//===----------------------------------------------------------------------===//
// Printer and Parser for HIVM Ops that follows Destination Style Op
// Interface
//===----------------------------------------------------------------------===//

static ParseResult handleOperandSegmentSizes(
    OpAsmParser &parser, OperationState &result,
    const SmallVector<OpAsmParser::UnresolvedOperand, 4> &inputsOperands,
    const SmallVector<OpAsmParser::UnresolvedOperand, 4> &outputsOperands) {
  // This is a bit complex because we're trying to be backward compatible with
  // operation syntax that mix the inherent attributes and the discardable
  // ones in the same dictionary. If the properties are used, we append the
  // operandSegmentSizes there directly. Otherwise we append it to the
  // discardable attributes dictionary where it is handled by the generic
  // Operation::create(...) method.
  if (result.propertiesAttr) {
    NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
    attrs.append("operandSegmentSizes",
                 parser.getBuilder().getDenseI32ArrayAttr(
                     {static_cast<int32_t>(inputsOperands.size()),
                      static_cast<int32_t>(outputsOperands.size())}));
    result.propertiesAttr = attrs.getDictionary(parser.getContext());
  } else {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputsOperands.size()),
                             static_cast<int32_t>(outputsOperands.size())}));
  }
  if (!result.propertiesAttr) {
    SMLoc attrsLoc = parser.getCurrentLocation();
    std::optional<RegisteredOperationName> info =
        result.name.getRegisteredInfo();
    if (info) {
      if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
            return parser.emitError(attrsLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          }))) {
        return failure();
      }
    }
  }
  return success();
}

static ParseResult parseDPSInputOutputs(OpAsmParser &parser,
                                        OperationState &result,
                                        SmallVectorImpl<Type> &inputTypes,
                                        SmallVectorImpl<Type> &outputTypes,
                                        bool addOperandSegmentSizes = true) {
  SMLoc inputsOperandsLoc;
  SMLoc outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater()) {
      return failure();
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen()) {
      return failure();
    }

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen()) {
      return failure();
    }
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen()) {
      return failure();
    }
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands)) {
    return failure();
  }
  if (addOperandSegmentSizes) {
    return handleOperandSegmentSizes(parser, result, inputsOperands,
                                     outputsOperands);
  }
  return success();
}

static ParseResult parseDPSResults(OpAsmParser &parser,
                                   SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes)) {
    return failure();
  }
  return success();
}

ParseResult hivm::detail::parseHIVMStructuredDPSOp(OpAsmParser &parser,
                                                   OperationState &result) {
  SmallVector<Type, 1> inputTypes;
  SmallVector<Type, 1> outputTypes;
  if (parseDPSInputOutputs(parser, result, inputTypes, outputTypes)) {
    return failure();
  }
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseDPSResults(parser, outputTensorsTypes)) {
    return failure();
  }
  result.addTypes(outputTensorsTypes);
  return success();
}

static void printDPSInputOutputs(OpAsmPrinter &p, ValueRange inputs,
                                 ValueRange outputs) {
  if (!inputs.empty()) {
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  }
  if (!outputs.empty()) {
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
  }
}

static void printDPSResults(OpAsmPrinter &p, TypeRange resultTypes) {
  if (resultTypes.empty()) {
    return;
  }
  p.printOptionalArrowTypeList(resultTypes);
}

void hivm::detail::printHIVMStructuredDPSOp(OpAsmPrinter &p, Operation *op,
                                            ValueRange inputs,
                                            ValueRange outputs) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  printDPSInputOutputs(p, inputs, outputs);
  printDPSResults(p, op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConvertLayoutOp
//===----------------------------------------------------------------------===//

void ConvertLayoutOp::build(OpBuilder &builder, OperationState &result,
                            Type resultType, Value source,
                            DataLayoutAttr srcLayout, DataLayoutAttr dstLayout,
                            ArrayRef<OpFoldResult> outputShape) {
  SmallVector<Value> dynamicDims;
  SmallVector<int64_t> staticDims;
  dispatchIndexOpFoldResults(outputShape, dynamicDims, staticDims);
  build(builder, result, resultType, source, srcLayout, dstLayout, staticDims,
        dynamicDims);
}

void ConvertLayoutOp::build(OpBuilder &builder, OperationState &result,
                            Type resultType, Value source,
                            DataLayoutAttr srcLayout,
                            DataLayoutAttr dstLayout) {
  auto staticDims = cast<ShapedType>(resultType).getShape();
  build(builder, result, resultType, source, srcLayout, dstLayout, staticDims,
        {});
}

LogicalResult ConvertLayoutOp::verify() {
  // Verify that the number of dynamic dims matches the number of kDynamic
  // entries in static_output_shape
  auto elementType = getElementTypeOrSelf(getResult());

  // Verify operand's element type matches first result's element type.
  for (auto operand : getOperands()) {
    if (!isa<ShapedType>(operand.getType()))
      continue;
    if (getElementTypeOrSelf(operand) != elementType)
      return emitOpError(
          "requires the same element type for all operands and results");
  }
  const size_t numDynamic = static_cast<size_t>(
      llvm::count_if(getStaticOutputShape(),
                     [](int64_t dim) { return ShapedType::isDynamic(dim); }));
  if (numDynamic != getOutputShape().size()) {
    return emitOpError("expected ")
           << numDynamic << " dynamic dimensions but got "
           << getOutputShape().size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

// CustomOp Methods
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

const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{};

// CustomMacroOp Methods
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

// Common CustomOp Methods
namespace {
ParseResult parseForCustomOps(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }

  // Parse attributes
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

    // Update operandSegmentSizes attribute
    const auto operandSegmentSizesAttr =
        parser.getBuilder().getDenseI32ArrayAttr(variadicArgsSizes);
    // This is a bit complex because we're trying to be backward compatible with
    // operation syntax that mix the inherent attributes and the discardable
    // ones in the same dictionary. If the properties are used, we append the
    // operandSegmentSizes there directly. Otherwise we append it to the
    // discardable attributes dictionary where it is handled by the generic
    // Operation::create(...) method.
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
} // namespace

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomOps(parser, result);
}

namespace {
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
} // namespace

ParseResult CustomMacroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomMacroOps(parser, result);
}

namespace {
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
} // namespace

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

namespace {
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
  // Check builtins
  if (op.isBuiltin())
    return verifyBuiltins(op);

  // Check core type attribute
  const auto coreType = op.getCoreType();
  if (!coreType)
    return op.emitOpError() << "Missing core type information";

  // Check VF mode attribute
  if (op.requiresVFMode() && !op.getVFMode())
    return op.emitOpError() << "Missing vf mode information";

  if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
    // Check pipe attribute
    if (op.getPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing pipe information";
  } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
    // Check input/output pip attributes
    if (op.getInPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing input pipe information";

    if (op.getOutPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing output pipe information";
  }

  // Check extra buffers attributes
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
} // namespace

LogicalResult CustomOp::verify() { return verifyCustomOp(*this); }

LogicalResult CustomMacroOp::verify() {
  if (failed(verifyCustomOp(*this)))
    return failure();
  return verifySyncEventSlots(*this);
}

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

LogicalResult GatherLoadOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    GatherLoadOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto dstType = dyn_cast<RankedTensorType>(adaptor.getDst().getType());
  if (dstType)
    inferredReturnTypes.push_back(dstType);
  return success();
}

LogicalResult GatherLoadOp::verify() {
  auto indicesType = getIndices().getType();
  if (auto mask = getMask()) {
    if (mask.getType().getShape() != indicesType.getShape()) {
      return emitOpError("mask of hivm::GatherLoadOp must have the same "
                         "shape and rank as indices");
    }
  }
  if (auto other = getOther()) {
    auto otherType = cast<RankedTensorType>(other.getType());
    if (otherType.getShape() != indicesType.getShape()) {
      return emitOpError("other of hivm::GatherLoadOp must have the same "
                         "shape and rank as indices");
    }
    auto otherElementType = otherType.getElementType();
    if (otherElementType != getElementTypeOrSelf(getBase())) {
      return emitOpError("other of hivm::GatherLoadOp must have the same "
                         "element type as base");
    }
  }
  return success();
}

void GatherLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ScatterStoreOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterStoreOp::verify() {
  auto dataType = getData().getType();
  if (auto mask = getMask()) {
    if (mask.getType().getShape() != dataType.getShape()) {
      return emitOpError("mask of hivm::ScatterStoreOp must have the same "
                         "shape and rank as data");
    }
  }
  return success();
}

void ScatterStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IndirectStoreOp
//===----------------------------------------------------------------------===//

LogicalResult IndirectStoreOp::verify() {
  auto dstType = getDst().getType();
  auto dstMemrefType = dyn_cast<MemRefType>(dstType);
  if (!dstMemrefType) {
    return emitError("dst must be a memref type");
  }

  auto offsetsType = getOffsets().getType();
  auto offsetsTensorType = dyn_cast<TensorType>(offsetsType);
  auto offsetsMemrefType = dyn_cast<MemRefType>(offsetsType);
  if (!(offsetsTensorType || offsetsMemrefType)) {
    return emitOpError("offset must be tensor or memref type");
  }

  auto srcType = getSrc().getType();
  auto srcTensorType = dyn_cast<TensorType>(srcType);
  auto srcMemrefType = dyn_cast<MemRefType>(srcType);
  if (!(srcTensorType || srcMemrefType)) {
    return emitOpError("src must be tensor or memref type");
  }

  auto dstElementType = dstMemrefType.getElementType();
  auto srcElementType = srcMemrefType ? srcMemrefType.getElementType()
                                      : srcTensorType.getElementType();
  if (srcElementType != dstElementType) {
    return emitOpError(
        "src of hivm::IndirectStoreOp must have the same element type as dst");
  }

  auto offsetShape = offsetsMemrefType ? offsetsMemrefType.getShape()
                                       : offsetsTensorType.getShape();
  auto srcShape =
      srcMemrefType ? srcMemrefType.getShape() : srcTensorType.getShape();
  if (offsetShape != srcShape) {
    return emitOpError("offsets of hivm::IndirectStoreOp must have the same "
                       "shape and rank as src");
  }

  if (auto mask = getMask()) {
    auto maskType = mask.getType();
    auto maskTensorType = dyn_cast<TensorType>(maskType);
    auto maskMemrefType = dyn_cast<MemRefType>(maskType);
    if (!(maskTensorType || maskMemrefType)) {
      return emitOpError("mask must be tensor or memref type");
    }

    auto maskShape =
        maskMemrefType ? maskMemrefType.getShape() : maskTensorType.getShape();
    if (maskShape != offsetShape) {
      return emitOpError("mask of hivm::IndirectStoreOp must have the same "
                         "shape and rank as offsets");
    }
  }

  return success();
}

void IndirectStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  if (getMask()) {
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getOperation()->getOpOperand(getODSOperandIndexAndLength(3).first),
        SideEffects::DefaultResource::get());
  }
}

namespace {
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
} // namespace

SmallVector<std::pair<Type, int64_t>> CustomOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}

SmallVector<std::pair<Type, int64_t>>
CustomMacroOp::getExtraBuffersInfo() const {
  return getCustomOpExtraBuffersInfo(*this);
}

namespace {
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

//===----------------------------------------------------------------------===//
// DebugOp
//===----------------------------------------------------------------------===//

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex,
                    Value arg) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, {}, {});
}

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex,
                    Value arg, hivm::TCoreTypeAttr tcoretype) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, tcoretype, {});
}

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex,
                    Value arg, hivm::AddressSpaceAttr memscope) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, {}, memscope);
}
