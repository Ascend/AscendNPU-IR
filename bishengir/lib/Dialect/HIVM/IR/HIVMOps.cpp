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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
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
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

// For function inliner support
#include "mlir/Transforms/InliningUtils.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

int64_t AddressSpaceAttr::getMappingId() const {
  return static_cast<int64_t>(getAddressSpace());
}

bool AddressSpaceAttr::isLinearMapping() const {
  llvm_unreachable("AddressSpaceAttr does not support linear mapping");
}

int64_t AddressSpaceAttr::getRelativeIndex() const {
  llvm_unreachable("AddressSpaceAttr does not support relative index");
}

//===----------------------------------------------------------------------===//
// DataLayoutAttr
//===----------------------------------------------------------------------===//

LogicalResult
DataLayoutAttr::verify(::llvm::function_ref<InFlightDiagnostic()> emitError,
                       hivm::DataLayout data_layout,
                       BoolAttr transpose,
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
  llvm_unreachable("HIVMSubBlockMappingAttr does not support linear mapping");
}

int64_t HIVMSubBlockMappingAttr::getRelativeIndex() const {
  llvm_unreachable("HIVMSubBlockMappingAttr does not support relative index");
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
  if (const std::size_t dynDims = getResult().getType().getNumDynamicDims(),
      dynDimValues = getDynamicSizes().size();
      dynDims != dynDimValues) {
    return emitOpError() << "expected " << dynDims
                         << " dynamic size operands, but found "
                         << dynDimValues;
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
                            DataLayoutAttr srcLayout,
                            DataLayoutAttr dstLayout,
                            ArrayRef<OpFoldResult> outputShape) {
  SmallVector<Value> dynamicDims;
  SmallVector<int64_t> staticDims;
  dispatchIndexOpFoldResults(outputShape, dynamicDims, staticDims);
  build(builder, result, resultType, source, srcLayout, dstLayout,
        staticDims, dynamicDims);
}

void ConvertLayoutOp::build(OpBuilder &builder, OperationState &result,
                            Type resultType, Value source,
                            DataLayoutAttr srcLayout,
                            DataLayoutAttr dstLayout) {
  auto staticDims = cast<ShapedType>(resultType).getShape();
  build(builder, result, resultType, source, srcLayout, dstLayout,
        staticDims, {});
}

LogicalResult ConvertLayoutOp::verify() {
  // Verify that the number of dynamic dims matches the number of kDynamic
  // entries in static_output_shape
  auto elementType = getElementTypeOrSelf(getResult());

  // Verify operand's element type matches first result's element type.
  for (auto operand : getOperands()) {
    if (!isa<ShapedType>(operand.getType())) continue;
    if (getElementTypeOrSelf(operand) != elementType)
      return emitOpError(
          "requires the same element type for all operands and results");
  }
  size_t numDynamic = llvm::count_if(getStaticOutputShape(), [](int64_t dim) {
    return ShapedType::isDynamic(dim);
  });
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

// Helper functions
void CustomOp::setPipe(PIPE pipe) {
  getOperation()->setAttr(PipeAttr::name, PipeAttr::get(getContext(), pipe));
}

std::optional<TCoreType> CustomOp::getCoreType() {
  if (const auto coreTypeAttr =
          getOperation()->template getAttrOfType<TCoreTypeAttr>(
              TCoreTypeAttr::name)) {
    return coreTypeAttr.getTcoretype();
  }

  return {};
}

void CustomOp::setCoreType(TCoreType coreType) {
  getOperation()->setAttr(TCoreTypeAttr::name,
                          TCoreTypeAttr::get(getContext(), coreType));
}

std::optional<VFMode> CustomOp::getVFMode() {
  if (const auto vfModeAttr =
          getOperation()->template getAttrOfType<VFModeAttr>(
              VFModeAttr::name)) {
    return vfModeAttr.getValue();
  }

  return {};
}

void CustomOp::setVFMode(VFMode vfMode) {
  getOperation()->setAttr(VFModeAttr::name,
                          VFModeAttr::get(getContext(), vfMode));
}

bool CustomOp::isBuiltin() { return kBuiltins.contains(getName()); }

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {
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
    auto parseVariadicArgs = [&](const std::string &nameHint) {
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

    if (failed(parseVariadicArgs("ins")) || failed(parseVariadicArgs("outs"))) {
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

void CustomOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "name"});

  p << " ";
  p.printString(getName());

  auto printVariadicArgs = [&](const auto &args, const std::string &nameHint) {
    if (!args.empty())
      p << " " << nameHint << "(" << args << " : " << args.getTypes() << ")";
  };

  printVariadicArgs(getInputs(), "ins");
  printVariadicArgs(getOutputs(), "outs");

  if (!getResults().empty())
    p.printOptionalArrowTypeList(getResultTypes());
}

static LogicalResult verifyBuiltins(CustomOp op) {
  const auto &builtinInfo = CustomOp::kBuiltins.at(op.getName());

  const auto &coreType = op.getCoreType();
  if (coreType && *coreType != builtinInfo.coreType)
    return op.emitOpError() << "Specified core type conflict with "
                            << op.getName() << "'s core type.";

  const auto &pipe = op.getPipe();
  if (pipe != PIPE::PIPE_UNASSIGNED && pipe != builtinInfo.pipe)
    return op.emitOpError()
           << "Specified pipe conflict with " << op.getName() << "'s pipe.";

  const auto &vfMode = op.getVFMode();
  if (vfMode && *vfMode != builtinInfo.vfMode)
    return op.emitOpError() << "Specified vf mode conflict with "
                            << op.getName() << "'s vf mode.";

  return success();
}

LogicalResult CustomOp::verify() {
  // Check builtins
  if (isBuiltin())
    return verifyBuiltins(*this);

  // Check core type attribute
  const auto coreType = getCoreType();
  if (!coreType)
    return emitOpError() << "Missing core type information";

  // Check pipe attribute
  if (getPipe() == PIPE::PIPE_UNASSIGNED)
    return emitOpError() << "Missing pipe information";

  // Check VF mode attribute
  if (*coreType != TCoreType::CUBE) {
    if (!getVFMode())
      return emitOpError() << "Missing vf mode information";
  } else { // Pure cube
    // Cube function ignores vf mode information
  }

  return success();
}

PIPE CustomOp::getPipe() {
  if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(PipeAttr::name))
    return pipAttr.getPipe();

  return PIPE::PIPE_UNASSIGNED;
}

const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{
    {"__builtin_gather_load", {TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT}}};
