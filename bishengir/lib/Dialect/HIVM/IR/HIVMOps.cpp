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
  llvm::report_fatal_error(
      "HIVMSubBlockMappingAttr does not support linear mapping");
}

int64_t HIVMSubBlockMappingAttr::getRelativeIndex() const {
  llvm::report_fatal_error(
      "HIVMSubBlockMappingAttr does not support relative index");
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

namespace {
bool shouldMapToUnsigned(IntegerType::SignednessSemantics val,
                         hivm::TypeFn casting) {
  if (hivm::TypeFn::cast_unsigned == casting)
    return true;

  switch (val) {
  case IntegerType::Signless:
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}
} // namespace

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

//===----------------------------------------------------------------------===//
// Debug Op helper
//===----------------------------------------------------------------------===//

namespace {
std::string debugCallNameMangleSuffix(Operation *op) {
  std::string suffix = "";
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return suffix;
  }
  TModuleCoreTypeAttr attr = dyn_cast_or_null<TModuleCoreTypeAttr>(
      moduleOp->getAttr(TModuleCoreTypeAttr::name));
  if (attr && attr.getModuleCoreType() == TModuleCoreType::MIX) {
    // getOpLibraryCallName is called in HIVMToStandard
    // where mix functions have already been splitted.
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return suffix;
    }
    std::optional<TFuncCoreType> funcCoreType = queryFuncCoreType(funcOp);
    if (funcCoreType.has_value()) {
      if (funcCoreType.value() == TFuncCoreType::AIC) {
        suffix = ".cube";
      } else if (funcCoreType.value() == TFuncCoreType::AIV) {
        suffix = ".vector";
      }
    }
  }
  return suffix;
}
} // namespace

//===----------------------------------------------------------------------===//
// DebugOp
//===----------------------------------------------------------------------===//

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex,
                    Value arg) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, {}, {});
}

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex, Value arg,
                    hivm::TCoreTypeAttr tcoretype) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, tcoretype, {});
}

void DebugOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    StringRef debugtype, StringRef prefix, bool hex, Value arg,
                    hivm::AddressSpaceAttr memscope) {
  build(odsBuilder, odsState, debugtype, prefix, hex, arg, {}, memscope);
}

//===----------------------------------------------------------------------===//
// StrideLoadOp
//===----------------------------------------------------------------------===//

LogicalResult StrideLoadOp::verify() {
  auto srcMemrefType = dyn_cast<MemRefType>(getSrc().getType());
  if (!srcMemrefType)
    return emitOpError("src must be a memref type");

  auto dstType = getDst().getType();
  auto dstTensorType = dyn_cast<TensorType>(dstType);
  auto dstMemrefType = dyn_cast<MemRefType>(dstType);
  if (!(dstTensorType || dstMemrefType))
    return emitOpError("dst must be tensor or memref type");

  auto dstRank =
      dstMemrefType ? dstMemrefType.getRank() : dstTensorType.getRank();
  if (dstRank < 1 || dstRank > 3)
    return emitOpError("only support 1-3D");

  if (static_cast<int64_t>(getStride().size()) != dstRank ||
      static_cast<int64_t>(getNumel().size()) != dstRank) {
    return emitOpError("stride and numel operand counts must match dst rank");
  }

  auto getIndexType = [&](ValueRange values,
                          StringRef name) -> FailureOr<Type> {
    if (values.empty())
      return emitOpError() << name << " operands must not be empty";
    Type type = values.front().getType();
    for (Value value : values) {
      if (value.getType() != type)
        return emitOpError() << name << " operands must have the same type";
    }
    return type;
  };

  Type indexType = getOffset().getType();
  FailureOr<Type> strideType = getIndexType(getStride(), "stride");
  FailureOr<Type> numelType = getIndexType(getNumel(), "numel");
  if (failed(strideType) || failed(numelType))
    return failure();
  if (indexType != *strideType || indexType != *numelType)
    return emitOpError(
        "offset, stride and numel operands must have the same type");

  auto srcElementType = srcMemrefType.getElementType();
  auto dstElementType = dstMemrefType ? dstMemrefType.getElementType()
                                      : dstTensorType.getElementType();
  if (dstElementType != srcElementType)
    return emitOpError(
        "dst of hivm::StrideLoadOp must have the same element type as src");
  if (getOther().getType() != srcElementType)
    return emitOpError("other must have the same element type as src");

  if (getResult()) {
    if (getResult().getType() != dstType)
      return emitOpError("result must have the same type as dst");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StrideStoreOp
//===----------------------------------------------------------------------===//

LogicalResult StrideStoreOp::verify() {
  auto dstMemrefType = dyn_cast<MemRefType>(getDst().getType());
  if (!dstMemrefType)
    return emitOpError("dst must be a memref type");

  auto srcType = getSrc().getType();
  auto srcTensorType = dyn_cast<TensorType>(srcType);
  auto srcMemrefType = dyn_cast<MemRefType>(srcType);
  if (!(srcTensorType || srcMemrefType))
    return emitOpError("src must be tensor or memref type");

  auto srcRank =
      srcMemrefType ? srcMemrefType.getRank() : srcTensorType.getRank();
  if (srcRank < 1 || srcRank > 3)
    return emitOpError("only support 1-3D");

  if (static_cast<int64_t>(getStride().size()) != srcRank ||
      static_cast<int64_t>(getNumel().size()) != srcRank) {
    return emitOpError("stride and numel operand counts must match src rank");
  }

  auto getIndexType = [&](ValueRange values,
                          StringRef name) -> FailureOr<Type> {
    if (values.empty())
      return emitOpError() << name << " operands must not be empty";
    Type type = values.front().getType();
    for (Value value : values) {
      if (value.getType() != type)
        return emitOpError() << name << " operands must have the same type";
    }
    return type;
  };

  Type indexType = getOffset().getType();
  FailureOr<Type> strideType = getIndexType(getStride(), "stride");
  FailureOr<Type> numelType = getIndexType(getNumel(), "numel");
  if (failed(strideType) || failed(numelType))
    return failure();
  if (indexType != *strideType || indexType != *numelType)
    return emitOpError(
        "offset, stride and numel operands must have the same type");

  auto dstElementType = dstMemrefType.getElementType();
  auto srcElementType = srcMemrefType ? srcMemrefType.getElementType()
                                      : srcTensorType.getElementType();
  if (srcElementType != dstElementType)
    return emitOpError(
        "src of hivm::StrideStoreOp must have the same element type as dst");

  return success();
}

//===----------------------------------------------------------------------===//
// GatherTOp
//===----------------------------------------------------------------------===//

void GatherTOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IndexPutOp
//===----------------------------------------------------------------------===//

void IndexPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ScatterTOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterTOp::verify() {
  auto valueType = getValue().getType();
  auto indexTileType = getIndexTile().getType();
  auto valueTensorType = dyn_cast<TensorType>(valueType);
  auto valueMemrefType = dyn_cast<MemRefType>(valueType);
  if (!(valueTensorType || valueMemrefType)) {
    return emitOpError("value must be tensor or memref type");
  }
  auto indexTileTensorType = dyn_cast<TensorType>(indexTileType);
  auto indexTileMemrefType = dyn_cast<MemRefType>(indexTileType);
  if (!(indexTileTensorType || indexTileMemrefType)) {
    return emitOpError("index_tile must be tensor or memref type");
  }

  auto valueShape =
      valueMemrefType ? valueMemrefType.getShape() : valueTensorType.getShape();
  auto indexTileShape = indexTileMemrefType ? indexTileMemrefType.getShape()
                                            : indexTileTensorType.getShape();

  if (valueShape != indexTileShape) {
    return emitOpError("tensor of value and index_tile of hivm::ScatterTOp "
                       "must have the same shape");
  }

  return success();
}

void ScatterTOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexTileMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AnchorOp
//===----------------------------------------------------------------------===//

struct AnchorResource
    : public mlir::SideEffects::Resource::Base<AnchorResource> {
  mlir::StringRef getName() final { return "<Anchor>"; }
};

void mlir::hivm::AnchorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), AnchorResource::get());
}

//===----------------------------------------------------------------------===//
// LocalLoadOp
//===----------------------------------------------------------------------===//

LogicalResult LocalLoadOp::verify() {
  auto inputType = getAddr().getType();
  auto outputType = getResult().getType();
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input address of hivm::LocalLoadOp must have the same "
                       "element type as output tensor");
  }
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input address of hivm::LocalLoadOp must have the same "
                       "shape as output tensor");
  }
  return success();
}

void LocalLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// LocalStoreOp
//===----------------------------------------------------------------------===//

LogicalResult LocalStoreOp::verify() {
  auto dstType = getAddr().getType();
  auto dataType = getData().getType();
  if (dstType.getElementType() != dataType.getElementType()) {
    return emitOpError("dst address of hivm::LocalStoreOp must have the same "
                       "element type as output tensor");
  }
  if (dstType.getShape() != dataType.getShape()) {
    return emitOpError("dst address of hivm::LocalStoreOp must have the same "
                       "shape as output tensor");
  }
  return success();
}

void LocalStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IndirectLoadOp
//===----------------------------------------------------------------------===//

LogicalResult IndirectLoadOp::verify() {
  auto srcType = getSrc().getType();
  auto srcMemrefType = dyn_cast<MemRefType>(srcType);
  if (!srcMemrefType) {
    return emitError("src must be a memref type");
  }

  auto offsetsType = getOffsets().getType();
  auto offsetsTensorType = dyn_cast<TensorType>(offsetsType);
  auto offsetsMemrefType = dyn_cast<MemRefType>(offsetsType);
  if (!(offsetsTensorType || offsetsMemrefType)) {
    return emitOpError("offset must be tensor or memref type");
  }

  auto dstType = getDst().getType();
  auto dstTensorType = dyn_cast<TensorType>(dstType);
  auto dstMemrefType = dyn_cast<MemRefType>(dstType);
  if (!(dstTensorType || dstMemrefType)) {
    return emitOpError("dst must be tensor or memref type");
  }

  auto srcElementType = srcMemrefType.getElementType();
  auto dstElementType = dstMemrefType ? dstMemrefType.getElementType()
                                      : dstTensorType.getElementType();
  if (dstElementType != srcElementType) {
    return emitOpError(
        "dst of hivm::IndirectLoadOp must have the same element type as src");
  }

  auto dstShape =
      dstMemrefType ? dstMemrefType.getShape() : dstTensorType.getShape();
  auto offsetShape = offsetsMemrefType ? offsetsMemrefType.getShape()
                                       : offsetsTensorType.getShape();
  if (dstShape != offsetShape) {
    return emitOpError(
        "dst of hivm::IndirectLoadOp must have the same shape as offsets");
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
      return emitOpError("mask of hivm::IndirectLoadOp must have the same "
                         "shape and rank as offsets");
    }
  }

  if (auto other = getOther()) {
    auto otherType = other.getType();
    auto otherTensorType = dyn_cast<TensorType>(otherType);
    auto otherMemrefType = dyn_cast<MemRefType>(otherType);
    if (!(otherTensorType || otherMemrefType)) {
      return emitOpError("other must be tensor or memref type");
    }

    auto otherShape = otherMemrefType ? otherMemrefType.getShape()
                                      : otherTensorType.getShape();
    if (otherShape != offsetShape) {
      return emitOpError("other of hfusion::IndirectLoadOp must have the same "
                         "shape and rank as offsets");
    }

    auto otherElementType = otherMemrefType ? otherMemrefType.getElementType()
                                            : otherTensorType.getElementType();
    if (otherElementType != srcElementType) {
      return emitOpError("other of hfusion::IndirectLoadOp must have the same "
                         "element type as src");
    }
  }
  return success();
}

void IndirectLoadOp::getEffects(
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
  if (getOther()) {
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getOperation()->getOpOperand(getODSOperandIndexAndLength(4).first),
        SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// EmbeddingGatherOp
//===----------------------------------------------------------------------===//

LogicalResult EmbeddingGatherOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(this->getSrc().getType());
  if (!srcTy) {
    return emitOpError("src of hivm::EmbeddingGatherOp must be MemrefType!");
  }
  auto idxTy = this->getIndex().getType();
  auto idxMemTy = dyn_cast<MemRefType>(idxTy);
  auto idxTenTy = dyn_cast<TensorType>(idxTy);
  if (!idxMemTy && !idxTenTy) {
    return emitOpError(
        "idx of hivm::EmbeddingGatherOp must be TensorOrMemRefType!");
  }
  auto dstTy = this->getDst().getType();
  auto dstMemTy = dyn_cast<MemRefType>(dstTy);
  auto dstTenTy = dyn_cast<TensorType>(dstTy);
  if (!dstMemTy && !dstTenTy) {
    return emitOpError(
        "dst of hivm::EmbeddingGatherOp must be TensorOrMemRefType!");
  }
  // TODO: supports more ranks?
  auto idxRank = idxMemTy ? idxMemTy.getRank() : idxTenTy.getRank();
  if (!(idxRank >= 1 && idxRank <= 2)) {
    return emitOpError(
        "idx of hivm::EmbeddingGatherOp must be of rank [1, 2]!");
  }
  auto dstRank = dstMemTy ? dstMemTy.getRank() : dstTenTy.getRank();
  if (dstRank != idxRank + 1) {
    return emitOpError("dst of hivm::EmbeddingGatherOp must be idxRank + 1!");
  }

  auto boundType = dyn_cast<IntegerType>(this->getBound().getType());
  if (!boundType ||
      (boundType.getWidth() != 64 && boundType.getWidth() != 32)) {
    return emitOpError("bound of hivm::EmbeddingGatherOp must be i64|i32!");
  }
  auto offsets = this->getOffsets();
  if (ssize_t(offsets.size()) != dstRank) {
    return emitOpError("The size of offsets of hivm::EmbeddingGatherOp must be "
                       "equal to the rank of dst!");
  }
  for (auto offset : offsets) {
    auto offsetType = dyn_cast<IntegerType>(offset.getType());
    auto bitWidth = offsetType.getWidth();
    if (!offsetType || (bitWidth != 64 && bitWidth != 32)) {
      return emitOpError("offsets of hivm::EmbeddingGatherOp must be i64|i32!");
    }
  }
  auto numels = this->getNumels();
  if (ssize_t(numels.size()) != dstRank) {
    return emitOpError("The size of numels of hivm::EmbeddingGatherOp must be "
                       "equal to the rank of dst!");
  }
  for (auto numel : numels) {
    auto numelType = dyn_cast<IntegerType>(numel.getType());
    auto bitWidth = numelType.getWidth();
    if (!numelType || (bitWidth != 64 && bitWidth != 32)) {
      return emitOpError("numels of hivm::EmbeddingGatherOp must be i64|i32!");
    }
  }

  return success();
}

void EmbeddingGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}
