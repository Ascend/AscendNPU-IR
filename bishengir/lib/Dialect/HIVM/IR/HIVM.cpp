//===- HIVM.cpp - HIVM ops implementation ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

// For function inliner support
#include "mlir/Transforms/InliningUtils.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivm;

#include "bishengir/Dialect/HIVM/IR/HIVMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"

#include "bishengir/Dialect/HIVM/IR/HIVMDialect.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// HIVMInlinerInterface Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct HIVMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in HIVM dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {}

  virtual ~HIVMInlinerInterface() = default;
};

} // namespace

//===----------------------------------------------------------------------===//
// HIVMDialect
//===----------------------------------------------------------------------===//

void hivm::HIVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMIntrinOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"
      >();
  // uncomment when adding types
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"
      >();

  // Add function inliner interfaces
  addInterfaces<HIVMInlinerInterface>();
}

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

void hivm::detail::printHIVMStructuredDPSOp(OpAsmPrinter &p, Operation *op,
                                            ValueRange inputs,
                                            ValueRange outputs) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  printDPSInputOutputs(p, inputs, outputs);
  printDPSResults(p, op->getResultTypes());
}

std::string hivm::detail::getTypeName(Location loc, Type type,
                                      hivm::TypeFn casting) {
  std::string unknown = "UNKNOWN";
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return "bool";
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness(), casting))
        return "uint" + std::to_string(iType.getWidth()) + "_t";
      else
        return "int" + std::to_string(iType.getWidth()) + "_t";
    default:
      emitError(loc, "unrecognized integer type: ") << type;
      return unknown;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 8:
      if (fType.isFloat8E4M3FN()) {
        return "float8_e4m3_t";
      } else if (fType.isFloat8E5M2()) {
        return "float8_e5m2_t";
      } else {
        emitError(loc, "unrecognized float8 type: ") << type;
        return unknown;
      }
    case 16:
      if (fType.isF16()) {
        return "half";
      } else if (fType.isBF16()) {
        return "bfloat16_t";
      } else {
        emitError(loc, "unrecognized float type: ") << type;
        return unknown;
      }
    case 32:
      return "float";
    case 64:
      return "double";
    default:
      emitError(loc, "unrecognized float type: ") << type;
      return unknown;
    }
  }
  emitError(loc, "unsupported type: ") << type;
  return unknown;
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
        suffix = "_mix_aic";
      } else if (funcCoreType.value() == TFuncCoreType::AIV) {
        suffix = "_mix_aiv";
      }
    }
  }
  return suffix;
}
} // namespace

//===----------------------------------------------------------------------===//
// DebugOp
//===----------------------------------------------------------------------===//

std::string DebugOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  std::string callName = this->getDebugtype().str();
  auto argTy = this->getArg().getType();
  if (isa<ShapedType>(argTy)) {
    auto argBufTy = cast<ShapedType>(argTy);
    int rank = argBufTy.getRank();
    int maxOpRank = getOpLibraryMaxRankImpl();
    if (rank > maxOpRank)
      this->emitError("DebugOp requires rank <= maxOpRank");
    std::string libCallDim = std::to_string(rank) + "d";
    std::string dataTypeStr =
        hivm::detail::getTypeName(this->getLoc(), argBufTy.getElementType());
    callName += "_" + libCallDim + "_" + dataTypeStr;
    // get and append address space
    // when getOpLibraryCallName is called from HIVMToStandard,
    // address space should only be GM (CUBE or VEC) or UB (VEC)
    if (isa<MemRefType>(argTy)) {
      Attribute argAttr = cast<MemRefType>(argTy).getMemorySpace();
      if (!isa<AddressSpaceAttr>(argAttr))
        this->emitError("print-to-libcall cannot find mem space");
      AddressSpace argAddrSpace =
          dyn_cast<AddressSpaceAttr>(argAttr).getAddressSpace();
      if (!(argAddrSpace == AddressSpace::GM ||
            argAddrSpace == AddressSpace::UB))
        this->emitError("print-to-libcall currently only supports GM and UB");
      if (argAddrSpace == AddressSpace::GM)
        callName = callName + "_" + "gm";
      else if (argAddrSpace == AddressSpace::UB)
        callName = callName + "_" + "ubuf";
    } else {
      this->emitError(
          "DebugOp::getOpLibraryCallName should only be called with memref");
    }
  } else {
    // Note: in this case "_mlir_ciface_" won't be automatically added by
    // mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
    std::string dataTypeStr = hivm::detail::getTypeName(this->getLoc(), argTy);
    callName = "_mlir_ciface_" + callName + "_scalar_" + dataTypeStr;
    callName += "_gm"; // currently, scalar can choose either gm or ubuf since
                       // they currently call the same core
  }
  return callName + debugCallNameMangleSuffix(this->getOperation());
}

//===----------------------------------------------------------------------===//
// InitDebugOp
//===----------------------------------------------------------------------===//

std::string InitDebugOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  return "_mlir_ciface_init_debug" +
         debugCallNameMangleSuffix(this->getOperation());
}

//===----------------------------------------------------------------------===//
// FinishDebugOp
//===----------------------------------------------------------------------===//

std::string FinishDebugOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  return "_mlir_ciface_finish_debug" +
         debugCallNameMangleSuffix(this->getOperation());
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

std::string
EmbeddingGatherOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto idxType = cast<ShapedType>(this->getIndex().getType());
  int rank = idxType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";
  // get embedding data type
  Type srcType = this->getSrc().getType();
  std::string srcTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));
  // get idx data type
  std::string idxTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(idxType));
  // make library function name
  return this->getOpName().str() + "_" + libCallDim + "_" + srcTypeStr + "_" +
         idxTypeStr;
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

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

LogicalResult GatherLoadOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    GatherLoadOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
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

std::string
IndirectLoadOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto offsetType = cast<ShapedType>(this->getOffsets().getType());
  int rank = offsetType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";

  Type srcType = this->getSrc().getType();
  std::string srcTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));

  std::string offsetTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(offsetType));

  return this->getOpName().str() + "_" + libCallDim + "_" + srcTypeStr + "_" +
         offsetTypeStr;
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

std::string
IndirectStoreOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto offsetType = cast<ShapedType>(this->getOffsets().getType());
  int rank = offsetType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";

  std::string hasMaskStr;
  if (this->getMask())
    hasMaskStr = "";
  else
    hasMaskStr = "_no_mask";

  Type srcType = this->getSrc().getType();
  std::string srcTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));

  std::string offsetTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(offsetType));

  return this->getOpName().str() + hasMaskStr + "_" + libCallDim + "_" +
         srcTypeStr + "_" + offsetTypeStr;
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
// GatherTOp
//===----------------------------------------------------------------------===//

std::string GatherTOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto idxType = cast<ShapedType>(this->getIndex().getType());
  int rank = idxType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";

  Type srcType = this->getSrc().getType();
  std::string srcTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));

  Type indexType = this->getIndex().getType();
  std::string idxTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(indexType));

  return this->getOpName().str() + "_" + libCallDim + "_" + srcTypeStr + "_" +
         idxTypeStr;
}

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

std::string IndexPutOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto valueType = dyn_cast<ShapedType>(this->getValue().getType());
  if (!valueType)
    llvm::report_fatal_error("IndexPutOp value must be a ShapedType");

  int rank = valueType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";

  Type dstType = this->getDst().getType();
  std::string dstTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(dstType));

  Type indexType = this->getIndex().getType();
  std::string indexTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(indexType));

  return this->getOpName().str() + "_" + libCallDim + "_" + dstTypeStr + "_" +
         indexTypeStr;
}

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

std::string ScatterTOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto indexTileType = cast<ShapedType>(this->getIndexTile().getType());
  int rank = indexTileType.getRank();
  std::string libCallDim = std::to_string(rank) + "d";
  std::string indexTileTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(indexTileType));

  Type valueType = this->getValue().getType();
  std::string valueTypeStr = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(valueType));

  return this->getOpName().str() + "_" + libCallDim + "_" + valueTypeStr + "_" +
         indexTileTypeStr;
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
  size_t numDynamic = static_cast<size_t>(llvm::count_if(getStaticOutputShape(), [](int64_t dim) {
    return ShapedType::isDynamic(dim);
  }));
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

static ParseResult parseForCustomOps(OpAsmParser &parser,
                                     OperationState &result) {
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

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomOps(parser, result);
}

ParseResult CustomMacroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForCustomOps(parser, result);
}

template <typename CustomOp>
static void printForCustomOps(CustomOp op, OpAsmPrinter &p) {
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

  if (!op.getResults().empty())
    p.printOptionalArrowTypeList(op.getResultTypes());
}

void CustomOp::print(OpAsmPrinter &p) { printForCustomOps(*this, p); }

void CustomMacroOp::print(OpAsmPrinter &p) { printForCustomOps(*this, p); }

template <typename CustomOpT>
static LogicalResult verifyBuiltins(CustomOpT op) {
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

template <typename CustomOpT>
static LogicalResult verifyCustomOp(CustomOpT op) {
  // Check builtins
  if (op.isBuiltin())
    return verifyBuiltins(op);

  // Check core type attribute
  const auto coreType = op.getCoreType();
  if (!coreType)
    return op.emitOpError() << "Missing core type information";

  // Check VF mode attribute
  if (*coreType != TCoreType::CUBE) {
    if (!op.getVFMode())
      return op.emitOpError() << "Missing vf mode information";
  } else { // Pure cube
    // Cube function ignores vf mode information
  }

  if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
    // Check pipe attribute
    if (op.getPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing pipe information";
  } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
    // Check input/output pipe attributes
    if (op.getInPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing input pipe information";

    if (op.getOutPipe() == PIPE::PIPE_UNASSIGNED)
      return op.emitOpError() << "Missing output pipe information";
  }

  return success();
}

LogicalResult CustomOp::verify() { return verifyCustomOp(*this); }

LogicalResult CustomMacroOp::verify() { return verifyCustomOp(*this); }

template <typename CustomOpT>
static std::string getGatherLoadLibraryCallName(CustomOpT op) {
  // Align to GatherTOp::getOpLibraryCallName()

  // Information from source
  const auto srcType = op->getOperand(0).getType();
  const std::string srcTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(srcType));

  // Information from idx
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

  auto srcRank =
      op->template getAttrOfType<StringAttr>("extra_attr").getValue().str();
  const std::string srcDim = srcRank.substr(srcRank.length() - 1, 1) + "d";

  // get embedding data type
  Type srcType = op->getOperand(0).getType();
  const std::string srcTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(srcType));
  // get idx data type
  const std::string idxTypeStr =
      hivm::detail::getTypeName(op->getLoc(), getElementTypeOrSelf(idxType));
  // make library function name
  return "index_select_" + srcDim + "_" + srcTypeStr + "_" + idxDim + "_" +
         idxTypeStr;
}

template <typename CustomOpT>
static std::string getCustomOpsLibraryCallName(CustomOpT op) {
  if (op.isBuiltin())
    return CustomOpT::kBuiltins.at(op.getName()).getOpLibraryCallName(op);

  // TODO: Extract from attributes (user provided hint)
  return "custom_todo";
}

std::string CustomOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getCustomOpsLibraryCallName<CustomOp>(*this);
}

std::string
CustomMacroOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getCustomOpsLibraryCallName<CustomMacroOp>(*this);
}

const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{
    {kBuiltinGatherLoadName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getGatherLoadLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinIndexSelectName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_V, VFMode::SIMT,
                 getIndexSelectLibraryCallName<CustomOp>,
                 /* GM Addr Args Indices */ {0})}};

const DenseMap<StringRef, CustomMacroOp::BuiltinInfo> CustomMacroOp::kBuiltins{
    {kBuiltinGatherLoadName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_MTE2, PIPE::PIPE_V, VFMode::SIMT,
                 getGatherLoadLibraryCallName<CustomMacroOp>,
                 /* GM Addr Args Indices */ {0})},
    {kBuiltinIndexSelectName,
     BuiltinInfo(TCoreType::VECTOR, PIPE::PIPE_MTE2, PIPE::PIPE_V, VFMode::SIMT,
                 getIndexSelectLibraryCallName<CustomMacroOp>,
                 /* GM Addr Args Indices */ {0})}};
