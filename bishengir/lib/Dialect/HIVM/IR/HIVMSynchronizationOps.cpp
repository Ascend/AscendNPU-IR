//===- HIVMSynchronizationOps.cpp - HIVM diaelct Sync. Ops Implementation -===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// Printing/parsing for EventID
//===----------------------------------------------------------------------===//

ParseResult hivm::parseEventID(
    OpAsmParser &parser, EventAttr &eventIDAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &eventIDValue) {
  OpAsmParser::UnresolvedOperand operand;
  auto res = parser.parseOptionalOperand(operand);
  if (res.has_value() && succeeded(res.value())) {
    eventIDValue = operand;
    return success();
  }
  eventIDValue = std::nullopt;
  if (parser.parseCustomAttributeWithFallback(eventIDAttr, Type{}))
    return failure();

  return success();
}

void hivm::printEventID(OpAsmPrinter &printer, Operation *op,
                        EventAttr eventIDAttr, Value eventIDValue) {
  if (eventIDAttr) {
    eventIDAttr.print(printer);
    return;
  }
  printer << eventIDValue;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for FlagID
//===----------------------------------------------------------------------===//

ParseResult
hivm::parseFlagID(OpAsmParser &parser, IntegerAttr &flagIDAttr,
                  std::optional<OpAsmParser::UnresolvedOperand> &flagIDValue) {
  OpAsmParser::UnresolvedOperand operand;
  auto res = parser.parseOptionalOperand(operand);
  if (res.has_value() && succeeded(res.value())) {
    flagIDValue = operand;
    return success();
  }
  flagIDValue = std::nullopt;
  int64_t integer;
  if (failed(parser.parseInteger(integer)))
    return failure();
  flagIDAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), integer);
  return success();
}

void hivm::printFlagID(OpAsmPrinter &printer, Operation *op,
                       IntegerAttr flagIDAttr, Value flagIDValue) {
  if (flagIDAttr) {
    printer << flagIDAttr.getValue();
    return;
  }
  printer << flagIDValue;
}

//===----------------------------------------------------------------------===//
// SetFlagOp
//===----------------------------------------------------------------------===//

LogicalResult SetFlagOp::verify() {
  auto eventIDAttr = getStaticEventId();
  auto eventID = getDynamicEventId();
  if (eventIDAttr.has_value() && eventID != TypedValue<IntegerType>{}) {
    return emitOpError("Only one Event ID is supported!");
  }

  if (!eventIDAttr.has_value() && eventID == TypedValue<IntegerType>{}) {
    return emitOpError("Event ID is needed!");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WaitFlagOp
//===----------------------------------------------------------------------===//

LogicalResult WaitFlagOp::verify() {
  auto eventIDAttr = getStaticEventId();
  auto eventID = getDynamicEventId();
  if (eventIDAttr.has_value() && eventID != TypedValue<IntegerType>{}) {
    return emitOpError("Only one Event ID is supported!");
  }

  if (!eventIDAttr.has_value() && eventID == TypedValue<IntegerType>{}) {
    return emitOpError("Event ID is needed!");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SyncBlockSetOp
//===----------------------------------------------------------------------===//

OpFoldResult SyncBlockSetOp::getFlagId() {
  if (auto attr = getStaticFlagId()) {
    return attr.value();
  }
  return getDynamicFlagId();
}

LogicalResult SyncBlockSetOp::verify() {
  auto flagIdIDAttr = getStaticFlagId();
  auto flagIdValue = getDynamicFlagId();
  if (flagIdIDAttr.has_value() && flagIdValue != TypedValue<IntegerType>{}) {
    return emitOpError("Only one flag ID is supported!");
  }

  if (!flagIdIDAttr.has_value() && flagIdValue == TypedValue<IntegerType>{}) {
    return emitOpError("Flag ID is needed!");
  }
  return success();
}

::mlir::ParseResult SyncBlockSetOp::parse(::mlir::OpAsmParser &parser,
                                          ::mlir::OperationState &result) {
  ::mlir::hivm::TCoreTypeAttr tcore_typeAttr;
  ::mlir::hivm::PipeAttr tpipeAttr;
  ::mlir::hivm::PipeAttr pipeAttr;
  ::mlir::IntegerAttr static_flag_idAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      dynamic_flag_idOperands;
  ::llvm::SMLoc dynamic_flag_idOperandsLoc;
  (void)dynamic_flag_idOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      ffts_base_addrOperands;
  ::llvm::SMLoc ffts_base_addrOperandsLoc;
  (void)ffts_base_addrOperandsLoc;
  ::mlir::hivm::SyncBlockInstrModeAttr tsync_instr_modeAttr;
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return ::mlir::failure();
  }
  if (parser.parseLSquare())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(tcore_typeAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (tcore_typeAttr)
    result.getOrAddProperties<SyncBlockSetOp::Properties>().tcore_type =
        tcore_typeAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(tpipeAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (tpipeAttr)
    result.getOrAddProperties<SyncBlockSetOp::Properties>().tpipe = tpipeAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pipeAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (pipeAttr)
    result.getOrAddProperties<SyncBlockSetOp::Properties>().pipe = pipeAttr;
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseKeyword("flag"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  {
    dynamic_flag_idOperandsLoc = parser.getCurrentLocation();
    ::std::optional<::mlir::OpAsmParser::UnresolvedOperand> dynamic_flag_idOperand;
    auto odsResult =
        parseFlagID(parser, static_flag_idAttr, dynamic_flag_idOperand);
    if (odsResult)
      return ::mlir::failure();
    if (static_flag_idAttr)
      result.getOrAddProperties<SyncBlockSetOp::Properties>().static_flag_id =
          static_flag_idAttr;
    if (dynamic_flag_idOperand.has_value())
      dynamic_flag_idOperands.push_back(*dynamic_flag_idOperand);
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("ffts_base_addr"))) {
    if (parser.parseEqual())
      return ::mlir::failure();

    {
      ffts_base_addrOperandsLoc = parser.getCurrentLocation();
      ::mlir::OpAsmParser::UnresolvedOperand operand;
      ::mlir::OptionalParseResult parseResult =
          parser.parseOptionalOperand(operand);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        ffts_base_addrOperands.push_back(operand);
      }
    }
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("sync_instr_mode"))) {
    if (parser.parseEqual())
      return ::mlir::failure();

    if (parser.parseCustomAttributeWithFallback(tsync_instr_modeAttr,
                                                ::mlir::Type{}))
      return ::mlir::failure();
    if (tsync_instr_modeAttr)
      result.getOrAddProperties<SyncBlockSetOp::Properties>().tsync_instr_mode =
          tsync_instr_modeAttr;
  }
  ::llvm::copy(
      ::llvm::ArrayRef<int32_t>(
          {static_cast<int32_t>(dynamic_flag_idOperands.size()),
           static_cast<int32_t>(ffts_base_addrOperands.size())}),
      result.getOrAddProperties<SyncBlockSetOp::Properties>()
          .operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(64);
  if (parser.resolveOperands(dynamic_flag_idOperands, odsBuildableType0,
                             dynamic_flag_idOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ffts_base_addrOperands, odsBuildableType0,
                             ffts_base_addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SyncBlockSetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("tcore_type");
  elidedAttrs.push_back("tpipe");
  elidedAttrs.push_back("pipe");
  elidedAttrs.push_back("static_flag_id");
  elidedAttrs.push_back("tsync_instr_mode");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << "[";
  _odsPrinter.printStrippedAttrOrType(getTcoreTypeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getTpipeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getPipeAttr());
  _odsPrinter << "]";
  _odsPrinter << ' ' << "flag";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  printFlagID(_odsPrinter, *this, getStaticFlagIdAttr(), getDynamicFlagId());
  if (getFftsBaseAddr()) {
    _odsPrinter << ' ' << "ffts_base_addr";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    if (::mlir::Value value = getFftsBaseAddr())
      _odsPrinter << value;
  }
  if (getTsyncInstrModeAttr()) {
    _odsPrinter << ' ' << "sync_instr_mode";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printStrippedAttrOrType(getTsyncInstrModeAttr());
  }
}

void SyncBlockSetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr, nullptr,
          /*tsync_instr_mode=*/{});
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          cast<Value>(flag_id), nullptr, /*tsync_instr_mode=*/{});
  }
}

void SyncBlockSetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id,
                           Value ffts_base_addr,
                           hivm::SyncBlockInstrModeAttr tsync_instr_mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr, ffts_base_addr, tsync_instr_mode);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          cast<Value>(flag_id), ffts_base_addr, tsync_instr_mode);
  }
}

//===----------------------------------------------------------------------===//
// SyncBlockWaitOp
//===----------------------------------------------------------------------===//

OpFoldResult SyncBlockWaitOp::getFlagId() {
  if (auto attr = getStaticFlagId()) {
    return attr.value();
  }
  return getDynamicFlagId();
}

LogicalResult SyncBlockWaitOp::verify() {
  auto flagIdIDAttr = getStaticFlagId();
  auto flagIdValue = getDynamicFlagId();
  if (flagIdIDAttr.has_value() && flagIdValue != TypedValue<IntegerType>{}) {
    return emitOpError("Only one flag ID is supported!");
  }

  if (!flagIdIDAttr.has_value() && flagIdValue == TypedValue<IntegerType>{}) {
    return emitOpError("Flag ID is needed!");
  }
  return success();
}

void SyncBlockWaitOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            TCoreTypeAttr tcore_type, PipeAttr tpipe,
                            PipeAttr pipe, OpFoldResult flag_id) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          cast<Value>(flag_id));
  }
}

//===----------------------------------------------------------------------===//
// SyncBlockOp
//===----------------------------------------------------------------------===//

LogicalResult SyncBlockOp::verify() {
  auto synBlockMode = getSyncBlockModeAttr().getSyncMode();
  if (synBlockMode == SyncBlockMode::BARRIER_CUBE ||
      synBlockMode == SyncBlockMode::BARRIER_VECTOR) {
    if (getTvectorPipeAttr() != nullptr) {
      return emitOpError("tvector_pipe should not be defined!");
    }
    if (getTcubePipeAttr() != nullptr) {
      return emitOpError("tcube_pipe should not be defined!");
    }
  }
  if (synBlockMode == SyncBlockMode::ALL_CUBE ||
      synBlockMode == SyncBlockMode::ALL) {
    if (getTcubePipeAttr() == nullptr) {
      return emitOpError("tcube_pipe should be defined!");
    }
    if (!checkPipeInferredCoreType(getTcubePipeAttr().getPipe(),
                                   TCoreType::CUBE)) {
      return emitOpError("tcube_pipe of should match CUBE core type!");
    }
  }
  if (synBlockMode == SyncBlockMode::ALL_VECTOR ||
      synBlockMode == SyncBlockMode::ALL) {
    if (getTvectorPipeAttr() == nullptr) {
      return emitOpError("tvector_pipe should be defined!");
    }
    if (!checkPipeInferredCoreType(getTvectorPipeAttr().getPipe(),
                                   TCoreType::VECTOR)) {
      return emitOpError(
          "tvector_pipe of ALL_VECTOR should match VECTOR core type!");
    }
  }
  if (synBlockMode == SyncBlockMode::ALL_SUB_VECTOR) {
    if (getTvectorPipeAttr() == nullptr) {
      return emitOpError("tvector_pipe should be defined!");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CreateSyncBlockLockOp
//===----------------------------------------------------------------------===//

LogicalResult CreateSyncBlockLockOp::verify() {
  MemRefType type = getType();
  if (type.getNumDynamicDims() > 0)
    return this->emitOpError(
        "'create_sync_block_lock' op should only support static shape");

  return success();
}
