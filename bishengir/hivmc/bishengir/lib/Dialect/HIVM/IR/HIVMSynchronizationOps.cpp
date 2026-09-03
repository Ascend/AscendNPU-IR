//===- HIVMSynchronizationOps.cpp - HIVM diaelct Sync. Ops Implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hivm;

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"

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
// Printing/parsing for SyncID
//===----------------------------------------------------------------------===//
ParseResult hivm::parseSyncID(
    OpAsmParser &parser, IntegerAttr &syncIDAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &syncIDValue) {
  OpAsmParser::UnresolvedOperand operand;
  auto res = parser.parseOptionalOperand(operand);
  if (res.has_value() && succeeded(res.value())) {
    syncIDValue = operand;
    return success();
  }
  syncIDValue = std::nullopt;
  if (parser.parseCustomAttributeWithFallback(syncIDAttr, Type{}))
    return failure();

  return success();
}

void hivm::printSyncID(OpAsmPrinter &printer, Operation *op,
                        IntegerAttr syncIDAttr, Value syncIDValue) {
  if (syncIDAttr) {
    printer << syncIDAttr.getValue();
    return;
  }
  printer << syncIDValue;
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

void SyncBlockSetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr, nullptr,
          /*tsync_instr_mode=*/{});
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          flag_id.get<Value>(), nullptr, /*tsync_instr_mode=*/{});
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
          flag_id.get<Value>(), ffts_base_addr, tsync_instr_mode);
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
          flag_id.get<Value>());
  }
}

void SyncBlockWaitOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id,
                           hivm::SyncBlockInstrModeAttr tsync_instr_mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr, tsync_instr_mode);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          flag_id.get<Value>(), tsync_instr_mode);
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
      synBlockMode == SyncBlockMode::ALL_SUB_VECTOR ||
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
