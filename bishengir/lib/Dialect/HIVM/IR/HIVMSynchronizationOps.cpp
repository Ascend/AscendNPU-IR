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
// Printing/parsing for SyncID
//===----------------------------------------------------------------------===//
ParseResult
hivm::parseSyncID(OpAsmParser &parser, IntegerAttr &syncIDAttr,
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
// TryWaitOp
//===----------------------------------------------------------------------===//

OpFoldResult TryWaitOp::getFlagId() {
  if (auto attr = getStaticFlagId())
    return *attr;
  return getDynamicFlagId();
}

LogicalResult TryWaitOp::verify() {
  auto flagIdAttr = getStaticFlagId();
  Value flagId = getDynamicFlagId();
  if (flagIdAttr.has_value() && flagId)
    return emitOpError("only one flag ID is supported");
  if (!flagIdAttr.has_value() && !flagId)
    return emitOpError("flag ID is needed");
  if (flagIdAttr &&
      (flagIdAttr->getValue().isNegative() || flagIdAttr->getValue().ugt(31)))
    return emitOpError("static flag ID must fit the 5-bit Xt field (0..31)");
  return success();
}

void TryWaitOp::build(OpBuilder &builder, OperationState &state,
                      OpFoldResult flagId, TryWaitSyncMode syncMode) {
  auto syncModeAttr = TryWaitSyncModeAttr::get(builder.getContext(), syncMode);
  if (auto attr = dyn_cast_if_present<Attribute>(flagId)) {
    build(builder, state, builder.getI64Type(), cast<IntegerAttr>(attr),
          nullptr, syncModeAttr);
    return;
  }
  build(builder, state, builder.getI64Type(), nullptr, cast<Value>(flagId),
        syncModeAttr);
}

//===----------------------------------------------------------------------===//
// Individual intra-block synchronization
//===----------------------------------------------------------------------===//

template <typename Op> static LogicalResult verifyBufferSync(Op op) {
  auto id = op.getStaticBufId();
  Value dynamic = op.getDynamicBufId();
  if (id.has_value() == static_cast<bool>(dynamic))
    return op.emitOpError(
        "exactly one static or dynamic buffer ID is required");
  if (!id && dynamic)
    if (auto constant = dynamic.getDefiningOp<arith::ConstantIntOp>())
      id = cast<IntegerAttr>(constant.getValue());
  if (id && (id->getValue().isNegative() || id->getValue().ugt(31)))
    return op.emitOpError("buffer ID must be in [0, 31]");
  auto pipe = op.getPipe().getPipe();
  bool valid = pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 ||
               pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1 ||
               pipe == PIPE::PIPE_FIX || pipe == PIPE::PIPE_V ||
               pipe == PIPE::PIPE_MTE3;
  if (!valid)
    return op.emitOpError("unsupported buffer synchronization pipe");
  return success();
}

LogicalResult GetBufOp::verify() { return verifyBufferSync(*this); }
LogicalResult RlsBufOp::verify() { return verifyBufferSync(*this); }

OpFoldResult GetBufOp::getBufId() {
  if (auto attr = getStaticBufId())
    return *attr;
  return getDynamicBufId();
}

OpFoldResult RlsBufOp::getBufId() {
  if (auto attr = getStaticBufId())
    return *attr;
  return getDynamicBufId();
}

void GetBufOp::build(OpBuilder &builder, OperationState &state, PipeAttr pipe,
                     OpFoldResult id, BoolAttr mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(id))
    build(builder, state, pipe, cast<IntegerAttr>(attr), Value{}, mode);
  else
    build(builder, state, pipe, IntegerAttr{}, cast<Value>(id), mode);
}

void RlsBufOp::build(OpBuilder &builder, OperationState &state, PipeAttr pipe,
                     OpFoldResult id, BoolAttr mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(id))
    build(builder, state, pipe, cast<IntegerAttr>(attr), Value{}, mode);
  else
    build(builder, state, pipe, IntegerAttr{}, cast<Value>(id), mode);
}

template <typename Op> static LogicalResult verifyIntraBlockSync(Op op) {
  auto attr = op.getStaticSyncId();
  Value value = op.getDynamicSyncId();
  if (attr.has_value() == static_cast<bool>(value))
    return op.emitOpError("exactly one static or dynamic sync ID is required");
  if (!attr && value)
    if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
      attr = cast<IntegerAttr>(constant.getValue());
  auto core = op.getTcoreType().getTcoretype();
  if (core != TCoreType::CUBE && core != TCoreType::VECTOR)
    return op.emitOpError("requires an explicit CUBE or VECTOR core type");
  if (op.getPipe().getPipe() == PIPE::PIPE_ALL ||
      !checkPipeInferredCoreType(op.getPipe().getPipe(), core))
    return op.emitOpError(
        "pipe must match the core type and cannot be PIPE_ALL");
  unsigned maximum = core == TCoreType::CUBE ? 31 : 15;
  if (attr && (attr->getValue().isNegative() || attr->getValue().ugt(maximum)))
    return op.emitOpError("sync ID out of range for the core type");
  return success();
}

OpFoldResult SyncIntraBlockSetOp::getSyncId() {
  if (auto attr = getStaticSyncId())
    return *attr;
  return getDynamicSyncId();
}

LogicalResult SyncIntraBlockSetOp::verify() {
  return verifyIntraBlockSync(*this);
}

void SyncIntraBlockSetOp::build(OpBuilder &builder, OperationState &state,
                                TCoreTypeAttr core, PipeAttr pipe,
                                OpFoldResult id) {
  if (auto attr = dyn_cast_if_present<Attribute>(id))
    build(builder, state, core, pipe, cast<IntegerAttr>(attr), Value{});
  else
    build(builder, state, core, pipe, IntegerAttr{}, cast<Value>(id));
}

OpFoldResult SyncIntraBlockWaitOp::getSyncId() {
  if (auto attr = getStaticSyncId())
    return *attr;
  return getDynamicSyncId();
}

LogicalResult SyncIntraBlockWaitOp::verify() {
  return verifyIntraBlockSync(*this);
}

void SyncIntraBlockWaitOp::build(OpBuilder &builder, OperationState &state,
                                 TCoreTypeAttr core, PipeAttr pipe,
                                 OpFoldResult id) {
  if (auto attr = dyn_cast_if_present<Attribute>(id))
    build(builder, state, core, pipe, cast<IntegerAttr>(attr), Value{});
  else
    build(builder, state, core, pipe, IntegerAttr{}, cast<Value>(id));
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

void SyncBlockWaitOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            TCoreTypeAttr tcore_type, PipeAttr tpipe,
                            PipeAttr pipe, OpFoldResult flag_id,
                            hivm::SyncBlockInstrModeAttr tsync_instr_mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          cast<IntegerAttr>(attr), nullptr, tsync_instr_mode);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe, nullptr,
          cast<Value>(flag_id), tsync_instr_mode);
  }
}

//===----------------------------------------------------------------------===//
// SyncBlockOp
//===----------------------------------------------------------------------===//

void SyncBlockOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        SyncBlockModeAttr sync_block_mode, IntegerAttr flag_id,
                        Value ffts_base_addr, PipeAttr tcube_pipe,
                        PipeAttr tvector_pipe) {
  build(odsBuilder, odsState, sync_block_mode, flag_id, ffts_base_addr,
        tcube_pipe, tcube_pipe, tvector_pipe, tvector_pipe);
}

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
    if (getVectorPipeAttr() != nullptr) {
      return emitOpError("vector_pipe should not be defined!");
    }
    if (getCubePipeAttr() != nullptr) {
      return emitOpError("cube_pipe should not be defined!");
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
    if (getCubePipeAttr() &&
        !checkPipeInferredCoreType(getCubePipeAttr().getPipe(),
                                   TCoreType::CUBE)) {
      return emitOpError("cube_pipe should match CUBE core type!");
    }
    if (synBlockMode == SyncBlockMode::ALL_CUBE) {
      if (getTvectorPipeAttr() != nullptr) {
        return emitOpError("tvector_pipe should not be defined!");
      }
      if (getVectorPipeAttr() != nullptr) {
        return emitOpError("vector_pipe should not be defined!");
      }
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
    if (getVectorPipeAttr() &&
        !checkPipeInferredCoreType(getVectorPipeAttr().getPipe(),
                                   TCoreType::VECTOR)) {
      return emitOpError("vector_pipe should match VECTOR core type!");
    }
    if (synBlockMode == SyncBlockMode::ALL_VECTOR ||
        synBlockMode == SyncBlockMode::ALL_SUB_VECTOR) {
      if (getTcubePipeAttr() != nullptr) {
        return emitOpError("tcube_pipe should not be defined!");
      }
      if (getCubePipeAttr() != nullptr) {
        return emitOpError("cube_pipe should not be defined!");
      }
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
