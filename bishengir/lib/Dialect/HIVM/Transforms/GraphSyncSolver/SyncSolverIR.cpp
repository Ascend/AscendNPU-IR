//===---------- SyncSolverIR.cpp ---- Graph Sync Solver -------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir::hivm::syncsolver {

int OperationBase::globalIndex = 0;

// Map OpType enum to human-readable strings for debugging output.
std::string getOpTypeStr(OpType opType) {
  const llvm::DenseMap<OpType, std::string> conv = {
      {OpType::OPERATION, "OperationBase"},
      {OpType::PLACE_HOLDER, "PlaceHolder"},
      {OpType::SCOPE, "Scope"},
      {OpType::FUNCTION, "Function"},
      {OpType::FUNCTION_BLOCK, "FunctionBlock"},
      {OpType::LOOP, "Loop"},
      {OpType::MMAD_SCOPE, "MmadLoop"},
      {OpType::CONDITION, "Condition"},
      {OpType::BARRIER_OP, "BarrierOp"},
      {OpType::SET_FLAG_OP, "SetFlagOp"},
      {OpType::WAIT_FLAG_OP, "WaitFlagOp"},
      {OpType::RW_OPERATION, "RWOperation"},
      {OpType::MMAD_OPERATION, "MmadOp"},
      {OpType::MMAD_LOAD_L0A_OPERATION, "LoadMmadL0AOp"},
      {OpType::MMAD_LOAD_L0B_OPERATION, "LoadMmadL0BOp"},
      {OpType::MMAD_LOAD_BIAS_OPERATION, "LoadMmadBias"},
      {OpType::RW_OPERATION_END, "RW_OPERATION_END"},
  };
  return conv.at(opType);
}

bool operator<(const SyncOp &op1, const SyncOp &op2) {
  return isa<WaitFlagOp>(&op1) && isa<SetFlagOp>(&op2);
}

struct Comma {
  bool comma = false;
  std::string get() {
    std::string ret = comma ? ", " : "";
    comma = true;
    return ret;
  }
};

// Provide readable string representations for IR nodes used in logs and dumps.
// Each specialized .str implementation documents what it prints.
std::string PlaceHolder::str(int indent, bool recursive) const {
  std::string opStr =
      (op != nullptr
           ? op2str(op)
           : llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType))) +
      std::to_string(this->id);
  return std::string(indent, ' ') + opStr;
}

std::string Scope::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      ret += op->str(indent + 2, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string Loop::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  if (isParallel) {
    ret += " parallel-loop";
  }
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      ret += op->str(indent + 2, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string Condition::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  ;
  if (isUnlikely) {
    ret += " unlikely-cond";
  }
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      if (op.get() == getTrueScope()) {
        ret += std::string(indent + 2, ' ') + "(trueScope)\n";
      } else if (op.get() == getFalseScope()) {
        ret += std::string(indent + 2, ' ') + "(falseScope)\n";
      }
      ret += op->str(indent + 2, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string RWOperation::str(int indent, bool recursive) const {
  std::string ret;
  std::string opStr =
      (op != nullptr
           ? op2str(op)
           : llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType))) +
      std::to_string(this->id);
  std::string coreTypeStr;
  if (coreType != hivm::TCoreType::CUBE_OR_VECTOR) {
    coreTypeStr = "[<" + stringifyTCoreType(coreType).str() + ">]";
  }
  std::string pipesStr;
  if (this->pipeRead != this->pipeWrite) {
    pipesStr = "[<" + stringifyPIPE(this->pipeRead).str() + ">, <" +
               stringifyPIPE(this->pipeRead).str() + ">]";
  } else {
    pipesStr = "[<" + stringifyPIPE(this->pipeRead).str() + ">]";
  }
  std::string unitFlag;
  if (!mergedUnitFlagInfo.disabledAsSet()) {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    ss << "unitFlagAsSet(";
    llvm::interleaveComma(
        mergedUnitFlagInfo.getUnitFlagModesAsSet(/*compress=*/true), ss);
    ss << ")";
    unitFlag += ss.str();
  }
  if (!mergedUnitFlagInfo.disabledAsWait()) {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    ss << "unitFlagAsWait(";
    llvm::interleaveComma(
        mergedUnitFlagInfo.getUnitFlagModesAsWait(/*compress=*/true), ss);
    ss << ")";
    unitFlag += (!unitFlag.empty() ? " " : "") + ss.str();
  }
  ret += std::string(indent, ' ') + opStr + " " + coreTypeStr + " " + pipesStr +
         " " + unitFlag + "\n";
  if (indent) {
    for (auto val : this->readMemVals) {
      ret += std::string(indent + 2, ' ') + "read: " + op2str(val) + "\n";
    }
    for (auto val : this->writeMemVals) {
      ret += std::string(indent + 2, ' ') + "write: " + op2str(val) + "\n";
    }
    for (auto &ptr : this->testReadMemVals) {
      ret += std::string(indent + 2, ' ') + "read: ptr(";
      Comma comma;
      for (auto val : ptr) {
        ret += comma.get() + std::to_string(val);
      }
      ret += ") \n";
    }
    for (auto &ptr : this->testWriteMemVals) {
      ret += std::string(indent + 2, ' ') + "write: ptr(";
      Comma comma;
      for (auto val : ptr) {
        ret += comma.get() + std::to_string(val);
      }
      ret += ") \n";
    }
  }
  ret.pop_back();
  return ret;
}

std::string SetFlagOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<";
  if (this->coreType != hivm::TCoreType::CUBE_OR_VECTOR) {
    ret += stringifyTCoreType(this->coreType).str() + ">, <";
  }
  ret += stringifyPIPE(this->pipeSrc).str() + ">, <" +
         stringifyPIPE(this->pipeDst).str() + ">, (";
  Comma comma;
  for (auto eventId : this->eventIds) {
    ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
  }
  ret += ")]";
  if (allAtOnce) {
    ret += " all-at-once";
  }
  if (checkFirstIter) {
    ret += " check-first-iter";
  }
  if (checkLastIter) {
    ret += " check-last-iter";
  }
  return ret;
}

std::string WaitFlagOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<";
  if (this->coreType != hivm::TCoreType::CUBE_OR_VECTOR) {
    ret += stringifyTCoreType(this->coreType).str() + ">, <";
  }
  ret += stringifyPIPE(this->pipeSrc).str() + ">, <" +
         stringifyPIPE(this->pipeDst).str() + ">, (";
  Comma comma;
  for (auto eventId : this->eventIds) {
    ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
  }
  ret += ")]";
  if (allAtOnce) {
    ret += " all-at-once";
  }
  if (checkFirstIter) {
    ret += " check-first-iter";
  }
  if (checkLastIter) {
    ret += " check-last-iter";
  }
  return ret;
}

std::string BarrierOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<" + stringifyPIPE(this->pipe).str() + ">]";
  return ret;
}

std::string ConflictPair::str() const {
  std::string ret;
  ret += "ConflictPair" + std::to_string(this->id);
  ret += " (" + std::to_string(this->startIndex) + ", " +
         std::to_string(this->endIndex) + ")";
  if (this->isBarrier()) {
    ret += " [<" + stringifyPIPE(setCorePipeInfo.pipe).str() + ">]";
  } else {
    if (setCorePipeInfo.coreType != hivm::TCoreType::CUBE_OR_VECTOR ||
        waitCorePipeInfo.coreType != hivm::TCoreType::CUBE_OR_VECTOR) {
      ret += "[<" + stringifyTCoreType(setCorePipeInfo.coreType).str() +
             ">, <" + stringifyTCoreType(waitCorePipeInfo.coreType).str() +
             ">]";
    }
    ret += " [<" + stringifyPIPE(setCorePipeInfo.pipe).str() + ">, <" +
           stringifyPIPE(waitCorePipeInfo.pipe).str() + ">, (";
    Comma comma;
    if (this->eventIdNode != nullptr) {
      for (auto eventId : this->eventIdNode->getEventIds()) {
        ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
      }
    }
    ret += ")]";
  }

  Comma comma;
  ret += " {";
  ret += (isBarrier() ? (comma.get() + "is-barrier") : "");
  ret += (isInnerBackward ? (comma.get() + "is-backward") : "");
  ret += (isUseless ? (comma.get() + "is-useless") : "");
  ret +=
      (replacedWithUnitFlag ? (comma.get() + "replaced-with-unit-flag") : "");
  ret += "}";

  ret += "\n";
  if (this->op1 != nullptr) {
    ret += this->op1->str(2, false) + '\n';
  }
  if (this->op2 != nullptr) {
    ret += this->op2->str(2, false) + '\n';
  }
  // ret += this->opSet->str(0, false) + '\n';
  // ret += this->opWait->str(0, false) + '\n';
  ret.pop_back();
  return ret;
}
} // namespace mlir::hivm::syncsolver
