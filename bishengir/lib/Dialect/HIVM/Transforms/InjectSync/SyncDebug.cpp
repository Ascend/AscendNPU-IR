//===------------- SyncDebug.ccp ----Provide print syncIR ------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {
struct Comma {
  bool comma = false;
  std::string get() {
    std::string ret = comma ? ", " : "";
    comma = true;
    return ret;
  }
};
} // namespace mlir::hivm

void SyncDebug::PrintSyncIr() {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);

  os << "-----------------------------syncIR-----------------------------"
     << "\n";
  os << "Num : " << syncIR.size() << "\n";
  for (auto &e : syncIR) {
    PrintInstanceElement(e.get(), os);
  }
  os << "\n";

  llvm::dbgs() << os.str();
}

void SyncDebug::PrintInstanceElement(const InstanceElement *e,
                                     raw_ostream &os) {
  PrintSyncOperation(e->pipeBefore, os);
  if (LoopInstanceElement::classof(e)) {
    PrintForIR(e, os);
  } else if (CompoundInstanceElement::classof(e)) {
    PrintCompoundIR(e, os);
  } else if (BranchInstanceElement::classof(e)) {
    PrintBranchIR(e, os);
  }
  PrintSyncOperation(e->pipeAfter, os);
}

void SyncDebug::PrintSyncOperation(const SyncOps &pipeType, raw_ostream &os) {
  for (auto &s : pipeType) {
    PrintSync(s, os);
  }
}

void SyncDebug::PrintSync(const SyncOperation *s, raw_ostream &os) {
  PrintIndent(os);
  if (s->isBarrierType()) {
    PrintPipeBarrierSync(s, os);
  } else if (s->GetType() == SyncOperation::TYPE::SET_EVENT) {
    PrintSetWaitSync(s, os);
  } else if (s->GetType() == SyncOperation::TYPE::WAIT_EVENT) {
    PrintSetWaitSync(s, os);
  } else if (s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET) {
    PrintSetWaitSync(s, os);
  } else if (s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    PrintSetWaitSync(s, os);
  } else if (s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
    PrintSetWaitSync(s, os);
  }
}

void SyncDebug::PrintIndent(raw_ostream &os) {
  for (unsigned i = 0; i < depth; i++) {
    os << " ";
  }
}

void SyncDebug::PrintPipeBarrierSync(const SyncOperation *s, raw_ostream &os) {
  os << SyncOperation::TypeName(s->GetType()) << "[" << s->GetSyncIndex() << "]"
     << "(" << PrintSyncType(s->GetSrcPipe()) << ") <";
  os << (s->GetForEndIndex() ? "-" + std::to_string(*s->GetForEndIndex()) : "");
  os << "> \tID: " << s->GetSyncIRIndex();
  if (s->GetDepSyncIRIndex() != 0) {
    os << "> \tID: " << s->GetDepSyncIRIndex();
  }
  os << (s->uselessSync ? " useless\n" : "\n");
}

std::string SyncDebug::PrintSyncType(hivm::PIPE type) const {
  auto name = kPipeToName.find(type);
  if (name != kPipeToName.end()) {
    return name->second;
  }
  return "";
}

void SyncDebug::PrintSetWaitSync(const SyncOperation *s, raw_ostream &os) {
  os << SyncOperation::TypeName(s->GetType()) << "[";
  os << s->GetSyncIndex() << "]";
  os << (s->GetForEndIndex() ? "-" + std::to_string(*s->GetForEndIndex()) : "");
  os << "(" << PrintSyncType(s->GetSrcPipe()) << ", "
     << PrintSyncType(s->GetDstPipe()) << ", ";
  for (auto eventID : s->eventIds) {
    os << eventID << ", ";
  }
  os << ") <";
  if (s->syncCoreType != TCoreType::CUBE_OR_VECTOR) {
    os << s->GetCoreTypeName(s->syncCoreType);
  }
  os << "> \tID: " << s->GetSyncIRIndex();
  os << " eventIdNum: " << s->eventIdNum;
  os << (s->uselessSync ? " useless\n" : "\n");
}

void SyncDebug::PrintCompoundIR(const InstanceElement *e, raw_ostream &os) {
  PrintIndent(os);
  assert(CompoundInstanceElement::classof(e));
  const auto *ptr = static_cast<const CompoundInstanceElement *>(e);
  assert(ptr != nullptr);
  os << "// "
     << "\tID: " << ptr->GetIndex() << "\n";
  PrintIndent(os);
  os << ptr->opName << " (";
  os << "defVec: ";
  for (const auto *memInfo : ptr->defVec) {
    os << "((" << memInfo->baseBuffer << ")"
       << " > "
       << " [ ";
    for (auto address : memInfo->baseAddresses) {
      os << address << " , ";
    }
    os << " | "
       << " , " << memInfo->allocateSize << "] ), ";
  }
  os << "useVec: ";
  for (const auto *memInfo : ptr->useVec) {
    os << "((";
    if (auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(
            memInfo->baseBuffer.getDefiningOp())) {
      forOp->print(os, mlir::OpPrintingFlags().skipRegions(true));
    } else if (auto whileOp = llvm::dyn_cast_if_present<scf::WhileOp>(
                   memInfo->baseBuffer.getDefiningOp())) {
      whileOp->print(os, mlir::OpPrintingFlags().skipRegions(true));
    } else if (auto ifOp = llvm::dyn_cast_if_present<scf::IfOp>(
                   memInfo->baseBuffer.getDefiningOp())) {
      ifOp->print(os, mlir::OpPrintingFlags().skipRegions(true));
    } else {
      os << memInfo->baseBuffer;
    }
    os << ")"
       << " > "
       << " [ ";
    for (auto address : memInfo->baseAddresses) {
      os << address << " , ";
    }
    os << " | "
       << " , " << memInfo->allocateSize << "] ), ";
  }

  if (ptr->compoundCoreType != TCoreType::CUBE_OR_VECTOR) {
    auto name = kCoreTypeToName.find(ptr->compoundCoreType);
    if (name != kCoreTypeToName.end()) {
      os << " , ----> " << name->second << "";
    }
  }
  os << ") ";

  printUnitFlag(ptr, os);
  os << "\n";
}

void SyncDebug::printUnitFlag(const CompoundInstanceElement *e,
                              raw_ostream &os) const {
  if (e->getUnitFlagMode() != UNIT_FLAG::DISABLED) {
    Comma comma;
    os << "unitFlag(";
    if (e->unitFlagModeAsSet == UNIT_FLAG::ENABLED_WITH_UPDATE) {
      os << comma.get() << "as-set";
    } else if (e->unitFlagModeAsSet == UNIT_FLAG::ENABLED_ONLY_LAST_ITER) {
      os << comma.get() << "as-set-only-last-iter";
    } else if (e->unitFlagModeAsSet == UNIT_FLAG::ENABLED_ONLY_FIRST_ITER) {
      os << comma.get() << "as-set-only-first-iter";
    }
    if (e->unitFlagModeAsWait == UNIT_FLAG::ENABLED_WITH_UPDATE) {
      os << comma.get() << "as-wait";
    } else if (e->unitFlagModeAsWait == UNIT_FLAG::ENABLED_ONLY_LAST_ITER) {
      os << comma.get() << "as-wait-only-last-iter";
    } else if (e->unitFlagModeAsWait == UNIT_FLAG::ENABLED_ONLY_FIRST_ITER) {
      os << comma.get() << "as-wait-only-first-iter";
    }
    os << ") ";
  }
}

void SyncDebug::PrintForIR(const InstanceElement *e, raw_ostream &os) {
  assert(LoopInstanceElement::classof(e));
  auto *ptr = dyn_cast<const LoopInstanceElement>(e);
  assert(ptr != nullptr);
  if (ptr->getLoopKind() == KindOfLoop::LOOP_BEGIN) {
    PrintIndent(os);
    depth += kTabSize;
    if (auto forOp = dyn_cast<scf::ForOp>(e->elementOp)) {
      os << "for () { ID: " << ptr->beginId << "\n";
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(e->elementOp)) {
      os << "while () { ID: " << ptr->beginId << "\n";
    } else {
      os << "loop () { ID: " << ptr->beginId << "\n";
    }
  } else {
    depth -= kTabSize;
    PrintIndent(os);
    os << "} ID: " << ptr->endId << "\n";
  }
}
void SyncDebug::PrintBranchIR(const InstanceElement *e, raw_ostream &os) {
  assert(BranchInstanceElement::classof(e));
  auto *ptr = dyn_cast<const BranchInstanceElement>(e);
  assert(ptr != nullptr);
  switch (ptr->getBranchKind()) {
  case KindOfBranch::IF_BEGIN: {
    PrintIndent(os);
    os << "if () {    //" << ptr->beginId << " | " << ptr->branchId << " | "
       << ptr->endId << " ID: " << ptr->GetIndex() << "\n";
    depth += kTabSize;
    break;
  }
  case KindOfBranch::ELSE_BEGIN: {
    depth -= kTabSize;
    PrintIndent(os);
    depth += kTabSize;
    os << "} else {"
       << " ID: " << ptr->GetIndex() << "\n";
    break;
  }
  case KindOfBranch::IF_END: {
    depth -= kTabSize;
    PrintIndent(os);
    os << "}\n";
    break;
  }
  }
}