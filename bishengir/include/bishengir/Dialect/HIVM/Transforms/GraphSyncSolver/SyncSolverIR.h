//===------------- SyncSolverIR.h ---- Graph Sync Solver ------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/UnitFlagInfoBase.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>

namespace mlir::hivm::syncsolver {

class OperationBase;
class Scope;
class Loop;
class Condition;
class RWOperation;
class MmadL0Operation;
using Body = std::vector<std::unique_ptr<OperationBase>>;

struct EventIdInfo {
  int64_t eventIdNum{0};
  LoopLikeOpInterface multibufferLoop{nullptr};
  LoopLikeOpInterface multibufferUnrollLoop1{nullptr};
  LoopLikeOpInterface multibufferUnrollLoop2{nullptr};
  EventIdInfo() {};
  explicit EventIdInfo(int64_t eventIdNum) : eventIdNum(eventIdNum) {};
};

enum struct OpType {
  OPERATION,
  PLACE_HOLDER,
  SCOPE,
  FUNCTION,
  FUNCTION_BLOCK,
  LOOP,
  LOOP_END,
  MMAD_SCOPE,
  CONDITION,
  SCOPE_END,
  SYNC_OP,
  BARRIER_OP,
  SW_FLAG_OP,
  SET_FLAG_OP,
  WAIT_FLAG_OP,
  SW_FLAG_OP_END,
  SYNC_OP_END,
  RW_OPERATION,
  MMAD_OPERATION,
  MMAD_LOAD_L0A_OPERATION,
  MMAD_LOAD_L0B_OPERATION,
  MMAD_LOAD_BIAS_OPERATION,
  RW_OPERATION_END
};

std::string getOpTypeStr(OpType opType);

class OperationBase {

public:
  int id{-1};
  const OpType opType;
  mlir::Operation *op{nullptr};
  OperationBase *parentOp{nullptr};

private:
  // Monotonic id allocator used to assign stable ids to in-memory ops.
  static int globalIndex;

public:
  OperationBase() = delete;
  OperationBase(const OpType &opType, Operation *op, OperationBase *parentOp)
      : opType(opType), op(op), parentOp(parentOp) {
    id = globalIndex++;
  }

public:
  virtual ~OperationBase() = default;

  // Return true when op1 and op2 share the same immediate parent operation.
  static bool sameScope(OperationBase *op1, OperationBase *op2);

  // Compute the depth (levels up to root) of the provided operation.
  int getDepth() const;

  // Return the ancestor `dist` levels above this operation.
  OperationBase *getNthParent(int dist);

  // Given two operations, return the pair of operations directly below their
  // LCA.
  static std::pair<OperationBase *, OperationBase *>
  getLCAPair(OperationBase *op1, OperationBase *op2);

  template <typename TyOp> TyOp *getParentOfType() {
    OperationBase *cur = this->parentOp;
    while (cur != nullptr && !isa<TyOp>(cur)) {
      cur = cur->parentOp;
    }
    return llvm::dyn_cast_if_present<TyOp>(cur);
  }

  // Find nearest parent operation that is a loop-like construct, or nullptr.
  static OperationBase *getParentloop(OperationBase *op);

  // Find the nearest parent condition operation, or nullptr.
  static OperationBase *getParentCondition(OperationBase *op);

  // Return true if this operation is a strict ancestor of `op`.
  bool isProperAncestor(OperationBase *op);

  // Collect and return all parent operations (walking upwards).
  llvm::SmallVector<OperationBase *> getAllParents();

  // Human-readable string representation (override in derived classes).
  virtual std::string str(int indent = 0, bool recursive = false) const = 0;

  static OperationBase *getUnlikelyParentCondition(OperationBase *op);
};

class PlaceHolder : public OperationBase {
public:
  mlir::Block *block{nullptr};
  OperationBase *beforeOp{nullptr};
  OperationBase *afterOp{nullptr};
  Scope *scopeBegin{nullptr};
  Scope *scopeEnd{nullptr};

public:
  PlaceHolder(Operation *op, OperationBase *parentOp)
      : OperationBase(OpType::PLACE_HOLDER, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::PLACE_HOLDER;
  }

  std::string str(int indent, bool recursive) const override;
};

class Scope : public OperationBase {

public:
  Body body;

public:
  Scope(const OpType &opType = OpType::SCOPE, Operation *op = nullptr,
        OperationBase *parentOp = nullptr)
      : OperationBase(opType, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SCOPE && e->opType < OpType::SCOPE_END;
  }

  std::string str(int indent, bool recursive) const override;
};

class FunctionBlock : public Scope {
public:
  FunctionBlock() : Scope(OpType::FUNCTION_BLOCK) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION_BLOCK;
  }
};

class Function : public Scope {
public:
  Function(Operation *op) : Scope(OpType::FUNCTION, op, nullptr) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION;
  }
};

class Loop : public Scope {

private:
public:
  bool isParallel{false};
  std::optional<int64_t> multibufferUnrollNum;
  Loop(Operation *op, OperationBase *parentOp)
      : Scope(OpType::LOOP, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::LOOP && e->opType < OpType::LOOP_END;
  }
};

class MmadL1LoopOp : public Scope {
private:
public:
  MmadL0Operation *mmadL0Op{nullptr};

  MmadL1LoopOp(Operation *op, OperationBase *parentOp)
      : Scope(OpType::MMAD_SCOPE, op, parentOp) {};

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_SCOPE;
  }
};

class Condition : public Scope {

private:
public:
  Scope *trueScope{nullptr};
  Scope *falseScope{nullptr};
  bool isUnlikely{false};
  Condition(Operation *op, OperationBase *parentOp,
            std::unique_ptr<Scope> trueScope, std::unique_ptr<Scope> falseScope)
      : Scope(OpType::CONDITION, op, parentOp) {
    if (trueScope != nullptr) {
      this->setTrueScope(std::move(trueScope));
    }
    if (falseScope != nullptr) {
      assert(this->trueScope != nullptr);
      this->setFalseScope(std::move(falseScope));
    }
  };

  Scope *getTrueScope() const {
    assert(this->trueScope != nullptr);
    return this->trueScope;
  }

  Scope *getFalseScope() const {
    assert(this->falseScope != nullptr);
    return this->falseScope;
  }

  void setFalseScope(std::unique_ptr<Scope> falseScope) {
    assert(falseScope != nullptr);
    falseScope->parentOp = this;
    this->falseScope = falseScope.get();
    this->body.push_back(std::move(falseScope));
  }

  void setTrueScope(std::unique_ptr<Scope> trueScope) {
    assert(trueScope != nullptr);
    trueScope->parentOp = this;
    this->trueScope = trueScope.get();
    this->body.push_back(std::move(trueScope));
  }

  bool hasFalseScope() const { return this->falseScope != nullptr; }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::CONDITION;
  }

  std::string str(int indent, bool recursive) const override;
};

class RWOperation : public OperationBase {
public:
  hivm::TCoreType coreType{hivm::TCoreType::CUBE_OR_VECTOR};
  hivm::PIPE pipeRead{hivm::PIPE::PIPE_UNASSIGNED};
  hivm::PIPE pipeWrite{hivm::PIPE::PIPE_UNASSIGNED};
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  llvm::SmallVector<llvm::SmallVector<int>> testReadMemVals;
  llvm::SmallVector<llvm::SmallVector<int>> testWriteMemVals;
  bool hasUnitFlagFeat{false};
  UnitFlagInfoBase mergedUnitFlagInfo;

private:
public:
  RWOperation(Operation *op, OperationBase *parentOp, hivm::TCoreType coreType,
              hivm::PIPE pipeRead, hivm::PIPE pipeWrite,
              const llvm::SmallVector<Value> &readMemVals,
              const llvm::SmallVector<Value> &writeMemVals,
              OpType opType = OpType::RW_OPERATION)
      : OperationBase(opType, op, parentOp), coreType(coreType),
        pipeRead(pipeRead), pipeWrite(pipeWrite), readMemVals(readMemVals),
        writeMemVals(writeMemVals) {};

  std::string str(int indent, bool recursive) const override;

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::RW_OPERATION &&
           e->opType < OpType::RW_OPERATION_END;
  }
};

class LoadL0AOp : public RWOperation {
private:
public:
  LoadL0AOp(Operation *op, OperationBase *parentOp, hivm::TCoreType coreType,
            hivm::PIPE pipeRead, hivm::PIPE pipeWrite,
            const llvm::SmallVector<Value> &readMemVals,
            const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_L0A_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_L0A_OPERATION;
  }
};

class LoadL0BOp : public RWOperation {
private:
public:
  LoadL0BOp(Operation *op, OperationBase *parentOp, hivm::TCoreType coreType,
            hivm::PIPE pipeRead, hivm::PIPE pipeWrite,
            const llvm::SmallVector<Value> &readMemVals,
            const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_L0B_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_L0B_OPERATION;
  }
};

class LoadBiasOp : public RWOperation {
private:
public:
  LoadBiasOp(Operation *op, OperationBase *parentOp, hivm::TCoreType coreType,
             hivm::PIPE pipeRead, hivm::PIPE pipeWrite,
             const llvm::SmallVector<Value> &readMemVals,
             const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_BIAS_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_BIAS_OPERATION;
  }
};

class MmadL0Operation : public RWOperation {
private:
public:
  MmadL0Operation(Operation *op, OperationBase *parentOp,
                  hivm::TCoreType coreType, hivm::PIPE pipeRead,
                  hivm::PIPE pipeWrite,
                  const llvm::SmallVector<Value> &readMemVals,
                  const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_OPERATION;
  }
};

class SyncOp : public OperationBase {
public:
  std::optional<int> debugId;
  SyncOp(const OpType &opType, Operation *op, OperationBase *parentOp)
      : OperationBase(opType, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SYNC_OP && e->opType < OpType::SYNC_OP_END;
  }
};

class SetWaitOp : public SyncOp {
public:
  llvm::SmallVector<int64_t> eventIds;
  hivm::TCoreType coreType{hivm::TCoreType::CUBE_OR_VECTOR};
  hivm::PIPE pipeSrc{hivm::PIPE::PIPE_UNASSIGNED};
  hivm::PIPE pipeDst{hivm::PIPE::PIPE_UNASSIGNED};
  EventIdInfo eventIdInfo;
  bool allAtOnce{false};
  bool checkFirstIter{false};
  bool checkLastIter{false};

  SetWaitOp(const OpType &opType, Operation *op, OperationBase *parentOp,
            const llvm::SmallVector<int64_t> &eventIds, hivm::PIPE pipeSrc,
            hivm::PIPE pipeDst)
      : SyncOp(opType, op, parentOp), eventIds(eventIds), pipeSrc(pipeSrc),
        pipeDst(pipeDst) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SW_FLAG_OP &&
           e->opType < OpType::SW_FLAG_OP_END;
  }
};

class SetFlagOp : public SetWaitOp {

private:
public:
  SetFlagOp(Operation *op, OperationBase *parentOp,
            const llvm::SmallVector<int64_t> &eventIds, hivm::PIPE pipeSrc,
            hivm::PIPE pipeDst)
      : SetWaitOp(OpType::SET_FLAG_OP, op, parentOp, eventIds, pipeSrc,
                  pipeDst) {}

  std::unique_ptr<SetFlagOp> clone() {
    return std::make_unique<SetFlagOp>(op, parentOp, eventIds, pipeSrc,
                                       pipeDst);
  }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::SET_FLAG_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

class WaitFlagOp : public SetWaitOp {

private:
public:
  WaitFlagOp(Operation *op, OperationBase *parentOp,
             const llvm::SmallVector<int64_t> &eventIds, hivm::PIPE pipeSrc,
             hivm::PIPE pipeDst)
      : SetWaitOp(OpType::WAIT_FLAG_OP, op, parentOp, eventIds, pipeSrc,
                  pipeDst) {}

  std::unique_ptr<WaitFlagOp> clone() {
    return std::make_unique<WaitFlagOp>(op, parentOp, eventIds, pipeSrc,
                                        pipeDst);
  }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::WAIT_FLAG_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

class BarrierOp : public SyncOp {

public:
  hivm::PIPE pipe{hivm::PIPE::PIPE_UNASSIGNED};

private:
public:
  BarrierOp(Operation *op, OperationBase *parentOp, hivm::PIPE pipe)
      : SyncOp(OpType::BARRIER_OP, op, parentOp), pipe(pipe) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::BARRIER_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

// Bool comparator for sync ops ordering (used for containers).
bool operator<(const SyncOp &op1, const SyncOp &op2);
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
