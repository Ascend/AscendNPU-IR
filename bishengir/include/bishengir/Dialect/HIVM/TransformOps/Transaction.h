//===- Transaction.h - HIVM transform transactions --------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_TRANSACTION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_TRANSACTION_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ThreadPool.h"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace hivm {
using BytecodeBuffer = llvm::SmallVector<char, 0>;

struct Ticket {
  uint64_t id = 0;
  ModuleOp root;
  BytecodeBuffer snapshot, library, options, candidate, result;
  Attribute optionsAttr;
  TransactionOutcome outcome = TransactionOutcome::error;
  std::string diagnostics;
  std::shared_future<void> future;
  bool done = false, claimed = false;
  std::string entry;

  void wait() const {
    if (future.valid())
      future.wait();
  }
};

using TicketRef = std::reference_wrapper<Ticket>;

struct CallbackCapture {
  TransactionOutcome outcome = TransactionOutcome::error;
  std::string diagnostics;
  BytecodeBuffer result;
};

using CallbackCaptureRef = std::reference_wrapper<CallbackCapture>;

std::optional<CallbackCaptureRef> getActiveCallbackCapture();

LogicalResult invokeCallback(transform::TransformState &state,
                             transform::NamedSequenceOp callee,
                             ValueRange operands);

struct CostOutcome {
  enum class Status { succeeded, rejected, error };
  Status status = Status::error;
  double cost = 0.0;
  std::string diagnostics;
};

FailureOr<CostOutcome> evaluateCost(transform::TransformState &state,
                                    transform::NamedSequenceOp cost,
                                    ModuleOp candidate, Attribute resultAttr,
                                    std::string &error);

LogicalResult serializeOperation(Operation *op, BytecodeBuffer &out,
                                 std::string &error);
LogicalResult serializeAttribute(Attribute attr, BytecodeBuffer &out,
                                 std::string &error);
OwningOpRef<ModuleOp> parseModule(ArrayRef<char> bytes, MLIRContext *context);
Attribute parseAttribute(ArrayRef<char> bytes, MLIRContext *context);

class TransactionSession : public transform::TransformState::Extension {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransactionSession)

  explicit TransactionSession(transform::TransformState &state);
  ~TransactionSession() override;

  TransactionSession(const TransactionSession &) = delete;
  TransactionSession &operator=(const TransactionSession &) = delete;

  static std::optional<std::reference_wrapper<TransactionSession>>
  get(transform::TransformState &state);
  static TransactionSession &getOrCreate(transform::TransformState &state);

  uint64_t getId() const { return id; }

  FailureOr<uint64_t> submit(ModuleOp root, Attribute options,
                             transform::NamedSequenceOp attempt, bool sync,
                             std::string &error);

  std::optional<TicketRef> getTicket(uint64_t ticket);

  void dropDescendantMappings(Operation *root);

private:
  void runWorker(Ticket &ticket);
  void initializeTicket(Ticket &ticket, ModuleOp root, Attribute options,
                        transform::NamedSequenceOp attempt);

  LogicalResult getLibrary(transform::NamedSequenceOp attempt,
                           BytecodeBuffer &out, std::string &error);

  uint64_t id;
  llvm::SmallVector<std::unique_ptr<Ticket>> tickets;

  DialectRegistry registry;
  llvm::DefaultThreadPool pool;
};

} // namespace hivm
} // namespace mlir

#endif
