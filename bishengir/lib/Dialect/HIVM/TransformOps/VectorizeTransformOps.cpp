//===- VectorizeTransformOps.cpp - Vectorizer transforms ------------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/Transaction.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cmath>
#include <optional>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
NamedSequenceOp lookupCallee(HIVMIncludeOp include) {
  return SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
      include, include.getTargetAttr());
}

enum class MappingKind { Unknown, Handle, ValueHandle, Parameter };

MappingKind getMappingKind(Type type) {
  if (isa<TransformHandleTypeInterface>(type))
    return MappingKind::Handle;
  if (isa<TransformValueHandleTypeInterface>(type))
    return MappingKind::ValueHandle;
  if (isa<TransformParamTypeInterface>(type))
    return MappingKind::Parameter;
  return MappingKind::Unknown;
}

LogicalResult verifyMappingCategories(HIVMIncludeOp include, TypeRange actual,
                                      TypeRange expected, StringRef kind) {
  for (auto [index, actualType, expectedType] :
       llvm::enumerate(actual, expected)) {
    MappingKind actualKind = getMappingKind(actualType);
    if (actualKind == MappingKind::Unknown ||
        actualKind != getMappingKind(expectedType))
      return include.emitOpError()
             << kind << " #" << index << " does not match the expected type";
  }
  return success();
}

LogicalResult verifyCallArity(HIVMIncludeOp include, FunctionType type) {
  if (include.getNumOperands() != type.getNumInputs())
    return include.emitOpError()
           << "number of operands (" << include.getNumOperands()
           << ") does not match the number of inputs (" << type.getNumInputs()
           << ")";
  if (include.getNumResults() != type.getNumResults())
    return include.emitOpError()
           << "number of results (" << include.getNumResults()
           << ") does not match the number of outputs (" << type.getNumResults()
           << ")";
  return success();
}

LogicalResult verifyIncludeGraph(NamedSequenceOp callee,
                                 llvm::SmallPtrSetImpl<Operation *> &active,
                                 llvm::SmallPtrSetImpl<Operation *> &done);

LogicalResult verifyIncludeBody(NamedSequenceOp callee,
                                llvm::SmallPtrSetImpl<Operation *> &active,
                                llvm::SmallPtrSetImpl<Operation *> &done) {
  for (auto &op : callee.getBody().front().without_terminator())
    if (!isa<TransformOpInterface>(&op))
      return op.emitOpError("expected an executable Transform operation");
  auto walk = callee.getBody().walk([&](Operation *op) {
    auto include = dyn_cast<HIVMIncludeOp>(op);
    if (!include)
      return WalkResult::advance();
    if (failed(include.verifyCall()) ||
        failed(verifyIncludeGraph(lookupCallee(include), active, done)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

LogicalResult verifyIncludeGraph(NamedSequenceOp callee,
                                 llvm::SmallPtrSetImpl<Operation *> &active,
                                 llvm::SmallPtrSetImpl<Operation *> &done) {
  if (done.contains(callee))
    return success();
  if (!active.insert(callee).second)
    return callee.emitOpError("recursive HIVM include");
  if (callee.isExternal())
    return callee.emitOpError("unresolved HIVM include");
  if (failed(verifyIncludeBody(callee, active, done)))
    return failure();
  active.erase(callee);
  done.insert(callee);
  return success();
}

SmallVector<SmallVector<MappedValue>>
prepareOperandMappings(HIVMIncludeOp include, TransformState &state) {
  SmallVector<SmallVector<MappedValue>> mappings;
  transform::detail::prepareValueMappings(mappings, include.getOperands(),
                                          state);
  return mappings;
}

LogicalResult mapOperandMappings(NamedSequenceOp callee,
                                 ArrayRef<SmallVector<MappedValue>> mappings,
                                 TransformState &state) {
  return state.mapBlockArguments(callee.getBody().front().getArguments(),
                                 mappings);
}

DiagnosedSilenceableFailure runCallee(NamedSequenceOp callee,
                                      TransformState &state) {
  for (auto &op : callee.getBody().front().without_terminator()) {
    auto transform = dyn_cast<TransformOpInterface>(op);
    if (!transform) {
      op.emitOpError("expected an executable Transform operation");
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    auto outcome = state.applyTransform(transform);
    if (!outcome.succeeded())
      return outcome;
  }
  return DiagnosedSilenceableFailure::success();
}

void forwardCalleeResults(HIVMIncludeOp include, NamedSequenceOp callee,
                          TransformResults &results, TransformState &state) {
  SmallVector<SmallVector<MappedValue>> mappings;
  transform::detail::prepareValueMappings(
      mappings, callee.getBody().front().getTerminator()->getOperands(), state);
  for (auto [result, mapping] : llvm::zip_equal(include.getResults(), mappings))
    results.setMappedValues(result, mapping);
}

bool isConsumedArgument(NamedSequenceOp callee, size_t index) {
  if (!callee || index >= callee.getFunctionType().getNumInputs())
    return false;
  return callee.getArgAttr(static_cast<unsigned>(index),
                           TransformDialect::kArgConsumedAttrName) != nullptr;
}

NamedSequenceOp lookupSequence(Operation *op, SymbolRefAttr ref) {
  return SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(op, ref);
}

bool isTransactionFamily(Operation *op) {
  return isa<HIVMTransactionOp, HIVMTransactionAwaitOp, HIVMTransactionSelectOp,
             HIVMTransactionExecuteOp>(op);
}

void captureDiagnosticText(std::string &out, Diagnostic &diag) {
  llvm::raw_string_ostream stream(out);
  stream << diag.str();
  for (const Diagnostic &note : diag.getNotes())
    stream << "\nnote: " << note.str();
  stream << "\n";
}

LogicalResult verifyCallbackSignature(Operation *user, NamedSequenceOp callee,
                                      StringRef kind) {
  if (callee.isExternal())
    return user->emitOpError() << kind << " must not be external";
  auto type = callee.getFunctionType();
  auto isExpectedType = type.getNumInputs() == 2 && type.getNumResults() == 1 &&
                        isa<TransformHandleTypeInterface>(type.getInput(0)) &&
                        isa<TransformParamTypeInterface>(type.getInput(1)) &&
                        isa<TransformParamTypeInterface>(type.getResult(0));
  if (!isExpectedType)
    return user->emitOpError()
           << kind
           << " must have type (!transform.any_op, !transform.any_param) -> "
              "!transform.any_param";
  for (unsigned index = 0; index < 2; ++index)
    if (callee.getArgAttr(index, TransformDialect::kArgConsumedAttrName))
      return user->emitOpError()
             << kind << " argument #" << index << " must not be consumed";
  return success();
}

LogicalResult verifyNoTransactionFamily(NamedSequenceOp callee) {
  auto walk = callee.getBody().walk([&](Operation *op) {
    if (!isTransactionFamily(op))
      return WalkResult::advance();
    op->emitOpError()
        << "transaction-family operation is not reachable from a transaction "
           "callback";
    return WalkResult::interrupt();
  });
  return walk.wasInterrupted() ? failure() : success();
}

LogicalResult verifyCallbackBody(NamedSequenceOp callee,
                                 llvm::SmallPtrSetImpl<Operation *> &active,
                                 llvm::SmallPtrSetImpl<Operation *> &done);

LogicalResult verifyIncludedCallback(Operation &op,
                                     llvm::SmallPtrSetImpl<Operation *> &active,
                                     llvm::SmallPtrSetImpl<Operation *> &done) {
  SymbolRefAttr target;
  if (auto include = dyn_cast<HIVMIncludeOp>(&op))
    target = include.getTargetAttr();
  else if (auto include = dyn_cast<transform::IncludeOp>(&op))
    target = include.getTargetAttr();
  else
    return success();

  if (!isa<FlatSymbolRefAttr>(target))
    return op.emitOpError() << "expected a flat symbol reference";
  auto included = lookupSequence(&op, target);
  if (!included)
    return op.emitOpError() << "target symbol '" << target << "' not found";
  if (included.isExternal())
    return op.emitOpError() << "unresolved include target '" << target << "'";
  return verifyCallbackBody(included, active, done);
}

LogicalResult
verifyCallbackOperations(NamedSequenceOp callee,
                         llvm::SmallPtrSetImpl<Operation *> &active,
                         llvm::SmallPtrSetImpl<Operation *> &done) {
  for (auto &op : callee.getBody().front().without_terminator()) {
    if (!isa<TransformOpInterface>(&op))
      return op.emitOpError("expected an executable Transform operation");
    if (failed(verifyIncludedCallback(op, active, done)))
      return failure();
  }
  return success();
}

LogicalResult verifyCallbackBody(NamedSequenceOp callee,
                                 llvm::SmallPtrSetImpl<Operation *> &active,
                                 llvm::SmallPtrSetImpl<Operation *> &done) {
  if (done.contains(callee))
    return success();
  if (!active.insert(callee).second)
    return callee.emitOpError("recursive transaction callback");
  if (failed(verifyNoTransactionFamily(callee)) ||
      failed(verifyCallbackOperations(callee, active, done)))
    return failure();

  active.erase(callee);
  done.insert(callee);
  return success();
}

LogicalResult verifyCallback(Operation *user, SymbolRefAttr ref,
                             StringRef kind) {
  if (!isa<FlatSymbolRefAttr>(ref))
    return user->emitOpError() << "expected a flat symbol reference";
  auto callee = lookupSequence(user, ref);
  if (!callee)
    return user->emitOpError() << "target symbol '" << ref << "' not found";
  if (failed(verifyCallbackSignature(user, callee, kind)))
    return failure();
  llvm::SmallPtrSet<Operation *, 8> active, done;
  return verifyCallbackBody(callee, active, done);
}

LogicalResult verifyTransactionRoot(Operation *user, TransformState &state,
                                    Value rootHandle) {
  auto roots = state.getPayloadOps(rootHandle);
  auto it = roots.begin();
  if (it == roots.end())
    return user->emitOpError() << "expected exactly one payload root";
  auto root = dyn_cast<ModuleOp>(*it);
  ++it;
  if (it != roots.end())
    return user->emitOpError() << "expected exactly one payload root";
  if (!root)
    return user->emitOpError() << "expected a payload module root";
  if (root.getOperation() != state.getTopLevel())
    return user->emitOpError() << "payload root must be the top-level module";
  return success();
}

LogicalResult resolveTransactionRoot(Operation *user, TransformState &state,
                                     Value rootHandle, ModuleOp &root) {
  if (failed(verifyTransactionRoot(user, state, rootHandle)))
    return failure();
  root = cast<ModuleOp>(*state.getPayloadOps(rootHandle).begin());
  return success();
}

LogicalResult resolveCallbackLibrary(Operation *user, ModuleOp root,
                                     NamedSequenceOp callee,
                                     StringRef errorMessage) {
  auto library = callee->getParentOfType<ModuleOp>();
  if (!library || library.getOperation() == root.getOperation())
    return user->emitOpError() << errorMessage;
  return success();
}

LogicalResult resolveCallback(Operation *user, ModuleOp root, SymbolRefAttr ref,
                              NamedSequenceOp &callee) {
  callee = lookupSequence(user, ref);
  if (!callee)
    return user->emitOpError() << "target symbol '" << ref << "' not found";
  return resolveCallbackLibrary(
      user, root, callee,
      "transaction library must be separate from the payload module");
}

struct TransactionInputs {
  ModuleOp root;
  Attribute options;
  NamedSequenceOp attempt;
};

FailureOr<TransactionInputs>
resolveTransactionInputs(Operation *user, TransformState &state,
                         Value rootHandle, Value optionsHandle,
                         SymbolRefAttr attemptRef) {
  TransactionInputs inputs;
  if (failed(resolveTransactionRoot(user, state, rootHandle, inputs.root)))
    return failure();

  auto params = state.getParams(optionsHandle);
  if (params.size() != 1) {
    user->emitOpError() << "expected exactly one options attribute";
    return failure();
  }
  inputs.options = params.front();

  if (failed(resolveCallback(user, inputs.root, attemptRef, inputs.attempt)))
    return failure();
  return inputs;
}

OwningOpRef<ModuleOp> parseCandidate(ArrayRef<char> bytes, MLIRContext *context,
                                     std::string &error) {
  std::string diagnostics;
  OwningOpRef<ModuleOp> candidate;
  {
    ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
      captureDiagnosticText(diagnostics, diag);
      return success();
    });
    candidate = parseModule(bytes, context);
  }
  if (!candidate)
    error = diagnostics.empty() ? "invalid transaction candidate bytecode"
                                : diagnostics;
  return candidate;
}

LogicalResult verifyCandidate(ModuleOp candidate, std::string &error) {
  std::string diagnostics;
  {
    ScopedDiagnosticHandler handler(candidate.getContext(),
                                    [&](Diagnostic &diag) {
                                      captureDiagnosticText(diagnostics, diag);
                                      return success();
                                    });
    if (failed(verify(candidate))) {
      error = diagnostics.empty() ? "transaction candidate failed verification"
                                  : diagnostics;
      return failure();
    }
  }
  return success();
}

bool sameRootIdentity(ModuleOp candidate, ModuleOp root) {
  return candidate->getName() == root->getName() &&
         candidate.getSymNameAttr() == root.getSymNameAttr();
}

bool checkPublishable(const Ticket &ticket, ModuleOp root,
                      const BytecodeBuffer &liveBytes) {
  return ticket.outcome == TransactionOutcome::succeeded && !ticket.claimed &&
         ticket.root == root && liveBytes == ticket.snapshot;
}

TransactionVerdictAttr makeVerdict(MLIRContext *context, uint64_t ticket,
                                   TransactionVerdictState state,
                                   std::optional<double> cost,
                                   StringRef reason = {}) {
  auto costAttr = cost ? Builder(context).getF64FloatAttr(*cost) : FloatAttr();
  auto reasonAttr =
      reason.empty() ? StringAttr() : StringAttr::get(context, reason);
  return TransactionVerdictAttr::get(context, ticket, state, costAttr,
                                     reasonAttr);
}

struct TransactionCandidate {
  TicketRef ticket;
  TransactionResultAttr record;
};

FailureOr<TicketRef> validateCandidate(Operation *user,
                                       TransactionSession &session,
                                       ModuleOp root,
                                       TransactionResultAttr record,
                                       llvm::DenseSet<uint64_t> &seen) {
  if (record.getSession() != session.getId()) {
    user->emitOpError() << "transaction result belongs to another session";
    return failure();
  }
  auto resolved = session.getTicket(record.getTicket());
  if (!resolved) {
    user->emitOpError() << "unknown transaction ticket";
    return failure();
  }
  auto &ticket = resolved->get();
  if (ticket.claimed) {
    user->emitOpError() << "transaction ticket has been claimed";
    return failure();
  }
  if (ticket.root != root) {
    user->emitOpError()
        << "transaction result belongs to a different payload root";
    return failure();
  }

  ticket.wait();

  if (ticket.outcome != record.getOutcome()) {
    user->emitOpError() << "transaction result is inconsistent with its ticket";
    return failure();
  }
  if (!seen.insert(record.getTicket()).second) {
    user->emitOpError() << "duplicate transaction result";
    return failure();
  }
  return std::ref(ticket);
}

LogicalResult collectCandidates(Operation *user, TransformState &state,
                                ModuleOp root, ValueRange candidates,
                                SmallVectorImpl<TransactionCandidate> &out) {
  if (candidates.empty())
    return success();

  auto session = TransactionSession::get(state);
  if (!session)
    return user->emitOpError() << "no transaction session is active";
  auto &activeSession = session->get();

  llvm::DenseSet<uint64_t> seen;
  for (auto value : candidates) {
    auto params = state.getParams(value);
    if (params.size() != 1)
      return user->emitOpError()
             << "expected exactly one transaction result per candidate";
    auto record = dyn_cast<TransactionResultAttr>(params.front());
    if (!record)
      return user->emitOpError() << "expected a transaction result";

    auto ticket = validateCandidate(user, activeSession, root, record, seen);
    if (failed(ticket))
      return failure();
    out.push_back({*ticket, record});
  }
  return success();
}

struct CandidateEvaluation {
  TransactionVerdictAttr verdict;
  std::optional<double> cost;
  bool shouldWarn = false;
  std::string warning;
};

CandidateEvaluation makeVerdictEvaluation(MLIRContext *context, uint64_t ticket,
                                          TransactionVerdictState state,
                                          StringRef reason,
                                          bool shouldWarn = false) {
  CandidateEvaluation evaluation;
  evaluation.verdict = makeVerdict(context, ticket, state, {}, reason);
  evaluation.shouldWarn = shouldWarn;
  if (shouldWarn)
    evaluation.warning = "transaction ticket " + std::to_string(ticket) +
                         " failed: " + reason.str();
  return evaluation;
}

CandidateEvaluation makeCostCallbackFailure(MLIRContext *context,
                                            uint64_t ticket,
                                            const CostOutcome &outcome) {
  auto verdictState = outcome.status == CostOutcome::Status::rejected
                          ? TransactionVerdictState::rejected
                          : TransactionVerdictState::error;
  auto evaluation =
      makeVerdictEvaluation(context, ticket, verdictState, outcome.diagnostics);
  evaluation.shouldWarn = true;
  evaluation.warning = "cost callback failed for transaction ticket " +
                       std::to_string(ticket) + ": " + outcome.diagnostics;
  return evaluation;
}

FailureOr<CandidateEvaluation>
evaluateSuccessfulCandidate(Operation *user, TransformState &state,
                            ModuleOp root, NamedSequenceOp cost,
                            const TransactionCandidate &candidate) {
  auto &ticket = candidate.ticket.get();
  auto *context = user->getContext();
  std::string error;

  auto candidateModule = parseCandidate(ticket.candidate, context, error);
  if (!candidateModule)
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::invalid, error);
  if (failed(verifyCandidate(*candidateModule, error)))
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::invalid, error);
  if (!sameRootIdentity(*candidateModule, root))
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::invalid,
                                 "candidate root identity mismatch");

  auto resultAttr = parseAttribute(ticket.result, context);
  if (!resultAttr)
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::invalid,
                                 "invalid transaction result serialization");

  auto costOutcome =
      evaluateCost(state, cost, *candidateModule, resultAttr, error);
  if (failed(costOutcome)) {
    user->emitError() << error;
    return failure();
  }
  if (costOutcome->status != CostOutcome::Status::succeeded)
    return makeCostCallbackFailure(context, ticket.id, *costOutcome);

  auto value = costOutcome->cost;
  if (!std::isfinite(value) || value <= 0.0)
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::rejected,
                                 "cost must be finite and positive");

  CandidateEvaluation evaluation;
  evaluation.verdict =
      makeVerdict(context, ticket.id, TransactionVerdictState::eligible, value);
  evaluation.cost = value;
  return evaluation;
}

FailureOr<CandidateEvaluation>
evaluateCandidate(Operation *user, TransformState &state, ModuleOp root,
                  NamedSequenceOp cost, const TransactionCandidate &candidate,
                  const BytecodeBuffer &liveBytes) {
  auto &ticket = candidate.ticket.get();
  auto *context = user->getContext();
  if (ticket.outcome == TransactionOutcome::rejected)
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::rejected,
                                 ticket.diagnostics);
  if (ticket.outcome == TransactionOutcome::error)
    return makeVerdictEvaluation(context, ticket.id,
                                 TransactionVerdictState::error,
                                 ticket.diagnostics, true);
  if (liveBytes != ticket.snapshot)
    return makeVerdictEvaluation(
        context, ticket.id, TransactionVerdictState::invalid, "stale snapshot");
  return evaluateSuccessfulCandidate(user, state, root, cost, candidate);
}

struct RankedCandidate {
  TicketRef ticket;
  double cost;
  size_t verdictIndex;
};

void recordEvaluation(const CandidateEvaluation &evaluation, TicketRef ticket,
                      SmallVectorImpl<TransactionVerdictAttr> &verdicts,
                      SmallVectorImpl<std::string> &warnings,
                      std::optional<RankedCandidate> &best) {
  auto verdictIndex = verdicts.size();
  verdicts.push_back(evaluation.verdict);
  if (evaluation.shouldWarn)
    warnings.push_back(evaluation.warning);
  if (!evaluation.cost)
    return;
  if (!best || *evaluation.cost > best->cost)
    best = RankedCandidate{ticket, *evaluation.cost, verdictIndex};
}

LogicalResult rankCandidates(Operation *user, TransformState &state,
                             ModuleOp root, NamedSequenceOp cost,
                             ArrayRef<TransactionCandidate> candidates,
                             const BytecodeBuffer &liveBytes,
                             SmallVectorImpl<TransactionVerdictAttr> &verdicts,
                             std::optional<RankedCandidate> &best,
                             SmallVectorImpl<std::string> &warnings) {
  for (auto &candidate : candidates) {
    auto evaluation =
        evaluateCandidate(user, state, root, cost, candidate, liveBytes);
    if (failed(evaluation))
      return failure();
    recordEvaluation(*evaluation, candidate.ticket, verdicts, warnings, best);
  }
  return success();
}

LogicalResult publishCandidate(Operation *user, TransactionSession &session,
                               ModuleOp root, Ticket &ticket,
                               const BytecodeBuffer &liveBytes) {
  if (root->isAncestor(user))
    return user->emitOpError()
           << "transform op must not be nested in the payload module";
  if (!checkPublishable(ticket, root, liveBytes))
    return user->emitOpError()
           << "transaction candidate is no longer publishable";

  std::string error;
  auto candidate = parseCandidate(ticket.candidate, user->getContext(), error);
  if (!candidate)
    return user->emitOpError() << error;
  if (failed(verifyCandidate(*candidate, error)))
    return user->emitOpError() << error;
  if (!sameRootIdentity(*candidate, root))
    return user->emitOpError()
           << "transaction candidate root identity mismatch";

  session.dropDescendantMappings(root.getOperation());
  auto *candidateOp = candidate->getOperation();
  root->setAttrs(candidateOp->getAttrDictionary());
  root->copyProperties(candidateOp->getPropertiesStorage());
  root->setLoc(candidateOp->getLoc());
  root.getBodyRegion().takeBody(candidate->getBodyRegion());
  return success();
}

TransactionSelectionAttr
createSelection(MLIRContext *context, bool applied, uint64_t ticket,
                std::optional<double> cost,
                ArrayRef<TransactionVerdictAttr> verdicts) {
  SmallVector<Attribute> verdictAttrs;
  for (auto verdict : verdicts)
    verdictAttrs.push_back(verdict);
  auto costAttr = cost ? Builder(context).getF64FloatAttr(*cost) : FloatAttr();
  return TransactionSelectionAttr::get(
      context, BoolAttr::get(context, applied), applied ? ticket : 0, costAttr,
      ArrayAttr::get(context, verdictAttrs), StringAttr::get(context, ""));
}

void claimCandidates(MutableArrayRef<TransactionCandidate> candidates) {
  for (auto &candidate : candidates) {
    auto &ticket = candidate.ticket.get();
    ticket.claimed = true;
    ticket.candidate = BytecodeBuffer();
    ticket.result = BytecodeBuffer();
  }
}

void emitSelectionResult(Operation *user, bool applied, uint64_t winnerId,
                         double winnerCost, ArrayRef<std::string> warnings) {
  for (auto &warning : warnings)
    user->emitWarning() << warning;
  if (applied)
    user->emitRemark() << "selected transaction ticket " << winnerId
                       << " with cost " << winnerCost;
  else
    user->emitRemark() << "no transaction candidate published";
}

FailureOr<TicketRef> resolveAwaitTicket(TransformState &state,
                                        TransactionTicketAttr ticketAttr,
                                        std::string &error) {
  auto session = TransactionSession::get(state);
  if (!session || ticketAttr.getSession() != session->get().getId()) {
    error = "transaction ticket belongs to another session";
    return failure();
  }
  auto resolved = session->get().getTicket(ticketAttr.getTicket());
  if (!resolved) {
    error = "unknown transaction ticket";
    return failure();
  }
  auto &ticket = resolved->get();
  if (ticket.claimed) {
    error = "transaction ticket has been claimed";
    return failure();
  }
  ticket.wait();
  return *resolved;
}

FailureOr<TransactionResultAttr> buildResultRecord(MLIRContext *context,
                                                   TransactionTicketAttr attr,
                                                   Ticket &ticket) {
  Attribute result;
  if (ticket.outcome == TransactionOutcome::succeeded)
    result = parseAttribute(ticket.result, context);
  if (ticket.outcome == TransactionOutcome::succeeded && !result)
    return failure();
  return TransactionResultAttr::get(
      context, attr.getSession(), attr.getTicket(), ticket.outcome,
      ticket.optionsAttr, result, StringAttr::get(context, ticket.diagnostics));
}

} // namespace

DiagnosedSilenceableFailure HIVMIncludeOp::apply(TransformRewriter &,
                                                 TransformResults &results,
                                                 TransformState &state) {
  if (failed(verifyCall()))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto callee = lookupCallee(*this);
  if (!callee)
    return emitDefiniteFailure()
           << "target symbol '" << getTargetAttr() << "' not found";

  auto mappings = prepareOperandMappings(*this, state);
  auto scope = state.make_region_scope(callee.getBody());
  if (failed(mapOperandMappings(callee, mappings, state)))
    return DiagnosedSilenceableFailure::definiteFailure();
  if (auto outcome = runCallee(callee, state); !outcome.succeeded())
    return outcome;
  forwardCalleeResults(*this, callee, results, state);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult HIVMIncludeOp::verify() {
  if (failed(verifyCall()))
    return failure();
  auto callee = lookupCallee(*this);
  llvm::SmallPtrSet<Operation *, 8> active, done;
  return verifyIncludeGraph(callee, active, done);
}

LogicalResult HIVMIncludeOp::verifyCall() {
  auto callee = lookupCallee(*this);
  if (!callee)
    return emitOpError() << "target symbol '" << getTargetAttr()
                         << "' not found";
  if (callee.isExternal())
    return emitOpError() << "callee is external";

  auto functionType = callee.getFunctionType();
  if (failed(verifyCallArity(*this, functionType)))
    return failure();
  if (failed(verifyMappingCategories(*this, getOperandTypes(),
                                     functionType.getInputs(), "operand")))
    return failure();
  return verifyMappingCategories(*this, getResultTypes(),
                                 functionType.getResults(), "result");
}

void HIVMIncludeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto callee = lookupCallee(*this);
  auto *operation = getOperation();
  for (auto [index, operand] : llvm::enumerate(operation->getOpOperands())) {
    if (isConsumedArgument(callee, index))
      consumesHandle(operand, effects);
    else
      onlyReadsHandle(operand, effects);
  }
  producesHandle(operation->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult HIVMTransactionOp::verifyAttempt() {
  if (failed(verifyCallback(getOperation(), getAttemptAttr(), "attempt")))
    return failure();
  auto sync = getOperation()->getAttr("sync");
  if (sync && !isa<UnitAttr>(sync))
    return emitOpError() << "sync must be a unit attribute";
  return success();
}

LogicalResult HIVMTransactionOp::verify() { return verifyAttempt(); }

DiagnosedSilenceableFailure HIVMTransactionOp::apply(TransformRewriter &,
                                                     TransformResults &results,
                                                     TransformState &state) {
  auto inputs = resolveTransactionInputs(getOperation(), state, getRoot(),
                                         getOptions(), getAttemptAttr());
  if (failed(inputs))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto sync = getOperation()->getAttr("sync") != nullptr;
  std::string error;
  auto &session = TransactionSession::getOrCreate(state);
  auto ticket = session.submit(inputs->root, inputs->options, inputs->attempt,
                               sync, error);
  if (failed(ticket))
    return emitDefiniteFailure() << error;

  results.setParams(
      cast<OpResult>(getTicket()),
      {TransactionTicketAttr::get(getContext(), session.getId(), *ticket)});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
HIVMTransactionAwaitOp::apply(TransformRewriter &, TransformResults &results,
                              TransformState &state) {
  auto params = state.getParams(getTicket());
  if (params.size() != 1)
    return emitDefiniteFailure() << "expected exactly one transaction ticket";
  auto ticketAttr = dyn_cast<TransactionTicketAttr>(params.front());
  if (!ticketAttr)
    return emitDefiniteFailure() << "expected a transaction ticket";

  std::string error;
  auto resolved = resolveAwaitTicket(state, ticketAttr, error);
  if (failed(resolved))
    return emitDefiniteFailure() << error;

  auto record = buildResultRecord(getContext(), ticketAttr, resolved->get());
  if (failed(record))
    return emitDefiniteFailure() << "invalid transaction result serialization";
  results.setParams(cast<OpResult>(getResult()), {*record});
  return DiagnosedSilenceableFailure::success();
}

LogicalResult HIVMTransactionSelectOp::verifyCost() {
  return verifyCallback(getOperation(), getCostAttr(), "cost");
}

LogicalResult HIVMTransactionSelectOp::verify() { return verifyCost(); }

DiagnosedSilenceableFailure
HIVMTransactionSelectOp::apply(TransformRewriter &, TransformResults &results,
                               TransformState &state) {
  ModuleOp root;
  if (failed(resolveTransactionRoot(getOperation(), state, getRoot(), root)))
    return DiagnosedSilenceableFailure::definiteFailure();

  SmallVector<TransactionCandidate> candidates;
  if (failed(collectCandidates(getOperation(), state, root, getCandidates(),
                               candidates)))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto *context = getContext();
  if (candidates.empty()) {
    results.set(cast<OpResult>(getNewRoot()), {root.getOperation()});
    results.setParams(cast<OpResult>(getSelection()),
                      {createSelection(context, false, 0, std::nullopt, {})});
    emitRemark() << "no transaction candidate published";
    return DiagnosedSilenceableFailure::success();
  }

  NamedSequenceOp cost;
  if (failed(resolveCallback(getOperation(), root, getCostAttr(), cost)))
    return DiagnosedSilenceableFailure::definiteFailure();

  BytecodeBuffer liveBytes;
  std::string error;
  if (failed(serializeOperation(root.getOperation(), liveBytes, error)))
    return emitDefiniteFailure() << error;

  SmallVector<TransactionVerdictAttr> verdicts;
  SmallVector<std::string> warnings;
  std::optional<RankedCandidate> best;
  if (failed(rankCandidates(getOperation(), state, root, cost, candidates,
                            liveBytes, verdicts, best, warnings)))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto applied = false;
  auto winnerId = uint64_t{0};
  auto winnerCost = 0.0;
  if (best) {
    auto session = TransactionSession::get(state);
    auto &winner = best->ticket.get();
    if (failed(publishCandidate(getOperation(), session->get(), root, winner,
                                liveBytes)))
      return DiagnosedSilenceableFailure::definiteFailure();
    verdicts[best->verdictIndex] = makeVerdict(
        context, winner.id, TransactionVerdictState::applied, best->cost);
    applied = true;
    winnerId = winner.id;
    winnerCost = best->cost;
  }

  claimCandidates(candidates);
  auto selectionCost =
      applied ? std::optional<double>(winnerCost) : std::nullopt;
  results.set(cast<OpResult>(getNewRoot()), {root.getOperation()});
  results.setParams(
      cast<OpResult>(getSelection()),
      {createSelection(context, applied, winnerId, selectionCost, verdicts)});
  emitSelectionResult(getOperation(), applied, winnerId, winnerCost, warnings);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult HIVMTransactionExecuteOp::verifyAttempt() {
  return verifyCallback(getOperation(), getAttemptAttr(), "attempt");
}

LogicalResult HIVMTransactionExecuteOp::verify() { return verifyAttempt(); }

DiagnosedSilenceableFailure
HIVMTransactionExecuteOp::apply(TransformRewriter &, TransformResults &results,
                                TransformState &state) {
  if (!getActiveCallbackCapture())
    return emitDefiniteFailure() << "transaction.execute is internal";
  auto attempt = lookupSequence(getOperation(), getAttemptAttr());
  if (!attempt)
    return emitDefiniteFailure()
           << "target symbol '" << getAttemptAttr() << "' not found";
  if (failed(invokeCallback(state, attempt, getOperands())))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.setParams(cast<OpResult>(getResult()), {});
  return DiagnosedSilenceableFailure::success();
}

void HIVMTransactionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRootMutable(), effects);
  onlyReadsHandle(getOptionsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

void HIVMTransactionAwaitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTicketMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

void HIVMTransactionSelectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getRootMutable(), effects);
  onlyReadsHandle(getOperation()->getOpOperands().drop_front(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void HIVMTransactionExecuteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getOperation()->getOpOperands(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

namespace {
class HIVMVectorizeTransformDialectExtension
    : public transform::TransformDialectExtension<
          HIVMVectorizeTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<mlir::hivm::HIVMDialect>();
    declareDependentDialect<HIVMVectorizeDialect>();
    declareDependentDialect<scf::SCFDialect>();

    declareGeneratedDialect<mlir::hivm::HIVMDialect>();
    declareGeneratedDialect<scope::ScopeDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<math::MathDialect>();
    declareGeneratedDialect<mathExt::MathExtDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.cpp.inc"

void mlir::hivm::registerVectorizeTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<HIVMVectorizeTransformDialectExtension>();
}
