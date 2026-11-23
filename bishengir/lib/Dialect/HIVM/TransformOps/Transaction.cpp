//===- Transaction.cpp - HIVM transform transactions --------------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/Transaction.h"

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
constexpr StringLiteral kEnvelopeAttrName = "hivm.transaction.value";
constexpr StringLiteral kWorkerEntryName = "__hivm_transaction_entry";

std::atomic<uint64_t> nextSessionId{1};

thread_local std::optional<CallbackCaptureRef> activeCapture;

class CaptureScope {
public:
  explicit CaptureScope(CallbackCapture &capture) : previous(activeCapture) {
    activeCapture = std::ref(capture);
  }
  ~CaptureScope() { activeCapture = previous; }

  CaptureScope(const CaptureScope &) = delete;
  CaptureScope &operator=(const CaptureScope &) = delete;

private:
  std::optional<CallbackCaptureRef> previous;
};

void appendDiagnostic(std::string &out, Diagnostic &diag) {
  llvm::raw_string_ostream stream(out);
  stream << diag.str();
  for (const Diagnostic &note : diag.getNotes())
    stream << "\nnote: " << note.str();
  stream << "\n";
}

void appendDiagnostics(std::string &out, const std::string &more) {
  if (more.empty())
    return;
  if (!out.empty() && out.back() != '\n')
    out.push_back('\n');
  out.append(more);
}

std::optional<StringRef> findUnsupportedTransport(Operation *root) {
  std::optional<StringRef> unsupported;
  AttrTypeWalker walker;
  walker.addWalk([&](Attribute attr) {
    if (unsupported)
      return;
    if (isa<DenseResourceElementsAttr>(attr))
      unsupported = "dense resource attributes";
    else if (isa<DistinctAttr>(attr))
      unsupported = "distinct attributes";
    else if (isa<OpaqueLoc>(attr))
      unsupported = "opaque locations";
  });

  root->walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    if (auto properties = op->getPropertiesAsAttribute())
      walker.walk(properties);
    walker.walk(op->getLoc());
    for (auto type : op->getOperandTypes())
      walker.walk(type);
    for (auto type : op->getResultTypes())
      walker.walk(type);
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto argument : block.getArguments()) {
          walker.walk(argument.getType());
          walker.walk(argument.getLoc());
        }
  });
  return unsupported;
}

LogicalResult checkSerializable(Operation *root, std::string &error) {
  auto unsupported = findUnsupportedTransport(root);
  if (!unsupported)
    return success();
  error = "transaction serialization does not support " + unsupported->str();
  return failure();
}

SymbolRefAttr getIncludeTarget(Operation *op) {
  if (auto include = dyn_cast<HIVMIncludeOp>(op))
    return include.getTargetAttr();
  if (auto include = dyn_cast<IncludeOp>(op))
    return include.getTargetAttr();
  return {};
}

LogicalResult collectClosure(NamedSequenceOp sequence,
                             SmallVectorImpl<NamedSequenceOp> &closure,
                             SmallPtrSetImpl<Operation *> &visited,
                             SmallPtrSetImpl<Operation *> &active,
                             std::string &error);

LogicalResult collectIncludeClosure(Operation *includeOp,
                                    FlatSymbolRefAttr target,
                                    SmallVectorImpl<NamedSequenceOp> &closure,
                                    SmallPtrSetImpl<Operation *> &visited,
                                    SmallPtrSetImpl<Operation *> &active,
                                    std::string &error) {
  auto callee =
      SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(includeOp, target);
  if (!callee) {
    error = "unresolved include target in transaction library";
    return failure();
  }
  return collectClosure(callee, closure, visited, active, error);
}

LogicalResult collectClosureBody(NamedSequenceOp sequence,
                                 SmallVectorImpl<NamedSequenceOp> &closure,
                                 SmallPtrSetImpl<Operation *> &visited,
                                 SmallPtrSetImpl<Operation *> &active,
                                 std::string &error) {
  auto walk = sequence.getBody().walk([&](Operation *op) {
    auto target = getIncludeTarget(op);
    if (!target)
      return WalkResult::advance();
    auto flat = dyn_cast<FlatSymbolRefAttr>(target);
    if (!flat) {
      error = "expected a flat symbol reference in transaction library";
      return WalkResult::interrupt();
    }
    if (failed(
            collectIncludeClosure(op, flat, closure, visited, active, error)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

LogicalResult collectClosure(NamedSequenceOp sequence,
                             SmallVectorImpl<NamedSequenceOp> &closure,
                             SmallPtrSetImpl<Operation *> &visited,
                             SmallPtrSetImpl<Operation *> &active,
                             std::string &error) {
  if (active.contains(sequence)) {
    error = "recursive include in transaction library";
    return failure();
  }
  if (!visited.insert(sequence).second)
    return success();
  if (sequence.isExternal()) {
    error = "transaction library contains an unresolved named sequence";
    return failure();
  }

  active.insert(sequence);
  closure.push_back(sequence);
  if (failed(collectClosureBody(sequence, closure, visited, active, error)))
    return failure();
  active.erase(sequence);
  return success();
}

OwningOpRef<ModuleOp> createLibraryModule(MLIRContext *context) {
  auto module = ModuleOp::create(UnknownLoc::get(context));
  module->setAttr("transform.with_named_sequence", UnitAttr::get(context));
  return module;
}

OwningOpRef<ModuleOp> cloneClosure(NamedSequenceOp entry, std::string &error) {
  SmallVector<NamedSequenceOp> closure;
  SmallPtrSet<Operation *, 8> visited, active;
  if (failed(collectClosure(entry, closure, visited, active, error)))
    return {};

  auto context = entry.getContext();
  auto module = createLibraryModule(context);
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module->getBody());
  for (auto sequence : closure)
    builder.insert(sequence->clone());
  return module;
}

void populateCaptureBody(Block &block, MLIRContext *context, StringRef target) {
  OpBuilder builder(context);
  builder.setInsertionPointToStart(&block);
  builder.create<HIVMTransactionExecuteOp>(
      block.getArgument(0).getLoc(), AnyParamType::get(context),
      FlatSymbolRefAttr::get(context, target), block.getArgument(0),
      block.getArgument(1));
  builder.create<YieldOp>(block.getArgument(0).getLoc());
}

NamedSequenceOp createCaptureWrapper(ModuleOp module, StringRef target) {
  auto context = module.getContext();
  std::string name(kWorkerEntryName.str());
  while (module.lookupSymbol(name))
    name += "_";

  OpBuilder builder(context);
  auto loc = UnknownLoc::get(context);
  auto rootType = AnyOpType::get(context);
  auto paramType = AnyParamType::get(context);

  OperationState state(loc, NamedSequenceOp::getOperationName());
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(NamedSequenceOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(builder.getFunctionType(
                         {rootType, paramType}, TypeRange())));
  auto &region = *state.addRegion();
  auto &block = region.emplaceBlock();
  block.addArgument(rootType, loc);
  block.addArgument(paramType, loc);
  populateCaptureBody(block, context, target);

  builder.setInsertionPointToEnd(module.getBody());
  return cast<NamedSequenceOp>(builder.create(state));
}

bool applyCallbackOperation(TransformState &state, Operation &op,
                            CallbackCapture &capture) {
  auto transform = dyn_cast<TransformOpInterface>(&op);
  if (!transform) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics,
                      "expected an executable Transform operation\n");
    return false;
  }

  auto outcome = state.applyTransform(transform);
  if (outcome.isDefiniteFailure()) {
    capture.outcome = TransactionOutcome::error;
    return false;
  }
  if (!outcome.isSilenceableFailure())
    return true;

  capture.outcome = TransactionOutcome::rejected;
  appendDiagnostics(capture.diagnostics, outcome.getMessage());
  return false;
}

void captureCallbackResult(ValueRange yielded, TransformState &state,
                           CallbackCapture &capture) {
  if (yielded.size() != 1) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics,
                      "expected exactly one yielded attribute\n");
    return;
  }

  auto params = state.getParams(yielded.front());
  if (params.size() != 1) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics,
                      "expected exactly one yielded attribute\n");
    return;
  }

  std::string error;
  if (failed(serializeAttribute(params.front(), capture.result, error))) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics, error + "\n");
    return;
  }
  capture.outcome = TransactionOutcome::succeeded;
}

CallbackCapture runCallbackBody(NamedSequenceOp callee, ValueRange operands,
                                TransformState &state) {
  CallbackCapture capture;
  if (callee.isExternal()) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics,
                      "transaction callback is external\n");
    return capture;
  }

  SmallVector<SmallVector<MappedValue>> mappings;
  transform::detail::prepareValueMappings(mappings, operands, state);
  auto scope = state.make_region_scope(callee.getBody());
  if (failed(state.mapBlockArguments(callee.getBody().front().getArguments(),
                                     mappings))) {
    capture.outcome = TransactionOutcome::error;
    appendDiagnostics(capture.diagnostics,
                      "failed to bind transaction callback arguments\n");
    return capture;
  }

  auto &body = callee.getBody().front();
  for (auto &op : body.without_terminator())
    if (!applyCallbackOperation(state, op, capture))
      return capture;
  captureCallbackResult(body.getTerminator()->getOperands(), state, capture);
  return capture;
}

struct CostInvocation {
  CallbackCapture capture;
  std::string diagnostics;
  bool applyFailed = false;
};

CostInvocation invokeCostCallback(TransformState &state, NamedSequenceOp cost,
                                  ModuleOp candidate, Attribute resultAttr) {
  auto *context = cost.getContext();
  std::string error;
  auto module = cloneClosure(cost, error);
  if (!module) {
    CostInvocation invocation;
    invocation.applyFailed = true;
    invocation.diagnostics = error;
    return invocation;
  }

  auto wrapper = createCaptureWrapper(*module, cost.getSymName());
  CostInvocation invocation;
  CaptureScope captureScope(invocation.capture);
  ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
    appendDiagnostic(invocation.diagnostics, diag);
    return success();
  });

  RaggedArray<MappedValue> bindings;
  bindings.push_back(ArrayRef<Operation *>{candidate.getOperation()});
  bindings.push_back(ArrayRef<Attribute>{resultAttr});
  invocation.applyFailed = failed(applyTransformNamedSequence(
      std::move(bindings), cast<TransformOpInterface>(wrapper.getOperation()),
      *module, state.getOptions()));
  return invocation;
}

FailureOr<double> convertCostAttribute(Attribute value, std::string &error) {
  if (!value) {
    error = "cost callback did not produce a valid attribute";
    return failure();
  }
  if (auto floatValue = dyn_cast<FloatAttr>(value))
    return floatValue.getValueAsDouble();
  if (auto integerValue = dyn_cast<IntegerAttr>(value))
    return static_cast<double>(integerValue.getInt());
  error = "cost callback must yield a FloatAttr or an IntegerAttr";
  return failure();
}

CostOutcome makeCostOutcome(CostOutcome::Status status,
                            std::string diagnostics = {}) {
  CostOutcome outcome;
  outcome.status = status;
  outcome.diagnostics = std::move(diagnostics);
  return outcome;
}

struct WorkerRequest {
  OwningOpRef<ModuleOp> payload;
  OwningOpRef<ModuleOp> library;
  Attribute options;
};

std::optional<WorkerRequest> parseWorkerRequest(Ticket &ticket,
                                                MLIRContext &context) {
  WorkerRequest request;
  request.payload = parseModule(ticket.snapshot, &context);
  request.library = parseModule(ticket.library, &context);
  request.options = parseAttribute(ticket.options, &context);
  if (!request.payload || !request.library || !request.options)
    return std::nullopt;
  return request;
}

CallbackCapture runWorkerAttempt(WorkerRequest &request, Ticket &ticket) {
  auto wrapper = createCaptureWrapper(*request.library, ticket.entry);
  RaggedArray<MappedValue> bindings;
  bindings.push_back(ArrayRef<Operation *>{request.payload->getOperation()});
  bindings.push_back(ArrayRef<Attribute>{request.options});

  CallbackCapture capture;
  CaptureScope captureScope(capture);
  if (failed(applyTransformNamedSequence(
          std::move(bindings),
          cast<TransformOpInterface>(wrapper.getOperation()), *request.library,
          TransformOptions())))
    capture.outcome = TransactionOutcome::error;
  return capture;
}

struct WorkerOutcome {
  TransactionOutcome outcome = TransactionOutcome::error;
  std::string diagnostics;
  BytecodeBuffer candidate;
  BytecodeBuffer result;
};

WorkerOutcome makeWorkerFailure(TransactionOutcome outcome,
                                std::string diagnostics) {
  WorkerOutcome result;
  result.outcome = outcome;
  result.diagnostics = std::move(diagnostics);
  return result;
}

WorkerOutcome finalizeWorkerCandidate(CallbackCapture &capture,
                                      WorkerRequest &request) {
  if (capture.outcome == TransactionOutcome::rejected)
    return makeWorkerFailure(TransactionOutcome::rejected, capture.diagnostics);
  if (capture.outcome != TransactionOutcome::succeeded)
    return makeWorkerFailure(TransactionOutcome::error, capture.diagnostics);

  if (failed(verify(*request.payload)))
    return makeWorkerFailure(TransactionOutcome::error,
                             "transaction candidate failed verification");

  WorkerOutcome result;
  std::string error;
  if (failed(serializeOperation(request.payload->getOperation(),
                                result.candidate, error)))
    return makeWorkerFailure(TransactionOutcome::error, error);

  result.outcome = TransactionOutcome::succeeded;
  result.result = std::move(capture.result);
  return result;
}

WorkerOutcome executeWorker(Ticket &ticket, MLIRContext &context,
                            std::string &diagnostics) {
  auto request = parseWorkerRequest(ticket, context);
  if (!request)
    return makeWorkerFailure(TransactionOutcome::error,
                             "failed to deserialize transaction request");

  auto capture = runWorkerAttempt(*request, ticket);
  auto result = finalizeWorkerCandidate(capture, *request);
  if (result.outcome == TransactionOutcome::error) {
    appendDiagnostics(result.diagnostics, diagnostics);
    if (result.diagnostics.empty())
      result.diagnostics = "transaction worker failed";
  }
  return result;
}

} // namespace

std::optional<CallbackCaptureRef> mlir::hivm::getActiveCallbackCapture() {
  return activeCapture;
}

LogicalResult mlir::hivm::serializeOperation(Operation *op, BytecodeBuffer &out,
                                             std::string &error) {
  if (failed(checkSerializable(op, error)))
    return failure();

  BytecodeBuffer bytes;
  llvm::raw_svector_ostream stream(bytes);
  if (failed(writeBytecodeToFile(op, stream))) {
    error = "failed to serialize transaction bytecode";
    return failure();
  }
  out = std::move(bytes);
  return success();
}

LogicalResult mlir::hivm::serializeAttribute(Attribute attr,
                                             BytecodeBuffer &out,
                                             std::string &error) {
  auto envelope = ModuleOp::create(UnknownLoc::get(attr.getContext()));
  envelope->setAttr(kEnvelopeAttrName, attr);
  return serializeOperation(envelope.getOperation(), out, error);
}

OwningOpRef<ModuleOp> mlir::hivm::parseModule(ArrayRef<char> bytes,
                                              MLIRContext *context) {
  if (bytes.empty())
    return {};
  StringRef data(bytes.data(), bytes.size());
  if (!isBytecode(llvm::MemoryBufferRef(data, "<transaction-bytecode>")))
    return {};
  auto buffer =
      llvm::MemoryBuffer::getMemBufferCopy(data, "<transaction-bytecode>");
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, ParserConfig(context));
}

Attribute mlir::hivm::parseAttribute(ArrayRef<char> bytes,
                                     MLIRContext *context) {
  auto envelope = parseModule(bytes, context);
  if (!envelope)
    return {};
  return envelope->getOperation()->getAttr(kEnvelopeAttrName);
}

LogicalResult mlir::hivm::invokeCallback(TransformState &state,
                                         NamedSequenceOp callee,
                                         ValueRange operands) {
  auto capture = getActiveCallbackCapture();
  if (!capture)
    return failure();

  auto result = runCallbackBody(callee, operands, state);
  capture->get() = std::move(result);
  return success();
}

FailureOr<CostOutcome> mlir::hivm::evaluateCost(TransformState &state,
                                                NamedSequenceOp cost,
                                                ModuleOp candidate,
                                                Attribute resultAttr,
                                                std::string &error) {
  auto invocation = invokeCostCallback(state, cost, candidate, resultAttr);
  if (invocation.applyFailed) {
    appendDiagnostics(invocation.diagnostics, invocation.capture.diagnostics);
    return makeCostOutcome(CostOutcome::Status::error,
                           std::move(invocation.diagnostics));
  }
  if (invocation.capture.outcome == TransactionOutcome::rejected) {
    appendDiagnostics(invocation.capture.diagnostics, invocation.diagnostics);
    return makeCostOutcome(CostOutcome::Status::rejected,
                           std::move(invocation.capture.diagnostics));
  }
  if (invocation.capture.outcome != TransactionOutcome::succeeded) {
    appendDiagnostics(invocation.diagnostics, invocation.capture.diagnostics);
    return makeCostOutcome(CostOutcome::Status::error,
                           std::move(invocation.diagnostics));
  }

  auto value = parseAttribute(invocation.capture.result, cost.getContext());
  auto converted = convertCostAttribute(value, error);
  if (failed(converted))
    return failure();

  auto outcome = makeCostOutcome(CostOutcome::Status::succeeded);
  outcome.cost = *converted;
  return outcome;
}

TransactionSession::TransactionSession(TransformState &state)
    : Extension(state), id(nextSessionId.fetch_add(1)) {
  auto *topLevel = state.getTopLevel();
  assert(topLevel && "transaction session requires a payload root");
  topLevel->getContext()->getDialectRegistry().appendTo(registry);
}

TransactionSession::~TransactionSession() { pool.wait(); }

std::optional<std::reference_wrapper<TransactionSession>>
TransactionSession::get(TransformState &state) {
  auto *session = state.getExtension<TransactionSession>();
  if (!session)
    return std::nullopt;
  return std::ref(*session);
}

TransactionSession &TransactionSession::getOrCreate(TransformState &state) {
  if (auto session = get(state))
    return session->get();
  return state.addExtension<TransactionSession>();
}

std::optional<TicketRef> TransactionSession::getTicket(uint64_t ticket) {
  if (ticket == 0 || ticket > tickets.size())
    return std::nullopt;
  return std::ref(*tickets[ticket - 1]);
}

void TransactionSession::dropDescendantMappings(Operation *root) {
  SmallVector<Value> values;
  root->walk([&](Operation *op) {
    llvm::append_range(values, op->getResults());
    for (auto &region : op->getRegions())
      for (auto &block : region)
        llvm::append_range(values, block.getArguments());
  });
  for (auto value : values)
    (void)replacePayloadValue(value, nullptr);

  SmallVector<Operation *> ops;
  root->walk([&](Operation *op) {
    if (op != root)
      ops.push_back(op);
  });
  for (auto *op : ops)
    (void)replacePayloadOp(op, nullptr);
}

LogicalResult TransactionSession::getLibrary(NamedSequenceOp attempt,
                                             BytecodeBuffer &out,
                                             std::string &error) {
  auto library = cloneClosure(attempt, error);
  if (!library)
    return failure();
  return serializeOperation(library->getOperation(), out, error);
}

void TransactionSession::initializeTicket(Ticket &ticket, ModuleOp root,
                                          Attribute options,
                                          NamedSequenceOp attempt) {
  ticket.id = tickets.size() + 1;
  ticket.root = root;
  ticket.optionsAttr = options;
  ticket.entry = attempt.getSymName().str();
}

FailureOr<uint64_t> TransactionSession::submit(ModuleOp root, Attribute options,
                                               NamedSequenceOp attempt,
                                               bool sync, std::string &error) {
  auto ticket = std::make_unique<Ticket>();
  initializeTicket(*ticket, root, options, attempt);

  if (failed(serializeOperation(root.getOperation(), ticket->snapshot, error)))
    return failure();
  if (failed(getLibrary(attempt, ticket->library, error)))
    return failure();
  if (failed(serializeAttribute(options, ticket->options, error)))
    return failure();

  auto &submitted = *ticket;
  tickets.push_back(std::move(ticket));
  if (sync || !root.getContext()->isMultithreadingEnabled()) {
    runWorker(submitted);
  } else {
    submitted.future = pool.async([this, &submitted] { runWorker(submitted); });
  }
  return submitted.id;
}

void TransactionSession::runWorker(Ticket &ticket) {
  ticket.outcome = TransactionOutcome::error;
  auto markDone = llvm::make_scope_exit([&ticket] { ticket.done = true; });

  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  std::string diagnostics;
  WorkerOutcome result;
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      appendDiagnostic(diagnostics, diag);
      return success();
    });
    result = executeWorker(ticket, context, diagnostics);
  }
  ticket.diagnostics = std::move(result.diagnostics);
  ticket.candidate = std::move(result.candidate);
  ticket.result = std::move(result.result);
  ticket.outcome = result.outcome;
}
