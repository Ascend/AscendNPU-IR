//===- TransactionTest.cpp - HIVM transaction unit tests -----------------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {

constexpr StringLiteral kPayload = R"mlir(
module {
  func.func @f(%arg0: index, %arg1: index) -> index {
    %sum = arith.addi %arg0, %arg1 : index
    return %sum : index
  }
}
)mlir";

constexpr StringLiteral kSuccessLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 7.5 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }
}
)mlir";

constexpr StringLiteral kRejectedLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.annotate %root "hivm.attempt_mutated" : !transform.any_op
    %addi = transform.structured.match ops{["arith.addi"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %missing = transform.get_result %addi[1]
        : (!transform.any_op) -> !transform.any_value
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }
}
)mlir";

constexpr StringLiteral kErrorLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %idx = transform.hivm.map_forall_to_blocks %fn
        : (!transform.any_op) -> !transform.any_op
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }
}
)mlir";

constexpr StringLiteral kClosureLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @helper(
      %root: !transform.any_op {transform.readonly}
  ) {
    transform.annotate %root "hivm.helper" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @attempt(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.hivm.include @helper failures(propagate) (%root)
        : (!transform.any_op) -> ()
    %score = transform.param.constant 2.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }
}
)mlir";

class TransactionSessionTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    transform::TransformDialect, hivm::HIVMDialect>();
    hivm::registerTransformDialectExtension(registry);
    hivm::registerVectorizeTransformDialectExtension(registry);
    linalg::registerTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  transform::NamedSequenceOp lookupAttempt(ModuleOp library) {
    return library.lookupSymbol<transform::NamedSequenceOp>("attempt");
  }

  hivm::Ticket &getTicket(hivm::TransactionSession &session,
                          uint64_t ticketId) {
    auto ticket = session.getTicket(ticketId);
    EXPECT_TRUE(ticket.has_value());
    return ticket->get();
  }

  void waitTicket(hivm::TransactionSession &session, uint64_t ticketId) {
    auto &ticket = getTicket(session, ticketId);
    ticket.wait();
    EXPECT_TRUE(ticket.done);
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(TransactionSessionTest, SerializeModuleRoundTrips) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  hivm::BytecodeBuffer bytes;
  std::string error;
  ASSERT_TRUE(succeeded(
      hivm::serializeOperation(payload->getOperation(), bytes, error)))
      << error;
  EXPECT_FALSE(bytes.empty());

  auto roundTrip = hivm::parseModule(bytes, context.get());
  ASSERT_TRUE(roundTrip.get());
  EXPECT_TRUE(succeeded(verify(*roundTrip)));

  hivm::BytecodeBuffer secondBytes;
  ASSERT_TRUE(succeeded(
      hivm::serializeOperation(roundTrip->getOperation(), secondBytes, error)))
      << error;
  EXPECT_EQ(bytes, secondBytes);
}

TEST_F(TransactionSessionTest, SerializeAttributeEnvelopeRoundTrips) {
  auto options = DictionaryAttr::get(
      context.get(), {NamedAttribute(StringAttr::get(context.get(), "tag"),
                                     StringAttr::get(context.get(), "value"))});

  hivm::BytecodeBuffer bytes;
  std::string error;
  ASSERT_TRUE(succeeded(hivm::serializeAttribute(options, bytes, error)))
      << error;

  MLIRContext otherContext(registry);
  auto roundTrip = hivm::parseAttribute(bytes, &otherContext);
  ASSERT_TRUE(static_cast<bool>(roundTrip));
  auto toString = [](Attribute attr) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    attr.print(stream);
    return stream.str();
  };
  EXPECT_EQ(toString(roundTrip), toString(options));
}

TEST_F(TransactionSessionTest, SerializeRejectsDistinctAttributes) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto distinct =
      DistinctAttr::create(StringAttr::get(context.get(), "identity"));
  payload->getOperation()->setAttr("hivm.distinct", distinct);

  hivm::BytecodeBuffer bytes;
  std::string error;
  EXPECT_TRUE(
      failed(hivm::serializeOperation(payload->getOperation(), bytes, error)));
  EXPECT_FALSE(error.empty());
}

TEST_F(TransactionSessionTest, ParseRejectsInvalidBytecode) {
  auto invalid = hivm::BytecodeBuffer{'n', 'o', 't', 'b', 'y', 't', 'e'};
  EXPECT_FALSE(static_cast<bool>(hivm::parseModule(invalid, context.get())));
  EXPECT_FALSE(static_cast<bool>(hivm::parseAttribute(invalid, context.get())));
  EXPECT_FALSE(static_cast<bool>(hivm::parseModule({}, context.get())));
  EXPECT_FALSE(static_cast<bool>(hivm::parseAttribute({}, context.get())));
}

TEST_F(TransactionSessionTest, SyncSubmitCompletesInline) {
  auto payload = parse(kPayload);
  auto library = parse(kSuccessLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());

  std::string error;
  auto ticketId =
      session.submit(*payload, options, attempt, /*sync=*/true, error);
  ASSERT_TRUE(succeeded(ticketId)) << error;

  auto &ticket = getTicket(session, *ticketId);
  EXPECT_TRUE(ticket.done);
  EXPECT_FALSE(ticket.future.valid());
  EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::succeeded);
  EXPECT_FALSE(ticket.candidate.empty());
  EXPECT_FALSE(ticket.result.empty());
}

TEST_F(TransactionSessionTest, AsyncSubmitsCompleteConcurrently) {
  auto payload = parse(kPayload);
  auto library = parse(kSuccessLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());

  SmallVector<uint64_t> ticketIds;
  for (unsigned index = 0; index < 8; ++index) {
    std::string error;
    auto ticketId =
        session.submit(*payload, options, attempt, /*sync=*/false, error);
    ASSERT_TRUE(succeeded(ticketId)) << error;
    ticketIds.push_back(*ticketId);
  }

  for (auto ticketId : ticketIds) {
    waitTicket(session, ticketId);
    auto &ticket = getTicket(session, ticketId);
    EXPECT_TRUE(ticket.future.valid());
    EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::succeeded);
  }
}

TEST_F(TransactionSessionTest, RejectedAttemptDoesNotMutateLivePayload) {
  auto payload = parse(kPayload);
  auto library = parse(kRejectedLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  hivm::BytecodeBuffer before;
  std::string error;
  ASSERT_TRUE(succeeded(
      hivm::serializeOperation(payload->getOperation(), before, error)))
      << error;

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());
  auto ticketId =
      session.submit(*payload, options, attempt, /*sync=*/true, error);
  ASSERT_TRUE(succeeded(ticketId)) << error;

  auto &ticket = getTicket(session, *ticketId);
  EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::rejected);
  EXPECT_TRUE(ticket.candidate.empty());
  EXPECT_FALSE(ticket.diagnostics.empty());
  EXPECT_FALSE(static_cast<bool>(
      payload->getOperation()->getAttr("hivm.attempt_mutated")));

  hivm::BytecodeBuffer after;
  ASSERT_TRUE(succeeded(
      hivm::serializeOperation(payload->getOperation(), after, error)))
      << error;
  EXPECT_EQ(before, after);
}

TEST_F(TransactionSessionTest, DefiniteFailureProducesErrorOutcome) {
  auto payload = parse(kPayload);
  auto library = parse(kErrorLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());

  std::string error;
  auto ticketId =
      session.submit(*payload, options, attempt, /*sync=*/true, error);
  ASSERT_TRUE(succeeded(ticketId)) << error;

  auto &ticket = getTicket(session, *ticketId);
  EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::error);
  EXPECT_TRUE(ticket.candidate.empty());
  EXPECT_NE(ticket.diagnostics.find("expect an scf.forall"), std::string::npos);
}

TEST_F(TransactionSessionTest, LibraryClosureResolvesIncludes) {
  auto payload = parse(kPayload);
  auto library = parse(kClosureLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());

  std::string error;
  auto ticketId =
      session.submit(*payload, options, attempt, /*sync=*/true, error);
  ASSERT_TRUE(succeeded(ticketId)) << error;

  auto &ticket = getTicket(session, *ticketId);
  EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::succeeded);
  auto candidate = hivm::parseModule(ticket.candidate, context.get());
  ASSERT_TRUE(candidate.get());
  EXPECT_TRUE(succeeded(verify(*candidate)));
  EXPECT_TRUE(candidate->getOperation()->hasAttr("hivm.helper"));
}

TEST_F(TransactionSessionTest, TicketIdsAreIndices) {
  auto payload = parse(kPayload);
  auto library = parse(kSuccessLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(context.get());

  std::string error;
  auto first = session.submit(*payload, options, attempt, true, error);
  auto second = session.submit(*payload, options, attempt, true, error);
  ASSERT_TRUE(succeeded(first)) << error;
  ASSERT_TRUE(succeeded(second)) << error;

  EXPECT_EQ(*first, 1u);
  EXPECT_EQ(*second, 2u);
  EXPECT_FALSE(session.getTicket(0).has_value());
  EXPECT_TRUE(session.getTicket(*first).has_value());
  EXPECT_TRUE(session.getTicket(*second).has_value());
}

TEST_F(TransactionSessionTest, SessionsHaveDistinctIds) {
  auto payload = parse(kPayload);
  auto library = parse(kSuccessLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto firstState = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto secondState = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &first = firstState.addExtension<hivm::TransactionSession>();
  auto &second = secondState.addExtension<hivm::TransactionSession>();

  EXPECT_NE(first.getId(), second.getId());
}

TEST_F(TransactionSessionTest, DestructorWaitsForOutstandingWork) {
  auto payload = parse(kPayload);
  auto library = parse(kSuccessLibrary);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  {
    auto state = transform::detail::makeTransformStateForTesting(
        attempt->getParentRegion(), payload.get());
    auto &session = state.addExtension<hivm::TransactionSession>();
    auto options = DictionaryAttr::get(context.get());
    std::string error;
    for (unsigned index = 0; index < 4; ++index)
      ASSERT_TRUE(
          succeeded(session.submit(*payload, options, attempt, false, error)))
          << error;
  }
  SUCCEED();
}

TEST_F(TransactionSessionTest, ThreadingDisabledRunsInline) {
  MLIRContext disabledContext(registry, MLIRContext::Threading::DISABLED);
  auto payload = parseSourceString<ModuleOp>(kPayload, &disabledContext);
  auto library = parseSourceString<ModuleOp>(kSuccessLibrary, &disabledContext);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(library.get());
  auto attempt = lookupAttempt(*library);
  ASSERT_TRUE(attempt.getOperation());

  auto state = transform::detail::makeTransformStateForTesting(
      attempt->getParentRegion(), payload.get());
  auto &session = state.addExtension<hivm::TransactionSession>();
  auto options = DictionaryAttr::get(&disabledContext);

  std::string error;
  auto ticketId =
      session.submit(*payload, options, attempt, /*sync=*/false, error);
  ASSERT_TRUE(succeeded(ticketId)) << error;

  auto &ticket = getTicket(session, *ticketId);
  EXPECT_TRUE(ticket.done);
  EXPECT_FALSE(ticket.future.valid());
  EXPECT_EQ(ticket.outcome, hivm::TransactionOutcome::succeeded);
}

} // namespace
