//===- OutlineTest.cpp - HIVM outlining and VF annotation tests -----------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/Outline.h"
#include "bishengir/Analysis/Partition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>

using namespace mlir;

namespace {

constexpr StringLiteral kPayload = R"mlir(
module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %0, %b : f32
    %2 = arith.subf %1, %a : f32
    return %2 : f32
  }
}
)mlir";

constexpr StringLiteral kNonContiguousPayload = R"mlir(
module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %a, %b : f32
    %2 = arith.subf %0, %a : f32
    %3 = arith.addf %2, %1 : f32
    return %3 : f32
  }
}
)mlir";

constexpr StringLiteral kSharedArgumentsPayload = R"mlir(
module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %a, %b : f32
    %2 = arith.subf %0, %1 : f32
    return %2 : f32
  }
}
)mlir";

constexpr StringLiteral kOutlineLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> (!transform.any_op, !transform.any_op) {
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2>, array<i64: 3>]>
        -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield %newRoot, %funcs : !transform.any_op, !transform.any_op
  }
}
)mlir";

constexpr StringLiteral kEmptyOutlineLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> (!transform.any_op, !transform.any_op) {
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield %newRoot, %funcs : !transform.any_op, !transform.any_op
  }
}
)mlir";

constexpr StringLiteral kOutlineAnnotateLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> (!transform.any_op, !transform.any_op) {
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2>, array<i64: 3>]>
        -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    %annotated = transform.hivm.annotate_vf %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield %annotated, %funcs : !transform.any_op, !transform.any_op
  }
}
)mlir";

constexpr StringLiteral kAnnotateNonFunctionLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> !transform.any_op {
    %annotated = transform.hivm.annotate_vf %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield %annotated : !transform.any_op
  }
}
)mlir";

constexpr StringLiteral kEmptyAnnotateLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> !transform.any_op {
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    %annotated = transform.hivm.annotate_vf %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield %annotated : !transform.any_op
  }
}
)mlir";

constexpr StringLiteral kExtractGroupsLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> !transform.any_param {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = 2.5 : f64, groups = #hivm_vectorize.groups<[array<i64: 2>, array<i64: 3>]>>
        -> !transform.any_param
    %groups = transform.hivm.extract_groups %plan
        : (!transform.any_param) -> !transform.any_param
    transform.yield %groups : !transform.any_param
  }
}
)mlir";

constexpr StringLiteral kExtractGroupsNonPlanLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> !transform.any_param {
    %plan = transform.param.constant 3 : i64 -> !transform.any_param
    %groups = transform.hivm.extract_groups %plan
        : (!transform.any_param) -> !transform.any_param
    transform.yield %groups : !transform.any_param
  }
}
)mlir";

class OutlineTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    transform::TransformDialect, hivm::HIVMDialect>();
    hivm::registerVectorizeTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadDialect<hivm::HIVMDialect>();
    context->loadDialect<arith::ArithDialect>();
    context->loadDialect<func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  SmallVector<Operation *> getBodyOps(ModuleOp module, StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function.getOperation());
    SmallVector<Operation *> ops;
    for (Operation &op : function.getBody().front().getOperations())
      ops.push_back(&op);
    return ops;
  }

  FailureOr<SmallVector<func::FuncOp>> runEngine(ModuleOp module,
                                                 ArrayRef<hivm::Group> groups) {
    std::string error;
    auto functions = hivm::outlineGroups(module, groups, error);
    EXPECT_TRUE(succeeded(functions)) << error;
    return functions;
  }

  struct OutlineOpResult {
    SmallVector<Operation *> newRoot;
    SmallVector<Operation *> functions;
  };

  FailureOr<OutlineOpResult> runOutlineOp(ModuleOp payload,
                                          StringRef librarySource) {
    auto library = parse(librarySource);
    if (!library)
      return failure();
    auto entry = library->lookupSymbol<transform::NamedSequenceOp>("entry");
    if (!entry)
      return failure();

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(&entry.getBody(),
                                                        payload.getOperation());
    Block &body = entry.getBody().front();
    if (failed(mapEntryRoot(body, payload, state)))
      return failure();
    if (failed(applyBody(body, state)))
      return failure();

    auto outlineOp = findOp<transform::HIVMOutlineOp>(body);
    if (!outlineOp)
      return failure();

    OutlineOpResult result;
    auto roots = state.getPayloadOps(outlineOp.getNewRoot());
    result.newRoot.assign(roots.begin(), roots.end());
    auto functions = state.getPayloadOps(outlineOp.getFuncs());
    result.functions.assign(functions.begin(), functions.end());
    return result;
  }

  FailureOr<SmallVector<Operation *>> runAnnotateOp(ModuleOp payload,
                                                    StringRef librarySource) {
    auto library = parse(librarySource);
    if (!library)
      return failure();
    auto entry = library->lookupSymbol<transform::NamedSequenceOp>("entry");
    if (!entry)
      return failure();

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(&entry.getBody(),
                                                        payload.getOperation());
    Block &body = entry.getBody().front();
    if (failed(mapEntryRoot(body, payload, state)))
      return failure();
    if (failed(applyBody(body, state)))
      return failure();

    auto annotateOp = findOp<transform::HIVMAnnotateVFOp>(body);
    if (!annotateOp)
      return failure();
    auto functions = state.getPayloadOps(annotateOp.getResult());
    return SmallVector<Operation *>(functions.begin(), functions.end());
  }

  FailureOr<SmallVector<Attribute>>
  runExtractGroupsOp(ModuleOp payload, StringRef librarySource) {
    auto library = parse(librarySource);
    if (!library)
      return failure();
    auto entry = library->lookupSymbol<transform::NamedSequenceOp>("entry");
    if (!entry)
      return failure();

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(&entry.getBody(),
                                                        payload.getOperation());
    Block &body = entry.getBody().front();
    if (failed(mapEntryRoot(body, payload, state)))
      return failure();
    if (failed(applyBody(body, state)))
      return failure();

    auto extractGroupsOp = findOp<transform::HIVMExtractGroupsOp>(body);
    if (!extractGroupsOp)
      return failure();
    auto groups = state.getParams(extractGroupsOp.getGroups());
    return SmallVector<Attribute>(groups.begin(), groups.end());
  }

  template <typename OpT> static OpT findOp(Block &body) {
    for (Operation &op : body.without_terminator())
      if (auto candidate = dyn_cast<OpT>(op))
        return candidate;
    return OpT();
  }

  static LogicalResult mapEntryRoot(Block &body, ModuleOp payload,
                                    transform::TransformState &state) {
    SmallVector<SmallVector<transform::MappedValue>> mappings;
    mappings.push_back(
        SmallVector<transform::MappedValue>{payload.getOperation()});
    return state.mapBlockArguments(body.getArguments(), mappings);
  }

  static LogicalResult applyBody(Block &body,
                                 transform::TransformState &state) {
    for (Operation &op : body.without_terminator()) {
      auto transformOp = dyn_cast<transform::TransformOpInterface>(&op);
      if (!transformOp)
        return failure();
      DiagnosedSilenceableFailure outcome = state.applyTransform(transformOp);
      if (!outcome.succeeded()) {
        (void)outcome.checkAndReport();
        return failure();
      }
    }
    return success();
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(OutlineTest, OutlineSingletonGroup) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions = runEngine(*payload, hivm::Groups{hivm::Group{ops[0]}});
  ASSERT_TRUE(succeeded(functions));
  ASSERT_EQ(functions->size(), 1u);

  func::FuncOp fused = functions->front();
  EXPECT_EQ(fused.getNumArguments(), 2u);
  EXPECT_EQ(fused.getNumResults(), 1u);

  auto bodyOps = getBodyOps(*payload, "f");
  ASSERT_EQ(bodyOps.size(), 4u);
  auto call = dyn_cast<func::CallOp>(bodyOps[0]);
  ASSERT_TRUE(call);
  EXPECT_EQ(call.getCallee(), fused.getSymName());
  EXPECT_EQ(call.getNumOperands(), 2u);
  EXPECT_EQ(call.getNumResults(), 1u);
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineMultiOpGroup) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions =
      runEngine(*payload, hivm::Groups{hivm::Group{ops[0], ops[1]}});
  ASSERT_TRUE(succeeded(functions));
  ASSERT_EQ(functions->size(), 1u);

  func::FuncOp fused = functions->front();
  EXPECT_EQ(fused.getNumArguments(), 2u);
  EXPECT_EQ(fused.getNumResults(), 1u);
  EXPECT_EQ(fused.getBody().front().getOperations().size(), 3u);

  auto bodyOps = getBodyOps(*payload, "f");
  ASSERT_EQ(bodyOps.size(), 3u);
  auto call = dyn_cast<func::CallOp>(bodyOps[0]);
  ASSERT_TRUE(call);
  EXPECT_EQ(call.getNumOperands(), 2u);
  EXPECT_EQ(call.getNumResults(), 1u);
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineNonContiguousGroup) {
  auto payload = parse(kNonContiguousPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions =
      runEngine(*payload, hivm::Groups{hivm::Group{ops[0], ops[2]}});
  ASSERT_TRUE(succeeded(functions));
  ASSERT_EQ(functions->size(), 1u);

  func::FuncOp fused = functions->front();
  EXPECT_EQ(fused.getNumArguments(), 2u);
  EXPECT_EQ(fused.getNumResults(), 1u);

  auto bodyOps = getBodyOps(*payload, "f");
  ASSERT_EQ(bodyOps.size(), 4u);
  auto call = dyn_cast<func::CallOp>(bodyOps[1]);
  ASSERT_TRUE(call);
  EXPECT_TRUE(isa<arith::MulFOp>(bodyOps[0]));
  EXPECT_EQ(call.getNumOperands(), 2u);
  EXPECT_EQ(call.getNumResults(), 1u);
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineMultipleGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions = runEngine(
      *payload, hivm::Groups{hivm::Group{ops[0]}, hivm::Group{ops[1]}});
  ASSERT_TRUE(succeeded(functions));
  ASSERT_EQ(functions->size(), 2u);

  EXPECT_TRUE(isa<arith::MulFOp>((*functions)[0].getBody().front().front()));
  EXPECT_TRUE(isa<arith::AddFOp>((*functions)[1].getBody().front().front()));

  auto bodyOps = getBodyOps(*payload, "f");
  ASSERT_EQ(bodyOps.size(), 4u);
  auto firstCall = dyn_cast<func::CallOp>(bodyOps[0]);
  auto secondCall = dyn_cast<func::CallOp>(bodyOps[1]);
  ASSERT_TRUE(firstCall);
  ASSERT_TRUE(secondCall);
  EXPECT_EQ(firstCall.getCallee(), (*functions)[1].getSymName());
  EXPECT_EQ(secondCall.getCallee(), (*functions)[0].getSymName());
  EXPECT_EQ(secondCall.getOperand(0), firstCall.getResult(0));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineEmptyGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");

  auto functions = runEngine(*payload, hivm::Groups{});
  ASSERT_TRUE(succeeded(functions));
  EXPECT_TRUE(functions->empty());

  auto after = getBodyOps(*payload, "f");
  EXPECT_EQ(before, after);
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, RejectEmptyGroupBeforeMutation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");
  std::string error;
  auto result =
      hivm::outlineGroups(*payload, hivm::Groups{{before[0]}, {}}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "outline group must not be empty");
  EXPECT_EQ(before, getBodyOps(*payload, "f"));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, RejectOverlappingGroupsBeforeMutation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");
  std::string error;
  auto result = hivm::outlineGroups(
      *payload, hivm::Groups{{before[0]}, {before[0], before[1]}}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "outline groups must be disjoint");
  EXPECT_EQ(before, getBodyOps(*payload, "f"));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, RejectUnorderedGroupBeforeMutation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");
  std::string error;
  auto result = hivm::outlineGroups(
      *payload, hivm::Groups{{before[1], before[0]}}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "outline group must be in topological order");
  EXPECT_EQ(before, getBodyOps(*payload, "f"));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, RejectTerminatorBeforeMutation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");
  std::string error;
  auto result =
      hivm::outlineGroups(*payload, hivm::Groups{{before.back()}}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "outline group must not contain a terminator");
  EXPECT_EQ(before, getBodyOps(*payload, "f"));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, RejectEarlyExternalUseBeforeMutation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto before = getBodyOps(*payload, "f");
  std::string error;
  auto result = hivm::outlineGroups(
      *payload, hivm::Groups{{before[0], before[2]}}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error,
            "outline group result has an external use before the last member");
  EXPECT_EQ(before, getBodyOps(*payload, "f"));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineLeavesNoVectorAnnotations) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions = runEngine(*payload, hivm::Groups{hivm::Group{ops[0]}});
  ASSERT_TRUE(succeeded(functions));

  payload->walk([&](func::FuncOp function) {
    EXPECT_FALSE(function->hasAttr("hivm.vector_function"));
    EXPECT_FALSE(function->hasAttr("no_inline"));
  });
}

TEST_F(OutlineTest, OutlineIsolatesOutlinerState) {
  auto payload = parse(kSharedArgumentsPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  auto functions = runEngine(
      *payload, hivm::Groups{hivm::Group{ops[0]}, hivm::Group{ops[1]}});
  ASSERT_TRUE(succeeded(functions));
  ASSERT_EQ(functions->size(), 2u);

  for (func::FuncOp function : *functions) {
    ASSERT_EQ(function.getNumArguments(), 2u);
    for (BlockArgument argument : function.getBody().front().getArguments())
      EXPECT_FALSE(argument.use_empty());
  }
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineOpReturnsFreshRootAndFunctions) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  auto result = runOutlineOp(*payload, kOutlineLibrary);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->newRoot.size(), 1u);
  EXPECT_EQ(result->newRoot.front(), payload->getOperation());
  ASSERT_EQ(result->functions.size(), 2u);
  for (Operation *function : result->functions)
    EXPECT_TRUE(isa<func::FuncOp>(function));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, OutlineOpAcceptsEmptyGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  auto result = runOutlineOp(*payload, kEmptyOutlineLibrary);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->newRoot.size(), 1u);
  EXPECT_EQ(result->newRoot.front(), payload->getOperation());
  EXPECT_TRUE(result->functions.empty());
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, AnnotateVFOpSetsAttributes) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  auto result = runOutlineOp(*payload, kOutlineAnnotateLibrary);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->functions.size(), 2u);
  for (Operation *function : result->functions) {
    EXPECT_TRUE(function->hasAttr("hivm.vector_function"));
    EXPECT_TRUE(function->hasAttr("no_inline"));
  }
  auto calls = payload->getOps<func::FuncOp>();
  for (auto function : calls)
    function.walk([&](func::CallOp call) {
      EXPECT_TRUE(call->hasAttr("hivm.vector_function"));
      EXPECT_TRUE(call->hasAttr("no_inline"));
    });
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, AnnotateVFOpRejectsNonFunction) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  EXPECT_TRUE(failed(runAnnotateOp(*payload, kAnnotateNonFunctionLibrary)));
}

TEST_F(OutlineTest, AnnotateVFOpAcceptsEmptyHandle) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  auto functions = runAnnotateOp(*payload, kEmptyAnnotateLibrary);
  ASSERT_TRUE(succeeded(functions));
  EXPECT_TRUE(functions->empty());
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(OutlineTest, ExtractGroupsOpExtractsGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  auto params = runExtractGroupsOp(*payload, kExtractGroupsLibrary);
  ASSERT_TRUE(succeeded(params));
  ASSERT_EQ(params->size(), 1u);

  auto groups = dyn_cast<hivm::GroupsAttr>(params->front());
  ASSERT_TRUE(groups);
  ASSERT_EQ(groups.getGroups().size(), 2u);
  EXPECT_EQ(groups.getGroups()[0].asArrayRef(), ArrayRef<int64_t>({2}));
  EXPECT_EQ(groups.getGroups()[1].asArrayRef(), ArrayRef<int64_t>({3}));
}

TEST_F(OutlineTest, ExtractGroupsOpRejectsNonPlan) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  EXPECT_TRUE(
      failed(runExtractGroupsOp(*payload, kExtractGroupsNonPlanLibrary)));
}

} // namespace
