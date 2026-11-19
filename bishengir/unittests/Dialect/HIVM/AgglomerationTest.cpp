//===- AgglomerationTest.cpp - HIVM agglomeration unit tests --------------===//
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

#include "bishengir/Analysis/Partition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/Agglomerate.h"
#include "bishengir/Dialect/HIVM/TransformOps/Groups.h"
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
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>

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

constexpr StringLiteral kHIVMPayload = R"mlir(
module {
  func.func @flat(%a: tensor<8xf32>, %b: tensor<8xf32>, %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    return %product : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kTwoFunctionPayload = R"mlir(
module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    return %0 : f32
  }
  func.func @g(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    return %0 : f32
  }
}
)mlir";

constexpr StringLiteral kTwoBlockPayload = R"mlir(
module {
  func.func @two_blocks(%cond: i1, %a: f32, %b: f32) -> f32 {
    %r = scf.if %cond -> (f32) {
      %x = arith.addf %a, %b : f32
      scf.yield %x : f32
    } else {
      %y = arith.subf %a, %b : f32
      scf.yield %y : f32
    }
    return %r : f32
  }
}
)mlir";

constexpr StringLiteral kNestedRegionPayload = R"mlir(
module {
  func.func @nested(%cond: i1, %a: f32, %b: f32) -> f32 {
    %r = scf.if %cond -> (f32) {
      %s = arith.addf %a, %b : f32
      scf.yield %s : f32
    } else {
      scf.yield %a : f32
    }
    return %r : f32
  }
}
)mlir";

constexpr StringLiteral kAgglomerateLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op {transform.readonly},
      %options: !transform.any_param {transform.readonly}
  ) -> !transform.any_param {
    %groups = transform.hivm.partition %root, %options
        {models = ["SameBlock", "UnseparatedByBarrier"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield %plan : !transform.any_param
  }
}
)mlir";

hivm::AgglomerationBlock fuseBlocks(ArrayRef<hivm::AgglomerationBlock> blocks,
                                    hivm::AgglomerationBlockIndex first,
                                    hivm::AgglomerationBlockIndex second) {
  hivm::AgglomerationBlock fused;
  for (Operation *op : blocks[first].getOps())
    fused.add(op);
  for (Operation *op : blocks[second].getOps())
    fused.add(op);
  return fused;
}

hivm::FusionCandidate makeCandidate(ArrayRef<hivm::AgglomerationBlock> blocks,
                                    hivm::AgglomerationBlockIndex first,
                                    hivm::AgglomerationBlockIndex second) {
  hivm::FusionCandidate candidate;
  candidate.sources = {first, second};
  candidate.fused = fuseBlocks(blocks, first, second);
  return candidate;
}

class LambdaSelector final : public hivm::PairSelector {
public:
  using SelectFn = std::function<SmallVector<hivm::FusionCandidate>(
      ArrayRef<hivm::AgglomerationBlock>)>;

  explicit LambdaSelector(SelectFn fn) : fn(std::move(fn)) {}

  FailureOr<SmallVector<hivm::FusionCandidate>>
  select(ArrayRef<hivm::AgglomerationBlock> blocks,
         const hivm::AgglomerationContext &, std::string &) const override {
    return fn(blocks);
  }

private:
  SelectFn fn;
};

class ConstantCostModel final : public hivm::FusionCostModel {
public:
  explicit ConstantCostModel(double value) : value(value) {}

  FailureOr<double> evaluate(const hivm::AgglomerationBlock &,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    return value;
  }

private:
  double value;
};

class LinearCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &block,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    return static_cast<double>(block.getOps().size());
  }
};

class BoundaryCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &block,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    return 1.0 + static_cast<double>(block.getInputs().size() +
                                     block.getOutputs().size());
  }
};

class QuadraticCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &block,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    double count = static_cast<double>(block.getOps().size());
    return count * count;
  }
};

class FailingCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &,
                             const hivm::AgglomerationContext &,
                             std::string &error) const override {
    error = "test cost model failure";
    return failure();
  }
};

class NonFiniteCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    return std::numeric_limits<double>::infinity();
  }
};

class NegativeCostModel final : public hivm::FusionCostModel {
public:
  FailureOr<double> evaluate(const hivm::AgglomerationBlock &,
                             const hivm::AgglomerationContext &,
                             std::string &) const override {
    return -1.0;
  }
};

class AgglomerationTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    transform::TransformDialect, hivm::HIVMDialect>();
    hivm::registerVectorizeTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadDialect<hivm::HIVMDialect, hivm::HIVMVectorizeDialect>();
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

  hivm::Groups makeGroups(ArrayRef<Operation *> ops) {
    hivm::Groups groups;
    for (Operation *op : ops)
      groups.push_back(hivm::Group{op});
    return groups;
  }

  FailureOr<hivm::AgglomerationResult>
  runEngine(ModuleOp module, ArrayRef<Operation *> members, StringRef selector,
            ArrayRef<StringRef> models, std::string &error) {
    return hivm::agglomerate(module, DictionaryAttr::get(context.get()),
                             makeGroups(members), selector, models, error);
  }

  void registerSelector(StringRef name,
                        std::unique_ptr<hivm::PairSelector> selector) {
    ASSERT_TRUE(succeeded(hivm::registerSelector(name, std::move(selector))));
  }

  void registerCostModel(StringRef name,
                         std::unique_ptr<hivm::FusionCostModel> model) {
    ASSERT_TRUE(
        succeeded(hivm::registerFusionCostModel(name, std::move(model))));
  }

  FailureOr<Attribute> runAgglomerateOp(ModuleOp payload,
                                        ArrayRef<Operation *> roots,
                                        Attribute options) {
    auto library = parse(kAgglomerateLibrary);
    if (!library)
      return failure();
    auto entry = library->lookupSymbol<transform::NamedSequenceOp>("entry");
    if (!entry)
      return failure();

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(&entry.getBody(),
                                                        payload.getOperation());
    Block &body = entry.getBody().front();
    transform::HIVMAgglomerateOp agglomerateOp;
    for (Operation &op : body.without_terminator())
      if (auto candidate = dyn_cast<transform::HIVMAgglomerateOp>(op))
        agglomerateOp = candidate;
    if (!agglomerateOp)
      return failure();

    SmallVector<SmallVector<transform::MappedValue>> mappings;
    mappings.push_back(
        SmallVector<transform::MappedValue>(roots.begin(), roots.end()));
    mappings.push_back(SmallVector<transform::MappedValue>{options});
    if (failed(state.mapBlockArguments(body.getArguments(), mappings)))
      return failure();

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

    auto params = state.getParams(agglomerateOp.getPlan());
    if (params.size() != 1)
      return failure();
    return params.front();
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(AgglomerationTest, RegistryRejectsEmptyAndDuplicateNames) {
  EXPECT_TRUE(failed(hivm::registerSelector(
      "",
      std::make_unique<LambdaSelector>([](ArrayRef<hivm::AgglomerationBlock>) {
        return SmallVector<hivm::FusionCandidate>{};
      }))));
  EXPECT_TRUE(failed(hivm::registerSelector(
      "preserve_groups",
      std::make_unique<LambdaSelector>([](ArrayRef<hivm::AgglomerationBlock>) {
        return SmallVector<hivm::FusionCandidate>{};
      }))));
  EXPECT_TRUE(failed(hivm::registerFusionCostModel(
      "", std::make_unique<ConstantCostModel>(1.0))));
  EXPECT_TRUE(failed(hivm::registerFusionCostModel(
      "uniform_cost", std::make_unique<ConstantCostModel>(1.0))));

  EXPECT_TRUE(hivm::lookupSelector("preserve_groups"));
  EXPECT_TRUE(hivm::lookupFusionCostModel("uniform_cost"));
  EXPECT_FALSE(hivm::lookupSelector("missing"));
  EXPECT_FALSE(hivm::lookupFusionCostModel("missing"));
}

TEST_F(AgglomerationTest, DefaultSelectorReturnsNoCandidates) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  std::string error;
  auto groups = makeGroups({ops[0], ops[1]});
  auto blocks = hivm::buildInitialBlocks(*payload, groups, error);
  ASSERT_TRUE(succeeded(blocks)) << error;

  auto selector = hivm::lookupSelector("preserve_groups");
  ASSERT_TRUE(selector);
  auto candidates = selector->get().select(
      *blocks, {*payload, DictionaryAttr::get(context.get())}, error);
  ASSERT_TRUE(succeeded(candidates)) << error;
  EXPECT_TRUE(candidates->empty());
}

TEST_F(AgglomerationTest, DefaultCostModelIsConstant) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  std::string error;
  auto groups = makeGroups({ops[0]});
  auto blocks = hivm::buildInitialBlocks(*payload, groups, error);
  ASSERT_TRUE(succeeded(blocks)) << error;

  auto model = hivm::lookupFusionCostModel("uniform_cost");
  ASSERT_TRUE(model);
  hivm::AgglomerationContext context{*payload,
                                     DictionaryAttr::get(this->context.get())};

  auto singleton = model->get().evaluate((*blocks)[0], context, error);
  ASSERT_TRUE(succeeded(singleton)) << error;
  EXPECT_DOUBLE_EQ(*singleton, 1.0);

  hivm::AgglomerationBlock fused = (*blocks)[0];
  fused.add(ops[1]);
  auto merged = model->get().evaluate(fused, context, error);
  ASSERT_TRUE(succeeded(merged)) << error;
  EXPECT_DOUBLE_EQ(*merged, 1.0);
}

TEST_F(AgglomerationTest, EngineWithoutCandidatesKeepsGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "preserve_groups",
                          {StringRef("uniform_cost")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  EXPECT_EQ(result->groups.size(), 2u);
  EXPECT_EQ(result->groups[0].size(), 1u);
  EXPECT_EQ(result->groups[1].size(), 1u);
  EXPECT_DOUBLE_EQ(result->cost, 2.0);
}

TEST_F(AgglomerationTest, EngineAppliesSingleCandidate) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.first_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.zero", std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.first_pair",
                          {StringRef("test.zero")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 1u);
  ASSERT_EQ(result->groups[0].size(), 2u);
  EXPECT_EQ(result->groups[0][0], ops[0]);
  EXPECT_EQ(result->groups[0][1], ops[1]);
  EXPECT_DOUBLE_EQ(result->cost, 0.0);
}

TEST_F(AgglomerationTest, EnginePicksBestAverageDelta) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.pair_of_pairs",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() != 3)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 1, 2),
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.boundary", std::make_unique<BoundaryCostModel>());

  std::string error;
  auto result =
      runEngine(*payload, {ops[0], ops[1], ops[2]}, "test.pair_of_pairs",
                {StringRef("test.boundary")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 2u);
  ASSERT_EQ(result->groups[0].size(), 2u);
  EXPECT_EQ(result->groups[0][0], ops[0]);
  EXPECT_EQ(result->groups[0][1], ops[1]);
  ASSERT_EQ(result->groups[1].size(), 1u);
  EXPECT_EQ(result->groups[1][0], ops[2]);
}

TEST_F(AgglomerationTest, EngineTieKeepsFirstCandidate) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.tie_pairs",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() != 3)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1),
                             makeCandidate(blocks, 1, 2)};
                       }));
  registerCostModel("test.tie_zero", std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1], ops[2]}, "test.tie_pairs",
                          {StringRef("test.tie_zero")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 2u);
  ASSERT_EQ(result->groups[0].size(), 2u);
  EXPECT_EQ(result->groups[0][0], ops[0]);
  EXPECT_EQ(result->groups[0][1], ops[1]);
  ASSERT_EQ(result->groups[1].size(), 1u);
  EXPECT_EQ(result->groups[1][0], ops[2]);
}

TEST_F(AgglomerationTest, EngineAcceptsEqualCost) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.equal_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.linear", std::make_unique<LinearCostModel>());

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.equal_pair",
                          {StringRef("test.linear")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  EXPECT_EQ(result->groups.size(), 1u);
}

TEST_F(AgglomerationTest, EngineStopsOnWorseningCandidate) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.worsening_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.quadratic", std::make_unique<QuadraticCostModel>());

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.worsening_pair",
                          {StringRef("test.quadratic")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  EXPECT_EQ(result->groups.size(), 2u);
}

TEST_F(AgglomerationTest, EngineStopsWithSingleBlock) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.always_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.always_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.always_pair",
                          {StringRef("test.always_zero")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 1u);
  EXPECT_EQ(result->groups[0][0], ops[0]);
  EXPECT_EQ(result->groups[0][1], ops[1]);
}

TEST_F(AgglomerationTest, ContractRejectsOutOfRangeSource) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.out_of_range",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         hivm::FusionCandidate candidate =
                             makeCandidate(blocks, 0, 1);
                         candidate.sources.second = 7;
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.range_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.out_of_range",
                          {StringRef("test.range_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "agglomeration candidate source index is out of range");
}

TEST_F(AgglomerationTest, ContractRejectsDuplicateSource) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.duplicate_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         hivm::FusionCandidate candidate =
                             makeCandidate(blocks, 0, 1);
                         candidate.sources.second = candidate.sources.first;
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.duplicate_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.duplicate_pair",
                          {StringRef("test.duplicate_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "agglomeration candidate contains a duplicate source index");
}

TEST_F(AgglomerationTest, ContractRejectsEmptyFusedBlock) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector(
      "test.empty_fused",
      std::make_unique<LambdaSelector>([](ArrayRef<hivm::AgglomerationBlock>) {
        hivm::FusionCandidate candidate;
        candidate.sources = {0, 1};
        return SmallVector<hivm::FusionCandidate>{candidate};
      }));
  registerCostModel("test.empty_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.empty_fused",
                          {StringRef("test.empty_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "agglomeration candidate fused block is empty");
}

TEST_F(AgglomerationTest, ContractRejectsFusedSubsetMismatch) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.subset_mismatch",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         hivm::FusionCandidate candidate;
                         candidate.sources = {0, 1};
                         candidate.fused = blocks[0];
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.subset_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.subset_mismatch",
                          {StringRef("test.subset_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error,
            "agglomeration candidate fused block does not match the union of "
            "its source blocks");
}

TEST_F(AgglomerationTest, ContractRejectsMultipleParentBlocks) {
  auto payload = parse(kTwoBlockPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "two_blocks");
  Operation *ifOp = ops[0];
  Operation *thenAdd = nullptr;
  Operation *elseSub = nullptr;
  ifOp->walk([&](arith::AddFOp op) { thenAdd = op.getOperation(); });
  ifOp->walk([&](arith::SubFOp op) { elseSub = op.getOperation(); });
  ASSERT_TRUE(thenAdd);
  ASSERT_TRUE(elseSub);

  registerSelector("test.multiple_parents",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         hivm::FusionCandidate candidate;
                         candidate.sources = {0, 1};
                         for (Operation *op : blocks[0].getOps())
                           candidate.fused.add(op);
                         for (Operation *op : blocks[1].getOps())
                           candidate.fused.add(op);
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.parent_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {thenAdd, elseSub}, "test.multiple_parents",
                          {StringRef("test.parent_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error,
            "agglomeration candidate fused block must be in one parent block");
}

TEST_F(AgglomerationTest, ContractRejectsOutOfOrderFusedBlock) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.out_of_order",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         hivm::FusionCandidate candidate;
                         candidate.sources = {0, 1};
                         candidate.fused = fuseBlocks(blocks, 1, 0);
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.order_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.out_of_order",
                          {StringRef("test.order_zero")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error,
            "agglomeration candidate fused block must be in topological order");
}

TEST_F(AgglomerationTest, EngineNormalizesSourceOrder) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.unsorted_sources",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         hivm::FusionCandidate candidate;
                         candidate.sources = {1, 0};
                         candidate.fused = fuseBlocks(blocks, 0, 1);
                         return SmallVector<hivm::FusionCandidate>{candidate};
                       }));
  registerCostModel("test.unsorted_zero",
                    std::make_unique<ConstantCostModel>(0.0));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.unsorted_sources",
                          {StringRef("test.unsorted_zero")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 1u);
  ASSERT_EQ(result->groups[0].size(), 2u);
  EXPECT_EQ(result->groups[0][0], ops[0]);
  EXPECT_EQ(result->groups[0][1], ops[1]);
}

TEST_F(AgglomerationTest, CostModelFailurePropagates) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.failing_model_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.failing", std::make_unique<FailingCostModel>());

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.failing_model_pair",
                          {StringRef("test.failing")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "test cost model failure");
}

TEST_F(AgglomerationTest, CostModelRejectsNonFiniteCost) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.nonfinite_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.nonfinite", std::make_unique<NonFiniteCostModel>());

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.nonfinite_pair",
                          {StringRef("test.nonfinite")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "fusion cost must be finite");
}

TEST_F(AgglomerationTest, CostModelRejectsNegativeCost) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.negative_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.negative", std::make_unique<NegativeCostModel>());

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.negative_pair",
                          {StringRef("test.negative")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "fusion cost must be non-negative");
}

TEST_F(AgglomerationTest, PlanCostAveragesModels) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.average_pair",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.one", std::make_unique<ConstantCostModel>(1.0));
  registerCostModel("test.three", std::make_unique<ConstantCostModel>(3.0));

  std::string error;
  auto result =
      runEngine(*payload, {ops[0], ops[1]}, "test.average_pair",
                {StringRef("test.one"), StringRef("test.three")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->groups.size(), 1u);
  EXPECT_DOUBLE_EQ(result->cost, 2.0);
}

TEST_F(AgglomerationTest, PlanCostRejectsNonFiniteSum) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerCostModel("test.huge", std::make_unique<ConstantCostModel>(1.0e308));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "preserve_groups",
                          {StringRef("test.huge")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "plan cost must be finite");
}

TEST_F(AgglomerationTest, CostModelRejectsNonFiniteDelta) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  registerSelector("test.delta_overflow",
                   std::make_unique<LambdaSelector>(
                       [](ArrayRef<hivm::AgglomerationBlock> blocks) {
                         if (blocks.size() < 2)
                           return SmallVector<hivm::FusionCandidate>{};
                         return SmallVector<hivm::FusionCandidate>{
                             makeCandidate(blocks, 0, 1)};
                       }));
  registerCostModel("test.delta_huge",
                    std::make_unique<ConstantCostModel>(1.0e308));

  std::string error;
  auto result = runEngine(*payload, {ops[0], ops[1]}, "test.delta_overflow",
                          {StringRef("test.delta_huge")}, error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "fusion cost delta must be finite");
}

TEST_F(AgglomerationTest, DecodeGroupsUsesPreorderOrdinals) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  SmallVector<DenseI64ArrayAttr> encoded{
      DenseI64ArrayAttr::get(context.get(), {2}),
      DenseI64ArrayAttr::get(context.get(), {3})};
  auto attr = hivm::GroupsAttr::get(context.get(), encoded);

  std::string error;
  auto groups = hivm::decodeGroups(*payload, attr, error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 2u);
  EXPECT_EQ((*groups)[0][0], ops[0]);
  EXPECT_EQ((*groups)[1][0], ops[1]);
}

TEST_F(AgglomerationTest, DecodeGroupsRejectsOutOfRangeOrdinal) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  SmallVector<DenseI64ArrayAttr> encoded{
      DenseI64ArrayAttr::get(context.get(), {99})};
  auto attr = hivm::GroupsAttr::get(context.get(), encoded);

  std::string error;
  auto groups = hivm::decodeGroups(*payload, attr, error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "group ordinal 99 is outside the payload root");
}

TEST_F(AgglomerationTest, DecodeGroupsRejectsModuleMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  SmallVector<DenseI64ArrayAttr> encoded{
      DenseI64ArrayAttr::get(context.get(), {0})};
  auto attr = hivm::GroupsAttr::get(context.get(), encoded);

  std::string error;
  auto groups = hivm::decodeGroups(*payload, attr, error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "group member must not be a module");
}

TEST_F(AgglomerationTest, DecodeGroupsRejectsFunctionMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  SmallVector<DenseI64ArrayAttr> encoded{
      DenseI64ArrayAttr::get(context.get(), {1})};
  auto attr = hivm::GroupsAttr::get(context.get(), encoded);

  std::string error;
  auto groups = hivm::decodeGroups(*payload, attr, error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "group member must not be a function");
}

TEST_F(AgglomerationTest, BuildGroupOrdinalsMatchesPreorder) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");
  auto function = payload->lookupSymbol<func::FuncOp>("f");

  std::string error;
  auto ordinals = hivm::buildGroupOrdinals(*payload, error);
  ASSERT_TRUE(succeeded(ordinals)) << error;
  EXPECT_EQ(ordinals->lookup(payload->getOperation()), 0);
  EXPECT_EQ(ordinals->lookup(function.getOperation()), 1);
  EXPECT_EQ(ordinals->lookup(ops[0]), 2);
  EXPECT_EQ(ordinals->lookup(ops[1]), 3);
  EXPECT_EQ(ordinals->lookup(ops[2]), 4);
}

TEST_F(AgglomerationTest, BuildGroupOrdinalsMatchesTwoFunctions) {
  auto payload = parse(kTwoFunctionPayload);
  ASSERT_TRUE(payload.get());
  auto firstOps = getBodyOps(*payload, "f");
  auto secondOps = getBodyOps(*payload, "g");

  std::string error;
  auto ordinals = hivm::buildGroupOrdinals(*payload, error);
  ASSERT_TRUE(succeeded(ordinals)) << error;
  EXPECT_EQ(ordinals->lookup(firstOps[0]), 2);
  EXPECT_EQ(ordinals->lookup(secondOps[0]), 5);
}

TEST_F(AgglomerationTest, EncodeGroupsUsesPreorderOrdinals) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  std::string error;
  auto encoded =
      hivm::encodeGroups(*payload, hivm::Groups{{ops[0]}, {ops[1]}}, error);
  ASSERT_TRUE(succeeded(encoded)) << error;
  ASSERT_EQ(encoded->getGroups().size(), 2u);
  EXPECT_EQ(encoded->getGroups()[0].asArrayRef(), ArrayRef<int64_t>({2}));
  EXPECT_EQ(encoded->getGroups()[1].asArrayRef(), ArrayRef<int64_t>({3}));
}

TEST_F(AgglomerationTest, EncodeGroupsRejectsForeignOperation) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto other = parse(kPayload);
  ASSERT_TRUE(other.get());
  auto otherOps = getBodyOps(*other, "f");

  std::string error;
  auto encoded =
      hivm::encodeGroups(*payload, hivm::Groups{{otherOps[0]}}, error);
  EXPECT_TRUE(failed(encoded));
  EXPECT_EQ(error, "group member is outside the payload root");
}

TEST_F(AgglomerationTest, EncodeGroupsRejectsModuleMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto encoded = hivm::encodeGroups(
      *payload, hivm::Groups{{payload->getOperation()}}, error);
  EXPECT_TRUE(failed(encoded));
  EXPECT_EQ(error, "group member must not be a module");
}

TEST_F(AgglomerationTest, EncodeGroupsRejectsFunctionMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto function = payload->lookupSymbol<func::FuncOp>("f");

  std::string error;
  auto encoded = hivm::encodeGroups(
      *payload, hivm::Groups{{function.getOperation()}}, error);
  EXPECT_TRUE(failed(encoded));
  EXPECT_EQ(error, "group member must not be a function");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsEmptyGroup) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto blocks =
      hivm::buildInitialBlocks(*payload, hivm::Groups{hivm::Group{}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "initial agglomeration group must not be empty");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsOverlappingGroups) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "f");

  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{ops[0]}, {ops[0]}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "initial agglomeration blocks must be disjoint");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsNestedGroups) {
  auto payload = parse(kNestedRegionPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "nested");
  Operation *ifOp = ops[0];
  Operation *innerAdd = nullptr;
  ifOp->walk([&](arith::AddFOp op) { innerAdd = op.getOperation(); });
  ASSERT_TRUE(innerAdd);

  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{ifOp}, {innerAdd}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "initial agglomeration blocks must not be nested");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsModuleMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{payload->getOperation()}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "group member must not be a module");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsFunctionMember) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto function = payload->lookupSymbol<func::FuncOp>("f");

  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{function.getOperation()}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "group member must not be a function");
}

TEST_F(AgglomerationTest, BuildInitialBlocksRejectsInterproceduralGroups) {
  auto payload = parse(kTwoFunctionPayload);
  ASSERT_TRUE(payload.get());
  auto firstOps = getBodyOps(*payload, "f");
  auto secondOps = getBodyOps(*payload, "g");

  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{firstOps[0], secondOps[0]}}, error);
  EXPECT_TRUE(failed(blocks));
  EXPECT_EQ(error, "agglomeration groups must belong to one function");
}

TEST_F(AgglomerationTest, BuildInitialBlocksAcceptsSeparateFunctions) {
  auto payload = parse(kTwoFunctionPayload);
  ASSERT_TRUE(payload.get());
  auto firstOps = getBodyOps(*payload, "f");
  auto secondOps = getBodyOps(*payload, "g");
  std::string error;
  auto blocks = hivm::buildInitialBlocks(
      *payload, hivm::Groups{{firstOps[0]}, {secondOps[0]}}, error);
  ASSERT_TRUE(succeeded(blocks)) << error;
  EXPECT_EQ(blocks->size(), 2u);
}

TEST_F(AgglomerationTest, EmptyGroupsProduceZeroCostPlan) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto result = hivm::agglomerate(*payload, DictionaryAttr::get(context.get()),
                                  hivm::Groups{}, "preserve_groups",
                                  {StringRef("uniform_cost")}, error);
  ASSERT_TRUE(succeeded(result)) << error;
  EXPECT_TRUE(result->groups.empty());
  EXPECT_DOUBLE_EQ(result->cost, 0.0);
}

TEST_F(AgglomerationTest, AgglomerateOpProducesPlan) {
  auto payload = parse(kHIVMPayload);
  ASSERT_TRUE(payload.get());
  auto function = payload->lookupSymbol<func::FuncOp>("flat");
  ASSERT_TRUE(function.getOperation());

  auto plan = runAgglomerateOp(*payload, {payload->getOperation()},
                               DictionaryAttr::get(context.get()));
  ASSERT_TRUE(succeeded(plan));

  auto planAttr = dyn_cast<hivm::PlanAttr>(*plan);
  ASSERT_TRUE(planAttr);
  EXPECT_DOUBLE_EQ(planAttr.getCost().getValueAsDouble(), 1.0);
  ASSERT_EQ(planAttr.getGroups().getGroups().size(), 1u);
  EXPECT_EQ(planAttr.getGroups().getGroups()[0].asArrayRef(),
            ArrayRef<int64_t>({2, 3}));
}

} // namespace
