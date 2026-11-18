//===- AgglomerationBlockTest.cpp - Agglomeration block unit tests --------===//
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

#include "bishengir/Analysis/AgglomerationBlock.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

using namespace mlir;

namespace {

constexpr StringLiteral kChainPayload = R"mlir(
module {
  func.func @chain(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %0, %b : f32
    return %1 : f32
  }
}
)mlir";

constexpr StringLiteral kConstantPayload = R"mlir(
module {
  func.func @constant(%a: f32) -> f32 {
    %c = arith.constant 1.000000e+00 : f32
    %0 = arith.addf %a, %c : f32
    return %0 : f32
  }
}
)mlir";

constexpr StringLiteral kNestedPayload = R"mlir(
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

constexpr StringLiteral kLoopPayload = R"mlir(
module {
  func.func @loop(%lb: index, %ub: index, %step: index, %init: f32,
                  %a: f32) -> f32 {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
      %next = arith.addf %acc, %a : f32
      scf.yield %next : f32
    }
    return %r : f32
  }
}
)mlir";

constexpr StringLiteral kMultiResultPayload = R"mlir(
module {
  func.func @multi(%cond: i1, %a: f32, %b: f32) -> f32 {
    %x, %y = scf.if %cond -> (f32, f32) {
      scf.yield %a, %b : f32, f32
    } else {
      scf.yield %b, %a : f32, f32
    }
    return %x : f32
  }
}
)mlir";

constexpr StringLiteral kInternalUsePayload = R"mlir(
module {
  func.func @internal(%cond: i1, %a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %r = scf.if %cond -> (f32) {
      scf.yield %0 : f32
    } else {
      scf.yield %0 : f32
    }
    return %r : f32
  }
}
)mlir";

class AgglomerationBlockTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
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

  SmallVector<Value> getArguments(ModuleOp module, StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function.getOperation());
    SmallVector<Value> arguments;
    for (BlockArgument argument : function.getArguments())
      arguments.push_back(argument);
    return arguments;
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(AgglomerationBlockTest, EmptyBlockHasNoBoundaries) {
  hivm::AgglomerationBlock block;
  EXPECT_TRUE(block.empty());
  EXPECT_EQ(block.size(), 0u);
  EXPECT_TRUE(block.getInputs().empty());
  EXPECT_TRUE(block.getOutputs().empty());
}

TEST_F(AgglomerationBlockTest, AddIgnoresNull) {
  hivm::AgglomerationBlock block;
  block.add(nullptr);
  EXPECT_TRUE(block.empty());
  EXPECT_TRUE(block.getInputs().empty());
  EXPECT_TRUE(block.getOutputs().empty());
}

TEST_F(AgglomerationBlockTest, AddPreservesOrderAndDeduplicates) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[1]);
  block.add(ops[0]);
  block.add(ops[1]);
  ASSERT_EQ(block.size(), 2u);
  EXPECT_EQ(block.getOps()[0], ops[1]);
  EXPECT_EQ(block.getOps()[1], ops[0]);
  EXPECT_TRUE(block.contains(ops[0]));
  EXPECT_TRUE(block.contains(ops[1]));
}

TEST_F(AgglomerationBlockTest, RemoveUpdatesMembership) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  block.add(ops[1]);
  block.remove(ops[0]);
  ASSERT_EQ(block.size(), 1u);
  EXPECT_EQ(block.getOps()[0], ops[1]);
  EXPECT_FALSE(block.contains(ops[0]));

  block.remove(ops[0]);
  EXPECT_EQ(block.size(), 1u);
  block.remove(ops[1]);
  EXPECT_TRUE(block.empty());
}

TEST_F(AgglomerationBlockTest, SingleOpBoundaries) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");
  auto arguments = getArguments(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  ASSERT_EQ(block.getInputs().size(), 2u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, ChainedBlockHidesInternalValue) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");
  auto arguments = getArguments(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  block.add(ops[1]);
  ASSERT_EQ(block.getInputs().size(), 2u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
  EXPECT_FALSE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, ConsumerOnlyBlockUsesProducerResult) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");
  auto arguments = getArguments(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[1]);
  ASSERT_EQ(block.getInputs().size(), 2u);
  EXPECT_TRUE(block.getInputs().contains(ops[0]->getResult(0)));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, ConstantsAndInternalDefinitionsAreNotInputs) {
  auto payload = parse(kConstantPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "constant");
  auto arguments = getArguments(*payload, "constant");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  block.add(ops[1]);
  ASSERT_EQ(block.getInputs().size(), 1u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_FALSE(block.getInputs().contains(ops[0]->getResult(0)));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
  EXPECT_FALSE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, NestedRegionValuesAreInputs) {
  auto payload = parse(kNestedPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "nested");
  auto arguments = getArguments(*payload, "nested");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  ASSERT_EQ(block.getInputs().size(), 3u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  EXPECT_TRUE(block.getInputs().contains(arguments[2]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, LoopBlockArgumentsAreNotInputs) {
  auto payload = parse(kLoopPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "loop");
  auto arguments = getArguments(*payload, "loop");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  ASSERT_EQ(block.getInputs().size(), 5u);
  for (Value argument : arguments)
    EXPECT_TRUE(block.getInputs().contains(argument));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, NestedResultUsedInsideBlockIsNotOutput) {
  auto payload = parse(kInternalUsePayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "internal");
  auto arguments = getArguments(*payload, "internal");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  block.add(ops[1]);
  ASSERT_EQ(block.getInputs().size(), 3u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  EXPECT_TRUE(block.getInputs().contains(arguments[2]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
  EXPECT_FALSE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, MultipleResultsOnlyEscapingAreOutputs) {
  auto payload = parse(kMultiResultPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "multi");
  auto arguments = getArguments(*payload, "multi");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  ASSERT_EQ(block.getInputs().size(), 3u);
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));
  EXPECT_FALSE(block.getOutputs().contains(ops[0]->getResult(1)));
}

TEST_F(AgglomerationBlockTest, ConstantSingletonHasNoInputs) {
  auto payload = parse(kConstantPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "constant");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  EXPECT_TRUE(block.getInputs().empty());
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, AddOrderDoesNotChangeBoundaries) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");
  auto arguments = getArguments(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[1]);
  block.add(ops[0]);
  ASSERT_EQ(block.getInputs().size(), 2u);
  EXPECT_TRUE(block.getInputs().contains(arguments[0]));
  EXPECT_TRUE(block.getInputs().contains(arguments[1]));
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, BoundariesRecomputeAfterAdd) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));

  block.add(ops[1]);
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));
  EXPECT_FALSE(block.getOutputs().contains(ops[0]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, BoundariesRecomputeAfterRemove) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  block.add(ops[1]);
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[1]->getResult(0)));

  block.remove(ops[1]);
  ASSERT_EQ(block.getOutputs().size(), 1u);
  EXPECT_TRUE(block.getOutputs().contains(ops[0]->getResult(0)));

  block.remove(ops[0]);
  EXPECT_TRUE(block.getInputs().empty());
  EXPECT_TRUE(block.getOutputs().empty());
}

TEST_F(AgglomerationBlockTest, CopyIsIndependent) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock original;
  original.add(ops[0]);
  hivm::AgglomerationBlock copy = original;
  copy.add(ops[1]);

  ASSERT_EQ(original.size(), 1u);
  ASSERT_EQ(original.getOutputs().size(), 1u);
  EXPECT_TRUE(original.getOutputs().contains(ops[0]->getResult(0)));

  ASSERT_EQ(copy.size(), 2u);
  ASSERT_EQ(copy.getOutputs().size(), 1u);
  EXPECT_TRUE(copy.getOutputs().contains(ops[1]->getResult(0)));
}

TEST_F(AgglomerationBlockTest, ConstQueriesRecomputeBoundaries) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());
  auto ops = getBodyOps(*payload, "chain");

  hivm::AgglomerationBlock block;
  block.add(ops[0]);
  const hivm::AgglomerationBlock &constBlock = block;
  ASSERT_EQ(constBlock.getInputs().size(), 2u);
  ASSERT_EQ(constBlock.getOutputs().size(), 1u);
  EXPECT_TRUE(constBlock.getOutputs().contains(ops[0]->getResult(0)));
}

} // namespace
