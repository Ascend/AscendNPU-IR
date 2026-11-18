//===- IntermediateValuesAnalyzerTest.cpp - Transfer analysis tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "bishengir/Dialect/HFusion/Analysis/IntermediateValuesAnalyzer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iterator>

using namespace mlir;
using namespace mlir::hfusion::detail;

namespace {

class IntermediateValuesAnalyzerTest : public ::testing::Test {
protected:
  MLIRContext context;
  ModuleOp module;
  Value value;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    value =
        builder.create<arith::ConstantIntOp>(UnknownLoc::get(&context), 0, 32);
  }

  void TearDown() override { module.erase(); }
};

TEST_F(IntermediateValuesAnalyzerTest,
       ConsumedMarkStartsUnmarkedAndJoinUsesOr) {
  ConsumedMark unmarked(value);
  ConsumedMark marked(value);

  EXPECT_FALSE(unmarked.isMarked);
  EXPECT_FALSE(marked.isMarked);
  EXPECT_EQ(unmarked.join(marked), ChangeResult::NoChange);

  EXPECT_EQ(marked.mark(), ChangeResult::Change);
  EXPECT_EQ(unmarked.join(marked), ChangeResult::Change);
  EXPECT_TRUE(unmarked.isMarked);
  EXPECT_EQ(unmarked.join(marked), ChangeResult::NoChange);
}

TEST_F(IntermediateValuesAnalyzerTest,
       ProducedMarkStartsUnmarkedAndMeetUsesOr) {
  ProducedMark unmarked(value);
  ProducedMark marked(value);
  (void)marked.mark();

  EXPECT_EQ(unmarked.meet(marked), ChangeResult::Change);
  EXPECT_TRUE(unmarked.isMarked);
  EXPECT_EQ(unmarked.meet(marked), ChangeResult::NoChange);
}

TEST_F(IntermediateValuesAnalyzerTest, TrivialFoldExtractTest) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %value = tensor.extract %arg[%c0] : tensor<1xf32>
        %empty = tensor.empty() : tensor<1xf32>
        %result = tensor.insert %value into %empty[%c0] : tensor<1xf32>
        return %result : tensor<1xf32>
      }
    }
  )mlir";

  context.loadDialect<func::FuncDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  IntermediateValuesAnalyzer analyzer(*parsed);
  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto extract = func.getBody().front().getOperations().begin();
  std::advance(extract, 1);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(extract->getResult(0)));
}

TEST_F(IntermediateValuesAnalyzerTest, LinearSequenceTest) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %value = tensor.extract %arg[%c0] : tensor<1xf32>
        %sum = arith.addf %value, %value : f32
        %empty = tensor.empty() : tensor<1xf32>
        %result = tensor.insert %sum into %empty[%c0] : tensor<1xf32>
        return %result : tensor<1xf32>
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto sum = func.getBody().front().getOperations().begin();
  std::advance(sum, 2);

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(sum->getResult(0)));
}

TEST_F(IntermediateValuesAnalyzerTest, IfAsIntermediateTest) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %cond: i1) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %init = tensor.extract %arg[%c0] : tensor<1xf32>
        %result = scf.if %cond -> (f32) {
          %then = arith.addf %init, %init : f32
          scf.yield %then : f32
        } else {
          scf.yield %init : f32
        }
        %empty = tensor.empty() : tensor<1xf32>
        %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
        return %output : tensor<1xf32>
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto ifOp = func.getBody().front().getOperations().begin();
  std::advance(ifOp, 2);
  ASSERT_TRUE(isa<scf::IfOp>(ifOp));
  auto result = ifOp->getResult(0);

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(result));
}

TEST_F(IntermediateValuesAnalyzerTest,
       OptimisticallyPropagateMarkOutsideRegion) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %cond: i1) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %result = scf.if %cond -> (f32) {
          %init = tensor.extract %arg[%c0] : tensor<1xf32>
          %then = arith.addf %init, %init : f32
          scf.yield %then : f32
        } else {
          scf.yield %cst_0 : f32
        }
        %empty = tensor.empty() : tensor<1xf32>
        %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
        return %output : tensor<1xf32>
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto ifOp = func.getBody().front().getOperations().begin();
  std::advance(ifOp, 2);
  ASSERT_TRUE(isa<scf::IfOp>(ifOp));
  auto result = ifOp->getResult(0);

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(result));
}

TEST_F(IntermediateValuesAnalyzerTest, DontMarkNonIntermediateIf) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %cond: i1) -> f32 {
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %result = scf.if %cond -> (f32) {
          %init = tensor.extract %arg[%c0] : tensor<1xf32>
          %then = arith.addf %init, %init : f32
          scf.yield %then : f32
        } else {
          scf.yield %cst_0 : f32
        }

        %output = arith.addf %result, %result : f32
        return %output : f32
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto ifOp = func.getBody().front().getOperations().begin();
  std::advance(ifOp, 2);
  ASSERT_TRUE(isa<scf::IfOp>(ifOp));
  auto result = ifOp->getResult(0);

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_FALSE(analyzer.isIntermediateTransfer(result));
}

TEST_F(IntermediateValuesAnalyzerTest, ScfForWithTransferInRegion) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                      %lb: index, %ub: index, %step: index) -> f32 {
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant 1.000000e+00 : f32
        %empty = tensor.empty() : tensor<1xf32>
        %result = scf.for %i = %lb to %ub step %step
                  iter_args(%acc = %cst_0) -> (f32) {
          %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
          %sum = arith.addf %inserted, %val : tensor<1xf32>
          %extracted = tensor.extract %sum[%c0] : tensor<1xf32>
          scf.yield %extracted : f32
        }
        %output = arith.addf %result, %result : f32
        return %output : f32
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto forOpIt = func.getBody().front().getOperations().begin();
  std::advance(forOpIt, 3);
  ASSERT_TRUE(isa<scf::ForOp>(forOpIt));
  auto forOp = dyn_cast<scf::ForOp>(*forOpIt);
  auto bbArg = forOp.getRegionIterArg(0);
  auto result = forOp.getResult(0);
  Block &body = forOp.getBodyRegion().front();
  auto addOp = body.getOperations().begin();
  std::advance(addOp, 1);
  ASSERT_TRUE(isa<arith::AddFOp>(addOp));

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(bbArg));
  EXPECT_FALSE(analyzer.isIntermediateTransfer(addOp->getResult(0)));
  EXPECT_FALSE(analyzer.isIntermediateTransfer(result));
}

TEST_F(IntermediateValuesAnalyzerTest, ConsumeValueProducedInLoop) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                      %lb: index, %ub: index, %step: index) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant 1.000000e+00 : f32
        %empty = tensor.empty() : tensor<1xf32>
        %result = scf.for %i = %lb to %ub step %step
                  iter_args(%acc = %cst_0) -> (f32) {
          %extracted = tensor.extract %val[%c0] : tensor<1xf32>
          scf.yield %extracted : f32
        }
        %inserted = tensor.insert %result into %empty[%c0] : tensor<1xf32>
        return %inserted : tensor<1xf32>
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto forOpIt = func.getBody().front().getOperations().begin();
  std::advance(forOpIt, 3);
  ASSERT_TRUE(isa<scf::ForOp>(forOpIt));
  auto forOp = dyn_cast<scf::ForOp>(*forOpIt);
  Block &body = forOp.getBodyRegion().front();
  auto extractOp = body.getOperations().begin();
  auto bbArg = forOp.getRegionIterArg(0);
  ASSERT_TRUE(isa<tensor::ExtractOp>(extractOp));

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_FALSE(analyzer.isIntermediateTransfer(bbArg));
  EXPECT_TRUE(analyzer.isIntermediateTransfer(forOp->getResult(0)));
}

TEST_F(IntermediateValuesAnalyzerTest, RejectIntermediateFor) {
  const char *source = R"mlir(
    module {
      func.func @test(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                      %lb: index, %ub: index, %step: index) -> tensor<1xf32> {
        %c0 = arith.constant 0 : index
        %cst_0 = arith.constant 1.000000e+00 : f32
        %empty = tensor.empty() : tensor<1xf32>
        %extracted = tensor.extract %val[%c0] : tensor<1xf32>
        %result = scf.for %i = %lb to %ub step %step
                  iter_args(%acc = %extracted) -> (f32) {
            %add = arith.addf %acc, %cst_0 : f32
          scf.yield %add : f32
        }
        %inserted = tensor.insert %result into %empty[%c0] : tensor<1xf32>
        return %inserted : tensor<1xf32>
      }
    }
  )mlir";

  context
      .loadDialect<func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
  auto parsed = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(parsed);

  auto func = parsed->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(func);
  auto forOpIt = func.getBody().front().getOperations().begin();
  std::advance(forOpIt, 4);
  ASSERT_TRUE(isa<scf::ForOp>(forOpIt));
  auto forOp = dyn_cast<scf::ForOp>(*forOpIt);
  auto bbArg = forOp.getRegionIterArg(0);

  IntermediateValuesAnalyzer analyzer(*parsed);
  EXPECT_TRUE(analyzer.isIntermediateTransfer(bbArg));
  EXPECT_TRUE(analyzer.isIntermediateTransfer(forOp->getResult(0)));
}

} // namespace
