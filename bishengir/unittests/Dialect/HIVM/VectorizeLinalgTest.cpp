//===- VectorizeLinalgTest.cpp - HIVM linalg vectorization tests ----------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeLinalg.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeResult.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr StringLiteral kUnaryPayload = R"mlir(
module {
  func.func private @vf(%arg: tensor<8xf32>, %init: tensor<8xf32>)
      -> tensor<8xf32> attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%arg : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp2 %in : f32
      linalg.yield %1 : f32
    } -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kChainPayload = R"mlir(
module {
  func.func private @vf(%lhs: tensor<8x16xf32>, %rhs: tensor<8x16xf32>,
                        %init: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<8x16xf32>)
        outs(%init : tensor<8x16xf32>) {
    ^bb0(%lhsIn: f32, %rhsIn: f32, %out: f32):
      %1 = arith.addf %lhsIn, %rhsIn : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
    %2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = math.exp2 %in : f32
      linalg.yield %3 : f32
    } -> tensor<8x16xf32>
    return %2 : tensor<8x16xf32>
  }
}
)mlir";

constexpr StringLiteral kBroadcastPayload = R"mlir(
module {
  func.func private @vf(%lhs: tensor<8x1xf32>, %rhs: tensor<8x16xf32>,
                        %init: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<8x1xf32>, tensor<8x16xf32>)
        outs(%init : tensor<8x16xf32>) {
    ^bb0(%lhsIn: f32, %rhsIn: f32, %out: f32):
      %1 = arith.addf %lhsIn, %rhsIn : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}
)mlir";

constexpr StringLiteral kMultiResultPayload = R"mlir(
module {
  func.func private @vf(%lhs: tensor<8xi32>, %rhs: tensor<8xi32>,
                        %low: tensor<8xi32>, %high: tensor<8xi32>)
      -> (tensor<8xi32>, tensor<8xi32>)
      attributes {hivm.vector_function, no_inline} {
    %0:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%low, %high : tensor<8xi32>, tensor<8xi32>) {
    ^bb0(%lhsIn: i32, %rhsIn: i32, %lowOut: i32, %highOut: i32):
      %1:2 = arith.mulsi_extended %lhsIn, %rhsIn : i32
      linalg.yield %1#0, %1#1 : i32, i32
    } -> (tensor<8xi32>, tensor<8xi32>)
    return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
}
)mlir";

constexpr StringLiteral kNonLinalgPayload = R"mlir(
module {
  func.func private @vf(%lhs: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                        %init: tensor<4x4xf32>) -> tensor<4x4xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = hivm.hir.vconcat dim(1)
        ins(%lhs, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir";

class VectorizeLinalgTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                math::MathDialect, scf::SCFDialect, tensor::TensorDialect,
                transform::TransformDialect, vector::VectorDialect,
                hivm::HIVMDialect, hfusion::HFusionDialect>();
    hivm::registerVectorizeTransformDialectExtension(registry);
    linalg::registerTransformDialectExtension(registry);
    linalg::registerTilingInterfaceExternalModels(registry);
    scf::registerTransformDialectExtension(registry);
    tensor::registerTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  LogicalResult vectorize(ModuleOp module, std::string &error) {
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) { functions.push_back(function); });
    return hivm::vectorizeFunctions(functions, hivm::VectorizeOptions(), error);
  }

  static func::FuncOp getOnlyFunction(ModuleOp module) {
    func::FuncOp result;
    module.walk([&](func::FuncOp function) { result = function; });
    return result;
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(VectorizeLinalgTest, VectorizesUnaryOp) {
  auto payload = parse(kUnaryPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  EXPECT_FALSE(
      function.walk([](linalg::LinalgOp) { return WalkResult::interrupt(); })
          .wasInterrupted());
  unsigned transfers = 0;
  function.walk([&](vector::TransferReadOp) { ++transfers; });
  function.walk([&](vector::TransferWriteOp) { ++transfers; });
  EXPECT_GT(transfers, 0u);
}

TEST_F(VectorizeLinalgTest, FlattensExistingMaskedComputations) {
  auto payload = parse(R"mlir(
    module {
      func.func @vf(%input: vector<1x4xf32>, %mask: vector<1x4xi1>,
                    %passthrough: vector<1x4xf16>) -> vector<1x4xf16> {
        %zero = arith.constant dense<0.0> : vector<1x4xf32>
        %sum = arith.addf %input, %zero : vector<1x4xf32>
        %narrow = arith.truncf %sum {round_mode = #hfusion.round_mode<rint>,
                                   enable_saturate = true}
            : vector<1x4xf32> to vector<1x4xf16>
        %result = arith.select %mask, %narrow, %passthrough
            : vector<1x4xi1>, vector<1x4xf16>
        return %result : vector<1x4xf16>
      }
    }
  )mlir");
  ASSERT_TRUE(payload.get());
  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  auto function = getOnlyFunction(*payload);
  EXPECT_EQ(cast<VectorType>(function.getArgument(0).getType()).getRank(), 2);
  unsigned computations = 0;
  function.walk([&](Operation *operation) {
    if (!operation->hasTrait<OpTrait::Vectorizable>())
      return;
    ++computations;
    for (auto type : operation->getResultTypes())
      if (auto vector = dyn_cast<VectorType>(type))
        EXPECT_EQ(vector.getRank(), 1);
  });
  EXPECT_EQ(computations, 3u);
  function.walk([&](arith::TruncFOp operation) {
    EXPECT_TRUE(operation->hasAttr("enable_saturate"));
    EXPECT_TRUE(operation->hasAttr("round_mode"));
  });
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
}

TEST_F(VectorizeLinalgTest, FusesIdentityChainIntoOneLoop) {
  auto payload = parse(kChainPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  ASSERT_TRUE(succeeded(hivm::cleanupFunctions({function}, error))) << error;
  unsigned loops = 0;
  function.walk([&](scf::ForOp) { ++loops; });
  EXPECT_EQ(loops, 1u);
  EXPECT_FALSE(
      function.walk([](linalg::LinalgOp) { return WalkResult::interrupt(); })
          .wasInterrupted());
}

TEST_F(VectorizeLinalgTest, VectorizesBroadcastGeneric) {
  auto payload = parse(kBroadcastPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  EXPECT_FALSE(
      function.walk([](linalg::LinalgOp) { return WalkResult::interrupt(); })
          .wasInterrupted());
  bool hasPermutationMap = false;
  function.walk([&](vector::TransferReadOp read) {
    hasPermutationMap |= !read.getPermutationMap().isIdentity();
  });
  EXPECT_TRUE(hasPermutationMap);
}

TEST_F(VectorizeLinalgTest, VectorizesMultiResultGeneric) {
  auto payload = parse(kMultiResultPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  EXPECT_FALSE(
      function.walk([](linalg::LinalgOp) { return WalkResult::interrupt(); })
          .wasInterrupted());
}

TEST_F(VectorizeLinalgTest, LeavesNonLinalgResidual) {
  auto payload = parse(kNonLinalgPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  EXPECT_TRUE(
      function.walk([](hivm::VConcatOp) { return WalkResult::interrupt(); })
          .wasInterrupted());
}

TEST_F(VectorizeLinalgTest, CheckResultRejectsHIVMResidual) {
  auto payload = parse(kNonLinalgPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  func::FuncOp function = getOnlyFunction(*payload);
  FailureOr<int64_t> cost = hivm::checkVectorizedFunctions({function}, error);
  EXPECT_TRUE(failed(cost));
  EXPECT_NE(error.find("vconcat"), std::string::npos);
}

TEST_F(VectorizeLinalgTest, CheckResultAcceptsVectorizedFunction) {
  auto payload = parse(kUnaryPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  ASSERT_TRUE(succeeded(vectorize(*payload, error))) << error;
  func::FuncOp function = getOnlyFunction(*payload);
  FailureOr<int64_t> cost = hivm::checkVectorizedFunctions({function}, error);
  ASSERT_TRUE(succeeded(cost)) << error;
  EXPECT_EQ(*cost, 1);
}

TEST_F(VectorizeLinalgTest, CheckResultRejectsLinalgLeftover) {
  auto payload = parse(kUnaryPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  func::FuncOp function = getOnlyFunction(*payload);
  FailureOr<int64_t> cost = hivm::checkVectorizedFunctions({function}, error);
  EXPECT_TRUE(failed(cost));
  EXPECT_FALSE(error.empty());
}

} // namespace
