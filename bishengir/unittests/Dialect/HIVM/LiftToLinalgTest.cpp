//===- LiftToLinalgTest.cpp - HIVM lift-to-linalg tests -------------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/LiftToLinalg.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr StringLiteral kUnaryPayload = R"mlir(
module {
  func.func @unary(%arg: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vexp2 ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kBinaryPayload = R"mlir(
module {
  func.func @binary(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                    %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kMulExtPayload = R"mlir(
module {
  func.func @mulext(%lhs: tensor<8xi32>, %rhs: tensor<8xi32>,
                    %low: tensor<8xi32>, %high: tensor<8xi32>) -> tensor<8xi32> {
    %0:2 = hivm.hir.vmulext
        ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%low, %high : tensor<8xi32>, tensor<8xi32>)
        -> tensor<8xi32>, tensor<8xi32>
    return %0#1 : tensor<8xi32>
  }
}
)mlir";

constexpr StringLiteral kIsInfPayload = R"mlir(
module {
  func.func @isinf(%arg: tensor<8xf32>, %init: tensor<8xi1>) -> tensor<8xi1> {
    %0 = hivm.hir.visinf ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xi1>) -> tensor<8xi1>
    return %0 : tensor<8xi1>
  }
}
)mlir";

constexpr StringLiteral kScalarOperandPayload = R"mlir(
module {
  func.func @scalar(%arg: tensor<8xf32>, %scalar: f32,
                    %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vadd ins(%arg, %scalar : tensor<8xf32>, f32)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kEmptyPayload = R"mlir(
module {
  func.func @empty(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kGluePayload = R"mlir(
module {
  func.func @glue(%arg: tensor<8xf32>) -> tensor<8xi32> {
    %0 = hivm.hir.bitcast %arg : tensor<8xf32> -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}
)mlir";

constexpr StringLiteral kMemRefPayload = R"mlir(
module {
  func.func @memref(%lhs: memref<8xf32>, %rhs: memref<8xf32>,
                    %init: memref<8xf32>) {
    hivm.hir.vadd ins(%lhs, %rhs : memref<8xf32>, memref<8xf32>)
        outs(%init : memref<8xf32>)
    return
  }
}
)mlir";

constexpr StringLiteral kBroadcastPayload = R"mlir(
module {
  func.func @broadcast(%lhs: tensor<8x1xf32>, %rhs: tensor<8x8xf32>,
                       %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x1xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) broadcast = [1] -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
)mlir";

constexpr StringLiteral kBadBroadcastPayload = R"mlir(
module {
  func.func @bad_broadcast(%lhs: tensor<8x2xf32>, %rhs: tensor<8x8xf32>,
                           %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x2xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) broadcast = [1] -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
)mlir";

constexpr StringLiteral kTransposePayload = R"mlir(
module {
  func.func @transpose(%lhs: tensor<4x8xf32>, %rhs: tensor<4x8xf32>,
                       %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init : tensor<8x4xf32>) transpose = [1, 0] -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
)mlir";

constexpr StringLiteral kBadTransposePayload = R"mlir(
module {
  func.func @bad_transpose(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>,
                           %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x4xf32>) transpose = [1, 0] -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
)mlir";

constexpr StringLiteral kTempBufferPayload = R"mlir(
module {
  func.func @temp(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                  %init: tensor<8xf32>, %temp: memref<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) temp_buffer(%temp : memref<8xf32>)
        -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kConcatPayload = R"mlir(
module {
  func.func @concat(%lhs: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                    %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = hivm.hir.vconcat dim(1) ins(%lhs, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir";

constexpr StringLiteral kCallPayload = R"mlir(
module {
  func.func private @vf(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32> attributes {hivm.vector_function, no_inline} {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @caller(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                    %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = call @vf(%lhs, %rhs, %init) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kLiftLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op
  ) -> !transform.any_op {
    %lifted = transform.hivm.lift_to_linalg %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield %lifted : !transform.any_op
  }
}
)mlir";

class LiftToLinalgTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                math::MathDialect, tensor::TensorDialect,
                transform::TransformDialect, hivm::HIVMDialect>();
    hivm::registerVectorizeTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadDialect<hivm::HIVMDialect>();
    context->loadDialect<arith::ArithDialect>();
    context->loadDialect<func::FuncDialect>();
    context->loadDialect<linalg::LinalgDialect>();
    context->loadDialect<math::MathDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  LogicalResult runEngine(ModuleOp module, std::string &error) {
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) { functions.push_back(function); });
    return hivm::liftFunctionsToLinalg(functions, error);
  }

  FailureOr<SmallVector<func::FuncOp>> runEngine(ModuleOp module) {
    std::string error;
    if (failed(runEngine(module, error)))
      return failure();
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) { functions.push_back(function); });
    return functions;
  }

  func::FuncOp lookupFunction(ModuleOp module, StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function.getOperation());
    return function;
  }

  linalg::GenericOp getOnlyGeneric(func::FuncOp function) {
    linalg::GenericOp generic;
    function.walk([&](linalg::GenericOp candidate) {
      EXPECT_FALSE(generic.getOperation());
      generic = candidate;
    });
    return generic;
  }

  FailureOr<SmallVector<Operation *>> runLiftOp(ModuleOp payload,
                                                Operation *handleTarget,
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
    if (failed(mapEntryRoot(body, handleTarget, state)))
      return failure();
    if (failed(applyBody(body, state)))
      return failure();

    auto liftOp = findOp<transform::HIVMLiftToLinalgOp>(body);
    if (!liftOp)
      return failure();
    auto functions = state.getPayloadOps(liftOp.getResult());
    return SmallVector<Operation *>(functions.begin(), functions.end());
  }

  template <typename OpT> static OpT findOp(Block &body) {
    for (Operation &op : body.without_terminator())
      if (auto candidate = dyn_cast<OpT>(op))
        return candidate;
    return OpT();
  }

  static LogicalResult mapEntryRoot(Block &body, Operation *target,
                                    transform::TransformState &state) {
    SmallVector<SmallVector<transform::MappedValue>> mappings;
    mappings.push_back(SmallVector<transform::MappedValue>{target});
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

TEST_F(LiftToLinalgTest, LiftsUnaryMathOp) {
  auto payload = parse(kUnaryPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic = getOnlyGeneric(lookupFunction(*payload, "unary"));
  ASSERT_TRUE(generic);
  EXPECT_EQ(generic.getNumLoops(), 1u);
  EXPECT_EQ(generic.getNumDpsInputs(), 1u);
  EXPECT_EQ(generic.getNumDpsInits(), 1u);

  Block &body = generic.getRegion().front();
  ASSERT_EQ(body.getNumArguments(), 2u);
  EXPECT_TRUE(isa<math::Exp2Op>(body.front()));
  EXPECT_TRUE(isa<linalg::YieldOp>(body.back()));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftsBinaryFloatOp) {
  auto payload = parse(kBinaryPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic =
      getOnlyGeneric(lookupFunction(*payload, "binary"));
  ASSERT_TRUE(generic);
  EXPECT_EQ(generic.getNumDpsInputs(), 2u);

  Block &body = generic.getRegion().front();
  auto add = dyn_cast<arith::AddFOp>(body.front());
  ASSERT_TRUE(add);
  EXPECT_EQ(add.getLhs().getType(), add.getResult().getType());
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftsMultiResultOp) {
  auto payload = parse(kMulExtPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic =
      getOnlyGeneric(lookupFunction(*payload, "mulext"));
  ASSERT_TRUE(generic);
  EXPECT_EQ(generic.getNumResults(), 2u);
  EXPECT_EQ(generic.getNumDpsInits(), 2u);

  Block &body = generic.getRegion().front();
  EXPECT_TRUE(isa<arith::MulSIExtendedOp>(body.front()));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftsIsInfToCmpF) {
  auto payload = parse(kIsInfPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic = getOnlyGeneric(lookupFunction(*payload, "isinf"));
  ASSERT_TRUE(generic);

  Block &body = generic.getRegion().front();
  auto constant = dyn_cast<arith::ConstantOp>(body.front());
  ASSERT_TRUE(constant);
  auto inf = dyn_cast<FloatAttr>(constant.getValue());
  ASSERT_TRUE(inf);
  EXPECT_TRUE(inf.getValue().isInfinity());

  arith::CmpFOp compare;
  for (Operation &op : body) {
    if (auto candidate = dyn_cast<arith::CmpFOp>(op)) {
      compare = candidate;
      break;
    }
  }
  ASSERT_TRUE(compare);
  EXPECT_EQ(compare.getPredicate(), arith::CmpFPredicate::OEQ);
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftsScalarOperand) {
  auto payload = parse(kScalarOperandPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic =
      getOnlyGeneric(lookupFunction(*payload, "scalar"));
  ASSERT_TRUE(generic);
  EXPECT_EQ(generic.getNumDpsInputs(), 2u);

  AffineMap scalarMap = generic.getIndexingMapsArray()[1];
  EXPECT_TRUE(scalarMap.getResults().empty());
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, AcceptsEmptyAndGlueBodies) {
  auto payload = parse(kEmptyPayload);
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(succeeded(runEngine(*payload)));

  auto glue = parse(kGluePayload);
  ASSERT_TRUE(glue.get());
  ASSERT_TRUE(succeeded(runEngine(*glue)));
  func::FuncOp function = lookupFunction(*glue, "glue");
  EXPECT_FALSE(getOnlyGeneric(function));
  function.walk(
      [](Operation *op) { EXPECT_FALSE(isa<linalg::GenericOp>(op)); });
  EXPECT_TRUE(succeeded(verify(glue.get())));
}

TEST_F(LiftToLinalgTest, RejectsMemRefForm) {
  auto payload = parse(kMemRefPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  EXPECT_TRUE(failed(runEngine(*payload, error)));
  EXPECT_NE(error.find("requires tensor results"), std::string::npos);
}

TEST_F(LiftToLinalgTest, LiftsInlineBroadcast) {
  auto payload = parse(kBroadcastPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic =
      getOnlyGeneric(lookupFunction(*payload, "broadcast"));
  ASSERT_TRUE(generic);
  ASSERT_EQ(generic.getIndexingMapsArray().size(), 3u);

  AffineMap inputMap = generic.getIndexingMapsArray().front();
  ASSERT_EQ(inputMap.getNumResults(), 2u);
  EXPECT_EQ(inputMap.getResult(0), getAffineDimExpr(0, context.get()));
  EXPECT_EQ(inputMap.getResult(1), getAffineConstantExpr(0, context.get()));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, RejectsInvalidBroadcast) {
  auto payload = parse(kBadBroadcastPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  EXPECT_TRUE(failed(runEngine(*payload, error)));
  EXPECT_NE(error.find("broadcast dim must be 1 or match the output"),
            std::string::npos);
}

TEST_F(LiftToLinalgTest, LiftsInlineTranspose) {
  auto payload = parse(kTransposePayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  linalg::GenericOp generic =
      getOnlyGeneric(lookupFunction(*payload, "transpose"));
  ASSERT_TRUE(generic);
  ASSERT_EQ(generic.getIndexingMapsArray().size(), 3u);

  AffineMap inputMap = generic.getIndexingMapsArray().front();
  ASSERT_EQ(inputMap.getNumResults(), 2u);
  EXPECT_EQ(inputMap.getResult(0), getAffineDimExpr(1, context.get()));
  EXPECT_EQ(inputMap.getResult(1), getAffineDimExpr(0, context.get()));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, RejectsInvalidTranspose) {
  auto payload = parse(kBadTransposePayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  EXPECT_TRUE(failed(runEngine(*payload, error)));
  EXPECT_NE(error.find("transpose shapes do not match"), std::string::npos);
}

TEST_F(LiftToLinalgTest, RejectsExtraOperands) {
  auto payload = parse(kTempBufferPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  EXPECT_TRUE(failed(runEngine(*payload, error)));
  EXPECT_NE(error.find("unexpected extra operands"), std::string::npos);
}

TEST_F(LiftToLinalgTest, LeavesConcatUntouched) {
  auto payload = parse(kConcatPayload);
  ASSERT_TRUE(payload.get());

  ASSERT_TRUE(succeeded(runEngine(*payload)));
  func::FuncOp function = lookupFunction(*payload, "concat");
  EXPECT_TRUE(isa<hivm::VConcatOp>(function.getBody().front().front()));
  EXPECT_FALSE(getOnlyGeneric(function));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftOpPreservesFunctionAndCall) {
  auto payload = parse(kCallPayload);
  ASSERT_TRUE(payload.get());

  auto lifted = runLiftOp(
      *payload, lookupFunction(*payload, "vf").getOperation(), kLiftLibrary);
  ASSERT_TRUE(succeeded(lifted));
  ASSERT_EQ(lifted->size(), 1u);
  EXPECT_EQ(lifted->front(), lookupFunction(*payload, "vf").getOperation());

  func::FuncOp function = lookupFunction(*payload, "vf");
  EXPECT_TRUE(function->hasAttr("hivm.vector_function"));
  EXPECT_TRUE(function->hasAttr("no_inline"));
  EXPECT_TRUE(getOnlyGeneric(function));

  func::FuncOp caller = lookupFunction(*payload, "caller");
  EXPECT_TRUE(isa<func::CallOp>(caller.getBody().front().front()));
  EXPECT_TRUE(succeeded(verify(payload.get())));
}

TEST_F(LiftToLinalgTest, LiftOpRejectsNonFunction) {
  auto payload = parse(kCallPayload);
  ASSERT_TRUE(payload.get());

  EXPECT_TRUE(
      failed(runLiftOp(*payload, payload->getOperation(), kLiftLibrary)));
}

} // namespace
