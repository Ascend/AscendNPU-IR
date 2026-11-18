//===- PartitionTest.cpp - HIVM partition unit tests ---------------------===//
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
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/HIVM/TransformOps/Partition.h"
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

#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr StringLiteral kPayload = R"mlir(
module {
  func.func @a() {
    return
  }
  func.func @b() {
    return
  }
}
)mlir";

constexpr StringLiteral kVectorizablePayload = R"mlir(
module {
  func.func @flat(%a: tensor<8xf32>, %b: tensor<8xf32>, %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
    return %product : tensor<8xf32>
  }
}
)mlir";

constexpr StringLiteral kPartitionLibrary = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @entry(
      %root: !transform.any_op {transform.readonly},
      %options: !transform.any_param {transform.readonly}
  ) {
    %groups = transform.hivm.partition %root, %options
        {models = ["SameBlock", "UnseparatedByBarrier"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}
)mlir";

class TestPartitionModel : public hivm::PartitionModel {
public:
  FailureOr<hivm::Groups> apply(ArrayRef<Operation *>,
                                const hivm::PartitionContext &,
                                std::string &) const override {
    return hivm::Groups{};
  }
};

hivm::Relation makeConstantRelation(hivm::Groups groups) {
  return
      [groups = std::move(groups)](
          ArrayRef<Operation *>) -> FailureOr<hivm::Groups> { return groups; };
}

class PartitionTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    transform::TransformDialect, hivm::HIVMDialect>();
    hivm::registerVectorizeTransformDialectExtension(registry);
    context = std::make_unique<MLIRContext>(registry);
    context->loadDialect<hivm::HIVMDialect, hivm::HIVMVectorizeDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  Operation *getFunction(ModuleOp module, StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function.getOperation());
    return function.getOperation();
  }

  hivm::PartitionContext makeContext(ModuleOp module) {
    return hivm::PartitionContext{module, DictionaryAttr::get(context.get())};
  }

  FailureOr<hivm::Groups> partition(ModuleOp module, std::string &error) {
    return hivm::partition(module, DictionaryAttr::get(context.get()),
                           {"SameBlock", "UnseparatedByBarrier"}, error);
  }

  FailureOr<Attribute> runPartitionOp(ModuleOp payload,
                                      ArrayRef<Operation *> roots,
                                      Attribute options) {
    auto library = parse(kPartitionLibrary);
    if (!library)
      return failure();
    auto entry = library->lookupSymbol<transform::NamedSequenceOp>("entry");
    if (!entry)
      return failure();

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(&entry.getBody(),
                                                        payload.getOperation());
    Block &body = entry.getBody().front();
    SmallVector<SmallVector<transform::MappedValue>> mappings;
    mappings.push_back(
        SmallVector<transform::MappedValue>(roots.begin(), roots.end()));
    mappings.push_back(SmallVector<transform::MappedValue>{options});
    if (failed(state.mapBlockArguments(body.getArguments(), mappings)))
      return failure();

    auto partitionOp = cast<transform::HIVMPartitionOp>(body.front());
    DiagnosedSilenceableFailure outcome = state.applyTransform(partitionOp);
    if (!outcome.succeeded()) {
      outcome.checkAndReport();
      return failure();
    }

    auto params = state.getParams(partitionOp.getGroups());
    if (params.size() != 1)
      return failure();
    return params.front();
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(PartitionTest, ApplyRelationRejectsEmptyInputGroup) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  hivm::Groups input{hivm::Group{}};
  auto result = hivm::applyRelation(input, makeConstantRelation({}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "partition input group must not be empty");
}

TEST_F(PartitionTest, ApplyRelationAcceptsEmptyGrouping) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto result = hivm::applyRelation({}, makeConstantRelation({}), error);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result->empty());
}

TEST_F(PartitionTest, ApplyRelationRejectsNullInput) {
  std::string error;
  hivm::Groups input{hivm::Group{nullptr}};
  auto result = hivm::applyRelation(input, makeConstantRelation({}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "partition input group contains a null operation");
}

TEST_F(PartitionTest, ApplyRelationRejectsDuplicateInput) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto *root = payload->getOperation();

  std::string error;
  hivm::Groups input{hivm::Group{root}, hivm::Group{root}};
  auto result = hivm::applyRelation(input, makeConstantRelation({}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "partition input group contains a duplicate operation");
}

TEST_F(PartitionTest, ApplyRelationRejectsEmptyOutputGroup) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  hivm::Groups input{hivm::Group{payload->getOperation()}};
  auto result =
      hivm::applyRelation(input, makeConstantRelation(hivm::Groups{{}}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "partition relation returned an empty group");
}

TEST_F(PartitionTest, ApplyRelationRejectsUnrelatedOperation) {
  auto payload = parse(kPayload);
  auto other = parse("module { func.func @other() { return } }");
  ASSERT_TRUE(payload.get());
  ASSERT_TRUE(other.get());
  auto *unrelated = getFunction(*other, "other");

  std::string error;
  hivm::Groups input{hivm::Group{payload->getOperation()}};
  auto result = hivm::applyRelation(
      input, makeConstantRelation(hivm::Groups{{unrelated}}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error,
            "partition relation returned an operation outside the input group");
}

TEST_F(PartitionTest, ApplyRelationRejectsDuplicateOutput) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto *a = getFunction(*payload, "a");

  std::string error;
  hivm::Groups input{hivm::Group{payload->getOperation()}};
  auto result = hivm::applyRelation(
      input, makeConstantRelation(hivm::Groups{{a}, {a}}), error);
  EXPECT_TRUE(failed(result));
  EXPECT_EQ(error, "partition relation returned a duplicate operation");
}

TEST_F(PartitionTest, ApplyRelationPreservesOutputOrder) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto *a = getFunction(*payload, "a");
  auto *b = getFunction(*payload, "b");

  std::string error;
  hivm::Groups input{hivm::Group{payload->getOperation()}};
  auto result = hivm::applyRelation(
      input, makeConstantRelation(hivm::Groups{{b}, {a}}), error);
  ASSERT_TRUE(succeeded(result)) << error;
  ASSERT_EQ(result->size(), 2u);
  EXPECT_EQ((*result)[0], hivm::Group{b});
  EXPECT_EQ((*result)[1], hivm::Group{a});
}

TEST_F(PartitionTest, ApplyRelationAllowsDropping) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  hivm::Groups input{hivm::Group{payload->getOperation()}};
  auto result = hivm::applyRelation(input, makeConstantRelation({}), error);
  ASSERT_TRUE(succeeded(result)) << error;
  EXPECT_TRUE(result->empty());
}

TEST_F(PartitionTest, SameBlockCollectsImmediateOperationsInIRorder) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());

  auto model = hivm::lookupPartitionModel("SameBlock");
  ASSERT_TRUE(model.has_value());

  std::string error;
  auto groups = model->get().apply({payload->getOperation()},
                                   makeContext(*payload), error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 1u);
  EXPECT_EQ((*groups)[0].size(), 3u);
  EXPECT_EQ(
      (*groups)[0].back(),
      &payload->lookupSymbol<func::FuncOp>("flat").getBody().front().back());
}

TEST_F(PartitionTest, SameBlockSeparatesNestedRegions) {
  auto payload = parse(R"mlir(
module {
  func.func @nested(%a: memref<8xf32>, %b: memref<8xf32>,
                    %c: memref<8xf32>, %cond: i1) {
    scf.if %cond {
      hivm.hir.vadd ins(%a, %b : memref<8xf32>, memref<8xf32>)
          outs(%c : memref<8xf32>)
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(payload.get());

  auto model = hivm::lookupPartitionModel("SameBlock");
  ASSERT_TRUE(model.has_value());

  std::string error;
  auto groups = model->get().apply({payload->getOperation()},
                                   makeContext(*payload), error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 2u);
  EXPECT_EQ((*groups)[0].size(), 2u);
  EXPECT_EQ((*groups)[1].size(), 2u);
  EXPECT_NE((*groups)[0].front()->getBlock(), (*groups)[1].front()->getBlock());
}

TEST_F(PartitionTest, SameBlockLeavesEmptyModuleEmpty) {
  auto payload = parse("module {}");
  ASSERT_TRUE(payload.get());

  auto model = hivm::lookupPartitionModel("SameBlock");
  ASSERT_TRUE(model.has_value());

  std::string error;
  auto groups = model->get().apply({payload->getOperation()},
                                   makeContext(*payload), error);
  ASSERT_TRUE(succeeded(groups)) << error;
  EXPECT_TRUE(groups->empty());
}

TEST_F(PartitionTest, BarrierModelLeavesBufferComputeOutsideGroups) {
  auto payload = parse(R"mlir(
module {
  func.func @ignored(%a: memref<8xf32>, %b: memref<8xf32>,
                     %c: memref<8xf32>) {
    %0 = arith.constant 0 : index
    %1 = arith.addi %0, %0 : index
    hivm.hir.vabs ins(%a : memref<8xf32>) outs(%c : memref<8xf32>)
    return
  }
}
)mlir");
  ASSERT_TRUE(payload.get());

  std::string error;
  auto groups = partition(*payload, error);
  ASSERT_TRUE(succeeded(groups)) << error;
  EXPECT_TRUE(groups->empty());
}

TEST_F(PartitionTest, BarrierModelSplitsAtInterveningPureOperation) {
  auto payload = parse(R"mlir(
    module {
      func.func @f(%a: tensor<8xf32>, %b: tensor<8xf32>,
                   %init: tensor<8xf32>) -> tensor<8xf32> {
        %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
            outs(%init : tensor<8xf32>) -> tensor<8xf32>
        %barrier = arith.constant 0 : index
        %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>)
            outs(%init : tensor<8xf32>) -> tensor<8xf32>
        return %product : tensor<8xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(payload.get());
  std::string error;
  auto groups = partition(*payload, error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 2u);
  EXPECT_EQ(groups->front().size(), 1u);
  EXPECT_EQ(groups->back().size(), 1u);
}

TEST_F(PartitionTest, BarrierModelDoesNotBridgeOmittedOperations) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());
  auto &block = payload->lookupSymbol<func::FuncOp>("flat").getBody().front();
  auto *first = &block.front();
  auto *second = first->getNextNode();
  auto model = hivm::lookupPartitionModel("UnseparatedByBarrier");
  ASSERT_TRUE(model.has_value());
  OpBuilder builder(second);
  builder.create<arith::ConstantIndexOp>(second->getLoc(), 0);
  std::string error;
  auto groups =
      model->get().apply({first, second}, makeContext(*payload), error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 2u);
}

TEST_F(PartitionTest, SameBlockRestoresOrderWithoutImportingSiblings) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());
  auto &block = payload->lookupSymbol<func::FuncOp>("flat").getBody().front();
  auto *first = &block.front();
  auto *second = first->getNextNode();
  auto model = hivm::lookupPartitionModel("SameBlock");
  ASSERT_TRUE(model.has_value());
  std::string error;
  auto groups =
      model->get().apply({second, first}, makeContext(*payload), error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 1u);
  EXPECT_EQ(groups->front(), (hivm::Group{first, second}));
}

TEST_F(PartitionTest, BarrierModelGroupsIndependentAdjacentCompute) {
  auto payload = parse(R"mlir(
    module {
      func.func @f(%a: tensor<8xf32>, %b: tensor<8xf32>,
                   %init: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
        %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
            outs(%init : tensor<8xf32>) -> tensor<8xf32>
        %product = hivm.hir.vmul ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
            outs(%init : tensor<8xf32>) -> tensor<8xf32>
        return %sum, %product : tensor<8xf32>, tensor<8xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(payload.get());
  std::string error;
  auto groups = partition(*payload, error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 1u);
  EXPECT_EQ(groups->front().size(), 2u);
}

TEST_F(PartitionTest, ModelsRejectNullMembers) {
  auto payload = parse("module {}");
  ASSERT_TRUE(payload.get());
  for (auto name : {"SameBlock", "UnseparatedByBarrier"}) {
    auto model = hivm::lookupPartitionModel(name);
    ASSERT_TRUE(model.has_value());
    std::string error;
    EXPECT_TRUE(
        failed(model->get().apply({nullptr}, makeContext(*payload), error)));
    EXPECT_EQ(error, std::string(name) + " received a null operation");
  }
}

TEST_F(PartitionTest, RegistryLooksUpKnownModel) {
  EXPECT_TRUE(hivm::lookupPartitionModel("SameBlock").has_value());
  EXPECT_TRUE(hivm::lookupPartitionModel("UnseparatedByBarrier").has_value());
  EXPECT_FALSE(hivm::lookupPartitionModel("vsingleton").has_value());
  EXPECT_FALSE(hivm::lookupPartitionModel("vtensor").has_value());
}

TEST_F(PartitionTest, RegistryRejectsUnknownName) {
  EXPECT_FALSE(hivm::lookupPartitionModel("missing").has_value());
}

TEST_F(PartitionTest, RegistryRejectsDuplicateName) {
  auto model = std::make_unique<TestPartitionModel>();
  ASSERT_TRUE(succeeded(
      hivm::registerPartitionModel("test_duplicate", std::move(model))));

  auto duplicate = std::make_unique<TestPartitionModel>();
  EXPECT_TRUE(failed(
      hivm::registerPartitionModel("test_duplicate", std::move(duplicate))));
}

TEST_F(PartitionTest, RegistryRejectsEmptyName) {
  auto model = std::make_unique<TestPartitionModel>();
  EXPECT_TRUE(failed(hivm::registerPartitionModel("", std::move(model))));
}

TEST_F(PartitionTest, PartitionAppliesExplicitModelList) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  StringRef models[] = {"SameBlock", "UnseparatedByBarrier"};
  auto groups = hivm::partition(*payload, DictionaryAttr::get(context.get()),
                                models, error);
  ASSERT_TRUE(succeeded(groups)) << error;
  ASSERT_EQ(groups->size(), 1u);
  EXPECT_EQ(groups->front().size(), 2u);
}

TEST_F(PartitionTest, PartitionRejectsEmptyModelList) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  auto groups = hivm::partition(*payload, DictionaryAttr::get(context.get()),
                                ArrayRef<StringRef>(), error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "partition requires at least one model");
}

TEST_F(PartitionTest, PartitionRejectsDuplicateModelNames) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  StringRef models[] = {"SameBlock", "SameBlock"};
  auto groups = hivm::partition(*payload, DictionaryAttr::get(context.get()),
                                models, error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "duplicate partition model 'SameBlock'");
}

TEST_F(PartitionTest, PartitionRejectsUnknownModel) {
  auto payload = parse(kVectorizablePayload);
  ASSERT_TRUE(payload.get());

  std::string error;
  StringRef models[] = {"missing"};
  auto groups = hivm::partition(*payload, DictionaryAttr::get(context.get()),
                                models, error);
  EXPECT_TRUE(failed(groups));
  EXPECT_EQ(error, "unknown partition model 'missing'");
}

TEST_F(PartitionTest, PartitionOpRejectsNonModuleRoot) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto *function = getFunction(*payload, "a");

  ScopedDiagnosticHandler handler(context.get(),
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(runPartitionOp(*payload, {function},
                                    DictionaryAttr::get(context.get()))));
}

TEST_F(PartitionTest, PartitionOpRejectsMultipleRoots) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto *a = getFunction(*payload, "a");
  auto *b = getFunction(*payload, "b");

  ScopedDiagnosticHandler handler(context.get(),
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(
      runPartitionOp(*payload, {a, b}, DictionaryAttr::get(context.get()))));
}

TEST_F(PartitionTest, PartitionOpRejectsMalformedOptions) {
  auto payload = parse(kPayload);
  ASSERT_TRUE(payload.get());
  auto options = IntegerAttr::get(IntegerType::get(context.get(), 64), 3);

  ScopedDiagnosticHandler handler(context.get(),
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(
      failed(runPartitionOp(*payload, {payload->getOperation()}, options)));
}

TEST_F(PartitionTest, PartitionOpProducesPreorderOrdinals) {
  auto payload = parse(R"mlir(
module {
  func.func @f(%a: tensor<8xf32>, %b: tensor<8xf32>, %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %barrier = arith.constant 0 : index
    %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    return %product : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(payload.get());

  auto groups = runPartitionOp(*payload, {payload->getOperation()},
                               DictionaryAttr::get(context.get()));
  ASSERT_TRUE(succeeded(groups));

  auto groupsAttr = dyn_cast<hivm::GroupsAttr>(*groups);
  ASSERT_TRUE(groupsAttr);
  ASSERT_EQ(groupsAttr.getGroups().size(), 2u);
  EXPECT_EQ(groupsAttr.getGroups()[0].asArrayRef(), ArrayRef<int64_t>({2}));
  EXPECT_EQ(groupsAttr.getGroups()[1].asArrayRef(), ArrayRef<int64_t>({4}));

  std::string text;
  llvm::raw_string_ostream stream(text);
  Attribute printable = *groups;
  printable.print(stream);
  EXPECT_EQ(text, "#hivm_vectorize.groups<[array<i64: 2>, array<i64: 4>]>");
}

TEST_F(PartitionTest, PartitionOpProducesEmptyGrouping) {
  auto payload = parse("module {}");
  ASSERT_TRUE(payload.get());

  auto groups = runPartitionOp(*payload, {payload->getOperation()},
                               DictionaryAttr::get(context.get()));
  ASSERT_TRUE(succeeded(groups));

  auto groupsAttr = dyn_cast<hivm::GroupsAttr>(*groups);
  ASSERT_TRUE(groupsAttr);
  EXPECT_TRUE(groupsAttr.getGroups().empty());

  std::string text;
  llvm::raw_string_ostream stream(text);
  Attribute printable = *groups;
  printable.print(stream);
  EXPECT_EQ(text, "#hivm_vectorize.groups<[]>");
}

TEST(HIVMVectorizeRegistrationTest, SharedDialectsDoNotRegisterVectorizer) {
  DialectRegistry registry;
  registry.insert<hivm::HIVMDialect, transform::TransformDialect>();
  hivm::registerTransformDialectExtension(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  EXPECT_TRUE(RegisteredOperationName::lookup(
      "transform.hivm.map_forall_to_blocks", &context));
  EXPECT_FALSE(context.getLoadedDialect("hivm_vectorize"));
  EXPECT_FALSE(
      RegisteredOperationName::lookup("transform.hivm.partition", &context));
  EXPECT_FALSE(
      RegisteredOperationName::lookup("transform.hivm.transaction", &context));
}

TEST_F(PartitionTest, GroupsAttributeRoundTrips) {
  Attribute attr = hivm::GroupsAttr::get(
      context.get(), {DenseI64ArrayAttr::get(context.get(), {3}),
                      DenseI64ArrayAttr::get(context.get(), {6})});

  std::string text;
  llvm::raw_string_ostream stream(text);
  attr.print(stream);
  EXPECT_EQ(text, "#hivm_vectorize.groups<[array<i64: 3>, array<i64: 6>]>");

  auto parsed =
      dyn_cast_or_null<hivm::GroupsAttr>(parseAttribute(text, context.get()));
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed, attr);
}

} // namespace
