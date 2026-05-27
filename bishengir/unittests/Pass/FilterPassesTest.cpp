//===- FilterPassesTest.cpp - filter-passes attribute unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

// A simple counting pass — records how many times it ran.
struct CountingPass : public PassWrapper<CountingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingPass)

  CountingPass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return "test-counting-pass"; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// A second counting pass with a different argument name.
struct OtherPass : public PassWrapper<OtherPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherPass)

  OtherPass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return "test-other-pass"; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class FilterPassesTest : public ::testing::Test {
protected:
  void SetUp() override {
    context
        .loadDialect<func::FuncDialect, mlir::annotation::AnnotationDialect>();
    builder = std::make_unique<OpBuilder>(&context);
  }

  mlir::annotation::FilterPassesAttr filterAttr(StringRef passes) {
    return mlir::annotation::FilterPassesAttr::get(
        &context, StringAttr::get(&context, passes));
  }

  // Build a module containing one function, optionally filtered.
  OwningOpRef<ModuleOp>
  makeModule(mlir::annotation::FilterPassesAttr attr = {}) {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "test_func", attr);
    return mod;
  }

  // Build a module with two functions, each with independent filter attrs.
  OwningOpRef<ModuleOp>
  makeModuleTwoFuncs(mlir::annotation::FilterPassesAttr attrA,
                     mlir::annotation::FilterPassesAttr attrB) {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "func_a", attrA);
    addFunc(mod, "func_b", attrB);
    return mod;
  }

  // Build a bare module op, optionally filtered.
  OwningOpRef<ModuleOp>
  makeModuleWithAttr(mlir::annotation::FilterPassesAttr attr = {}) {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    if (attr)
      mod->setAttr(mlir::annotation::FilterPassesAttr::name, attr);
    return mod;
  }

private:
  void addFunc(mlir::ModuleOp mod, llvm::StringRef name,
               mlir::annotation::FilterPassesAttr attr) {
    auto loc = builder->getUnknownLoc();
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, name, funcType);
    func.addEntryBlock();
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<func::ReturnOp>(loc);
    if (attr)
      func->setAttr(mlir::annotation::FilterPassesAttr::name, attr);
  }

protected:

  bishengir::BiShengIRCompileConfigBase config;
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

// ── func.func tests ──────────────────────────────────────────────────────────

// No filter attribute → both passes run on the function.
TEST_F(FilterPassesTest, FuncNoFilterRunsAllPasses) {
  int countA = 0, countB = 0;
  auto mod = makeModule();

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 1);
}

// filter_passes lists only "test-counting-pass" → only CountingPass runs.
TEST_F(FilterPassesTest, FuncFilterAllowsOnlyListedPass) {
  int countA = 0, countB = 0;
  auto mod = makeModule(filterAttr("test-counting-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 0);
}

// filter_passes lists both passes → both run.
TEST_F(FilterPassesTest, FuncFilterAllowsBothPasses) {
  int countA = 0, countB = 0;
  auto mod = makeModule(filterAttr("test-counting-pass,test-other-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 1);
}

// filter_passes lists a pass not in the pipeline → nothing runs.
TEST_F(FilterPassesTest, FuncFilterUnknownPassSkipsAll) {
  int countA = 0, countB = 0;
  auto mod = makeModule(filterAttr("some-other-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 0);
  EXPECT_EQ(countB, 0);
}

// Whitespace around pass names in the list is trimmed correctly.
TEST_F(FilterPassesTest, FuncFilterTrimsWhitespace) {
  int countA = 0, countB = 0;
  auto mod = makeModule(filterAttr(" test-counting-pass , test-other-pass "));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 1);
}

// ── module tests ─────────────────────────────────────────────────────────────

// No filter on module → pass runs.
TEST_F(FilterPassesTest, ModuleNoFilterRunsPass) {
  int countA = 0;
  auto mod = makeModuleWithAttr();

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingPass>(countA));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
}

// filter_passes on module allows only the listed pass.
TEST_F(FilterPassesTest, ModuleFilterAllowsOnlyListedPass) {
  int countA = 0, countB = 0;
  auto mod = makeModuleWithAttr(filterAttr("test-counting-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingPass>(countA));
  pm.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 0);
}

// filter_passes on module blocks all passes when none match.
TEST_F(FilterPassesTest, ModuleFilterBlocksAllWhenNoneMatch) {
  int countA = 0, countB = 0;
  auto mod = makeModuleWithAttr(filterAttr("unrelated-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingPass>(countA));
  pm.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 0);
  EXPECT_EQ(countB, 0);
}

// Filter on func does not affect module-level passes, and vice versa.
TEST_F(FilterPassesTest, FilterScopeIsPerOperation) {
  int modCount = 0, funcCount = 0;
  // Module has no filter; function filters out funcCount's pass.
  auto mod = makeModule(filterAttr("unrelated-pass"));

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingPass>(modCount));
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<OtherPass>(funcCount));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(modCount, 1);  // module has no filter → runs
  EXPECT_EQ(funcCount, 0); // func filtered out test-other-pass
}

// ── module-pass + filtered children tests ────────────────────────────────────

// A module pass running on a module whose only function is filtered out should
// not see that function.
TEST_F(FilterPassesTest, ModulePassSkipsFilteredChildFunc) {
  // func_a filtered to only allow "test-other-pass", so "test-counting-pass"
  // should not see it.
  auto mod = makeModule(filterAttr("test-other-pass"));

  llvm::SmallVector<llvm::StringRef> seen;
  struct SniffPass : public PassWrapper<SniffPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SniffPass)
    SniffPass(llvm::SmallVector<llvm::StringRef> &seen) : seen(seen) {}
    StringRef getArgument() const override { return "test-counting-pass"; }
    void runOnOperation() override {
      getOperation()->walk<mlir::WalkOrder::PreOrder>(
          [&](func::FuncOp f) { seen.push_back(f.getName()); });
    }
    llvm::SmallVector<llvm::StringRef> &seen;
  };

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<SniffPass>(seen));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_TRUE(seen.empty()); // test_func was hidden from the pass
}

// A module with two functions: one filtered, one not. The module pass should
// only see the unfiltered function.
TEST_F(FilterPassesTest, ModulePassSeesOnlyUnfilteredFuncs) {
  // func_a: only allow "test-other-pass" → hidden from "test-counting-pass"
  // func_b: no filter → visible to all passes
  auto mod = makeModuleTwoFuncs(filterAttr("test-other-pass"), {});

  llvm::SmallVector<llvm::StringRef> seen;
  struct SniffPass : public PassWrapper<SniffPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SniffPass)
    SniffPass(llvm::SmallVector<llvm::StringRef> &seen) : seen(seen) {}
    StringRef getArgument() const override { return "test-counting-pass"; }
    void runOnOperation() override {
      getOperation()->walk<mlir::WalkOrder::PreOrder>(
          [&](func::FuncOp f) { seen.push_back(f.getName()); });
    }
    llvm::SmallVector<llvm::StringRef> &seen;
  };

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<SniffPass>(seen));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  ASSERT_EQ(seen.size(), 1u);
  EXPECT_EQ(seen[0], "func_b");
}

// After the module pass runs, filtered functions must still be present in the
// module (they were only temporarily removed).
TEST_F(FilterPassesTest, FilteredFuncsRestoredAfterModulePass) {
  auto mod = makeModuleTwoFuncs(filterAttr("test-other-pass"), {});

  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  // A no-op module pass that triggers the hide/restore logic.
  struct NoopPass : public PassWrapper<NoopPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoopPass)
    StringRef getArgument() const override { return "test-counting-pass"; }
    void runOnOperation() override {}
  };
  pm.addPass(std::make_unique<NoopPass>());

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  // Both functions must still be in the module after the pass.
  llvm::SmallVector<llvm::StringRef> names;
  mod->walk<mlir::WalkOrder::PreOrder>(
      [&](func::FuncOp f) { names.push_back(f.getName()); });
  ASSERT_EQ(names.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(names, "func_a"));
  EXPECT_TRUE(llvm::is_contained(names, "func_b"));
}

} // namespace
