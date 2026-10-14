//===- PassExecutionPolicyTest.cpp - pipeline-mode filter tests ----------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Pass/PassExecutionPolicy.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#include "gtest/gtest.h"
#include <string>

using namespace mlir;

namespace {

// A counting pass on a denylisted argument (as registered below).
struct CountingPass : public PassWrapper<CountingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingPass)

  CountingPass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return "test-counting-pass"; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// A counting pass on an argument that is not denylisted.
struct OtherPass : public PassWrapper<OtherPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherPass)

  OtherPass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return "test-other-pass"; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// An adaptor pass with an empty argument (orchestrates nested pipelines).
struct AdaptorPass : public PassWrapper<AdaptorPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AdaptorPass)

  AdaptorPass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return ""; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// A counting pass whose argument is configurable, so a production denylist
// entry can be exercised through the real handler.
struct NamedCountingPass
    : public PassWrapper<NamedCountingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NamedCountingPass)

  NamedCountingPass(llvm::StringRef arg, int &counter)
      : arg(arg.str()), counter(counter) {}
  StringRef getArgument() const override { return arg; }
  void runOnOperation() override { ++counter; }

  std::string arg;
  int &counter;
};

// A module-scope pass on a denylisted argument.
struct CountingModulePass
    : public PassWrapper<CountingModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingModulePass)

  CountingModulePass(int &counter) : counter(counter) {}
  StringRef getArgument() const override { return "test-module-pass"; }
  void runOnOperation() override { ++counter; }

  int &counter;
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class PassExecutionPolicyTest : public ::testing::Test {
protected:
  // Register the test passes in the process-wide Aiv denylist. The real
  // entries are registered by the HIVM RegBase pipeline library, which is
  // not linked into this unit test, so the tests register their own
  // representative entries.
  void SetUp() override {
    context
        .loadDialect<func::FuncDialect, mlir::annotation::AnnotationDialect>();
    builder = std::make_unique<OpBuilder>(&context);
    registerTestDenylist();
  }

  // Build a module containing one function, optionally with a `mix_mode`
  // attribute.
  OwningOpRef<ModuleOp> makeModule(StringRef mixMode = "") {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "test_func", mixMode);
    return mod;
  }

  // Build a module with two functions carrying independent `mix_mode`
  // attributes.
  OwningOpRef<ModuleOp> makeModuleTwoFuncs(StringRef modeA, StringRef modeB) {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "func_a", modeA);
    addFunc(mod, "func_b", modeB);
    return mod;
  }

  // Register the test passes in the process-wide Aiv denylist. The real
  // entries are registered by the HIVM RegBase pipeline library, which is
  // not linked into this unit test, so the tests register their own
  // representative entries.
  static void registerTestDenylist() {
    bishengir::registerAivDisabledPasses(
        {"test-counting-pass", "test-module-pass"});
  }

private:
  void addFunc(mlir::ModuleOp mod, llvm::StringRef name,
               llvm::StringRef mixMode) {
    auto loc = builder->getUnknownLoc();
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<func::FuncOp>(loc, name, funcType);
    func.addEntryBlock();
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<func::ReturnOp>(loc);
    if (!mixMode.empty())
      func->setAttr("mix_mode", StringAttr::get(&context, mixMode));
  }

protected:
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

// ── detectPipelineMode ───────────────────────────────────────────────────────

// No function carries mix_mode → default full pipeline.
TEST_F(PassExecutionPolicyTest, DetectNoAttrDefaultsToMix) {
  auto mod = makeModule();
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Mix);
}

// A single mix_mode="aiv" function → Aiv mode.
TEST_F(PassExecutionPolicyTest, DetectSingleAivFunc) {
  auto mod = makeModule("aiv");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Aiv);
}

// mix_mode="mix" keeps the full pipeline.
TEST_F(PassExecutionPolicyTest, DetectMixValue) {
  auto mod = makeModule("mix");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Mix);
}

// Functions disagreeing on mix_mode fail safe to the full pipeline.
TEST_F(PassExecutionPolicyTest, DetectDisagreeingFuncsFallBackToMix) {
  auto mod = makeModuleTwoFuncs("aiv", "mix");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Mix);
}

// An unknown mix_mode value fails safe to the full pipeline.
TEST_F(PassExecutionPolicyTest, DetectUnknownValueFallsBackToMix) {
  auto mod = makeModule("future-mode");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Mix);
}

// ── Action-handler integration ───────────────────────────────────────────────

// Mix mode is the default: a denylisted pass still runs unchanged.
TEST_F(PassExecutionPolicyTest, MixModeDenylistedPassRuns) {
  int countA = 0, countB = 0;
  auto mod = makeModule();

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 1);
}

// Aiv mode: denylisted pass is skipped, other passes run.
TEST_F(PassExecutionPolicyTest, AivModeDenylistedPassSkipped) {
  int countA = 0, countB = 0;
  auto mod = makeModule("aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 0);
  EXPECT_EQ(countB, 1);
}

// The config-carrying 4-argument constructor registers the same handler as
// the config-less 3-argument one: Aiv filtering must behave identically on
// both construction paths (bishengir-compile vs hivmc/tests).
TEST_F(PassExecutionPolicyTest, AivModeFilteringOnConfigCtorPath) {
  int countA = 0, countB = 0;
  auto mod = makeModule("aiv");

  bishengir::BiShengIRCompileConfigBase config;
  bishengir::BiShengIRPassManager pm(config, &context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 0);
  EXPECT_EQ(countB, 1);
}

// A production Aiv denylist entry is skipped in Aiv mode and runs in Mix
// mode, through the real policy/handler path.
TEST_F(PassExecutionPolicyTest, ProductionDenylistEntrySkipsInAivOnly) {
  mlir::hivm::regbase::registerAivDisabledPassesForRegbase();
  int countAiv = 0, countMix = 0;

  {
    auto mod = makeModule("aiv");
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    pm.nest<func::FuncOp>().addPass(
        std::make_unique<NamedCountingPass>("hivm-cross-core-gss", countAiv));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  }
  {
    auto mod = makeModule("mix");
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    pm.nest<func::FuncOp>().addPass(
        std::make_unique<NamedCountingPass>("hivm-cross-core-gss", countMix));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  }

  EXPECT_EQ(countAiv, 0);
  EXPECT_EQ(countMix, 1);
}

// A module without any function has no mix_mode to read: Mix (full pipeline).
TEST_F(PassExecutionPolicyTest, DetectEmptyModuleDefaultsToMix) {
  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Mix);
}

// Aiv mode: an adaptor pass (empty argument) always executes so nested
// pipelines keep running.
TEST_F(PassExecutionPolicyTest, AivModeAdaptorPassAlwaysRuns) {
  int countAdaptor = 0;
  auto mod = makeModule("aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<AdaptorPass>(countAdaptor));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countAdaptor, 1);
}

// Aiv mode: a module-scope denylisted pass is skipped.
TEST_F(PassExecutionPolicyTest, AivModeModulePassSkipped) {
  int countMod = 0;
  auto mod = makeModule("aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingModulePass>(countMod));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countMod, 0);
}

// The FilterPassesAttr per-op whitelist still applies on top of the Aiv
// denylist: a pass that is not denylisted is filtered as before.
TEST_F(PassExecutionPolicyTest, AivModeFilterPassesAttrStillFilters) {
  int countB = 0;
  auto mod = makeModule("aiv");
  mod.get().getBody()->front().setAttr(
      mlir::annotation::FilterPassesAttr::name,
      mlir::annotation::FilterPassesAttr::get(
          &context, StringAttr::get(&context, "unrelated-pass")));

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<OtherPass>(countB));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countB, 0);
}

// The FilterPassesAttr whitelist wins over the Aiv denylist for ops the
// module-level hide/restore machinery touches: a module pass that is NOT
// denylisted still hides the filtered child and restores it afterwards.
TEST_F(PassExecutionPolicyTest, AivModeModuleHideRestoreStillWorks) {
  int countMod = 0;
  auto mod = makeModule("aiv");
  mod.get().getBody()->front().setAttr(
      mlir::annotation::FilterPassesAttr::name,
      mlir::annotation::FilterPassesAttr::get(
          &context, StringAttr::get(&context, "test-other-pass")));

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<OtherPass>(countMod));
  pm.addPass(std::make_unique<CountingModulePass>(countMod));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countMod, 1); // module pass ran (not denylisted)
  // The filtered func must still be there after the module pass restored it.
  bool funcPresent = false;
  mod->walk([&](func::FuncOp f) {
    if (f.getName() == "test_func")
      funcPresent = true;
  });
  EXPECT_TRUE(funcPresent);
}

// A module whose functions carry a backup function attribute is exempt from
// the Aiv denylist: the delayed cross-core GSS closed loop must survive.
TEST_F(PassExecutionPolicyTest, AivModeBackupFunctionExempt) {
  int countA = 0;
  auto mod = makeModule("aiv");
  mod.get().getBody()->front().setAttr("hivm.backup_function",
                                       UnitAttr::get(&context));

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(std::make_unique<CountingPass>(countA));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countA, 1);
}

// Running Mix then Aiv then Mix pipelines on the same context must not leak
// the mode across runs: the policy is per pass manager.
TEST_F(PassExecutionPolicyTest, ModeDoesNotLeakAcrossRuns) {
  int countMix = 0, countAiv = 0, countMixAgain = 0;

  auto runPipeline = [&](OwningOpRef<ModuleOp> mod, int &counter) {
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(std::make_unique<CountingPass>(counter));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  };

  runPipeline(makeModule(), countMix);      // Mix
  runPipeline(makeModule("aiv"), countAiv); // Aiv
  runPipeline(makeModule(), countMixAgain); // Mix again

  EXPECT_EQ(countMix, 1);
  EXPECT_EQ(countAiv, 0);
  EXPECT_EQ(countMixAgain, 1);
}

// A handler left on the context by a destroyed manager must not keep
// filtering: it may read neither destroyed state nor the old Aiv mode.
TEST_F(PassExecutionPolicyTest, DestroyedManagerHandlerIsInert) {
  int countAiv = 0, countPlain = 0;
  {
    auto mod = makeModule("aiv");
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    pm.nest<func::FuncOp>().addPass(std::make_unique<CountingPass>(countAiv));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  } // the manager is destroyed here; its handler stays registered

  {
    // A plain mlir::PassManager on the same context dispatches through that
    // handler. The denylisted module pass must still execute.
    auto mod = makeModule("aiv");
    mlir::PassManager pm(&context);
    pm.addPass(std::make_unique<CountingModulePass>(countPlain));
    ASSERT_TRUE(succeeded(pm.run(mod.get())));
  }

  EXPECT_EQ(countAiv, 0);
  EXPECT_EQ(countPlain, 1);
}

// The production RegBase denylist must be reachable through the real
// registration entry point the pipeline builders call (not only through the
// entries this test suite registers for itself).
TEST_F(PassExecutionPolicyTest, ProductionRegbaseDenylistRegisteredByBuilder) {
  mlir::hivm::regbase::registerAivDisabledPassesForRegbase();

  const llvm::StringSet<> &disabled = bishengir::getAivDisabledPasses();
  EXPECT_TRUE(disabled.contains("hivm-cross-core-gss"));
  EXPECT_TRUE(disabled.contains("hivm-delayed-cross-core-gss"));
  EXPECT_TRUE(disabled.contains("hivm-insert-anchors-and-backup"));
  EXPECT_TRUE(disabled.contains("hivm-split-mix-kernel"));
  EXPECT_TRUE(disabled.contains("hivm-inline-fixpipe"));
  // Deliberately not denylisted: it drives the whole flow and self-gates.
  EXPECT_FALSE(disabled.contains("mark-real-core-type"));
}

} // namespace
