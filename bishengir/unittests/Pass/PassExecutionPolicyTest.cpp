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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#include "gtest/gtest.h"
#include <optional>
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

// Replaces hivm.module_core_type so a later decision in the same run sees
// the new value.
struct RetagModuleCoreTypePass
    : public PassWrapper<RetagModuleCoreTypePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RetagModuleCoreTypePass)

  RetagModuleCoreTypePass(hivm::TModuleCoreType coreType)
      : coreType(coreType) {}
  StringRef getArgument() const override {
    return "test-retag-module-core-type";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->setAttr(
        hivm::TModuleCoreTypeAttr::name,
        hivm::TModuleCoreTypeAttr::get(module.getContext(), coreType));
  }

  hivm::TModuleCoreType coreType;
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class PassExecutionPolicyTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<func::FuncDialect, hivm::HIVMDialect,
                        mlir::annotation::AnnotationDialect>();
    builder = std::make_unique<OpBuilder>(&context);
    registerTestKernelKinds();
  }

  // Build a module containing one function. `moduleCoreType` is `aiv`,
  // `aic`, `mix`, empty (no attribute), or any other string (an unrecognized
  // attribute with the core-type name).
  OwningOpRef<ModuleOp> makeModule(StringRef moduleCoreType = "") {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "test_func");
    setModuleCoreType(mod, moduleCoreType);
    return mod;
  }

  // Two functions that still carry `mix_mode`. The filter must ignore that
  // function attribute and read only the module core type.
  OwningOpRef<ModuleOp> makeModuleTwoFuncs(StringRef modeA, StringRef modeB) {
    auto loc = builder->getUnknownLoc();
    auto mod = builder->create<ModuleOp>(loc);
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.getBody());
    addFunc(mod, "func_a", modeA);
    addFunc(mod, "func_b", modeB);
    return mod;
  }

  void setModuleCoreType(ModuleOp mod, StringRef moduleCoreType) {
    if (moduleCoreType.empty())
      return;
    std::optional<hivm::TModuleCoreType> coreType;
    if (moduleCoreType == "aiv")
      coreType = hivm::TModuleCoreType::AIV;
    else if (moduleCoreType == "aic")
      coreType = hivm::TModuleCoreType::AIC;
    else if (moduleCoreType == "mix")
      coreType = hivm::TModuleCoreType::MIX;
    if (!coreType) {
      mod->setAttr(hivm::TModuleCoreTypeAttr::name,
                   StringAttr::get(&context, moduleCoreType));
      return;
    }
    mod->setAttr(hivm::TModuleCoreTypeAttr::name,
                 hivm::TModuleCoreTypeAttr::get(&context, *coreType));
  }

  // Register the test passes for mix and pure AIC only. The real entries
  // are registered by the HIVM RegBase pipeline library, which is not linked
  // into this unit test, so the tests register their own representative
  // entries. Pure AIV is omitted, so an AIV module skips these passes.
  static void registerTestKernelKinds() {
    bishengir::registerKernelKindPasses(
        bishengir::PipelineMode::Mix,
        {"test-counting-pass", "test-module-pass"});
    bishengir::registerKernelKindPasses(
        bishengir::PipelineMode::Aic,
        {"test-counting-pass", "test-module-pass"});
  }

private:
  void addFunc(mlir::ModuleOp mod, llvm::StringRef name,
               llvm::StringRef mixMode = "") {
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

// No module core type → no kernel kind, so the filter does not run.
TEST_F(PassExecutionPolicyTest, DetectNoAttrIsUnspecified) {
  auto mod = makeModule();
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Unspecified);
}

// hivm.module_core_type = AIV → Aiv mode.
TEST_F(PassExecutionPolicyTest, DetectModuleCoreTypeAiv) {
  auto mod = makeModule("aiv");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Aiv);
}

// MIX and AIC each select that kernel kind.
TEST_F(PassExecutionPolicyTest, DetectMixAndAic) {
  auto mix = makeModule("mix");
  auto aic = makeModule("aic");
  EXPECT_EQ(bishengir::detectPipelineMode(mix.get()),
            bishengir::PipelineMode::Mix);
  EXPECT_EQ(bishengir::detectPipelineMode(aic.get()),
            bishengir::PipelineMode::Aic);
}

// Function mix_mode is not the filter key, even when every function agrees.
TEST_F(PassExecutionPolicyTest, DetectMixModeOnFunctionsIsIgnored) {
  auto mod = makeModuleTwoFuncs("aiv", "aiv");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Unspecified);
  setModuleCoreType(mod.get(), "aiv");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Aiv);
}

// An unrecognized attribute with the core-type name selects no kernel kind.
TEST_F(PassExecutionPolicyTest, DetectUnknownValueIsUnspecified) {
  auto mod = makeModule("future-mode");
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Unspecified);
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

// hivm-cross-core-gss lists Mix and pure AIC. AIV skips it; MIX and AIC run
// it, through the real policy/handler path.
TEST_F(PassExecutionPolicyTest, CrossCoreGssFollowsKernelKindList) {
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();
  int countAiv = 0, countMix = 0, countAic = 0;

  auto run = [&](StringRef coreType, int &counter) {
    auto mod = makeModule(coreType);
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    pm.nest<func::FuncOp>().addPass(
        std::make_unique<NamedCountingPass>("hivm-cross-core-gss", counter));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  };
  run("aiv", countAiv);
  run("mix", countMix);
  run("aic", countAic);

  EXPECT_EQ(countAiv, 0);
  EXPECT_EQ(countMix, 1);
  EXPECT_EQ(countAic, 1);
}

// A pass registered only for pure AIV does not run when the module core
// type is AIC or MIX. No core type still runs it.
TEST_F(PassExecutionPolicyTest, PassSkippedWhenKernelKindOmitted) {
  bishengir::registerKernelKindPasses(bishengir::PipelineMode::Aiv,
                                      {"test-aiv-only-pass"});
  int countAiv = 0, countAic = 0, countMix = 0, countNone = 0;

  auto run = [&](StringRef coreType, int &counter) {
    auto mod = makeModule(coreType);
    bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                       mlir::PassManager::Nesting::Implicit);
    pm.nest<func::FuncOp>().addPass(
        std::make_unique<NamedCountingPass>("test-aiv-only-pass", counter));
    ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  };
  run("aiv", countAiv);
  run("aic", countAic);
  run("mix", countMix);
  run("", countNone);

  EXPECT_EQ(countAiv, 1);
  EXPECT_EQ(countAic, 0);
  EXPECT_EQ(countMix, 0);
  EXPECT_EQ(countNone, 1);
}

// A module without a core-type attribute selects no kernel kind.
TEST_F(PassExecutionPolicyTest, DetectEmptyModuleIsUnspecified) {
  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  EXPECT_EQ(bishengir::detectPipelineMode(mod.get()),
            bishengir::PipelineMode::Unspecified);
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

// A pass that replaces hivm.module_core_type changes the decision for the
// next pass in the same run.
TEST_F(PassExecutionPolicyTest, ModuleCoreTypeChangeRefreshesFilter) {
  int countBefore = 0, countAfter = 0;
  auto mod = makeModule("aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<CountingModulePass>(countBefore));
  pm.addPass(
      std::make_unique<RetagModuleCoreTypePass>(hivm::TModuleCoreType::MIX));
  pm.addPass(std::make_unique<CountingModulePass>(countAfter));

  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(countBefore, 0);
  EXPECT_EQ(countAfter, 1);
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

// A pass registered as classify-kernel gets the callback after it runs.
TEST_F(PassExecutionPolicyTest, ClassifyKernelCallbackRunsAfterPass) {
  int ran = 0;
  bishengir::registerKernelKindPasses(bishengir::PipelineMode::Mix,
                                      {"test-classify-kernel-pass"});
  bishengir::registerKernelKindPasses(bishengir::PipelineMode::Aic,
                                      {"test-classify-kernel-pass"});
  bishengir::registerClassifyKernelPasses({"test-classify-kernel-pass"});
  bishengir::registerClassifyKernelCallback([](Operation *op) -> LogicalResult {
    op->setAttr("classified", UnitAttr::get(op->getContext()));
    return success();
  });

  auto mod = makeModule();
  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.nest<func::FuncOp>().addPass(
      std::make_unique<NamedCountingPass>("test-classify-kernel-pass", ran));
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(ran, 1);
  func::FuncOp func = *mod.get().getOps<func::FuncOp>().begin();
  EXPECT_TRUE(func->hasAttr("classified"));
}

// A pass skipped by the Aiv denylist did not run, so it must not classify.
TEST_F(PassExecutionPolicyTest, SkippedAivPassDoesNotClassify) {
  int ran = 0;
  bishengir::registerKernelKindPasses(bishengir::PipelineMode::Mix,
                                      {"test-classify-kernel-pass"});
  bishengir::registerKernelKindPasses(bishengir::PipelineMode::Aic,
                                      {"test-classify-kernel-pass"});
  bishengir::registerClassifyKernelPasses({"test-classify-kernel-pass"});
  bishengir::registerClassifyKernelCallback([](Operation *op) -> LogicalResult {
    op->setAttr("classified", UnitAttr::get(op->getContext()));
    return success();
  });

  auto mod = makeModule("aiv");
  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.nest<func::FuncOp>().addPass(
      std::make_unique<NamedCountingPass>("test-classify-kernel-pass", ran));
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));
  EXPECT_EQ(ran, 0);
  func::FuncOp func = *mod.get().getOps<func::FuncOp>().begin();
  EXPECT_FALSE(func->hasAttr("classified"));
}

// convert-to-hivm-op lowers memref.copy to an HIVM op and has
// classifyKernel set, so InferFuncCoreType must stamp core-type attrs.
TEST_F(PassExecutionPolicyTest, ConvertToHIVMOpRunsInferFuncCoreType) {
  context.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                      hivm::HIVMDialect>();
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();
  ASSERT_TRUE(bishengir::passClassifiesKernel("convert-to-hivm-op"));

  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto memTy = MemRefType::get({4}, builder->getF32Type());
    auto func = builder->create<func::FuncOp>(
        loc, "copy_to_hivm", builder->getFunctionType({memTy, memTy}, {}));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<memref::CopyOp>(loc, func.getArgument(0),
                                    func.getArgument(1));
    builder->create<func::ReturnOp>(loc);
  }

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(mlir::hivm::createConvertToHIVMOpPass());
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  func::FuncOp func = *mod.get().getOps<func::FuncOp>().begin();
  EXPECT_TRUE(func.walk([](hivm::LoadOp) { return WalkResult::interrupt(); })
                  .wasInterrupted());
  EXPECT_TRUE(func->hasAttr("hivm.func_core_type"));
  EXPECT_TRUE(mod.get()->hasAttr("hivm.module_core_type"));
}

// hivm-inline-fixpipe's classifyKernel follow-up replaces a core type that
// was stamped before inlining.
TEST_F(PassExecutionPolicyTest, InlineFixpipeReplacesFuncCoreType) {
  context.loadDialect<hivm::HIVMDialect, memref::MemRefDialect>();
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();
  ASSERT_TRUE(bishengir::passClassifiesKernel("hivm-inline-fixpipe"));

  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto memTy = MemRefType::get({4}, builder->getF32Type());
    auto func = builder->create<func::FuncOp>(
        loc, "load_only", builder->getFunctionType({memTy, memTy}, {}));
    func->setAttr(
        hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(&context, hivm::TFuncCoreType::MIX));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<hivm::LoadOp>(loc, TypeRange{}, func.getArgument(0),
                                  func.getArgument(1));
    builder->create<func::ReturnOp>(loc);
  }

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(mlir::hivm::createInlineFixpipePass());
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  func::FuncOp func = *mod.get().getOps<func::FuncOp>().begin();
  auto coreType = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  ASSERT_TRUE(coreType);
  EXPECT_EQ(coreType.getFuncCoreType(), hivm::TFuncCoreType::AIV);
}

// A function already split out of a mix kernel keeps its core type. That
// member does not keep a sibling function's stale core type, and the module
// core type stays.
TEST_F(PassExecutionPolicyTest, PartOfMixCoreTypeIsNotCleared) {
  context.loadDialect<hivm::HIVMDialect, memref::MemRefDialect>();
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();

  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  mod.get()->setAttr(
      hivm::TModuleCoreTypeAttr::name,
      hivm::TModuleCoreTypeAttr::get(&context, hivm::TModuleCoreType::MIX));
  auto memTy = MemRefType::get({4}, builder->getF32Type());
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto func = builder->create<func::FuncOp>(
        loc, "mix_aic", builder->getFunctionType({memTy, memTy}, {}));
    func->setAttr(
        hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(&context, hivm::TFuncCoreType::AIC));
    func->setAttr(hivm::TPartOfMixAttr::name, UnitAttr::get(&context));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<hivm::LoadOp>(loc, TypeRange{}, func.getArgument(0),
                                  func.getArgument(1));
    builder->create<func::ReturnOp>(loc);
  }
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto func = builder->create<func::FuncOp>(
        loc, "plain_load", builder->getFunctionType({memTy, memTy}, {}));
    func->setAttr(
        hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(&context, hivm::TFuncCoreType::MIX));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    builder->create<hivm::LoadOp>(loc, TypeRange{}, func.getArgument(0),
                                  func.getArgument(1));
    builder->create<func::ReturnOp>(loc);
  }

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(mlir::hivm::createInlineFixpipePass());
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  func::FuncOp aicFunc, plainFunc;
  for (func::FuncOp func : mod.get().getOps<func::FuncOp>()) {
    if (func.getName() == "mix_aic")
      aicFunc = func;
    else if (func.getName() == "plain_load")
      plainFunc = func;
  }
  ASSERT_TRUE(aicFunc);
  ASSERT_TRUE(plainFunc);
  auto aicType = aicFunc->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  auto plainType = plainFunc->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  ASSERT_TRUE(aicType);
  ASSERT_TRUE(plainType);
  EXPECT_EQ(aicType.getFuncCoreType(), hivm::TFuncCoreType::AIC);
  EXPECT_EQ(plainType.getFuncCoreType(), hivm::TFuncCoreType::AIV);
  auto moduleType = mod.get()->getAttrOfType<hivm::TModuleCoreTypeAttr>(
      hivm::TModuleCoreTypeAttr::name);
  ASSERT_TRUE(moduleType);
  EXPECT_EQ(moduleType.getModuleCoreType(), hivm::TModuleCoreType::MIX);
}

// The production RegBase kernel-kind lists must be reachable through the
// real registration entry point the pipeline builders call.
TEST_F(PassExecutionPolicyTest, ProductionRegbaseKernelKindsRegistered) {
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();

  auto applies = [](StringRef pass, bishengir::PipelineMode kind) {
    return bishengir::passAppliesToKernelKind(pass, kind);
  };
  // [MixKernel, PureAicKernel]: not pure AIV.
  for (StringRef pass : {"create-preload",
                         "hivm-peel-preload",
                         "cv-pipelining",
                         "hivm-bind-sub-block",
                         "hivm-combine-optimized-convert-layout",
                         "hivm-convert-layout-to-transpose",
                         "hivm-cross-core-gss",
                         "hivm-delayed-cross-core-gss",
                         "hivm-hoist-tightly-coupled-alloc",
                         "hivm-inline-fixpipe",
                         "hivm-insert-anchors-and-backup",
                         "hivm-insert-convert-layout",
                         "hivm-insert-cv-tight-coupled-buffer",
                         "hivm-insert-fixpipe",
                         "hivm-insert-l12ub-for-debug",
                         "hivm-insert-nz2nd-for-debug",
                         "hivm-mark-real-core-type",
                         "hivm-mark-sync-block-lock-with-subblock",
                         "hivm-mark-tightly-coupled-buffer",
                         "hivm-merge-same-preload-scopes",
                         "hivm-normalize-convops",
                         "hivm-normalize-matmul",
                         "hivm-partition-and-bind-sub-block",
                         "hivm-propagate-convert-layout",
                         "hivm-sink-exclusive-preload-work",
                         "hivm-split-mix-kernel",
                         "hivm-sub-block-guard-cleanup",
                         "hivm-tile-batchmm-into-loop",
                         "insert-workspace-for-mix-cv",
                         "hivm-insert-load-store-for-mix-cv",
                         "hivm-inject-block-sync",
                         "hivm-insert-load-store-for-scalar"}) {
    EXPECT_TRUE(applies(pass, bishengir::PipelineMode::Mix)) << pass.str();
    EXPECT_TRUE(applies(pass, bishengir::PipelineMode::Aic)) << pass.str();
    EXPECT_FALSE(applies(pass, bishengir::PipelineMode::Aiv)) << pass.str();
  }
  // Default kernel set, including the AIV-side core-type inference.
  for (StringRef pass : {"hivm-inject-sync", "mark-simt-scope-no-inline",
                         "hivm-infer-func-core-type"}) {
    EXPECT_TRUE(applies(pass, bishengir::PipelineMode::Mix)) << pass.str();
    EXPECT_TRUE(applies(pass, bishengir::PipelineMode::Aiv)) << pass.str();
    EXPECT_TRUE(applies(pass, bishengir::PipelineMode::Aic)) << pass.str();
  }
}

// Skipping hivm-insert-load-store-for-mix-cv on a pure-AIV module runs the
// VECTOR load stamp and does not insert the mix-CV boundary copies.
TEST_F(PassExecutionPolicyTest,
       InsertLoadStoreForMixCVAivSubstituteStampsVector) {
  context.loadDialect<hivm::HIVMDialect, memref::MemRefDialect>();
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();

  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  hivm::LoadOp defaultLoad, cubeLoad;
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto memTy = MemRefType::get({4}, builder->getF32Type());
    auto func = builder->create<func::FuncOp>(
        loc, "aiv_loads",
        builder->getFunctionType({memTy, memTy, memTy, memTy}, {}));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    defaultLoad = builder->create<hivm::LoadOp>(
        loc, TypeRange{}, func.getArgument(0), func.getArgument(1));
    cubeLoad = builder->create<hivm::LoadOp>(
        loc, TypeRange{}, func.getArgument(2), func.getArgument(3));
    cubeLoad.setTcoretypeAttr(
        hivm::TCoreTypeAttr::get(&context, hivm::TCoreType::CUBE));
    builder->create<func::ReturnOp>(loc);
  }
  setModuleCoreType(mod.get(), "aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.nest<func::FuncOp>().addPass(
      mlir::hivm::createInsertLoadStoreForMixCVPass());
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  EXPECT_EQ(defaultLoad.getTcoretype().getTcoretype(), hivm::TCoreType::VECTOR);
  EXPECT_EQ(cubeLoad.getTcoretype().getTcoretype(), hivm::TCoreType::CUBE);
  func::FuncOp func = *mod.get().getOps<func::FuncOp>().begin();
  EXPECT_FALSE(func.walk([](hivm::StoreOp) { return WalkResult::interrupt(); })
                   .wasInterrupted());
}

// Skipping hivm-mark-real-core-type on a pure-AIV module stamps VECTOR on
// scalar-pipe and InferCoreTypeInterface ops instead of cloning and
// splitting. An explicit core type is left unchanged. The cleanup
// invocation still removes the stamps.
TEST_F(PassExecutionPolicyTest, MarkRealCoreTypeAivSubstituteStampsVector) {
  context.loadDialect<arith::ArithDialect, hivm::HIVMDialect,
                      memref::MemRefDialect>();
  mlir::hivm::regbase::registerKernelKindPassesForRegbase();

  auto loc = builder->getUnknownLoc();
  OwningOpRef<ModuleOp> mod = builder->create<ModuleOp>(loc);
  memref::LoadOp scalarLoad;
  hivm::LoadOp defaultLoad, cubeLoad;
  {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointToEnd(mod.get().getBody());
    auto memTy = MemRefType::get({4}, builder->getF32Type());
    auto func = builder->create<func::FuncOp>(
        loc, "aiv_mark",
        builder->getFunctionType({memTy, memTy, memTy, memTy}, {}));
    func.addEntryBlock();
    builder->setInsertionPointToEnd(&func.getBody().front());
    auto idx = builder->create<arith::ConstantIndexOp>(loc, 0);
    scalarLoad = builder->create<memref::LoadOp>(loc, func.getArgument(0),
                                                 ValueRange{idx});
    defaultLoad = builder->create<hivm::LoadOp>(
        loc, TypeRange{}, func.getArgument(1), func.getArgument(2));
    cubeLoad = builder->create<hivm::LoadOp>(
        loc, TypeRange{}, func.getArgument(1), func.getArgument(3));
    cubeLoad.setTcoretypeAttr(
        hivm::TCoreTypeAttr::get(&context, hivm::TCoreType::CUBE));
    builder->create<func::ReturnOp>(loc);
  }
  setModuleCoreType(mod.get(), "aiv");

  bishengir::BiShengIRPassManager pm(&context, "builtin.module",
                                     mlir::PassManager::Nesting::Implicit);
  pm.addPass(mlir::hivm::createMarkRealCoreTypePass());
  ASSERT_TRUE(succeeded(static_cast<mlir::PassManager &>(pm).run(mod.get())));

  auto scalarAttr =
      scalarLoad->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  ASSERT_TRUE(scalarAttr);
  EXPECT_EQ(scalarAttr.getTcoretype(), hivm::TCoreType::VECTOR);
  auto defaultAttr = defaultLoad->getAttrOfType<hivm::TCoreTypeAttr>(
      hivm::TCoreTypeAttr::name);
  ASSERT_TRUE(defaultAttr);
  EXPECT_EQ(defaultAttr.getTcoretype(), hivm::TCoreType::VECTOR);
  EXPECT_EQ(cubeLoad.getTcoretype().getTcoretype(), hivm::TCoreType::CUBE);

  MarkRealCoreTypeOptions cleanupOpts;
  cleanupOpts.removeCoreTypeAttrs = true;
  bishengir::BiShengIRPassManager cleanupPm(
      &context, "builtin.module", mlir::PassManager::Nesting::Implicit);
  cleanupPm.addPass(mlir::hivm::createMarkRealCoreTypePass(cleanupOpts));
  ASSERT_TRUE(
      succeeded(static_cast<mlir::PassManager &>(cleanupPm).run(mod.get())));

  EXPECT_FALSE(scalarLoad->hasAttr(hivm::TCoreTypeAttr::name));
  EXPECT_FALSE(defaultLoad->hasAttr(hivm::TCoreTypeAttr::name));
  EXPECT_FALSE(cubeLoad->hasAttr(hivm::TCoreTypeAttr::name));
}

} // namespace
