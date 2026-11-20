//===- CustomOpMemoryEffectsTest.cpp - CustomOp memory effect tests -------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

using namespace mlir;

namespace {

/// The memory effects declared on an operand, as reported by
/// MemoryEffectOpInterface.
struct OperandEffects {
  bool read = false;
  bool write = false;
};

/// Collect the effects that are attached to a specific operand. Resource-level
/// effects carry no operand and are skipped, so an entry stays default
/// constructed when the op declares nothing for that operand.
SmallVector<OperandEffects> collectOperandEffects(Operation *op) {
  auto interface = cast<MemoryEffectOpInterface>(op);
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  interface.getEffects(effects);

  SmallVector<OperandEffects> perOperand(op->getNumOperands());
  for (const auto &effect : effects) {
    auto *operand = effect.getEffectValue<OpOperand *>();
    if (!operand)
      continue;

    OperandEffects &slot = perOperand[operand->getOperandNumber()];
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      slot.read = true;
    else if (isa<MemoryEffects::Write>(effect.getEffect()))
      slot.write = true;
  }
  return perOperand;
}

/// Return the single hivm.hir.custom / hivm.hir.custom_macro op in `module`.
Operation *getCustomOp(ModuleOp module) {
  Operation *result = nullptr;
  module.walk([&](Operation *op) {
    if (isa<hivm::CustomOp, hivm::CustomMacroOp>(op))
      result = op;
  });
  return result;
}

/// A CustomOp whose first input is bound in place to the output. The second
/// input is an ordinary, read-only operand.
constexpr StringLiteral kInplaceCustomPayload = R"mlir(
module {
  func.func @inplace(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "k_inplace",
         arg_attrs = [{inplace_outs = 0 : i32}, {}]}
        "user.inplace"
        ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>)
        outs(%arg0 : memref<256xf32>)
    return
  }
}
)mlir";

/// The same op without inplace_outs: every input is read only.
constexpr StringLiteral kNoInplaceCustomPayload = R"mlir(
module {
  func.func @no_inplace(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "k_no_inplace"}
        "user.no_inplace"
        ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>)
        outs(%arg1 : memref<256xf32>)
    return
  }
}
)mlir";

/// A CustomMacroOp with the same inplace binding.
constexpr StringLiteral kInplaceMacroPayload = R"mlir(
module {
  func.func @inplace_macro(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "k_inplace_macro",
         arg_attrs = [{inplace_outs = 0 : i32}, {}]}
        "user.inplace_macro"
        ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>)
        outs(%arg0 : memref<256xf32>)
    return
  }
}
)mlir";

class CustomOpMemoryEffectsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<hivm::HIVMDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadDialect<hivm::HIVMDialect>();
    context->loadDialect<func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    auto module = parseSourceString<ModuleOp>(source, context.get());
    if (module)
      EXPECT_TRUE(succeeded(verify(module.get())));
    return module;
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
};

TEST_F(CustomOpMemoryEffectsTest, InplaceInputIsReadAndWritten) {
  auto module = parse(kInplaceCustomPayload);
  ASSERT_TRUE(module.get());

  Operation *custom = getCustomOp(*module);
  ASSERT_TRUE(custom);
  ASSERT_EQ(custom->getNumOperands(), 3u);

  auto effects = collectOperandEffects(custom);

  // The inplace input carries the result of the computation.
  EXPECT_TRUE(effects[0].read);
  EXPECT_TRUE(effects[0].write);

  // An input without inplace_outs has no operand-attached effect of its own.
  EXPECT_FALSE(effects[1].read);
  EXPECT_FALSE(effects[1].write);
}

TEST_F(CustomOpMemoryEffectsTest, InplaceEffectsStayOnTheInput) {
  auto module = parse(kInplaceCustomPayload);
  ASSERT_TRUE(module.get());

  Operation *custom = getCustomOp(*module);
  ASSERT_TRUE(custom);

  auto effects = collectOperandEffects(custom);

  // The bound output is the same storage, but the alias is declared on the
  // input, so only the input reports the operand-level effects.
  EXPECT_FALSE(effects[2].read);
  EXPECT_FALSE(effects[2].write);
}

TEST_F(CustomOpMemoryEffectsTest, NonInplaceInputsHaveNoOperandEffects) {
  auto module = parse(kNoInplaceCustomPayload);
  ASSERT_TRUE(module.get());

  Operation *custom = getCustomOp(*module);
  ASSERT_TRUE(custom);
  ASSERT_EQ(custom->getNumOperands(), 3u);

  auto effects = collectOperandEffects(custom);
  for (const OperandEffects &operandEffects : effects) {
    EXPECT_FALSE(operandEffects.read);
    EXPECT_FALSE(operandEffects.write);
  }
}

TEST_F(CustomOpMemoryEffectsTest, ResourceLevelEffectsAreAlwaysDeclared) {
  auto module = parse(kNoInplaceCustomPayload);
  ASSERT_TRUE(module.get());

  Operation *custom = getCustomOp(*module);
  ASSERT_TRUE(custom);

  auto interface = cast<MemoryEffectOpInterface>(custom);
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  interface.getEffects(effects);

  // Scheduling still relies on the conservative resource-level Read / Write,
  // so those must be present even when no operand declares inplace_outs.
  bool hasResourceRead = false;
  bool hasResourceWrite = false;
  for (const auto &effect : effects) {
    if (effect.getEffectValue<OpOperand *>())
      continue;
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      hasResourceRead = true;
    if (isa<MemoryEffects::Write>(effect.getEffect()))
      hasResourceWrite = true;
  }
  EXPECT_TRUE(hasResourceRead);
  EXPECT_TRUE(hasResourceWrite);
}

TEST_F(CustomOpMemoryEffectsTest, InplaceInputIsReadAndWrittenOnMacroOp) {
  auto module = parse(kInplaceMacroPayload);
  ASSERT_TRUE(module.get());

  Operation *custom = getCustomOp(*module);
  ASSERT_TRUE(custom);
  ASSERT_TRUE(isa<hivm::CustomMacroOp>(custom));
  ASSERT_EQ(custom->getNumOperands(), 3u);

  auto effects = collectOperandEffects(custom);
  EXPECT_TRUE(effects[0].read);
  EXPECT_TRUE(effects[0].write);
  EXPECT_FALSE(effects[1].read);
  EXPECT_FALSE(effects[1].write);
}

} // namespace
