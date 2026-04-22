//===- SelectRoundModeTest.cpp - selectRoundMode unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace {

class SelectRoundModeTest : public ::testing::Test {
protected:
  template <typename RoundModeT>
  void expectRoundMode(Type inputType, Type outputType, RoundModeT expected) {
    EXPECT_EQ(utils::selectRoundMode<RoundModeT>(inputType, outputType),
              expected);
  }

  MLIRContext context;
  Builder builder{&context};
};

TEST_F(SelectRoundModeTest, FloatConversionsUseRint) {
  using RoundMode = hivm::RoundMode;

  expectRoundMode(builder.getF32Type(), builder.getF16Type(), RoundMode::RINT);
  expectRoundMode(builder.getF32Type(), builder.getBF16Type(),
                  RoundMode::RINT);
  expectRoundMode(builder.getF32Type(), builder.getF32Type(), RoundMode::RINT);
  expectRoundMode(builder.getF16Type(), builder.getF32Type(), RoundMode::RINT);
}

TEST_F(SelectRoundModeTest, Int8ToF16UsesRint) {
  expectRoundMode(builder.getI8Type(), builder.getF16Type(),
                  hivm::RoundMode::RINT);
}

TEST_F(SelectRoundModeTest, Int16ToInt8UsesTruncWithOverflow) {
  expectRoundMode(builder.getI16Type(), builder.getI8Type(),
                  hivm::RoundMode::TRUNCWITHOVERFLOW);
}

TEST_F(SelectRoundModeTest, FloatToIntegerUsesTrunc) {
  expectRoundMode(builder.getF16Type(), builder.getI32Type(),
                  hivm::RoundMode::TRUNC);
  expectRoundMode(builder.getF32Type(), builder.getI16Type(),
                  hivm::RoundMode::TRUNC);
}

TEST_F(SelectRoundModeTest, IntegerToFloatUsesTrunc) {
  expectRoundMode(builder.getI32Type(), builder.getF16Type(),
                  hivm::RoundMode::TRUNC);
  expectRoundMode(builder.getI64Type(), builder.getF32Type(),
                  hivm::RoundMode::TRUNC);
}

TEST_F(SelectRoundModeTest, IntegerToIntegerDefaultsToRint) {
  expectRoundMode(builder.getI32Type(), builder.getI16Type(),
                  hivm::RoundMode::RINT);
  expectRoundMode(builder.getI64Type(), builder.getI32Type(),
                  hivm::RoundMode::RINT);
}

} // namespace
