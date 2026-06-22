//===- SelectRoundModeTest.cpp - selectRoundMode unit tests ---------------===//
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

#include "gtest/gtest.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
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

TEST_F(SelectRoundModeTest, IntegerToIntegerDefaultsToRint) {
  expectRoundMode(builder.getI16Type(), builder.getI8Type(),
                  hivm::RoundMode::RINT);
  expectRoundMode(builder.getI32Type(), builder.getI16Type(),
                  hivm::RoundMode::RINT);
  expectRoundMode(builder.getI64Type(), builder.getI32Type(),
                  hivm::RoundMode::RINT);
}

TEST_F(SelectRoundModeTest, FloatToIntegerUsesTrunc) {
  expectRoundMode(builder.getF16Type(), builder.getI32Type(),
                  hivm::RoundMode::TRUNC);
  expectRoundMode(builder.getF32Type(), builder.getI16Type(),
                  hivm::RoundMode::TRUNC);
}

TEST_F(SelectRoundModeTest, IntegerToFloatUsesRint) {
  expectRoundMode(builder.getI32Type(), builder.getF16Type(),
                  hivm::RoundMode::RINT);
  expectRoundMode(builder.getI64Type(), builder.getF32Type(),
                  hivm::RoundMode::RINT);
}

} // namespace
