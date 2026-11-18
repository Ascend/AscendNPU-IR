//===- L12L0BuilderTest.cpp - Whole-view copy construction
//-----------------===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

#include <utility>

using namespace mlir;
using namespace mlir::hivm;

namespace {
class L12L0BuilderTest : public ::testing::Test {
protected:
  L12L0BuilderTest() : builder(&context) {
    context.loadDialect<HIVMDialect>();
    builder.setInsertionPointToEnd(&block);
  }

  Value buffer(ArrayRef<int64_t> shape, Type element, AddressSpace space) {
    auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface{},
                                AddressSpaceAttr::get(&context, space));
    return block.addArgument(type, builder.getUnknownLoc());
  }

  void expectWholeView(L12L0Op op, Value src, Value dst, int64_t mn, int64_t k,
                       bool transposed) {
    EXPECT_TRUE(succeeded(verify(op.getOperation())));
    EXPECT_EQ(op.getSrc(), src);
    EXPECT_EQ(op.getDst(), dst);
    EXPECT_EQ(op->getNumOperands(), 2u);
    EXPECT_EQ(op->getNumResults(), 0u);
    EXPECT_EQ(op.getKPartIdxAttr().getInt(), 0);
    EXPECT_EQ(op.getKPartLoopAttr().getInt(), 1);
    EXPECT_EQ(op.getKPartAttr().getInt(), k);
    EXPECT_EQ(op.getKPartCeilAttr().getInt(), k);
    EXPECT_EQ(op.getKPartActualAttr().getInt(), k);
    EXPECT_EQ(op.getMAttr().getInt(), mn);
    EXPECT_EQ(op.getIsTransposed(), transposed);
    EXPECT_FALSE(op->hasAttr("actual_batch"));
  }

  MLIRContext context;
  OpBuilder builder;
  Block block;
};

TEST_F(L12L0BuilderTest, LogicalViewsUseDestinationAxes) {
  for (Type element : SmallVector<Type>{
           builder.getF16Type(), builder.getBF16Type(), builder.getF32Type()}) {
    for (bool toA : {false, true}) {
      for (bool transposed : {false, true}) {
        SmallVector<int64_t> dstShape =
            toA ? SmallVector<int64_t>{48, 32} : SmallVector<int64_t>{32, 48};
        auto srcShape = dstShape;
        if (transposed)
          std::swap(srcShape[0], srcShape[1]);
        Value src = buffer(srcShape, element, AddressSpace::L1);
        Value dst = buffer(dstShape, element,
                           toA ? AddressSpace::L0A : AddressSpace::L0B);
        auto op = builder.create<L12L0Op>(builder.getUnknownLoc(), src, dst,
                                          transposed);
        expectWholeView(op, src, dst, 48, 32, transposed);
      }
    }
  }
}

TEST_F(L12L0BuilderTest, PhysicalViewsRespectElementBlockSize) {
  for (Type element : SmallVector<Type>{
           builder.getF16Type(), builder.getBF16Type(), builder.getF32Type()}) {
    int64_t c0 = 256 / element.getIntOrFloatBitWidth();
    for (bool toA : {false, true}) {
      for (bool transposed : {false, true}) {
        bool sourceKOuter = toA != transposed;
        Value src = buffer({(sourceKOuter ? 32 : 48) / c0,
                            (sourceKOuter ? 48 : 32) / 16, 16, c0},
                           element, AddressSpace::L1);
        Value dst = buffer({32 / c0, 3, 16, c0}, element,
                           toA ? AddressSpace::L0A : AddressSpace::L0B);
        auto op = builder.create<L12L0Op>(builder.getUnknownLoc(), src, dst,
                                          transposed);
        expectWholeView(op, src, dst, 48, 32, transposed);
      }
    }
  }
}

TEST_F(L12L0BuilderTest, DefaultTransposeAndParentStride) {
  auto type = [&](AddressSpace space, int64_t outerStride, int64_t offset) {
    return MemRefType::get(
        {2, 3, 16, 16}, builder.getF16Type(),
        StridedLayoutAttr::get(&context, offset, {outerStride, 256, 16, 1}),
        AddressSpaceAttr::get(&context, space));
  };
  Value src = block.addArgument(type(AddressSpace::L1, 2048, 2304),
                                builder.getUnknownLoc());
  Value dst = block.addArgument(type(AddressSpace::L0A, 1024, 1280),
                                builder.getUnknownLoc());
  auto op = builder.create<L12L0Op>(builder.getUnknownLoc(), src, dst);
  expectWholeView(op, src, dst, 48, 32, false);
}

TEST_F(L12L0BuilderTest, ExplicitPartitionBuilderRetainsLegacySemantics) {
  Value src = buffer({6, 3, 16, 16}, builder.getF16Type(), AddressSpace::L1);
  Value dst = buffer({2, 3, 16, 16}, builder.getF16Type(), AddressSpace::L0A);
  auto attr = [&](int64_t value) { return builder.getI64IntegerAttr(value); };
  auto op = builder.create<L12L0Op>(
      builder.getUnknownLoc(), TypeRange{}, src, dst, attr(2), attr(32),
      attr(16), attr(3), attr(13), attr(48), builder.getBoolAttr(false));
  EXPECT_TRUE(succeeded(verify(op.getOperation())));
  EXPECT_EQ(op.getKPartIdxAttr().getInt(), 2);
  EXPECT_EQ(op.getKPartLoopAttr().getInt(), 3);
  EXPECT_EQ(op.getKPartAttr().getInt(), 32);
  EXPECT_EQ(op.getKPartCeilAttr().getInt(), 16);
  EXPECT_EQ(op.getKPartActualAttr().getInt(), 13);
  EXPECT_EQ(op.getMAttr().getInt(), 48);
}
} // namespace
