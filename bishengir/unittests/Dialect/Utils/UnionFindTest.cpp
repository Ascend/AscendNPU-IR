//===- UnionFindTest.cpp - UnionFind unit tests ---------------------------===//
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
#include "bishengir/Dialect/Utils/UnionFind.h"

namespace {

class UnionFindTest : public ::testing::Test {
protected:
  UnionFindBase dsu{16};
};

TEST_F(UnionFindTest, FindNegativeInputReturnsNegative) {
  EXPECT_EQ(dsu.find(-1), -1);
  EXPECT_EQ(dsu.find(-100), -100);
  EXPECT_EQ(dsu.find(-2147483647), -2147483647);
}

TEST_F(UnionFindTest, FindWithValidPositiveInput) {
  EXPECT_EQ(dsu.find(0), 0);
  EXPECT_EQ(dsu.find(5), 5);
  EXPECT_EQ(dsu.find(15), 15);
}

TEST_F(UnionFindTest, JoinTwoSets) {
  dsu.join(2, 3);
  EXPECT_EQ(dsu.find(2), dsu.find(3));
}

TEST_F(UnionFindTest, JoinMultipleSets) {
  dsu.join(1, 2);
  dsu.join(2, 3);
  dsu.join(3, 4);
  EXPECT_EQ(dsu.find(1), dsu.find(4));
  EXPECT_EQ(dsu.find(1), dsu.find(2));
  EXPECT_EQ(dsu.find(1), dsu.find(3));
}

TEST_F(UnionFindTest, FindAfterAllocateMinimum) {
  dsu.allocateMinimum(50);
  EXPECT_EQ(dsu.find(30), 30);
  EXPECT_EQ(dsu.find(50), 50);
}

TEST_F(UnionFindTest, JoinAfterAllocateMinimum) {
  dsu.allocateMinimum(100);
  dsu.join(50, 60);
  EXPECT_EQ(dsu.find(50), dsu.find(60));
}

TEST_F(UnionFindTest, MinIndexIsCorrectAfterJoin) {
  dsu.join(5, 10);
  int root = dsu.find(5);
  EXPECT_LE(dsu.minIndex[root], 5);
  EXPECT_LE(dsu.minIndex[root], 10);
}

TEST_F(UnionFindTest, FindDoesNotCrashOnNegativeInput) {
  EXPECT_NO_FATAL_FAILURE(dsu.find(-1));
  EXPECT_NO_FATAL_FAILURE(dsu.find(-1000));
  EXPECT_NO_FATAL_FAILURE(dsu.find(-2147483647));
}

TEST_F(UnionFindTest, FindNegativeDoesNotGrowStructure) {
  size_t originalSize = dsu.minIndex.size();
  dsu.find(-1);
  EXPECT_EQ(dsu.minIndex.size(), originalSize);
}

} // namespace
