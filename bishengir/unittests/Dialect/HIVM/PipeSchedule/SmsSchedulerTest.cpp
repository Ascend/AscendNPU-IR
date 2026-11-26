//===- SmsSchedulerTest.cpp - Pure C++ SMS scheduler tests ----------------===//
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

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DepthHeight.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/MRT.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/ResMII.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SwingOrder.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bishengir::hivm::sms {
namespace {

OpNode node(int ordinal, const std::string &name, const std::string &pipeline,
            int latency) {
  const bool parallel = pipeline == "mte1" || pipeline == "mte2" ||
                        pipeline == "mte3" || pipeline == "fixpipe";
  return OpNode{ordinal, name, pipeline, latency, latency, parallel};
}

DDG makeDiamond() {
  DDG ddg;
  ddg.addNode(node(0, "mte2_load", "mte2", 2));
  ddg.addNode(node(1, "cube_mmad", "cube", 4));
  ddg.addNode(node(2, "vec_exp", "vector", 3));
  ddg.addNode(node(3, "fixpipe_st", "fixpipe", 1));
  ddg.addEdge(0, 1);
  ddg.addEdge(0, 2);
  ddg.addEdge(1, 3);
  ddg.addEdge(2, 3);
  return ddg;
}

DDG makeChain() {
  DDG ddg;
  ddg.addNode(node(0, "mte2_in", "mte2", 2));
  ddg.addNode(node(1, "cube_comp", "cube", 5));
  ddg.addNode(node(2, "fixpipe_out", "fixpipe", 1));
  ddg.addEdge(0, 1);
  ddg.addEdge(1, 2);
  return ddg;
}

DDG makeWide() {
  DDG ddg;
  ddg.addNode(node(0, "mte2_load0", "mte2", 2));
  ddg.addNode(node(1, "mte2_load1", "mte2", 2));
  ddg.addNode(node(2, "cube_mmad", "cube", 4));
  ddg.addNode(node(3, "vec_add", "vector", 2));
  ddg.addNode(node(4, "vec_mul", "vector", 3));
  ddg.addNode(node(5, "fixpipe_st", "fixpipe", 1));
  ddg.addEdge(0, 2);
  ddg.addEdge(1, 2);
  ddg.addEdge(2, 3);
  ddg.addEdge(2, 4);
  ddg.addEdge(3, 5);
  ddg.addEdge(4, 5);
  return ddg;
}

void expectValidSchedule(const DDG &ddg, const ScheduleResult &result) {
  ASSERT_GE(result.ii, computeResMII(ddg));
  ASSERT_EQ(result.nodeToCycle.size(), static_cast<size_t>(ddg.size()));
  ASSERT_EQ(result.nodeToStage.size(), static_cast<size_t>(ddg.size()));
  for (int i = 0; i < ddg.size(); ++i) {
    ASSERT_TRUE(result.nodeToCycle.count(i));
    ASSERT_TRUE(result.nodeToStage.count(i));
    EXPECT_GE(result.nodeToCycle.at(i), 0);
    EXPECT_EQ(result.nodeToStage.at(i), result.nodeToCycle.at(i) / result.ii);
  }
  for (const DDGEdge &edge : ddg.getEdges()) {
    if (!result.nodeToCycle.count(edge.src) ||
        !result.nodeToCycle.count(edge.dst))
      continue;
    EXPECT_GE(result.nodeToCycle.at(edge.dst),
              result.nodeToCycle.at(edge.src) + edge.latency);
  }
  EXPECT_GE(result.stageCount, 1);
}

TEST(DDGTest, ConstructionMaintainsAdjacencyAndEdgeLatency) {
  DDG ddg = makeDiamond();
  ASSERT_EQ(ddg.size(), 4);
  ASSERT_EQ(ddg.getEdges().size(), 4u);
  EXPECT_EQ(ddg.getPreds()[3], (std::vector<int>{1, 2}));
  EXPECT_EQ(ddg.getSuccs()[0], (std::vector<int>{1, 2}));
  EXPECT_EQ(ddg.getInEdges()[1][0].latency, 2);
  EXPECT_EQ(ddg.getOutEdges()[1][0].latency, 4);
}

TEST(ResMIITest, SumsOnlySingleOccupancyPipelines) {
  EXPECT_EQ(computeResMII(makeDiamond()), 4);
  EXPECT_EQ(computeResMII(makeChain()), 5);
  EXPECT_EQ(computeResMII(makeWide()), 5);

  DDG parallelOnly;
  parallelOnly.addNode(node(0, "load", "mte2", 7));
  EXPECT_EQ(computeResMII(parallelOnly), 1);
}

TEST(AsapAlapTest, ComputesDiamondIntermediateValues) {
  DDG ddg = makeDiamond();
  const std::vector<int> asap = computeASAP(ddg);
  EXPECT_EQ(asap, (std::vector<int>{0, 2, 2, 6}));

  const int horizon = *std::max_element(asap.begin(), asap.end()) + 4;
  const std::vector<int> alap = computeALAP(ddg, horizon);
  EXPECT_EQ(horizon, 10);
  EXPECT_EQ(alap, (std::vector<int>{3, 5, 6, 9}));
  EXPECT_EQ(computeMOV(asap, alap), (std::vector<int>{3, 3, 4, 3}));
}

TEST(AsapAlapTest, ComputesChainIntermediateValues) {
  DDG ddg = makeChain();
  const std::vector<int> asap = computeASAP(ddg);
  EXPECT_EQ(asap, (std::vector<int>{0, 2, 7}));
  EXPECT_EQ(computeALAP(ddg, 12), (std::vector<int>{4, 6, 11}));
  EXPECT_EQ(computeMOV(asap, computeALAP(ddg, 12)),
            (std::vector<int>{4, 4, 4}));
}

TEST(DepthHeightTest, UsesCurrentNodeLatency) {
  EXPECT_EQ(computeDepth(makeDiamond()), (std::vector<int>{0, 4, 3, 5}));
  EXPECT_EQ(computeHeight(makeDiamond()), (std::vector<int>{6, 4, 3, 0}));
  EXPECT_EQ(computeDepth(makeChain()), (std::vector<int>{0, 5, 6}));
  EXPECT_EQ(computeHeight(makeChain()), (std::vector<int>{7, 5, 0}));
  EXPECT_EQ(computeDepth(makeWide()), (std::vector<int>{0, 0, 4, 6, 7, 8}));
  EXPECT_EQ(computeHeight(makeWide()), (std::vector<int>{9, 9, 7, 2, 3, 0}));
}

TEST(SwingOrderTest, CoversEveryNodeAndUsesExpectedSeed) {
  DDG ddg = makeDiamond();
  const std::vector<int> asap = computeASAP(ddg);
  const int horizon =
      *std::max_element(asap.begin(), asap.end()) + computeResMII(ddg);
  const std::vector<int> mov = computeMOV(asap, computeALAP(ddg, horizon));
  const std::vector<OrderEntry> order =
      buildScheduleOrder(ddg, asap, mov, computeDepth(ddg), computeHeight(ddg));

  ASSERT_EQ(order.size(), 4u);
  EXPECT_EQ(order.front().nodeIdx, 3);
  EXPECT_FALSE(order.front().topDown);
  std::set<int> covered;
  for (const OrderEntry &entry : order)
    covered.insert(entry.nodeIdx);
  EXPECT_EQ(covered, (std::set<int>{0, 1, 2, 3}));
}

TEST(SwingOrderTest, BreaksSeedTieBySmallestNodeIndex) {
  DDG ddg;
  ddg.addNode(node(0, "left", "mte2", 1));
  ddg.addNode(node(1, "right", "mte3", 1));
  const std::vector<int> values{0, 0};
  const std::vector<OrderEntry> order =
      buildScheduleOrder(ddg, values, values, values, values);
  ASSERT_EQ(order.size(), 2u);
  // Python: max(range(n), key=lambda i: (asap[i], -mov[i], -i))
  // All equal except -i: -0 > -1, so seed = node 0.
  EXPECT_EQ(order.front().nodeIdx, 0);
}

TEST(MRTTest, ReservesModuloIntervalsOnSinglePipelines) {
  MRT mrt(5);
  EXPECT_FALSE(mrt.isParallel("vector"));
  EXPECT_TRUE(mrt.isIntervalFree(4, "vector", 2));
  mrt.reserve(4, "vector", 17, 2);
  EXPECT_FALSE(mrt.isIntervalFree(4, "vector", 1));
  EXPECT_FALSE(mrt.isIntervalFree(0, "vector", 1));
  EXPECT_TRUE(mrt.isIntervalFree(1, "vector", 2));
}

TEST(MRTTest, ParallelPipelinesNeverConflictAndCanBeRebuilt) {
  MRT mrt(4);
  EXPECT_TRUE(mrt.isParallel("mte2"));
  mrt.reserve(0, "mte2", 0, 4);
  EXPECT_TRUE(mrt.isIntervalFree(0, "mte2", 4));

  DDG ddg = makeDiamond();
  std::map<int, int> scheduled{{0, 1}, {1, 3}};
  mrt.rebuild(6, scheduled, ddg);
  EXPECT_EQ(mrt.getII(), 6);
  EXPECT_FALSE(mrt.isIntervalFree(3, "cube", 1));
}

TEST(SmsSchedulerTest, SchedulesAllFixedGraphs) {
  for (const DDG &ddg : {makeDiamond(), makeChain(), makeWide()}) {
    const std::optional<ScheduleResult> result = smsSchedule(ddg);
    ASSERT_TRUE(result.has_value());
    expectValidSchedule(ddg, *result);
  }
}

TEST(SmsSchedulerTest, HonorsClosedIISearchInterval) {
  DDG schedulable;
  schedulable.addNode(node(0, "single", "cube", 2));
  // maxIISearch=0 still attempts ResMII exactly once.
  ASSERT_TRUE(smsSchedule(schedulable, 0).has_value());

  EXPECT_FALSE(smsSchedule(makeDiamond(), -1).has_value());
}

TEST(DagVerifierTest, AcceptsValidDag) {
  VerifyResult result = verifyDag(makeDiamond());
  EXPECT_TRUE(result.ok) << result.error;
  EXPECT_TRUE(result.error.empty());
}

TEST(DagVerifierTest, RejectsEmptyCyclicAndInvalidNodes) {
  EXPECT_FALSE(verifyDag(DDG{}).ok);

  DDG cyclic = makeChain();
  cyclic.addEdge(2, 0);
  EXPECT_FALSE(verifyDag(cyclic).ok);

  DDG badPipeline;
  badPipeline.addNode(node(0, "bad", "scalar", 1));
  EXPECT_FALSE(verifyDag(badPipeline).ok);

  DDG zeroLatency;
  zeroLatency.addNode(node(0, "zero", "cube", 0));
  EXPECT_FALSE(verifyDag(zeroLatency).ok);

  DDG zeroIssue;
  zeroIssue.addNode(node(0, "zero_issue", "cube", 1));
  zeroIssue.getNodes()[0].issueTime = 0;
  EXPECT_FALSE(verifyDag(zeroIssue).ok);
}

TEST(SmsSchedulerTest, IsDeterministic) {
  const auto first = smsSchedule(makeWide());
  const auto second = smsSchedule(makeWide());
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->ii, second->ii);
  EXPECT_EQ(first->nodeToCycle, second->nodeToCycle);
  EXPECT_EQ(first->nodeToStage, second->nodeToStage);
  EXPECT_EQ(first->stageCount, second->stageCount);
}

} // namespace
} // namespace bishengir::hivm::sms
