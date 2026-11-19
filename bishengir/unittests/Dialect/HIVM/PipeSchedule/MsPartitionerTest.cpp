//===- MsPartitionerTest.cpp - WorkItem schedule conversion tests
//-----------===//
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

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsPartitioner.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/WorklistBuilder.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <limits>

namespace mlir {
namespace hivm {
namespace {

class MsPartitionerTest : public ::testing::Test {
protected:
  void SetUp() override {
    context
        .loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    module = parseSourceString<ModuleOp>(R"mlir(
      func.func @test() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        scf.for %i = %c0 to %c8 step %c1 {
        }
        return
      }
    )mlir",
                                         &context);
    ASSERT_TRUE(module);
    module->walk([&](scf::ForOp op) { loop = op; });
    ASSERT_TRUE(loop);
  }

  // Explicit node linkage isolates conversion from WorklistBuilder and costs.
  void build(ArrayRef<int> stages) {
    schedule.ii = 2;
    schedule.stageCount = 1;
    for (int stage : stages) {
      auto item = std::make_shared<WorkItem>();
      item->core = TCoreType::VECTOR;
      item->multibuffer = 2;
      dag.defaultWorklist.push_back(item);
      int index = addNode(item.get(), stage);
      dag.workItemToDdgIndex[item.get()] = index;
    }
  }

  int addNode(WorkItem *item, int stage, bool synthetic = false) {
    int index = dag.ddg.size();
    MsNode node;
    node.ddgIndex = index;
    node.workItem = item;
    node.kind = synthetic ? MsNodeKind::Transfer : MsNodeKind::VectorGroup;
    node.isSynthetic = synthetic;
    node.pipeline = synthetic ? "mte2" : "vector";
    node.storageTag = synthetic ? "GM_to_L1" : "";
    dag.nodes.push_back(node);
    sms::OpNode ddgNode;
    ddgNode.ordinal = index;
    ddgNode.pipeline = node.pipeline;
    ddgNode.latency = 1;
    ddgNode.issueTime = 1;
    dag.ddg.addNode(ddgNode);
    schedule.nodeToStage[index] = stage;
    schedule.nodeToCycle[index] = stage * schedule.ii;
    schedule.stageCount = std::max(schedule.stageCount, stage + 1);
    return index;
  }

  FailureOr<MsPartitionResult> partition() {
    return MsPartitioner(loop, dag, schedule).partition();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  scf::ForOp loop;
  MsDagBuildResult dag;
  sms::ScheduleResult schedule;
};

TEST_F(MsPartitionerTest, SparseStagesPreserveGaps) {
  build({0, 3});
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->stages.size(), 4u);
  EXPECT_EQ(result->pipelineDepth, 4);
  EXPECT_TRUE(result->stages[1].workItems.empty());
  EXPECT_TRUE(result->stages[2].workItems.empty());
  EXPECT_EQ(result->stages[0].preloadCoordinate, 3);
  EXPECT_EQ(result->stages[3].preloadCoordinate, 0);
}

TEST_F(MsPartitionerTest, ShiftOnlyLeadingOffset) {
  build({2, 4, 7});
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->stages.size(), 6u);
  EXPECT_EQ(result->pipelineDepth, 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(result->stages[i].rawMsStage, i + 2);
    EXPECT_EQ(result->stages[i].stageOrdinal, i);
    EXPECT_EQ(result->stages[i].preloadCoordinate, 5 - i);
  }
  EXPECT_EQ(result->stages[0].workItems.front(), dag.defaultWorklist[0].get());
  EXPECT_EQ(result->stages[2].workItems.front(), dag.defaultWorklist[1].get());
  EXPECT_EQ(result->stages[5].workItems.front(), dag.defaultWorklist[2].get());
  for (int stage : {1, 3, 4})
    EXPECT_TRUE(result->stages[stage].workItems.empty());
  for (int i = 0; i < 3; ++i) {
    auto timing =
        result->workItemSchedules.lookup(dag.defaultWorklist[i].get());
    EXPECT_EQ(timing.ddgIndex, i);
    EXPECT_EQ(timing.rawStage, schedule.nodeToStage.at(i));
    EXPECT_EQ(timing.cycle, schedule.nodeToCycle.at(i));
  }
}

TEST_F(MsPartitionerTest, SharedNonzeroStageAndIIPreserved) {
  build({4, 4});
  schedule.ii = 7;
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->pipelineDepth, 1);
  EXPECT_EQ(result->ii, 7);
  ASSERT_EQ(result->stages.size(), 1u);
  EXPECT_EQ(result->stages[0].preloadCoordinate, 0);
  ASSERT_EQ(result->stages[0].workItems.size(), 2u);
  EXPECT_EQ(result->stages[0].workItems[0], dag.defaultWorklist[0].get());
  EXPECT_EQ(result->stages[0].workItems[1], dag.defaultWorklist[1].get());
}

TEST_F(MsPartitionerTest, TransfersDoNotChangeWorkItemDepth) {
  build({2, 4});
  addNode(nullptr, 0, true);
  int last = addNode(nullptr, 9, true);
  // A synthetic endpoint is not interpreted as a WorkItem ordering edge.
  dag.ddg.addEdge(1, last);
  dag.ddg.addEdge(last, 0);
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->pipelineDepth, 3);
  ASSERT_EQ(result->transferSchedules.size(), 2u);
  EXPECT_EQ(result->transferSchedules[0].stage, 0);
  EXPECT_EQ(result->transferSchedules[1].stage, 9);
  EXPECT_EQ(result->transferSchedules[1].cycle, 18);
  EXPECT_EQ(result->transferSchedules[1].pipeline, "mte2");
  EXPECT_EQ(result->transferSchedules[1].storageTag, "GM_to_L1");
}

TEST_F(MsPartitionerTest, RejectsNonPositiveII) {
  build({0});
  schedule.ii = 0;
  EXPECT_TRUE(failed(partition()));
  schedule.ii = -1;
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsNonPositiveStageCount) {
  build({0});
  schedule.stageCount = 0;
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsMissingScheduleEntries) {
  build({0});
  schedule.nodeToCycle.erase(0);
  EXPECT_TRUE(failed(partition()));
  schedule.nodeToCycle[0] = 0;
  schedule.nodeToStage.erase(0);
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsNegativeSchedules) {
  build({0});
  schedule.nodeToCycle[0] = -1;
  EXPECT_TRUE(failed(partition()));
  schedule.nodeToCycle[0] = 0;
  schedule.nodeToStage[0] = -1;
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsExtraneousScheduleEntries) {
  build({0});
  schedule.nodeToCycle[1] = 0;
  EXPECT_TRUE(failed(partition()));
  schedule.nodeToCycle.erase(1);
  schedule.nodeToStage[-1] = 0;
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsMultipleComputeNodesForWorkItem) {
  build({0});
  addNode(dag.defaultWorklist.front().get(), 3);
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsMissingComputeNode) {
  build({0});
  dag.defaultWorklist.push_back(std::make_shared<WorkItem>());
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsNullAndForeignNodeOwners) {
  build({0});
  dag.nodes[0].workItem = nullptr;
  EXPECT_TRUE(failed(partition()));
  auto foreign = std::make_shared<WorkItem>();
  dag.nodes[0].workItem = foreign.get();
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsNullAndDuplicateWorkItems) {
  build({0});
  dag.defaultWorklist.push_back(nullptr);
  EXPECT_TRUE(failed(partition()));
  dag.defaultWorklist.back() = dag.defaultWorklist.front();
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsInvalidNodeRecords) {
  build({0, 1});
  dag.nodes[1].ddgIndex = 0;
  EXPECT_TRUE(failed(partition()));
  dag.nodes[1].ddgIndex = 2;
  EXPECT_TRUE(failed(partition()));
  dag.nodes.pop_back();
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsEmptyInputs) {
  build({});
  EXPECT_TRUE(failed(partition()));
  addNode(nullptr, 0, true);
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, RejectsUnrepresentableDepth) {
  build({0, 1});
  schedule.nodeToStage[1] = std::numeric_limits<int>::max();
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, ForwardEdgesAndInternalEdgesPass) {
  build({0, 2});
  dag.ddg.addEdge(0, 1);
  dag.ddg.addEdge(0, 0);
  EXPECT_TRUE(succeeded(partition()));
}

TEST_F(MsPartitionerTest, RejectsBackwardWorkItemEdge) {
  build({0, 2});
  dag.ddg.addEdge(1, 0);
  EXPECT_TRUE(failed(partition()));
}

TEST_F(MsPartitionerTest, ConversionPreservesInputs) {
  build({2, 4});
  // A shared IR operation tests preservation of multi-owner mappings only;
  // partitioning does not inspect or validate its contents.
  Operation *op = loop.getBody()->getTerminator();
  for (const auto &item : dag.defaultWorklist)
    item->ops.insert(op);
  dag.defaultOpToWorkItemMap[op] = {dag.defaultWorklist[0].get(),
                                    dag.defaultWorklist[1].get()};
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->opToWorkItemMap, dag.defaultOpToWorkItemMap);
  for (size_t i = 0; i < result->worklist.size(); ++i) {
    EXPECT_EQ(result->worklist[i], dag.defaultWorklist[i]);
    EXPECT_EQ(result->worklist[i]->ops.size(), 1u);
    EXPECT_EQ(result->worklist[i]->multibuffer, 2u);
    EXPECT_EQ(result->worklist[i]->core, TCoreType::VECTOR);
  }
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(before, after);
}

static constexpr const char *kBasicIR = R"mlir(
  func.func @basic(%bound: i32) {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %bound step %step : i32 {
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %a, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cube_out = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %cube_out_cast = memref.memory_space_cast %cube_out : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_out_tensor = bufferization.to_tensor %cube_out_cast restrict writable : memref<16x16xf32>
      %fix = hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_out_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v_init = tensor.empty() : tensor<16x16xf32>
      %v = hivm.hir.vexp ins(%fix : tensor<16x16xf32>) outs(%v_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield
    }
    return
  }
)mlir";

TEST_F(MsPartitionerTest, WorklistBuilderDagIntegration) {
  context.loadDialect<
      hivm::HIVMDialect, scope::ScopeDialect, memref::MemRefDialect,
      tensor::TensorDialect, bufferization::BufferizationDialect,
      annotation::AnnotationDialect, bishengir::memref_ext::MemRefExtDialect>();
  context.allowUnregisteredDialects();
  module = parseSourceString<ModuleOp>(kBasicIR, &context);
  ASSERT_TRUE(module);
  module->walk([&](scf::ForOp op) { loop = op; });
  ASSERT_TRUE(loop);
  WorklistBuilder worklistBuilder(loop, /*numMultibuffer=*/2);
  auto defaultBuild = worklistBuilder.build();
  ASSERT_TRUE(succeeded(defaultBuild));
  ASSERT_GE(defaultBuild->worklist.size(), 2u);

  struct FixedLatencyProvider : LatencyProvider {
    int getLatency(Operation *) const override { return 2; }
    int getIssueTime(Operation *) const override { return 2; }
  } latencyProvider;
  DenseMap<Value, Operation *> noClones;
  MsDagBuilder builder(loop, latencyProvider, noClones, *defaultBuild);
  auto builtDag = builder.build();
  ASSERT_TRUE(succeeded(builtDag));
  dag = std::move(*builtDag);
  SmallVector<int> rawStages;
  for (const auto &item : defaultBuild->worklist) {
    int stage = 2 + 3 * rawStages.size();
    rawStages.push_back(stage);
    int count = 0;
    for (const MsNode &node : dag.nodes)
      if (!node.isSynthetic && node.workItem == item.get())
        ++count;
    EXPECT_EQ(count, 1);
  }
  schedule.ii = 2;
  schedule.stageCount = rawStages.back() + 1;
  for (const MsNode &node : dag.nodes) {
    int stage = node.isSynthetic ? 0 : rawStages[node.workItemIndex];
    schedule.nodeToStage[node.ddgIndex] = stage;
    schedule.nodeToCycle[node.ddgIndex] = stage * schedule.ii;
  }
  auto result = partition();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->pipelineDepth, rawStages.back() - rawStages.front() + 1);
  EXPECT_EQ(result->opToWorkItemMap, defaultBuild->opToWorkItemMap);
  ASSERT_EQ(result->worklist.size(), defaultBuild->worklist.size());
  for (size_t i = 0; i < result->worklist.size(); ++i) {
    EXPECT_EQ(result->worklist[i], defaultBuild->worklist[i]);
    int normalized = rawStages[i] - rawStages.front();
    ASSERT_EQ(result->stages[normalized].workItems.size(), 1u);
    EXPECT_EQ(result->stages[normalized].workItems.front(),
              result->worklist[i].get());
    EXPECT_EQ(result->stages[normalized].preloadCoordinate,
              rawStages.back() - rawStages[i]);
  }
  EXPECT_TRUE(result->stages[1].workItems.empty());
}

} // namespace
} // namespace hivm
} // namespace mlir
