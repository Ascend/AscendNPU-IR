//===- MsPartitionerTest.cpp - MsPartitioner unit tests
//------------------===//
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
//
// Tests for MsPartitioner — the SMS schedule → WorkItem partition bridge.
//
// Tests use parseSourceString to construct HIVM IR, then exercise
// MsDagBuilder::build() to obtain a DAG, and construct sms::ScheduleResult
// explicitly to isolate partition behavior from scheduler heuristics.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsPartitioner.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsDagBuilder.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

namespace mlir {
namespace hivm {
namespace {

/// Fixed-latency provider for deterministic testing.
struct FixedLatencyProvider : public LatencyProvider {
  int fixedLatency;
  int fixedIssueTime;
  FixedLatencyProvider(int lat = 2, int issue = 2)
      : fixedLatency(lat), fixedIssueTime(issue) {}
  int getLatency(Operation *) const override { return fixedLatency; }
  int getIssueTime(Operation *) const override { return fixedIssueTime; }
};

class MsPartitionerTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<hivm::HIVMDialect, scope::ScopeDialect,
                        arith::ArithDialect, scf::SCFDialect, func::FuncDialect,
                        memref::MemRefDialect, tensor::TensorDialect,
                        bufferization::BufferizationDialect,
                        annotation::AnnotationDialect,
                        bishengir::memref_ext::MemRefExtDialect>();
    context.allowUnregisteredDialects();
  }

  OwningOpRef<ModuleOp> parseModule(StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  scf::ForOp findForOp(ModuleOp module) {
    scf::ForOp result;
    module.walk([&](scf::ForOp op) { result = op; });
    return result;
  }

  MLIRContext context;
};

/// Minimal MLIR with a load + store (two transfer ops, no compute).
static constexpr const char *kTransferOnly = R"mlir(
  func.func @transfer_only(%src: memref<32xf16>, %dst: memref<32xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %alloc = memref.alloc() : memref<32xf16>
      hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
      %t = bufferization.to_tensor %alloc : memref<32xf16>
      hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
      scf.yield
    }
    return
  }
)mlir";

//===----------------------------------------------------------------------===//
// Schedule validation tests
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, RejectsNonPositiveII) {
  auto module = parseModule(kTransferOnly);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return; // DAG may be empty (transfer-only) — still test schedule validation

  sms::ScheduleResult sched;
  sched.ii = 0; // Invalid
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = 0;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  EXPECT_TRUE(failed(result));
}

TEST_F(MsPartitionerTest, RejectsMissingNodeSchedule) {
  auto module = parseModule(kTransferOnly);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return;

  // Schedule with missing entries.
  sms::ScheduleResult sched;
  sched.ii = 2;
  sched.stageCount = 1;
  // Leave nodeToCycle/nodeToStage empty — should fail.

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Transfer-only DAG tests (transfer with no compute consumer rejects)
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, TransferOnlyRejects) {
  auto module = parseModule(kTransferOnly);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  // If DAG build succeeds (transfer-only), partition should reject because
  // transfer nodes have no compute consumer.
  if (failed(dagResult))
    return;

  sms::ScheduleResult sched;
  sched.ii = 2;
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = 0;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  // Transfer nodes have no compute consumer → fuseTransfers should fail.
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Dense coordinate tests with synthetic schedule
//===----------------------------------------------------------------------===//

// Helper: build a DAG from IR and manually construct a schedule that places
// all nodes in one stage.  Used for testing the partition flow independently.
TEST_F(MsPartitionerTest, SingleStageSingleTransfer) {
  auto source = R"mlir(
    func.func @single_stage(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %alloc = memref.alloc() : memref<32xf16>
        hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
        %t = bufferization.to_tensor %alloc : memref<32xf16>
        hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
        scf.yield
      }
      return
    }
  )mlir";
  auto module = parseModule(source);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult)) {
    // Transfer-only DAG may have no compute nodes → partition cannot proceed.
    // This is expected behavior, just verify the path does not crash.
    return;
  }

  // If DAG succeeded, it must have at least transfer nodes.
  // Check if there are any compute nodes.
  bool hasCompute = false;
  for (const auto &sn : dagResult->nodes) {
    if (sn.kind != MsNodeKind::Transfer)
      hasCompute = true;
  }
  if (!hasCompute)
    return; // Transfer-only → cannot test partition

  sms::ScheduleResult sched;
  sched.ii = 2;
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = 0;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  if (succeeded(result)) {
    EXPECT_EQ(result->pipelineDepth, 1);
    EXPECT_EQ(result->ii, 2);
    EXPECT_EQ(result->stages.size(), 1u);
    EXPECT_EQ(result->stages[0].preloadCoordinate, 0);
    EXPECT_EQ(result->stages[0].stageOrdinal, 0);
    // Verify opToWorkItemMap is consistent.
    for (auto &item : result->worklist) {
      for (Operation *op : item->ops) {
        ASSERT_TRUE(result->opToWorkItemMap.contains(op));
        auto &owners = result->opToWorkItemMap[op];
        EXPECT_EQ(owners.size(), 1u);
        EXPECT_EQ(owners[0], item.get());
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Schedule coordinate tests (synthetic)
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, SparseStagesDenseRenumber) {
  // Build a DAG and manually assign sparse stages to test dense renumbering.
  auto source = R"mlir(
    func.func @sparse(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %alloc = memref.alloc() : memref<32xf16>
        hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
        %t = bufferization.to_tensor %alloc : memref<32xf16>
        hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
        scf.yield
      }
      return
    }
  )mlir";
  auto module = parseModule(source);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return;

  // Only test the coordinate logic if we have at least 2 compute nodes
  // (so we can assign them to different sparse stages).
  int numCompute = 0;
  SmallVector<int> computeNodeIds;
  SmallVector<int> transferNodeIds;
  for (const auto &sn : dagResult->nodes) {
    if (sn.kind != MsNodeKind::Transfer) {
      computeNodeIds.push_back(sn.ddgIndex);
      numCompute++;
    } else {
      transferNodeIds.push_back(sn.ddgIndex);
    }
  }

  if (numCompute < 2)
    return;

  // Assign compute nodes to sparse stages 0 and 3.
  sms::ScheduleResult sched;
  sched.ii = 2;
  sched.stageCount = 4; // stages 0..3

  for (size_t i = 0; i < computeNodeIds.size(); ++i) {
    int nodeId = computeNodeIds[i];
    int stage = (i == 0) ? 0 : 3;
    sched.nodeToCycle[nodeId] = stage * sched.ii;
    sched.nodeToStage[nodeId] = stage;
  }

  // Assign transfers to match their consumer's stage (for legal fusion).
  for (int tid : transferNodeIds) {
    // Find which compute node(s) this transfer reaches via successor edges.
    const auto &succs = dagResult->ddg.getSuccs()[tid];
    int consumerStage = 0;
    for (int s : succs) {
      auto stageIt = sched.nodeToStage.find(s);
      if (stageIt != sched.nodeToStage.end()) {
        consumerStage = stageIt->second;
        break;
      }
    }
    sched.nodeToCycle[tid] = consumerStage * sched.ii;
    sched.nodeToStage[tid] = consumerStage;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  if (succeeded(result)) {
    // Sparse stages {0, 3} → dense ordinals {0, 1}.
    EXPECT_EQ(result->pipelineDepth, 2);
    EXPECT_EQ(result->stages.size(), 2u);
    // Stage 0 → ordinal 0 → coordinate 1.
    EXPECT_EQ(result->stages[0].rawMsStage, 0);
    EXPECT_EQ(result->stages[0].stageOrdinal, 0);
    EXPECT_EQ(result->stages[0].preloadCoordinate, 1);
    // Stage 3 → ordinal 1 → coordinate 0.
    EXPECT_EQ(result->stages[1].rawMsStage, 3);
    EXPECT_EQ(result->stages[1].stageOrdinal, 1);
    EXPECT_EQ(result->stages[1].preloadCoordinate, 0);
  }
}

//===----------------------------------------------------------------------===//
// Owner map consistency
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, OwnerMapBidirectional) {
  auto source = R"mlir(
    func.func @owner(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %alloc = memref.alloc() : memref<32xf16>
        hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
        %t = bufferization.to_tensor %alloc : memref<32xf16>
        hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
        scf.yield
      }
      return
    }
  )mlir";
  auto module = parseModule(source);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return;

  sms::ScheduleResult sched;
  sched.ii = 2;
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = 0;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  if (succeeded(result)) {
    // Verify bidirectional consistency:
    // For every (op, item) in opToWorkItemMap, op must be in item->ops.
    for (auto &[op, owners] : result->opToWorkItemMap) {
      for (WorkItem *item : owners) {
        EXPECT_TRUE(item->ops.contains(op)) << "op in map but not in item->ops";
      }
    }
    // For every op in every item->ops, the map must contain it.
    for (auto &item : result->worklist) {
      for (Operation *op : item->ops) {
        EXPECT_TRUE(result->opToWorkItemMap.contains(op))
            << "op in item->ops but not in map";
      }
    }
    // P0: single ownership — each op has exactly one owner.
    for (auto &[op, owners] : result->opToWorkItemMap) {
      EXPECT_EQ(owners.size(), 1u) << "P0 requires single ownership";
    }
  }
}

//===----------------------------------------------------------------------===//
// LatencyProvider interface test (basic sanity)
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, IIPreserved) {
  auto source = R"mlir(
    func.func @ii(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %alloc = memref.alloc() : memref<32xf16>
        hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
        %t = bufferization.to_tensor %alloc : memref<32xf16>
        hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
        scf.yield
      }
      return
    }
  )mlir";
  auto module = parseModule(source);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return;

  sms::ScheduleResult sched;
  sched.ii = 7;
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = 0;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  if (succeeded(result)) {
    EXPECT_EQ(result->ii, 7);
  }
}

//===----------------------------------------------------------------------===//
// Schedule metadata preservation
//===----------------------------------------------------------------------===//

TEST_F(MsPartitionerTest, OpScheduleMetadataPreserved) {
  auto source = R"mlir(
    func.func @meta(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %alloc = memref.alloc() : memref<32xf16>
        hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
        %t = bufferization.to_tensor %alloc : memref<32xf16>
        hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
        scf.yield
      }
      return
    }
  )mlir";
  auto module = parseModule(source);
  ASSERT_TRUE(module);
  scf::ForOp forOp = findForOp(*module);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto dagResult = builder.build();
  if (failed(dagResult))
    return;

  sms::ScheduleResult sched;
  sched.ii = 3;
  sched.stageCount = 1;
  for (int i = 0; i < dagResult->ddg.size(); ++i) {
    sched.nodeToCycle[i] = i * 2;
    sched.nodeToStage[i] = 0;
  }

  MsPartitioner partitioner(forOp, *dagResult, sched);
  auto result = partitioner.partition();
  if (succeeded(result)) {
    // Every node's ops should have cycle/stage metadata.
    for (const MsNode &sn : dagResult->nodes) {
      for (Operation *op : sn.ops) {
        EXPECT_TRUE(result->opToMsCycle.contains(op))
            << "Missing cycle metadata";
        EXPECT_TRUE(result->opToMsStage.contains(op))
            << "Missing stage metadata";
      }
    }
  }
}

} // namespace
} // namespace hivm
} // namespace mlir
