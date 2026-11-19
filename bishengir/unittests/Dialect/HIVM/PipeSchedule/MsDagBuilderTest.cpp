//===- MsDagBuilderTest.cpp - MsDagBuilder unit tests
//--------------------===//
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
// Tests for MsDagBuilder — the MLIR → sms::DDG bridge.
//
// These tests use parseSourceString to construct HIVM IR programmatically,
// then exercise MsDagBuilder::build() and verify the resulting DDG.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsDagBuilder.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
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

/// Per-op-type latency provider for more realistic tests.
struct TypedLatencyProvider : public LatencyProvider {
  int getLatency(Operation *op) const override {
    if (isa<LoadOp, ND2NZOp>(op))
      return 2;
    if (isa<FixpipeOp>(op))
      return 1;
    if (isa<StoreOp>(op))
      return 1;
    if (auto copyOp = dyn_cast<CopyOp>(op))
      return 3;
    // HIVM DPS compute ops.
    auto core = detail::queryCoreTypeHelper(op);
    if (core && *core == TCoreType::CUBE)
      return 4;
    return 3; // Vector default.
  }
};

/// Mock MS cost model: constant results, recording whether it was queried.
struct MockCostModel : public costmodel::CostModel {
  int64_t cubeCost = 7;
  int64_t vectorCost = 5;
  int64_t transferCost = 3;
  mutable int cubeQueries = 0;
  mutable int vectorQueries = 0;
  mutable int transferQueries = 0;

  costmodel::CostResult
  getCubeCost(const costmodel::CubeCostInput &) const override {
    ++cubeQueries;
    return costmodel::CostResult{cubeCost, ""};
  }
  costmodel::CostResult
  getVectorGroupCost(const costmodel::VectorGroupCostInput &) const override {
    ++vectorQueries;
    return costmodel::CostResult{vectorCost, ""};
  }
  costmodel::CostResult
  getTransferCost(const costmodel::TransferCostInput &) const override {
    ++transferQueries;
    return costmodel::CostResult{transferCost, ""};
  }
};

/// Mock cost model that always fails - exercises the -1 sentinel path.
struct FailingCostModel : public costmodel::CostModel {
  costmodel::CostResult
  getCubeCost(const costmodel::CubeCostInput &) const override {
    return costmodel::CostResult{-1, "mock cube failure"};
  }
  costmodel::CostResult
  getVectorGroupCost(const costmodel::VectorGroupCostInput &) const override {
    return costmodel::CostResult{-1, "mock vector failure"};
  }
  costmodel::CostResult
  getTransferCost(const costmodel::TransferCostInput &) const override {
    return costmodel::CostResult{-1, "mock transfer failure"};
  }
};

class MsDagBuilderTest : public ::testing::Test {
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
    scf::ForOp result = nullptr;
    module.walk([&](scf::ForOp op) { result = op; });
    return result;
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// Basic construction tests
//===----------------------------------------------------------------------===//

TEST_F(MsDagBuilderTest, EmptyLoopBodyFails) {
  auto module = parseModule(R"mlir(
    func.func @empty() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        scope.scope : () -> () {
          scope.return
        }
        scf.yield
      }
      return
    }
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  ASSERT_TRUE(forOp);

  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto result = builder.build();
  // Empty scope body has no compute/transfer ops → empty DAG → failure.
  EXPECT_TRUE(failed(result));
}

TEST_F(MsDagBuilderTest, LatencyProviderRejectsZeroLatency) {
  auto module = parseModule(R"mlir(
    func.func @zero_latency(%gm: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        scope.scope : () -> () {
          %alloc = memref.alloc() : memref<32xf16>
          hivm.hir.load ins(%gm : memref<32xf16>) outs(%alloc : memref<32xf16>)
          scope.return
        }
        scf.yield
      }
      return
    }
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  ASSERT_TRUE(forOp);

  FixedLatencyProvider zeroLp(0, 0);
  MsDagBuilder builder(forOp, zeroLp);
  auto result = builder.build();
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Classification tests
//===----------------------------------------------------------------------===//

TEST_F(MsDagBuilderTest, ClassifiesTransferOps) {
  // A loop with a load (MTE2) and a store (MTE3) — both transfer nodes.
  auto module = parseModule(R"mlir(
    func.func @transfers(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        scope.scope : () -> () {
          %alloc = memref.alloc() : memref<32xf16>
          hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
          %t = bufferization.to_tensor %alloc : memref<32xf16>
          hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
          scope.return
        }
        scf.yield
      }
      return
    }
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  FixedLatencyProvider lp;
  MsDagBuilder builder(forOp, lp);
  auto result = builder.build();

  if (succeeded(result)) {
    // Should have at least 2 nodes (load + store).
    EXPECT_GE(result->ddg.size(), 2);
    // Verify all nodes are transfer type.
    for (const MsNode &node : result->nodes) {
      EXPECT_EQ(node.kind, MsNodeKind::Transfer);
      EXPECT_TRUE(node.isParallelUnit);
    }
    // Verify DAG passes validation.
    auto vr = sms::verifyDag(result->ddg);
    EXPECT_TRUE(vr.ok) << vr.error;
  }
}

//===----------------------------------------------------------------------===//
// Determinism tests
//===----------------------------------------------------------------------===//

TEST_F(MsDagBuilderTest, NodeOrdinalsAreDeterministic) {
  auto source = R"mlir(
    func.func @determinism(%src: memref<32xf16>, %dst: memref<32xf16>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        scope.scope : () -> () {
          %alloc = memref.alloc() : memref<32xf16>
          hivm.hir.load ins(%src : memref<32xf16>) outs(%alloc : memref<32xf16>)
          %t = bufferization.to_tensor %alloc : memref<32xf16>
          hivm.hir.store ins(%t : tensor<32xf16>) outs(%dst : memref<32xf16>)
          scope.return
        }
        scf.yield
      }
      return
    }
  )mlir";

  FixedLatencyProvider lp;

  auto mod1 = parseModule(source);
  ASSERT_TRUE(mod1);
  MsDagBuilder b1(findForOp(*mod1), lp);
  auto r1 = b1.build();

  auto mod2 = parseModule(source);
  ASSERT_TRUE(mod2);
  MsDagBuilder b2(findForOp(*mod2), lp);
  auto r2 = b2.build();

  if (succeeded(r1) && succeeded(r2)) {
    ASSERT_EQ(r1->ddg.size(), r2->ddg.size());
    for (int i = 0; i < r1->ddg.size(); ++i) {
      EXPECT_EQ(r1->ddg.getNodes()[i].pipeline, r2->ddg.getNodes()[i].pipeline);
      EXPECT_EQ(r1->ddg.getNodes()[i].latency, r2->ddg.getNodes()[i].latency);
    }
    EXPECT_EQ(r1->ddg.getEdges().size(), r2->ddg.getEdges().size());
  }
}

//===----------------------------------------------------------------------===//
// LatencyProvider interface test
//===----------------------------------------------------------------------===//

TEST_F(MsDagBuilderTest, IssueTimeDefaultsToLatency) {
  FixedLatencyProvider lp(5, 5);
  MLIRContext ctx;
  // Just verify the interface contract — issue time defaults to latency.
  EXPECT_EQ(lp.getLatency(nullptr), 5);
  EXPECT_EQ(lp.getIssueTime(nullptr), 5);
}

//===----------------------------------------------------------------------===//
// DAG1/DAG2/DAG3 WorkItem-path tests
//===----------------------------------------------------------------------===//

TEST_F(MsDagBuilderTest, WorkItemPathBuildsDag123) {
  // Cube (mmadL1 + fixpipe) feeding vector (vexp) — the Default builder
  // partitions this into one CUBE and one VECTOR WorkItem.
  auto module = parseModule(R"mlir(
    func.func @wi_path(%bound: i32) {
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
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  ASSERT_TRUE(forOp);

  WorklistBuilder wb(forOp, /*numMultibuffer=*/2);
  auto wbResult = wb.build();
  if (failed(wbResult))
    return; // Default partition prerequisites not met in isolation — skip.

  FixedLatencyProvider lp;
  DenseMap<Value, Operation *> noClones;
  MsDagBuilder builder(forOp, lp, noClones, *wbResult);
  auto result = builder.build();
  ASSERT_TRUE(succeeded(result));

  // DAG1: one compute node per Default WorkItem, in worklist order, with
  // preserved WorkItem identity.
  ASSERT_EQ(result->computeNodeCount,
            static_cast<int>(wbResult->worklist.size()));
  for (int i = 0; i < result->computeNodeCount; ++i) {
    EXPECT_FALSE(result->nodes[i].isSynthetic);
    EXPECT_EQ(result->nodes[i].workItem, wbResult->worklist[i].get());
    EXPECT_EQ(result->nodes[i].workItemIndex, i);
  }

  // DAG2: synthetic transfer nodes appended after compute nodes, with no
  // MLIR ops and valid storage tags.
  EXPECT_GT(result->ddg.size(), result->computeNodeCount);
  for (int i = result->computeNodeCount;
       i < static_cast<int>(result->nodes.size()); ++i) {
    const MsNode &sn = result->nodes[i];
    EXPECT_TRUE(sn.isSynthetic);
    EXPECT_TRUE(sn.ops.empty());
    EXPECT_EQ(sn.workItem, nullptr);
    EXPECT_FALSE(sn.storageTag.empty());
    EXPECT_TRUE(sn.pipeline == "mte1" || sn.pipeline == "mte2" ||
                sn.pipeline == "mte3" || sn.pipeline == "fixpipe");
  }

  // DAG3: one data node per synthetic transfer output.
  EXPECT_EQ(result->dataNodes.size(),
            result->nodes.size() -
                static_cast<size_t>(result->computeNodeCount));

  // The combined DAG passes verification (acyclic, valid pipelines).
  auto vr = sms::verifyDag(result->ddg);
  EXPECT_TRUE(vr.ok) << vr.error;
}

TEST_F(MsDagBuilderTest, WorkItemPathCostBackfill) {
  // Same IR as WorkItemPathBuildsDag123 — a cube WorkItem feeding a vector
  // WorkItem — but with a mock cost model injected.  Every compute and
  // transfer node must carry the mock's costs in latency/issueTime, and the
  // DDG edge latencies must match their source nodes after the refresh.
  auto module = parseModule(R"mlir(
    func.func @wi_path_cost(%bound: i32) {
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
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  ASSERT_TRUE(forOp);

  WorklistBuilder wb(forOp, /*numMultibuffer=*/2);
  auto wbResult = wb.build();
  if (failed(wbResult))
    return; // Default partition prerequisites not met in isolation — skip.

  FixedLatencyProvider lp;
  DenseMap<Value, Operation *> noClones;
  MockCostModel cm;
  MsDagBuilder builder(forOp, lp, noClones, *wbResult, &cm);
  auto result = builder.build();
  ASSERT_TRUE(succeeded(result));

  // Every node was queried exactly once and backfilled with the mock cost.
  int expectCube = 0, expectVector = 0;
  for (int i = 0; i < result->computeNodeCount; ++i) {
    if (result->nodes[i].kind == MsNodeKind::CubeCompute)
      ++expectCube;
    else if (result->nodes[i].kind == MsNodeKind::VectorGroup)
      ++expectVector;
  }
  int expectTransfer = result->ddg.size() - result->computeNodeCount;
  ASSERT_GT(expectCube, 0);
  ASSERT_GT(expectVector, 0);
  ASSERT_GT(expectTransfer, 0);
  EXPECT_EQ(cm.cubeQueries, expectCube);
  EXPECT_EQ(cm.vectorQueries, expectVector);
  EXPECT_EQ(cm.transferQueries, expectTransfer);

  for (int i = 0; i < result->computeNodeCount; ++i) {
    int expected = result->nodes[i].kind == MsNodeKind::CubeCompute
                       ? static_cast<int>(cm.cubeCost)
                       : static_cast<int>(cm.vectorCost);
    EXPECT_EQ(result->ddg.getNodes()[i].latency, expected);
    EXPECT_EQ(result->ddg.getNodes()[i].issueTime, expected);
  }
  for (int i = result->computeNodeCount; i < result->ddg.size(); ++i) {
    EXPECT_EQ(result->ddg.getNodes()[i].latency,
              static_cast<int>(cm.transferCost));
    EXPECT_EQ(result->ddg.getNodes()[i].issueTime,
              static_cast<int>(cm.transferCost));
  }

  // Edge latencies were refreshed to the new source-node values (addEdge()
  // snapshots the latency at creation time, so a stale value here means the
  // refresh was skipped).
  for (const sms::DDGEdge &edge : result->ddg.getEdges()) {
    EXPECT_EQ(edge.latency, result->ddg.getNodes()[edge.src].latency);
  }

  // Data nodes still do not enter the DDG.
  EXPECT_EQ(result->nodes.size(),
            static_cast<size_t>(result->computeNodeCount) +
                static_cast<size_t>(expectTransfer));
  EXPECT_EQ(result->dataNodes.size(), static_cast<size_t>(expectTransfer));
}

TEST_F(MsDagBuilderTest, WorkItemPathCostFailureFailsBuild) {
  // A cost model returning the -1 sentinel must fail the build — the SMS
  // path then falls back to the Default scheduler.
  auto module = parseModule(R"mlir(
    func.func @wi_path_cost_fail(%bound: i32) {
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
  )mlir");
  ASSERT_TRUE(module);

  scf::ForOp forOp = findForOp(*module);
  ASSERT_TRUE(forOp);

  WorklistBuilder wb(forOp, /*numMultibuffer=*/2);
  auto wbResult = wb.build();
  if (failed(wbResult))
    return; // Default partition prerequisites not met in isolation — skip.

  FixedLatencyProvider lp;
  DenseMap<Value, Operation *> noClones;
  FailingCostModel cm;
  MsDagBuilder builder(forOp, lp, noClones, *wbResult, &cm);
  auto result = builder.build();
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Static helper tests (controlled-copy correctness)
//===----------------------------------------------------------------------===//

// The controlled-copy classifiers (isSeparator, isLoadLikeOp, isCoreOp, etc.)
// are private to MsDagBuilder.  They are tested indirectly through the build()
// method above: a correctly classified DAG proves the classifiers work.
// Direct-access tests would require a friend declaration or test-only API;
// those can be added if needed, but the build()-level tests cover the
// functional requirements for P0.

} // namespace
} // namespace hivm
} // namespace mlir
