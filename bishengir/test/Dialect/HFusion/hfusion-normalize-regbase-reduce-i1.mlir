// RUN: bishengir-opt --hfusion-normalize-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_reduce_i1_add_to_select_max_compare
// CHECK: hfusion.select
// CHECK: linalg.reduce
// CHECK: arith.maxsi
// CHECK: hfusion.compare
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_reduce_i1_add_to_select_max_compare(%arg0: tensor<8xi1> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}) -> tensor<i1> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant false

    %0 = bufferization.alloc_tensor() : tensor<i1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<i1>) -> tensor<i1>
    %reduced = linalg.reduce ins(%arg0 : tensor<8xi1>) outs(%1 : tensor<i1>) dimensions = [0]
        (%in: i1, %init: i1) {
          %2 = arith.addi %in, %init : i1
          linalg.yield %2 : i1
        }
    return %reduced : tensor<i1>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_reduce_i1_and_to_i16_andi(%arg0: tensor<1024xi1>) -> tensor<1xi8> 
  attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"}{
    %cst_0 = arith.constant true
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<i1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<i1>) -> tensor<i1>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xi1>) outs(%1 : tensor<i1>) dimensions = [0]
        (%in: i1, %init: i1) {
          %2 = arith.andi %in, %init : i1
          linalg.yield %2 : i1
        }
    %extracted = tensor.extract %reduced[] : tensor<i1>
    %10 = arith.extui %extracted : i1 to i8
    %11 = tensor.empty() : tensor<1xi8>
    %inserted = tensor.insert %10 into %11[%c0] : tensor<1xi8>
    return %inserted : tensor<1xi8>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_reduce_i1_and_to_i16_ori(%arg0: tensor<1024xi1>) -> tensor<1xi8> 
  attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"}{
    %cst_0 = arith.constant true
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<i1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<i1>) -> tensor<i1>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xi1>) outs(%1 : tensor<i1>) dimensions = [0]
        (%in: i1, %init: i1) {
          %2 = arith.ori %in, %init : i1
          linalg.yield %2 : i1
        }
    %extracted = tensor.extract %reduced[] : tensor<i1>
    %10 = arith.extui %extracted : i1 to i8
    %11 = tensor.empty() : tensor<1xi8>
    %inserted = tensor.insert %10 into %11[%c0] : tensor<1xi8>
    return %inserted : tensor<1xi8>
  }
}

// -----

// Regression test for partial i1->i16 reduce conversion: the converted init
// must preserve the rank of the original reduce's init tensor. Previously,
// `HFusionReduceI1AndOrToI16Traits::createReduceInit` hardcoded the init to a
// 0-rank `tensor<i16>` regardless of the source shape, producing a malformed
// `linalg.reduce` (rank-2 input, rank-0 init/output, `dimensions=[1]`) that
// tripped the assert in `DimensionAnalyzerBase::processDecreasingDimensions`
// (ProcessOperations.cpp:369) during subsequent FlattenOpsPass.

// CHECK-LABEL: func.func @test_reduce_i1_partial_or_to_i16_ori
// CHECK: linalg.fill ins({{.*}} : i16) outs({{.*}} : tensor<2xi16>)
// CHECK: linalg.reduce ins({{.*}} : tensor<2x1024xi16>) outs({{.*}} : tensor<2xi16>) dimensions = [1]
// CHECK: arith.minsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_reduce_i1_partial_or_to_i16_ori(%arg0: tensor<2x1024xi1>) -> tensor<2xi1>
  attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant false
    %0 = tensor.empty() : tensor<2xi1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<2xi1>) -> tensor<2xi1>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x1024xi1>) outs(%1 : tensor<2xi1>) dimensions = [1]
        (%in: i1, %init: i1) {
          %2 = arith.ori %in, %init : i1
          linalg.yield %2 : i1
        }
    return %reduced : tensor<2xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_reduce_i1_partial_and_to_i16_andi
// CHECK: linalg.fill ins({{.*}} : i16) outs({{.*}} : tensor<2xi16>)
// CHECK: linalg.reduce ins({{.*}} : tensor<2x1024xi16>) outs({{.*}} : tensor<2xi16>) dimensions = [1]
// CHECK: arith.maxsi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_reduce_i1_partial_and_to_i16_andi(%arg0: tensor<2x1024xi1>) -> tensor<2xi1>
  attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant true
    %0 = tensor.empty() : tensor<2xi1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<2xi1>) -> tensor<2xi1>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x1024xi1>) outs(%1 : tensor<2xi1>) dimensions = [1]
        (%in: i1, %init: i1) {
          %2 = arith.andi %in, %init : i1
          linalg.yield %2 : i1
        }
    return %reduced : tensor<2xi1>
  }
}