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