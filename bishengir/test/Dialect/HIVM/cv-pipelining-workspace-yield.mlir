// RUN: bishengir-opt -cv-pipelining="pipeline-mode=unroll" -allow-unregistered-dialect -verify-diagnostics -split-input-file %s | FileCheck %s

// Verify cv-pipelining reverts when a yielded workspace output has a cross-workitem dependency via iter_arg.

func.func @test_workspace_yield_cross_workitem_revert(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %7 = tensor.empty() : tensor<64x32xf32>
  %12 = "some_op"() : () -> i32

  // expected-warning @+1 {{cannot pipeline loop: loop-carried tensor iter_arg #0 is produced by one work item but consumed by another work item across the iteration boundary; skipping pipelining}}
  %24 = scf.for %arg13 = %c0_i32 to %12 step %c1_i32 iter_args(%arg14 = %7) -> (tensor<64x32xf32>) : i32 {
    // expected-note @+1 {{and consumed here by another work item in the next iteration}}
    %41 = hivm.hir.load ins(%arg14 : tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) {"inserted-load"} core_type = <CUBE> -> tensor<64x32xf32>
    %alloc = memref.alloc() : memref<64x64xf32>
    %55 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %56 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%55, %41, %true, %c64, %c64, %c32 : tensor<64x64xf32>, tensor<64x32xf32>, i1, index, index, index) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %57 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %57 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %58 = bufferization.to_tensor %57 restrict writable : memref<64x32xf32>
    %59 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%56 : tensor<64x32xf32>) outs(%58 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %60 = hivm.hir.load ins(%59 : tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<64x32xf32>
    %alloc_5 = memref.alloc() : memref<64x32xf32>
    %79 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x32xf32>
    %80 = hivm.hir.vsub ins(%79, %60 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %81 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %81 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %82 = bufferization.to_tensor %81 restrict writable : memref<64x32xf32>
    %83 = hivm.hir.store ins(%80 : tensor<64x32xf32>) outs(%82 : tensor<64x32xf32>) {"inserted-store"} -> tensor<64x32xf32>
    %84 = hivm.hir.load ins(%83 : tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) {"inserted-load"} core_type = <CUBE> -> tensor<64x32xf32>
    %alloc_9 = memref.alloc() : memref<64x64xf32>
    %86 = bufferization.to_tensor %alloc_9 restrict writable : memref<64x64xf32>
    %88 = hivm.hir.mmadL1 {a_transpose, fixpipe_for_result_already_inserted = true} ins(%86, %84, %true, %c64, %c64, %c32 : tensor<64x64xf32>, tensor<64x32xf32>, i1, index, index, index) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %89 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %89 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %90 = bufferization.to_tensor %89 restrict writable : memref<64x32xf32>
    // expected-note @+1 {{loop-carried value produced here}}
    %91 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%88 : tensor<64x32xf32>) outs(%90 : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %91 : tensor<64x32xf32>
  }
  return
}

// -----

// Verify cv-pipelining extracts from the last slot when there is no cross-workitem iter_arg dependency.

// CHECK-LABEL: func.func @test_workspace_yield_same_workitem
// CHECK: scf.for
// CHECK: hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: bufferization.to_tensor %{{.*}} : memref<4x64x32xf32>
// CHECK: %[[LAST_SLOT:.*]] = arith.constant 3 : index
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %{{.*}}[%[[LAST_SLOT]], 0, 0] [1, 64, 32] [1, 1, 1] : tensor<4x64x32xf32> to tensor<64x32xf32>
// CHECK: scf.yield %[[EXTRACTED]] : tensor<64x32xf32>

func.func @test_workspace_yield_same_workitem(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %7 = tensor.empty() : tensor<64x32xf32>
  %12 = "some_op"() : () -> i32

  %24 = scf.for %arg13 = %c0_i32 to %12 step %c1_i32 iter_args(%arg14 = %7) -> (tensor<64x32xf32>) : i32 {
    %alloc = memref.alloc() : memref<64x64xf32>
    %55 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %56 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%55, %7, %true, %c64, %c64, %c32 : tensor<64x64xf32>, tensor<64x32xf32>, i1, index, index, index) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %57 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %57 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %58 = bufferization.to_tensor %57 restrict writable : memref<64x32xf32>
    %59 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%56 : tensor<64x32xf32>) outs(%58 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %60 = hivm.hir.load ins(%59 : tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<64x32xf32>
    %alloc_5 = memref.alloc() : memref<64x32xf32>
    %79 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x32xf32>
    %80 = hivm.hir.vsub ins(%79, %60 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %81 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %81 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %82 = bufferization.to_tensor %81 restrict writable : memref<64x32xf32>
    %83 = hivm.hir.store ins(%80 : tensor<64x32xf32>) outs(%82 : tensor<64x32xf32>) {"inserted-store"} -> tensor<64x32xf32>
    %84 = hivm.hir.load ins(%83 : tensor<64x32xf32>) outs(%7 : tensor<64x32xf32>) {"inserted-load"} core_type = <CUBE> -> tensor<64x32xf32>
    %alloc_9 = memref.alloc() : memref<64x64xf32>
    %86 = bufferization.to_tensor %alloc_9 restrict writable : memref<64x64xf32>
    %88 = hivm.hir.mmadL1 {a_transpose, fixpipe_for_result_already_inserted = true} ins(%86, %84, %true, %c64, %c64, %c32 : tensor<64x64xf32>, tensor<64x32xf32>, i1, index, index, index) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %89 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<64x32xf32>
    annotation.mark %89 {hivm.multi_buffer = 4 : i32} : memref<64x32xf32>
    %90 = bufferization.to_tensor %89 restrict writable : memref<64x32xf32>
    %91 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%88 : tensor<64x32xf32>) outs(%90 : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %91 : tensor<64x32xf32>
  }
  return
}
