// REQUIRES: regbase
// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// A loop whose tensor iter_arg is produced on the CUBE core (fixpipe result)
// and consumed on the VECTOR core (vbrc/insert_slice/transpose/copy chain) the
// next iteration. CV pipelining would split the body into a VECTOR loop and a
// CUBE loop that each run to completion, so the VECTOR stage would read the
// loop-entry value instead of the previous iteration's CUBE result -- a silent
// miscompile. The pass must detect this cross-core loop-carried dependency and
// leave the loop un-pipelined.

// CHECK-LABEL: func.func @cross_core_loop_carry
// The loop must remain a single, un-split scf.for: no multibuffer expansion and
// no cv_unrolled_loop marker.
// CHECK-NOT: hivm.cv_pipelined_multi_buffer
// CHECK-NOT: cv_unrolled_loop
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK: scf.yield
// CHECK-NOT: cv_unrolled_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cross_core_loop_carry(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x16x8xf32>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<2x2xf32, strided<[2, 1]>>) outs(%alloc_1 : memref<1x1x16x8xf32>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x16x8xf32>
    %alloc_4 = memref.alloc() : memref<2x2xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_4 : memref<2x2xf32, #hivm.address_space<ub>> to memref<2x2xf32>
    %7 = bufferization.to_tensor %memspacecast restrict writable : memref<2x2xf32>
    // expected-warning@+1 {{loop-carried tensor iter_arg #0 is produced on the CUBE core but consumed on the VECTOR core across the iteration boundary}}
    %8 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %7) -> (tensor<2x2xf32>)  : i32 {
      %11 = tensor.empty() : tensor<16x8xf32>
      %12 = hivm.hir.vbrc ins(%cst : f32) outs(%11 : tensor<16x8xf32>) -> tensor<16x8xf32>
      // expected-note@+1 {{and consumed here on the VECTOR core the next iteration}}
      %inserted_slice = tensor.insert_slice %arg10 into %12[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<16x8xf32>
      %expanded = tensor.expand_shape %inserted_slice [[0], [1, 2]] output_shape [16, 1, 8] : tensor<16x8xf32> into tensor<16x1x8xf32>
      %13 = tensor.empty() : tensor<1x16x8xf32>
      %14 = hivm.hir.vtranspose ins(%expanded : tensor<16x1x8xf32>) outs(%13 : tensor<1x16x8xf32>) permutation = [1, 0, 2] -> tensor<1x16x8xf32>
      %expanded_6 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [1, 1, 16, 8] : tensor<1x16x8xf32> into tensor<1x1x16x8xf32>
      %alloc_7 = memref.alloc() : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
      %memspacecast_8 = memref.memory_space_cast %alloc_7 : memref<1x1x16x8xf32, #hivm.address_space<cbuf>> to memref<1x1x16x8xf32>
      %15 = bufferization.to_tensor %memspacecast_8 restrict writable : memref<1x1x16x8xf32>
      hivm.hir.copy ins(%expanded_6 : tensor<1x1x16x8xf32>) outs(%memspacecast_8 : memref<1x1x16x8xf32>) {"inserted-copy"}
      %16 = tensor.empty() : tensor<1x1x16x16xf32>
      %17 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%15, %3, %true, %c2, %c2, %c2 : tensor<1x1x16x8xf32>, tensor<1x1x16x8xf32>, i1, index, index, index) outs(%16 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %alloc_9 = memref.alloc() : memref<2x2xf32, #hivm.address_space<ub>>
      %memspacecast_10 = memref.memory_space_cast %alloc_9 : memref<2x2xf32, #hivm.address_space<ub>> to memref<2x2xf32>
      // expected-note@+1 {{loop-carried value produced here on the CUBE core}}
      %18 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<2x2xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%17 : tensor<1x1x16x16xf32>) outs(%alloc_9 : memref<2x2xf32, #hivm.address_space<ub>>)
      scf.yield %18 : tensor<2x2xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
    hivm.hir.store ins(%8 : tensor<2x2xf32>) outs(%reinterpret_cast_5 : memref<2x2xf32, strided<[2, 1]>>)
    return
  }
}
