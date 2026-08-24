// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Test 1: Cross-core loop carry (CUBE producer -> VECTOR consumer).
// In skew mode, the pass pipelines the loop into VECTOR and CUBE scopes,
// assigning preload_num = 1 to the independent VECTOR stage and preload_num = 0
// to the CUBE stage that produces the carried iter_arg.

// CHECK-LABEL: func.func @cross_core_loop_carry_skew
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (tensor<2x2xf32>)
// CHECK:   %[[VEC_RES:.*]] = scope.scope : () -> tensor<1x1x16x8xf32> {
// CHECK:     tensor.insert_slice %[[ITER_ARG]]
// CHECK:     hivm.hir.vtranspose
// CHECK:     hivm.hir.copy
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK:   %[[CUBE_RES:.*]] = scope.scope : () -> tensor<2x2xf32> {
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
// CHECK:   scf.yield %[[CUBE_RES]] : tensor<2x2xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cross_core_loop_carry_skew(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
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
    %8 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %7) -> (tensor<2x2xf32>)  : i32 {
      %11 = tensor.empty() : tensor<16x8xf32>
      %12 = hivm.hir.vbrc ins(%cst : f32) outs(%11 : tensor<16x8xf32>) -> tensor<16x8xf32>
      %inserted_slice = tensor.insert_slice %arg10 into %12[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<16x8xf32>
      %expanded = tensor.expand_shape %inserted_slice [[0], [1, 2]] output_shape [16, 1, 8] : tensor<16x8xf32> into tensor<16x1x8xf32>
      %13 = tensor.empty() : tensor<1x16x8xf32>
      %14 = hivm.hir.vtranspose ins(%expanded : tensor<16x1x8xf32>) outs(%13 : tensor<1x16x8xf32>) permutation = [1, 0, 2] -> tensor<1x16x8xf32>
      %expanded_6 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [1, 1, 16, 8] : tensor<1x16x8xf32> into tensor<1x1x16x8xf32>
      %alloc_7 = memref.alloc() : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
      %memspacecast_8 = memref.memory_space_cast %alloc_7 : memref<1x1x16x8xf32, #hivm.address_space<cbuf>> to memref<1x1x16x8xf32>
      %15 = bufferization.to_tensor %memspacecast_8 restrict writable : memref<1x1x16x8xf32>
      hivm.hir.copy ins(%expanded_6 : tensor<1x1x16x8xf32>) outs(%memspacecast_8 : memref<1x1x16x8xf32>) {"hivm.inserted-copy"}
      %16 = tensor.empty() : tensor<1x16x16xf32>
      %17 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%15, %3, %true, %c2, %c2, %c2 : tensor<1x1x16x8xf32>, tensor<1x1x16x8xf32>, i1, index, index, index) outs(%16 : tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
      %alloc_9 = memref.alloc() : memref<2x2xf32, #hivm.address_space<ub>>
      %memspacecast_10 = memref.memory_space_cast %alloc_9 : memref<2x2xf32, #hivm.address_space<ub>> to memref<2x2xf32>
      %18 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<2x2xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%17 : tensor<1x16x16xf32>) outs(%alloc_9 : memref<2x2xf32, #hivm.address_space<ub>>)
      scf.yield %18 : tensor<2x2xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
    hivm.hir.store ins(%8 : tensor<2x2xf32>) outs(%reinterpret_cast_5 : memref<2x2xf32, strided<[2, 1]>>)
    return
  }
}

// -----

// Test 2: Multi-stage pipeline with loop carry (CUBE independent -> VECTOR LCD consumer/producer).
// CUBE mmad + fixpipe computes intermediate tensor (independent stage, preload_num = 1).
// VECTOR vadd accumulates the carried tensor with the CUBE output (LCD stage, preload_num = 0).

// CHECK-LABEL: func.func @multi_stage_loop_carry_skew
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ACC:.*]] = %{{.*}}) -> (tensor<16x16xf32>)
// CHECK:   scope.scope : () -> () {
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK:   %[[VEC_OUT:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.vadd
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
// CHECK:   scf.yield %[[VEC_OUT]] : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @multi_stage_loop_carry_skew(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<true> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %0 = "some_op"() : () -> memref<16x16xf16>
    %1 = bufferization.to_tensor %0 : memref<16x16xf16>
    %2 = "some_op"() : () -> memref<16x16xf16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %3 = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %4 = tensor.empty() : tensor<16x16xf32>
    %result = scf.for %arg1 = %c0_i32 to %3 step %c1_i32 iter_args(%arg2 = %4) -> (tensor<16x16xf32>)  : i32 {
      %k0_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%2 : memref<16x16xf16>) outs(%k0_alloc : memref<16x16xf16>)
      %k0 = bufferization.to_tensor %k0_alloc : memref<16x16xf16>
      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%1, %k0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load0_init = tensor.empty() : tensor<16x16xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<16x16xf32>) outs(%load0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v_out_init = tensor.empty() : tensor<16x16xf32>
      %v_out = hivm.hir.vadd ins(%load0, %arg2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %v_out : tensor<16x16xf32>
    }
    "some_consume"(%result) : (tensor<16x16xf32>) -> ()
    return
  }
}
