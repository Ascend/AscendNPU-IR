// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

func.func @test_sync_block(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c127_i32 = arith.constant 127 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %c-2139095040_i32 = arith.constant -2139095040 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = tensor.empty() : tensor<64x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %7 = arith.addi %arg9, %c127_i32 : i32
  %8 = arith.divsi %7, %c128_i32 : i32
  %9 = arith.index_cast %arg9 : i32 to index
  %10 = tensor.empty() : tensor<64x64xi32>
  %11 = linalg.fill ins(%c2147483647_i32 : i32) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %12 = hivm.hir.bitcast %6 : tensor<64x64xf32> -> tensor<64x64xi32>
  %mapped = linalg.map { arith.andi } ins(%12, %11 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%10 : tensor<64x64xi32>)
  %13 = linalg.fill ins(%c-2139095040_i32 : i32) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %14 = linalg.add ins(%mapped, %13 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %15 = linalg.fill ins(%c1_i32 : i32) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %16 = linalg.min ins(%14, %15 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%14 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %17 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %18 = linalg.max ins(%16, %17 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%16 : tensor<64x64xi32>) -> tensor<64x64xi32>
  %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%18 : tensor<64x64xi32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %20 = tensor.empty() : tensor<64x64xi1>
  %21 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%19, %cst : tensor<64x64xf32>, f32) outs(%20 : tensor<64x64xi1>) -> tensor<64x64xi1>
  %22 = linalg.select ins(%21, %cst_0, %cst : tensor<64x64xi1>, f32, f32) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %23 = tensor.empty() : tensor<64xf32>
  scf.for %arg13 = %c0 to %c2 step %c1 {
    %24 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg13]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    annotation.mark %1 {logical_block_num} : i32
    %25 = arith.divsi %3, %4 : i32
    %26 = arith.remsi %25, %arg10 : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
    %27 = arith.muli %26, %c64_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%28], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<64xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>>) outs(%alloc : memref<64xf32>) eviction_policy = <EvictFirst>
    %29 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
    %30 = arith.muli %26, %arg9 : i32
    %31 = arith.index_cast %30 : i32 to index
    scf.for %arg14 = %c0_i32 to %8 step %c1_i32  : i32 {
      %32 = arith.muli %arg14, %c128_i32 : i32
      %33 = arith.index_cast %32 : i32 to index
      %34 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%33]
      %35 = arith.maxsi %33, %9 : index
      %36 = arith.minsi %34, %35 : index
      %37 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %33]
      %38 = arith.minsi %24, %37 : index
      %39 = affine.apply affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>()[%38, %36, %33]
      %40 = arith.minsi %39, %c64 : index
      %alloc_1 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_1 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %41 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
      %42 = hivm.hir.bitcast %41 : tensor<64x64xf32> -> tensor<64x64xi32>
      // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
      %mapped_2 = linalg.map { arith.andi } ins(%42, %11 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%10 : tensor<64x64xi32>)
      // CHECK: %[[FUSED_PRODUCER:.*]] = linalg.generic
      // CHECK-SAME: iterator_types = ["parallel", "parallel"]
      // CHECK:      ^bb0(%{{.*}}: i32, %{{.*}}: f32, %{{.*}}: f32):
      // CHECK:        arith.addi
      // CHECK:        arith.minsi
      // CHECK:        arith.maxsi
      // CHECK:        arith.sitofp
      // CHECK:        arith.cmpf une
      // CHECK:        arith.select
      // CHECK:        linalg.yield
      %43 = linalg.add ins(%mapped_2, %13 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
      %44 = linalg.min ins(%43, %15 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%43 : tensor<64x64xi32>) -> tensor<64x64xi32>
      %45 = linalg.max ins(%44, %17 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%44 : tensor<64x64xi32>) -> tensor<64x64xi32>
      %46 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%45 : tensor<64x64xi32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %47 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%46, %cst : tensor<64x64xf32>, f32) outs(%20 : tensor<64x64xi1>) -> tensor<64x64xi1>
      %48 = linalg.select ins(%47, %cst_0, %41 : tensor<64x64xi1>, f32, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_S>] flag = 0
      %49 = linalg.max ins(%48, %22 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %broadcasted = linalg.broadcast ins(%29 : tensor<64xf32>) outs(%5 : tensor<64x64xf32>) dimensions = [1]
      %50 = linalg.mul ins(%49, %broadcasted : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %51 = linalg.fill ins(%cst : f32) outs(%23 : tensor<64xf32>) -> tensor<64xf32>
      %reduced = linalg.reduce ins(%50 : tensor<64x64xf32>) outs(%51 : tensor<64xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %53 = arith.addf %in, %init : f32
          linalg.yield %53 : f32
        }
      %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%31, %33]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      %subview = memref.subview %reinterpret_cast_3[%24] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %reduced[0] [%40] [1] : tensor<64xf32> to tensor<?xf32>
      %subview_4 = memref.subview %subview[0] [%40] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>>) {tiled_op}
    }
  } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return
}