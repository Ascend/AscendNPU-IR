// RUN: bishengir-opt %s -hivm-partition-and-bind-sub-block=enable-load-balanced=false -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @solve_tril_cv_pipelined

// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK:           %[[UB:.*]] = memref.alloc() : memref<2x16x16xf32, #hivm.address_space<ub>>
// CHECK:           annotation.mark %[[UB]] {hivm.cv_pipelined_multi_buffer}
// CHECK:           memref.alloc() : memref<2x2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:           hivm.hir.get_sub_block_idx
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           arith.cmpi eq, %{{.*}}, %[[C0]]
// CHECK:           scf.if
// CHECK:             scf.for %{{.*}} iter_args
// CHECK:               hivm.hir.vmul
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           arith.cmpi eq, %{{.*}}, %[[C1]]
// CHECK:           scf.if
// CHECK:             scf.for %{{.*}} iter_args
// CHECK:               hivm.hir.vmul
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

// CHECK:           scf.for
// CHECK:             hivm.hir.mmadL1
// CHECK:             memref.subview %[[UB]]
// CHECK:             hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : tensor<16x16xf32>) outs(
// CHECK-NOT:         sub_block_idx
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

// CHECK:           %[[W:.*]] = bufferization.to_tensor %{{.*}} : memref<2x16x16xf32>
// CHECK:           scf.for
// CHECK:             tensor.extract_slice %[[W]]
// CHECK:             hivm.hir.get_sub_block_idx
// CHECK:             %[[RC0:.*]] = arith.constant 0 : index
// CHECK:             arith.cmpi eq, %{{.*}}, %[[RC0]]
// CHECK:             scf.if
// CHECK:               hivm.hir.vmul
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @solve_tril_cv_pipelined(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xf32>,
    %cb: tensor<16x16xf32>, %vin0: tensor<16x16xf32>, %vin1: tensor<16x16xf32>, %sc: f32)
    attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>, parallel_mode = "simd"} {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64_i32 = arith.constant 64 : i32

  scf.for %arg8 = %c0_i32 to %c64_i32 step %c4_i32 : i32 {
    %ub = memref.alloc() : memref<2x16x16xf32, #hivm.address_space<ub>>
    annotation.mark %ub {hivm.cv_pipelined_multi_buffer} : memref<2x16x16xf32, #hivm.address_space<ub>>
    %cbBuf = memref.alloc() : memref<2x2x1x16x8xf32, #hivm.address_space<cbuf>>
    annotation.mark %cbBuf {hivm.cv_pipelined_multi_buffer} : memref<2x2x1x16x8xf32, #hivm.address_space<cbuf>>

    scf.for %arg9 = %c0 to %c2 step %c1 {
      %e0 = tensor.empty() : tensor<16x16xf32>
      %s0 = scope.scope : () -> tensor<16x16xf32> {
        %r0 = scf.for %arg10 = %c0 to %c2 step %c1 iter_args(%acc = %vin0) -> tensor<16x16xf32> {
          %v = hivm.hir.vmul ins(%acc, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %v : tensor<16x16xf32>
        }
        scope.return %r0 : tensor<16x16xf32>
      } {noinline, pipeline.veconly, sub_block = 0 : i32}
      %e1 = tensor.empty() : tensor<16x16xf32>
      %s1 = scope.scope : () -> tensor<16x16xf32> {
        %r1 = scf.for %arg10 = %c0 to %c2 step %c1 iter_args(%acc = %vin1) -> tensor<16x16xf32> {
          %v = hivm.hir.vmul ins(%acc, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %v : tensor<16x16xf32>
        }
        scope.return %r1 : tensor<16x16xf32>
      } {noinline, pipeline.veconly, sub_block = 1 : i32}
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

    %cbcast = memref.memory_space_cast %cbBuf : memref<2x2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x2x1x16x8xf32>
    %cbt = bufferization.to_tensor %cbcast restrict writable : memref<2x2x1x16x8xf32>
    scf.for %arg9 = %c0 to %c2 step %c1 {
      %aslice = tensor.extract_slice %cbt[%arg9, 0, 0, 0, 0] [1, 2, 1, 16, 8] [1, 1, 1, 1, 1] : tensor<2x2x1x16x8xf32> to tensor<2x1x16x8xf32>
      %ec = tensor.empty() : tensor<16x16xf32>
      %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true} ins(%aslice, %cb, %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%ec : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sv = memref.subview %ub[%arg9, 0, 0] [1, 16, 16] [1, 1, 1] : memref<2x16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mm : tensor<16x16xf32>) outs(%sv : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

    %wcast = memref.memory_space_cast %ub : memref<2x16x16xf32, #hivm.address_space<ub>> to memref<2x16x16xf32>
    %whole = bufferization.to_tensor %wcast restrict writable : memref<2x16x16xf32>
    scf.for %arg9 = %c0 to %c2 step %c1 {
      %sl = tensor.extract_slice %whole[%arg9, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<2x16x16xf32> to tensor<16x16xf32>
      %er = tensor.empty() : tensor<16x16xf32>
      %r = hivm.hir.vmul ins(%sl, %sc : tensor<16x16xf32>, f32) outs(%er : tensor<16x16xf32>) -> tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  }
  return
}
}
