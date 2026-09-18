// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true" -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Registered LIT shape without bypass: vcast-of-load is delayed and bundled
// with the consumer VECTOR work item.
// CHECK-LABEL: func.func @cross_core_loop_carry_skew
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (tensor<16x16xf32>)
// CHECK:   scope.scope : () -> () {
// CHECK:     hivm.hir.load
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK:   %[[VEC_RES:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.load
// CHECK:     hivm.hir.vcast
// CHECK:     hivm.hir.vadd
// CHECK:     scope.return
// CHECK:   } {hivm.has_loop_carried_dep, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
// CHECK:   scf.yield %[[VEC_RES]] : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cross_core_loop_carry_skew(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<true> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %0 = "some_op"() : () -> memref<16x16xbf16>
    %1 = bufferization.to_tensor %0 : memref<16x16xbf16>
    %2 = "some_op"() : () -> memref<16x16xbf16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %3 = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %4 = tensor.empty() : tensor<16x16xf32>
    %5 = scf.for %arg1 = %c0_i32 to %3 step %c1_i32 iter_args(%arg2 = %4) -> (tensor<16x16xf32>)  : i32 {
      %alloc = memref.alloc() : memref<16x16xbf16>
      hivm.hir.load ins(%2 : memref<16x16xbf16>) outs(%alloc : memref<16x16xbf16>)
      %6 = bufferization.to_tensor %alloc : memref<16x16xbf16>
      %7 = tensor.empty() : tensor<16x16xf32>
      %8 = hivm.hir.mmadL1 ins(%1, %6, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %9 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<16x16xf32>
      %10 = bufferization.to_tensor %9 restrict writable : memref<16x16xf32>
      %11 = hivm.hir.fixpipe ins(%8 : tensor<16x16xf32>) outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load_alloc = memref.alloc() : memref<16x16xbf16>
      hivm.hir.load ins(%2 : memref<16x16xbf16>) outs(%load_alloc : memref<16x16xbf16>)
      %load_tensor = bufferization.to_tensor %load_alloc : memref<16x16xbf16>
      %vcast_init = tensor.empty() : tensor<16x16xf32>
      %vcast = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%load_tensor : tensor<16x16xbf16>) outs(%vcast_init : tensor<16x16xf32>) -> tensor<16x16xf32>

      %v_out_init = tensor.empty() : tensor<16x16xf32>
      %v_out = hivm.hir.vadd ins(%vcast, %arg2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %v_out : tensor<16x16xf32>
    }
    "some_consume"(%5) : (tensor<16x16xf32>) -> ()
    return
  }
}
