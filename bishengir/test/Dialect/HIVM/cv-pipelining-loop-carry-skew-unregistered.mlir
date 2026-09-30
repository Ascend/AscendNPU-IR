// Off-registry intra-WI LCD (VECTOR acc) must not take the LCD backup /
// preload path. The same IR pipelines under --bypass-shape-registry.
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true set-depth-in-unroll-mode=2" -allow-unregistered-dialect -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true set-depth-in-unroll-mode=2 bypass-shape-registry=true" -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=BYPASS

// CHECK-LABEL: func.func @unregistered_intra_wi_acc_lcd
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.preload_num
// CHECK-NOT: hivm.has_loop_carried_dep
// CHECK-NOT: hivm.cv_pipelined_loop
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vadd
// CHECK: scf.yield
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.preload_num
//
// BYPASS-LABEL: func.func @unregistered_intra_wi_acc_lcd
// BYPASS: scf.for
// BYPASS:   scope.scope
// BYPASS:     hivm.hir.mmadL1
// BYPASS:     hivm.hir.fixpipe
// BYPASS:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// BYPASS:   scope.scope
// BYPASS:     hivm.hir.vadd
// BYPASS:   } {hivm.has_loop_carried_dep, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unregistered_intra_wi_acc_lcd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<true> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
