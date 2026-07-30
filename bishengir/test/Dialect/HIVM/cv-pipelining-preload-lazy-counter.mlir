// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-lazy-loading=true" -allow-unregistered-dialect %s | FileCheck %s

// Lazy loading clones the GM load into both CUBE work items, dragging the i32
// address counter chain along. Each cloned reader must advance its own private
// iter_arg, otherwise the stage-1 scope addresses GM with the stage-3 counter.

// CHECK-LABEL: func.func @lazy_load_shared_counter
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// The clone gets a third iter_arg, initialized like the counter it shadows.
// CHECK:         scf.for %{{.*}} = %[[C0]] to %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[CNT:.*]] = %[[C0]], %[[PRIV:.*]] = %[[C0]]) -> (tensor<16x16xf32>, i32, i32){{.*}}{

// Stage 3 keeps the original counter iter_arg.
// CHECK:           %[[ADV3:.*]] = scope.scope : () -> i32 {
// CHECK:             arith.index_cast %[[CNT]] : i32 to index
// CHECK:             hivm.hir.mmadL1
// CHECK:             %[[NEXT3:.*]] = arith.addi %[[CNT]], %{{.*}} : i32
// CHECK:             scope.return %[[NEXT3]] : i32
// CHECK:           } {{.*}}hivm.preload_num = 3 : i32

// CHECK:           scope.scope
// CHECK:             hivm.hir.store
// CHECK:           } {{.*}}hivm.preload_num = 2 : i32

// Stage 1 is the lazy-load clone: it reads and advances only its own counter.
// CHECK:           %[[ADV1:.*]] = scope.scope : () -> i32 {
// CHECK:             arith.index_cast %[[PRIV]] : i32 to index
// CHECK-NOT:         %[[CNT]]
// CHECK:             %[[NEXT1:.*]] = arith.addi %[[PRIV]], %{{.*}} : i32
// CHECK:             scope.return %[[NEXT1]] : i32
// CHECK:           } {{.*}}hivm.preload_num = 1 : i32

// CHECK:           %[[VEC:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:           } {{.*}}hivm.preload_num = 0 : i32

// Both counters reach the loop's yield: the owner's slot and the private one.
// CHECK:           scf.yield %[[VEC]], %[[ADV3]], %[[ADV1]] : tensor<16x16xf32>, i32, i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @lazy_load_shared_counter(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %gm: memref<?xi8>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true, true]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %stride = arith.constant 512 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %init = tensor.empty() : tensor<16x16xf32>
    %result:2 = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init, %cnt = %c0) -> (tensor<16x16xf32>, i32) : i32 {
      // GM address of the shared load, derived from the loop-carried counter.
      %off = arith.index_cast %cnt : i32 to index
      %k_view = memref.view %gm[%off][] : memref<?xi8> to memref<16x16xf16>
      %k_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_view : memref<16x16xf16>) outs(%k_alloc : memref<16x16xf16>)
      %k = bufferization.to_tensor %k_alloc : memref<16x16xf16>

      // Stage 3 (CUBE): first consumer of the shared load.
      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      // Stage 2 (VECTOR).
      %load0_init = tensor.empty() : tensor<16x16xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<16x16xf32>) outs(%load0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_init = tensor.empty() : tensor<16x16xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<16x16xf32>) outs(%v0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_f16_init = tensor.empty() : tensor<16x16xf16>
      %v0_f16 = hivm.hir.vcast ins(%v0 : tensor<16x16xf32>) outs(%v0_f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf16>
      annotation.mark %ws1 {hivm.multi_buffer = 4 : i32} : memref<16x16xf16>
      %ws1_tensor = bufferization.to_tensor %ws1 restrict writable : memref<16x16xf16>
      %store1 = hivm.hir.store ins(%v0_f16 : tensor<16x16xf16>) outs(%ws1_tensor : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Stage 1 (CUBE): second consumer of the same shared load.
      %load1_init = tensor.empty() : tensor<16x16xf16>
      %load1 = hivm.hir.load ins(%store1 : tensor<16x16xf16>) outs(%load1_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot1_init = tensor.empty() : tensor<16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%load1, %k, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws2 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws2 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws2_tensor = bufferization.to_tensor %ws2 restrict writable : memref<16x16xf32>
      %fix1 = hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf32>) outs(%ws2_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      // Stage 0 (VECTOR).
      %load2_init = tensor.empty() : tensor<16x16xf32>
      %load2 = hivm.hir.load ins(%fix1 : tensor<16x16xf32>) outs(%load2_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v1_init = tensor.empty() : tensor<16x16xf32>
      %v1 = hivm.hir.vexp ins(%load2 : tensor<16x16xf32>) outs(%v1_init : tensor<16x16xf32>) -> tensor<16x16xf32>

      %cnt_next = arith.addi %cnt, %stride : i32
      scf.yield %v1, %cnt_next : tensor<16x16xf32>, i32
    }
    "some_consume"(%result#0) : (tensor<16x16xf32>) -> ()
    return
  }
}
