// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @a5_preload_workspace
// CHECK-DAG: %[[WS0:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf32>
// CHECK-DAG: %[[WS1:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf16>
// CHECK-DAG: %[[WS2:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf32>
// CHECK: scope.scope
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf32>
// CHECK:   hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
// CHECK: bufferization.to_tensor {{.*}} : memref<4x16x16xf32>
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf32> to tensor<16x16xf32>
// CHECK:   hivm.hir.vexp
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf16>
// CHECK:   hivm.hir.store
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf16> to tensor<16x16xf16>
// CHECK:   hivm.hir.mmadL1
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf32>
// CHECK:   hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf32> to tensor<16x16xf32>
// CHECK:   hivm.hir.vexp
// CHECK:   scope.return
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @a5_preload_workspace(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %k_mem = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %init = tensor.empty() : tensor<16x16xf32>
    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      %k0_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k0_alloc : memref<16x16xf16>)
      %k0 = bufferization.to_tensor %k0_alloc : memref<16x16xf16>
      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

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

      %load1_init = tensor.empty() : tensor<16x16xf16>
      %load1 = hivm.hir.load ins(%store1 : tensor<16x16xf16>) outs(%load1_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %k1_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k1_alloc : memref<16x16xf16>)
      %k1 = bufferization.to_tensor %k1_alloc : memref<16x16xf16>
      %dot1_init = tensor.empty() : tensor<16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%load1, %k1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws2 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws2 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws2_tensor = bufferization.to_tensor %ws2 restrict writable : memref<16x16xf32>
      %fix1 = hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf32>) outs(%ws2_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load2_init = tensor.empty() : tensor<16x16xf32>
      %load2 = hivm.hir.load ins(%fix1 : tensor<16x16xf32>) outs(%load2_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v1_init = tensor.empty() : tensor<16x16xf32>
      %v1 = hivm.hir.vexp ins(%load2 : tensor<16x16xf32>) outs(%v1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %v1 : tensor<16x16xf32>
    }
    "some_consume"(%result) : (tensor<16x16xf32>) -> ()
    return
  }

  // Minimal preload case for the TCB mixed-CV inserted-load path:
  //   CUBE mmadL1 -> fixpipe(nz2nd) -> VECTOR inserted-load.
  // The TCB tensor view must be rebuilt after the CUBE scope.
  // CHECK-LABEL: func.func @preload_tcb_fixpipe_inserted_load
  // CHECK: scf.for
  // CHECK: %[[TCB_ALLOC:.*]] = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
  // CHECK: scope.scope : () -> () {
  // CHECK: hivm.hir.mmadL1
  // CHECK: annotation.mark %[[TCB_ALLOC]] {{.*}}hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>{{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.fixpipe
  // CHECK: scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
  // CHECK: %[[TCB_CAST:.*]] = memref.memory_space_cast %[[TCB_ALLOC]] : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
  // CHECK-NEXT: %[[TCB_TENSOR:.*]] = bufferization.to_tensor %[[TCB_CAST]] restrict writable : memref<128x128xf32>
  // CHECK: scope.scope
  // CHECK: hivm.hir.load ins(%[[TCB_TENSOR]] : tensor<128x128xf32>)
  func.func @preload_tcb_fixpipe_inserted_load(%arg0: tensor<8x8x16x16xf16>, %arg1: tensor<8x8x16x16xf16>) attributes {func_dyn_memref_args = dense<[false, false]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %init = tensor.empty() : tensor<128x128xf32>
    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<128x128xf32> : i32 {
      %dot0_init = tensor.empty() : tensor<8x8x16x16xf32>
      %dot0 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32>
      %tcb_alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %tcb_alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x128xf32, #hivm.address_space<ub>>
      %tcb_tensor = bufferization.to_tensor %tcb_alloc restrict writable : memref<128x128xf32, #hivm.address_space<ub>>
      %fix0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<8x8x16x16xf32>) outs(%tcb_tensor : tensor<128x128xf32>) -> tensor<128x128xf32>

      %load0_init = tensor.empty() : tensor<128x128xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<128x128xf32>) outs(%load0_init : tensor<128x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<128x128xf32>
      %v0_init = tensor.empty() : tensor<128x128xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<128x128xf32>) outs(%v0_init : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %v0 : tensor<128x128xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    "some_consume"(%result) : (tensor<128x128xf32>) -> ()
    return
  }
}
