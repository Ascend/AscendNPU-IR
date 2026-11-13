// Registered jagged-DK kernel enables 3160 TCB-view rematerialize without
// --bypass-shape-registry.
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true set-depth-in-unroll-mode=2" -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @_jagged_dense_flash_attention_bwd_dk_kernel
// CHECK: %[[TCB_STORAGE:.*]] = memref.alloc() : memref<65536xi8, #hivm.address_space<ub>>
// CHECK: scope.scope : () -> () {
// CHECK: hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK-NOT: bufferization.to_tensor %[[TCB_STORAGE]]
// CHECK: scope.scope
// CHECK: %[[TCB_VIEW:.*]] = memref.view %[[TCB_STORAGE]]
// CHECK: %[[TCB_CAST:.*]] = memref.memory_space_cast %[[TCB_VIEW]]
// CHECK: %[[TCB_TENSOR:.*]] = bufferization.to_tensor %[[TCB_CAST]] restrict writable
// CHECK: hivm.hir.load ins(%[[TCB_TENSOR]] : tensor<?x128xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_jagged_dense_flash_attention_bwd_dk_kernel(
      %rows: index,
      %arg0: tensor<8x8x16x16xf16>,
      %arg1: tensor<8x8x16x16xf16>,
      %cond: i1) attributes {
        func_dyn_memref_args = dense<[false, false, false, false]> : vector<4xi1>,
        global_kernel = "local",
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %c0_index = arith.constant 0 : index
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %loop_init = tensor.empty(%rows) : tensor<?x128xf32>
    %result = scf.for %i = %c0 to %bound step %step
        iter_args(%acc = %loop_init) -> tensor<?x128xf32> : i32 {
      %dot_init = tensor.empty() : tensor<8x8x16x16xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32>
      %storage = memref.alloc() : memref<65536xi8, #hivm.address_space<ub>>
      annotation.mark %storage {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<65536xi8, #hivm.address_space<ub>>
      %view = memref.view %storage[%c0_index][%rows] : memref<65536xi8, #hivm.address_space<ub>> to memref<?x128xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %view : memref<?x128xf32, #hivm.address_space<ub>> to memref<?x128xf32>
      %tensor = bufferization.to_tensor %cast restrict writable : memref<?x128xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<8x8x16x16xf32>) outs(%tensor : tensor<?x128xf32>) -> tensor<?x128xf32>
      %nested = scf.if %cond -> tensor<?x128xf32> {
        %nested_init = tensor.empty(%rows) : tensor<?x128xf32>
        %nested_load = hivm.hir.load ins(%fix : tensor<?x128xf32>) outs(%nested_init : tensor<?x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<?x128xf32>
        scf.yield %nested_load : tensor<?x128xf32>
      } else {
        scf.yield %acc : tensor<?x128xf32>
      }
      %load_init = tensor.empty(%rows) : tensor<?x128xf32>
      %load = hivm.hir.load ins(%fix : tensor<?x128xf32>) outs(%load_init : tensor<?x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<?x128xf32>
      %vector_init = tensor.empty(%rows) : tensor<?x128xf32>
      %vector = hivm.hir.vadd ins(%load, %nested : tensor<?x128xf32>, tensor<?x128xf32>) outs(%vector_init : tensor<?x128xf32>) -> tensor<?x128xf32>
      scf.yield %vector : tensor<?x128xf32>
    }
    "some_consume"(%result) : (tensor<?x128xf32>) -> ()
    return
  }
}
