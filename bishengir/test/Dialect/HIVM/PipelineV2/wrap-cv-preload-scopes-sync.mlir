// RUN: bishengir-opt %s -wrap-cv-preload-scopes -hivm-cross-core-gss | FileCheck %s

// Same-stage producer/consumer dependencies retain their cross-core set/wait
// synchronization inside the outer preload scope.
// CHECK-LABEL: func.func @same_stage_sync
// CHECK: scf.for
// CHECK: scope.scope : () -> () {
// CHECK: scope.scope : () -> () {
// CHECK: hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[FLAG:[0-9]+]]
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[FLAG]]
// CHECK-NEXT: scope.scope : () -> () {
// CHECK: hivm.hir.vadd
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @same_stage_sync(%mat: tensor<1x1x16x16xf32>, %other: tensor<16x16xf32>, %dst: memref<16x16xf32, #hivm.address_space<gm>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %buf : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %tensor = bufferization.to_tensor %cast restrict writable : memref<16x16xf32>
    scope.scope : () -> () {
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mat : tensor<1x1x16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%tensor, %other : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.store ins(%sum : tensor<16x16xf32>) outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}
}
