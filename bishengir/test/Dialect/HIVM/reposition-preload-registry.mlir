// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s
// RUN: sed 's/kernel_sdpa_bwd_q/parallel_path_fwd_kernel/g' %s | bishengir-opt -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s
// RUN: sed 's/kernel_sdpa_bwd_q/chunk_gated_delta_product_fwd_kernel_h_blockdim64/g' %s | bishengir-opt -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s
// RUN: sed 's/kernel_sdpa_bwd_q/off/g' %s | bishengir-opt -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s --check-prefix=OFF --implicit-check-not=hivm.independent_preload_prefix
// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=false set-workspace-multibuffer=2" | FileCheck %s --check-prefix=OFF --implicit-check-not=hivm.independent_preload_prefix
// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefix=OFF --implicit-check-not=hivm.independent_preload_prefix

// Check the shared pass gate with registered prefixes and a decorated name.
// The independent multiply hoists; the cube-dependent add stays with its use.
// CHECK-LABEL: func.func @{{.*}}_registry
// CHECK: scope.scope
// CHECK: hivm.hir.fixpipe
// CHECK: %[[CUBE:.*]] = bufferization.to_tensor
// CHECK: scope.scope
// CHECK: hivm.hir.vmul
// CHECK: scope.return
// CHECK: } {{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 2
// CHECK: scope.scope
// CHECK: hivm.hir.vadd ins(%[[CUBE]]
// CHECK: hivm.hir.vsub
// CHECK-NOT: hivm.hir.vmul
// CHECK: } {{.*}}hivm.preload_num = 2
// OFF-LABEL: func.func @{{.*}}_registry
// OFF: hivm.hir.vadd
// OFF: hivm.hir.vmul
// OFF: hivm.hir.vsub
module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
func.func @kernel_sdpa_bwd_q_registry(%p: tensor<16x16xf32>, %q: tensor<16x16xf32>, %buf: memref<16x16xf32, #hivm.address_space<ub>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () {
      hivm.hir.fixpipe ins(%p : tensor<16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %buf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<16x16xf32>
      %add = hivm.hir.vadd ins(%cube, %p : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %mul = hivm.hir.vmul ins(%q, %q : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sub = hivm.hir.vsub ins(%add, %mul : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.fixpipe ins(%p : tensor<16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
  }
  return
}
}
