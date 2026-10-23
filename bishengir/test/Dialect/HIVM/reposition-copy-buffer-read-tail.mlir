// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s
// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s
// RUN: sed 's/to_tensor %buffer/to_tensor %external/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/to_tensor %buffer/to_tensor %external/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/ins(%%state, %r/ins(%other, %r/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/ins(%%state, %r/ins(%other, %r/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/ins(%%state, %r/ins(%old_view, %r/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/ins(%%state, %r/ins(%old_view, %r/' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: bishengir-opt %s -hivm-reposition-preload-work | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/@private_result(/@parallel_path_fwd_kernel_result(/' %s > %t
// RUN: bishengir-opt %t -hivm-reposition-preload-work="enable-preload=false set-workspace-multibuffer=2" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: bishengir-opt %t -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix
// RUN: sed 's/hivm.preload_num = 0/hivm.preload_num = 1/g' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=KEEP --implicit-check-not=hivm.independent_preload_prefix

// Variants retain the scope when the tail reads external storage or old %other
// (directly or through %old_view). Returning %packed writes back to %other.
// KEEP-LABEL: func.func @
// KEEP: %{{.*}}:2 = scope.scope
// KEEP: hivm.hir.copy
// KEEP: hivm.hir.vsub
// KEEP: scope.return %{{.*}}, %{{.*}}
// KEEP: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
// Split at the copy, forward the other state directly, and keep the tail destination allocation local.
// CHECK-LABEL: func.func @private_result(
// CHECK: %[[PREFIX:.*]] = scope.scope
// CHECK: %[[PACKED:.*]] = hivm.hir.vadd
// CHECK-NOT: tensor.empty
// CHECK: hivm.hir.copy
// CHECK-NEXT: scope.return %[[PACKED]]
// CHECK: } {{.*}}hivm.independent_preload_prefix
// CHECK: %[[TAIL:.*]] = scope.scope
// CHECK: tensor.empty
// CHECK: %[[SUB:.*]] = hivm.hir.vsub
// CHECK-NEXT: scope.return %[[SUB]] : tensor<16x16xf32>
// CHECK: scf.yield %[[TAIL]], %[[PREFIX]]
func.func @private_result(%init: tensor<16x16xf32>, %p: tensor<16x16xf32>, %cube: tensor<16x16xf32>, %external: memref<16x16xf32, #hivm.address_space<ub>>, %n: index) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out:2 = scf.for %i = %c0 to %n step %c1 iter_args(%state = %init, %other = %init) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
    %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    %l1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      hivm.hir.fixpipe ins(%cube : tensor<16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 1 : i32, hivm.preload_num = 0 : i32, no_inline}
    %r = bufferization.to_tensor %buffer restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
    %next:2 = scope.scope : () -> (tensor<16x16xf32>, tensor<16x16xf32>) {
      %empty = tensor.empty() : tensor<16x16xf32>
      %packed = hivm.hir.vadd ins(%p, %other : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %old_view = tensor.cast %other : tensor<16x16xf32> to tensor<16x16xf32>
      %tail_empty = tensor.empty() : tensor<16x16xf32>
      hivm.hir.copy ins(%packed : tensor<16x16xf32>) outs(%l1 : memref<16x16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %sub = hivm.hir.vsub ins(%state, %r : tensor<16x16xf32>, tensor<16x16xf32>) outs(%tail_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %sub, %packed : tensor<16x16xf32>, tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 1 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next#0, %next#1 : tensor<16x16xf32>, tensor<16x16xf32>
  }
  return %out#0, %out#1 : tensor<16x16xf32>, tensor<16x16xf32>
}
}
