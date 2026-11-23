// RUN: bishengir-opt %s -hivm-merge-same-preload-scopes="bypass-shape-registry=true" | FileCheck %s

// A marked prefix between two equal-stage scopes must not be pulled into the
// merged span. Matching scopes within each side can still be combined.
// CHECK-LABEL: func.func @prefix_is_merge_boundary
// CHECK: scope.scope
// CHECK: hivm.hir.load ins(%arg0
// CHECK: hivm.hir.load ins(%arg1
// CHECK: } {{.*}}hivm.preload_num = 0
// CHECK: %[[VIEW:.*]] = bufferization.to_tensor
// CHECK: scope.scope
// CHECK: hivm.hir.load ins(%arg2
// CHECK: hivm.hir.vadd ins(%[[VIEW]]
// CHECK: } {{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK: hivm.hir.load ins(%arg3
// CHECK: hivm.hir.load ins(%arg4
// CHECK: } {{.*}}hivm.preload_num = 0
// CHECK-NOT: scope.scope
func.func @prefix_is_merge_boundary(%a: memref<16xf32>, %b: memref<16xf32>, %c: memref<16xf32>, %d: memref<16xf32>, %e: memref<16xf32>, %dst: memref<16xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () {
      hivm.hir.load ins(%a : memref<16xf32>) outs(%dst : memref<16xf32>)
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.load ins(%b : memref<16xf32>) outs(%dst : memref<16xf32>)
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
    %view = bufferization.to_tensor %dst restrict writable : memref<16xf32>
    scope.scope : () -> () {
      hivm.hir.load ins(%c : memref<16xf32>) outs(%dst : memref<16xf32>)
      %empty = tensor.empty() : tensor<16xf32>
      %sum = hivm.hir.vadd ins(%view, %view : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
      scope.return
    } {hivm.independent_preload_prefix, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.load ins(%d : memref<16xf32>) outs(%dst : memref<16xf32>)
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.load ins(%e : memref<16xf32>) outs(%dst : memref<16xf32>)
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
