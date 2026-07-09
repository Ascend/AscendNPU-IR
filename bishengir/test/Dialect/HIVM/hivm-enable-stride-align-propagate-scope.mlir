// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// Test that EnableStrideAlign propagates aligned memref type through scope.scope
// when the annotated alloc is returned via scope.return. The scope result type
// is updated to the aligned strided type and downstream users receive it correctly.

// CHECK-LABEL: func @test_stride_align_through_scope_scope
func.func @test_stride_align_through_scope_scope(%arg0: memref<2x15xf16, #hivm.address_space<gm>>) {
  // CHECK: scope.scope : () -> memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>
  %result = scope.scope : () -> memref<2x15xf16, #hivm.address_space<ub>> {
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x15x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x15xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    // CHECK: scope.return %[[SUBVIEW]] : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>
    scope.return %alloc : memref<2x15xf16, #hivm.address_space<ub>>
  }
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%result : memref<2x15xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<2x15xf16, #hivm.address_space<gm>>)
  return
}

// -----

// Test that EnableStrideAlign propagates through scope.scope with multiple results,
// where each result carries an independently stride-aligned memref.

// CHECK-LABEL: func @test_scope_scope_multi_result_stride_align
func.func @test_scope_scope_multi_result_stride_align(%arg0: memref<2x15xf16, #hivm.address_space<gm>>, %arg1: memref<2x15xf16, #hivm.address_space<gm>>) {
  // CHECK: %{{.*}}:2 = scope.scope : () -> (memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>, memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>)
  %result:2 = scope.scope : () -> (memref<2x15xf16, #hivm.address_space<ub>>, memref<2x15xf16, #hivm.address_space<ub>>) {
    // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x15x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC0]]
    %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x15xf16, #hivm.address_space<ub>>
    // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x15x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW1:.*]] = memref.subview %[[ALLOC1]]
    %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x15xf16, #hivm.address_space<ub>>
    annotation.mark %alloc0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    annotation.mark %alloc1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.vexp ins(%[[SUBVIEW0]] : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>) outs(%[[SUBVIEW1]] : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>)
    hivm.hir.vexp ins(%alloc0 : memref<2x15xf16, #hivm.address_space<ub>>) outs(%alloc1 : memref<2x15xf16, #hivm.address_space<ub>>)
    // CHECK: scope.return %[[SUBVIEW0]], %[[SUBVIEW1]]
    scope.return %alloc0, %alloc1 : memref<2x15xf16, #hivm.address_space<ub>>, memref<2x15xf16, #hivm.address_space<ub>>
  }
  // CHECK: hivm.hir.store ins(%{{.*}}#0 : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%result#0 : memref<2x15xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<2x15xf16, #hivm.address_space<gm>>)
  // CHECK: hivm.hir.store ins(%{{.*}}#1 : memref<2x15xf16, strided<[240, 16]>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%result#1 : memref<2x15xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<2x15xf16, #hivm.address_space<gm>>)
  return
}
