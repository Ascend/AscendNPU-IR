// RUN: bishengir-opt %s -hivm-tensor-copy-insertion="analysis-heuristic=top-down allow-return-allocs-from-loops bufferize-function-boundaries" -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s

// When `bufferization.alloc_tensor (copy...)` is rewritten into an
// `alloc_tensor` + `hivm.hir.copy` pair, the new dst carries over the copy
// source's annotation marks. For a dynamically shaped source without marks, a
// buffer_size_in_byte upper bound is derived from its statically-shaped
// ancestor (256 * 16bit = 512 bytes).
// CHECK-LABEL: func.func @test_copy_mark_derived_from_static_ancestor
func.func @test_copy_mark_derived_from_static_ancestor(%arg0: tensor<256xf16>, %arg1: index) -> tensor<?xf16> {
  %0 = tensor.extract_slice %arg0[0] [%arg1] [1] : tensor<256xf16> to tensor<?xf16>
  %1 = bufferization.alloc_tensor() copy(%0) : tensor<?xf16>
  // CHECK: %[[ALLOC_DERIVE:.*]] = memref.alloc({{.*}}) {{.*}} : memref<?xf16>
  // CHECK-NEXT: annotation.mark %[[ALLOC_DERIVE]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHECK-NEXT: hivm.hir.copy ins({{.*}} : memref<?xf16, strided<[1]>>) outs(%[[ALLOC_DERIVE]] : memref<?xf16>)
  return %1 : tensor<?xf16>
}

// -----

// Non-size marks are cloned as well; since the cloned mark carries no
// buffer_size_in_byte, the upper bound is still derived separately.
// CHECK-LABEL: func.func @test_copy_mark_cloned_non_size_mark
func.func @test_copy_mark_cloned_non_size_mark(%arg0: tensor<256xf16>, %arg1: index) -> tensor<?xf16> {
  %0 = tensor.extract_slice %arg0[0] [%arg1] [1] : tensor<256xf16> to tensor<?xf16>
  annotation.mark %0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : tensor<?xf16>
  %1 = bufferization.alloc_tensor() copy(%0) : tensor<?xf16>
  // CHECK: %[[ALLOC_NONSIZE:.*]] = memref.alloc({{.*}}) {{.*}} : memref<?xf16>
  // CHECK-NEXT: annotation.mark %[[ALLOC_NONSIZE]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<?xf16>
  // CHECK-NEXT: annotation.mark %[[ALLOC_NONSIZE]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHECK-NEXT: hivm.hir.copy ins({{.*}} : memref<?xf16, strided<[1]>>) outs(%[[ALLOC_NONSIZE]] : memref<?xf16>)
  return %1 : tensor<?xf16>
}

// -----

// A mark combining buffer_size_in_byte with other attributes is cloned
// wholesale; no extra derived mark is added.
// CHECK-LABEL: func.func @test_copy_mark_cloned_combined_mark
func.func @test_copy_mark_cloned_combined_mark(%arg0: tensor<256xf16>, %arg1: index) -> tensor<?xf16> {
  %0 = tensor.extract_slice %arg0[0] [%arg1] [1] : tensor<256xf16> to tensor<?xf16>
  annotation.mark %0 {buffer_size_in_byte = 384 : i64, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : tensor<?xf16>
  %1 = bufferization.alloc_tensor() copy(%0) : tensor<?xf16>
  // CHECK: %[[ALLOC_COMBINED:.*]] = memref.alloc({{.*}}) {{.*}} : memref<?xf16>
  // CHECK-NEXT: annotation.mark %[[ALLOC_COMBINED]] {buffer_size_in_byte = 384 : i64, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<?xf16>
  // CHECK-NEXT: hivm.hir.copy ins({{.*}} : memref<?xf16, strided<[1]>>) outs(%[[ALLOC_COMBINED]] : memref<?xf16>)
  return %1 : tensor<?xf16>
}
