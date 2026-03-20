// RUN: bishengir-opt %s -canonicalize -allow-unregistered-dialect | FileCheck %s

func.func @fold_buffer_size_mark_for_static_alloc(%d : index) {
  %alloc = memref.alloc(%d) : memref<?xf32>
  // CHECK: annotation.mark
  annotation.mark %alloc {buffer_size_in_byte = 1 : i64} : memref<?xf32>

  %alloc_0 = memref.alloc() : memref<10xf32>
  // CHECK-NOT: annotation.mark
  annotation.mark %alloc_0 {buffer_size_in_byte = 1 : i64} : memref<10xf32>

  "some_use"(%alloc) : (memref<?xf32>) -> ()
  "some_other_use"(%alloc_0) : (memref<10xf32>) -> ()
  return
}

// -----

func.func @fold_buffer_size_mark_to_source_alloc(%offset : index, %size : index, %stride : index) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %alloc = memref.alloc(%size) : memref<?xf32>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview
  %subview = memref.subview %alloc[%offset] [%size] [%stride] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
  // annotation.mark %[[ALLOC]] {buffer_size_in_byte = 1 : i64} : memref<?xf32>
  annotation.mark %subview {buffer_size_in_byte = 1 : i64} : memref<?xf32, strided<[?], offset: ?>>
  "some_use"(%subview) : (memref<?xf32, strided<[?], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL:   @fold_redundant_mark(
func.func @fold_redundant_mark(%arg1: tensor<256x64xf32>) ->tensor<256x64xf32> {
  %1 = tensor.empty() : tensor<256x64xf32>
  %2 = hivm.hir.vabs ins(%arg1 : tensor<256x64xf32>) outs(%1 : tensor<256x64xf32>) -> tensor<256x64xf32>
  // CHECK-NOT: annotation.mark
  annotation.mark %2 : tensor<256x64xf32>
  return %2 : tensor<256x64xf32>
}

// -----

// CHECK-LABEL:   @mark_with_mem_write_read(
func.func @mark_with_mem_write_read(%arg0: tensor<16x16xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark
  annotation.mark %alloc {effects = ["write", "read"]} : memref<16x16xf32, #hivm.address_space<ub>>
  "some_operation" (%alloc) : (memref<16x16xf32, #hivm.address_space<ub>>) -> ()
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  annotation.mark %alloc_0 : memref<16x16xf32, #hivm.address_space<ub>>
  "some_operation" (%alloc_0) : (memref<16x16xf32, #hivm.address_space<ub>>) -> ()
  return
}
