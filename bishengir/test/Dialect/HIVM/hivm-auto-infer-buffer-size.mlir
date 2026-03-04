// RUN: bishengir-opt -hivm-auto-infer-buffer-size -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_auto_infer_buffer_size
func.func @test_auto_infer_buffer_size(%arg0: index) attributes {enable_auto_mark_buffer_size} {
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc(%arg0) {alignment = 64 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc(%arg0) {alignment = 64 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc(%arg0) {alignment = 64 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc(%arg0) {alignment = 64 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC_1]] {buffer_size_in_byte = 21824 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC_0]] {buffer_size_in_byte = 21824 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  annotation.mark %alloc_0 {buffer_size_in_byte = 21824 : i64} : memref<1x1x?xf32, #hivm.address_space<ub>>
  return
}