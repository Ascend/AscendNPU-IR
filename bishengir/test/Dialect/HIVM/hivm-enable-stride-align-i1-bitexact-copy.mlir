// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// A stride-align bridge copy between i1 buffers whose layout the DMA copy
// library cannot express must not become an hivm.hir.copy; it is outlined
// into a private hivm.vector_function doing an elementwise vector<1xi1>
// transfer copy instead. Byte-expressible layouts keep the library copy.

// CHECK: func.func private @i1_bitexact_copy_2d_{{.*}}({{.*}}: memref<13x19xi1, #hivm.address_space<ub>>, {{.*}}: memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline}
// CHECK: func.func private @i1_bitexact_copy_2d_{{.*}}({{.*}}: memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>, {{.*}}: memref<13x19xi1, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline}
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @i1_bridge_bitexact
func.func @i1_bridge_bitexact() {
  %0 = memref.alloc() : memref<13x19xi1, #hivm.address_space<ub>>
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<13x19xi1, #hivm.address_space<ub>>
  %c0 = arith.constant 0 : index
  %1 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [247], strides: [1] : memref<13x19xi1, #hivm.address_space<ub>> to memref<247xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %false = arith.constant false
  %v = vector.transfer_read %1[%c0], %false {in_bounds = [true]} : memref<247xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>
  vector.transfer_write %v, %1[%c0] {in_bounds = [true]} : vector<1xi1>, memref<247xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  return
}
// The forward bridge and the write-back bridge are both calls to the
// outlined fallback; no DMA copy is left behind.
// CHECK: memref.alloc() : memref<13x256x1xi1, #hivm.address_space<ub>>
// CHECK: %[[SV:.*]] = memref.subview %{{.*}}[0, 0, 0] [13, 19, 1] [1, 1, 1] : memref<13x256x1xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>
// CHECK: %[[DENSE:.*]] = memref.alloc() : memref<13x19xi1, #hivm.address_space<ub>>
// CHECK: call @i1_bitexact_copy_2d_{{.*}}(%[[SV]], %[[DENSE]]) {hivm.vector_function}
// CHECK-NOT: hivm.hir.copy
// CHECK: memref.reinterpret_cast %[[DENSE]]
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: call @i1_bitexact_copy_2d_{{.*}}(%[[DENSE]], %[[SV]]) {hivm.vector_function}
// CHECK-NOT: hivm.hir.copy

// -----

// Byte-expressible i1 layout (last dim 8, row stride 256): the DMA copy
// library handles it losslessly, so the bridges stay hivm.hir.copy and no
// fallback function is outlined.
// CHECK-NOT: i1_bitexact_copy
// CHECK-LABEL: func.func @byte_expressible_i1_bridge_keeps_copy
func.func @byte_expressible_i1_bridge_keeps_copy() {
  %0 = memref.alloc() : memref<13x8xi1, #hivm.address_space<ub>>
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<13x8xi1, #hivm.address_space<ub>>
  %c0 = arith.constant 0 : index
  %1 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [104], strides: [1] : memref<13x8xi1, #hivm.address_space<ub>> to memref<104xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %false = arith.constant false
  %v = vector.transfer_read %1[%c0], %false {in_bounds = [true]} : memref<104xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>
  vector.transfer_write %v, %1[%c0] {in_bounds = [true]} : vector<1xi1>, memref<104xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  return
}
// CHECK: %[[SV2:.*]] = memref.subview %{{.*}}[0, 0, 0] [13, 8, 1] [1, 1, 1] : memref<13x256x1xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
// CHECK: %[[DENSE2:.*]] = memref.alloc() : memref<13x8xi1, #hivm.address_space<ub>>
// CHECK: hivm.hir.copy ins(%[[SV2]] : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%[[DENSE2]] : memref<13x8xi1, #hivm.address_space<ub>>)
// CHECK: memref.reinterpret_cast %[[DENSE2]]
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: hivm.hir.copy ins(%[[DENSE2]] : memref<13x8xi1, #hivm.address_space<ub>>) outs(%[[SV2]] : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
// CHECK-NOT: i1_bitexact_copy
