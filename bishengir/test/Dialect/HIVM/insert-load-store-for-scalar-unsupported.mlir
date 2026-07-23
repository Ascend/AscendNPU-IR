// RUN: bishengir-opt -hivm-insert-load-store-for-scalar %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// XFAIL: *
// CHECK-LABEL: @extract_i1_direct_load_for_cube_init
// CHECK: bufferization.to_tensor
// CHECK: tensor.extract
// CHECK-SAME: tensor<16x16xi1>
// CHECK: hivm.hir.store
// CHECK-SAME: tensor<16x16xi8>
// CHECK-SAME: "hivm.inserted-store"
// CHECK: DuplicateTensorExtractForCube::replacementLabel
// CHECK: hivm.hir.mmadL1
func.func @extract_i1_direct_load_for_cube_init(
    %src: memref<16x16xi1>, %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %alloc = memref.alloc() : memref<16x16xi1>
  hivm.hir.load ins(%src : memref<16x16xi1>) outs(%alloc : memref<16x16xi1>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<16x16xi1>
  %cond = tensor.extract %tensor[%c0, %c0] : tensor<16x16xi1>
  %cube_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %cond, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}
