// RUN: bishengir-opt -hivm-insert-load-store-for-scalar %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// CHECK-LABEL: @extract_i1_for_cube_init
// CHECK: tensor.extract
// CHECK-SAME: tensor<16x16xi1>
// CHECK: hivm.hir.store
// CHECK-SAME: tensor<1xi8>
// CHECK-SAME: "inserted-store"
// CHECK: DuplicateTensorExtractForCube::replacementLabel
// CHECK: hivm.hir.mmadL1
func.func @extract_i1_for_cube_init(
    %arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>,
    %lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>,
    %out: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %cmp_empty = tensor.empty() : tensor<16x16xi1>
  %cmp = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%cmp_empty : tensor<16x16xi1>) compare_mode = <lt> -> tensor<16x16xi1>
  %cond = tensor.extract %cmp[%c0, %c0] : tensor<16x16xi1>
  %cube_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %cond, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  return
}
// -----

// CHECK-LABEL: @extract_for_index_use
// CHECK: %[[EXTRACT:.*]] = tensor.extract
// CHECK-SAME: tensor<16xi64>
// CHECK: %[[IDX:.*]] = arith.index_cast %[[EXTRACT]]
// CHECK-SAME: i64 to index
// CHECK-NOT: inserted-store
// CHECK: memref.load %{{.*}}[%[[IDX]]]
// CHECK-SAME: memref<16xf32>
func.func @extract_for_index_use(
    %indices: tensor<16xi64>, %data: memref<16xf32>,
    %lhs: tensor<16x16xf32>, %rhs: tensor<16x16xf32>,
    %out: memref<16x16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %cube_empty = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%cube_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
  %raw_index = tensor.extract %indices[%c0] : tensor<16xi64>
  %idx = arith.index_cast %raw_index : i64 to index
  %value = memref.load %data[%idx] : memref<16xf32>
  return %value : f32
}
