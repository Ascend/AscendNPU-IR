// RUN: bishengir-opt --hivm-insert-load-store-for-scalar -split-input-file %s | FileCheck %s

// -----
// CHECK-LABEL: @test_extract
func.func @test_extract(%arg0: memref<8xi32>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>, %arg3: i32, %arg4: i32, %arg5: i32) -> tensor<16x16xf32>{
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = arith.muli %arg3, %arg4 : i32
  %1 = arith.muli %0, %arg5 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = bufferization.to_tensor %arg0 restrict writable : memref<8xi32>
  %3 = bufferization.alloc_tensor() : tensor<i32>
  %expanded = tensor.expand_shape %3 [] output_shape [1] : tensor<i32> into tensor<1xi32>
  %4 = hivm.hir.vreduce <sum> ins(%2 : tensor<8xi32>) outs(%expanded : tensor<1xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1xi32>
  %collapsed = tensor.collapse_shape %4 [] : tensor<1xi32> into tensor<i32>
  %extracted = tensor.extract %collapsed[] : tensor<i32>
  // CHECK: tensor.extract %{{.*}}[] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<i32>
  // CHECK: memref_ext.alloc_workspace() : memref<i32>
  // CHECK: bufferization.to_tensor %{{.*}} restrict writable : memref<i32>
  // CHECK: hivm.hir.store ins(%{{.*}} : tensor<i32>) outs(%{{.*}} : tensor<i32>) -> tensor<i32>
  // CHECK: annotation.mark %{{.*}} {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<i32>
  // CHECK: tensor.extract %{{.*}}[] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<i32>
  // CHECK: annotation.mark %{{.*}} {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%{{.*}} : i32] : i32
  %5 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %6 = bufferization.to_tensor %arg2 restrict writable : memref<16x16xf16>
  %7 = tensor.empty() : tensor<16x16xf32>
  %8 = scf.for %arg6 = %c0_i32 to %extracted step %c8_i32 iter_args(%arg7 = %7) -> (tensor<16x16xf32>)  : i32 {
     %9 = hivm.hir.mmadL1 ins(%5, %6, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg7 : tensor<16x16xf32>) -> tensor<16x16xf32>
     scf.yield %9 : tensor<16x16xf32>
  }
  return %8 : tensor<16x16xf32>
}