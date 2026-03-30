// RUN: bishengir-opt %s --execution-engine-convert-hfusion-to-upstream --convert-linalg-to-loops | FileCheck %s

module {
  // ==========================================
  // 1D tensor histogram
  // ==========================================
  // CHECK-LABEL: func.func @histogram_kernel_1d
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK: %[[IN:.*]] = bufferization.to_tensor
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[EMPTY]] : tensor<4xi32>)
  // CHECK: %[[RES:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[FILL]]) -> (tensor<4xi32>) {
  // CHECK:   %[[EXTRACT:.*]] = tensor.extract %[[IN]][%{{.*}}] : tensor<16xi32>
  // CHECK:   %[[IDX:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
  // CHECK:   scf.if %{{.*}} -> (tensor<4xi32>) {
  // CHECK:     %[[OLD_VAL:.*]] = tensor.extract %[[ITER]][%[[IDX]]] : tensor<4xi32>
  // CHECK:     %[[NEW_VAL:.*]] = arith.addi %[[OLD_VAL]], %[[ONE]] : i32
  // CHECK:     %[[INSERTED:.*]] = tensor.insert %[[NEW_VAL]] into %[[ITER]][%[[IDX]]] : tensor<4xi32>
  // CHECK:     scf.yield %[[INSERTED]] : tensor<4xi32>
  // CHECK:   } else {
  // CHECK:     scf.yield %[[ITER]] : tensor<4xi32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return %[[RES]] : tensor<4xi32>
  func.func @histogram_kernel_1d(%arg0: memref<16xi32>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
    %1 = hfusion.histogram %0, 4 : tensor<16xi32> -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // ==========================================
  // 2D tensor histogram 
  // ==========================================
  // CHECK-LABEL: func.func @histogram_kernel_2d
  // CHECK: %[[IN:.*]] = bufferization.to_tensor
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1{{\]\]}} : tensor<2x16xi32> into tensor<32xi32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<4xi32>) {
  // CHECK:   tensor.extract %[[COLLAPSED]][%{{.*}}] : tensor<32xi32>
  func.func @histogram_kernel_2d(%arg0: memref<2x16xi32>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x16xi32>
    %1 = hfusion.histogram %0, 4 : tensor<2x16xi32> -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // ==========================================
  // 3D tensor histogram 
  // ==========================================
  // CHECK-LABEL: func.func @histogram_kernel_3d
  // CHECK: %[[IN:.*]] = bufferization.to_tensor
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1, 2{{\]\]}} : tensor<2x3x16xi32> into tensor<96xi32>
  // CHECK: scf.for
  func.func @histogram_kernel_3d(%arg0: memref<2x3x16xi32>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x3x16xi32>
    %1 = hfusion.histogram %0, 4 : tensor<2x3x16xi32> -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // ==========================================
  // Edge Case 1: Minimal input size (Size-1 Tensor)
  // ==========================================
  // CHECK-LABEL: func.func @histogram_boundary_size1
  // CHECK: tensor.empty() : tensor<4xi32>
  // CHECK: scf.for
  func.func @histogram_boundary_size1(%arg0: memref<1xi32>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<1xi32>
    %1 = hfusion.histogram %0, 4 : tensor<1xi32> -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // ==========================================
  // Edge Case 2: Extreme bin count (num_bins = 1)
  // ==========================================
  // CHECK-LABEL: func.func @histogram_boundary_1_bin
  // CHECK: tensor.empty() : tensor<1xi32>
  // CHECK: scf.for
  func.func @histogram_boundary_1_bin(%arg0: memref<16xi32>) -> tensor<1xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
    %1 = hfusion.histogram %0, 1 : tensor<16xi32> -> tensor<1xi32>
    return %1 : tensor<1xi32>
  }

  // ==========================================
  // Test 3: Dynamic shape handling (Dynamic dimension resolution)
  // ==========================================
  // CHECK-LABEL: func.func @histogram_negative_dynamic
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IN:.*]] = bufferization.to_tensor
  // CHECK: %[[DIM:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?xi32>
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[DIM]] step %{{.*}}
  func.func @histogram_negative_dynamic(%arg0: memref<?xi32>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<?xi32>
    %1 = hfusion.histogram %0, 4 : tensor<?xi32> -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // ==========================================
  // Test 4: 1D tensor histogram WITH Mask
  // ==========================================
  // CHECK-LABEL: func.func @histogram_mask_1d
  // CHECK: %[[IN:.*]] = bufferization.to_tensor %arg0
  // CHECK: %[[MASK:.*]] = bufferization.to_tensor %arg1
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %{{.*}}) -> (tensor<4xi32>) {
  // CHECK:   %[[ELEM:.*]] = tensor.extract %[[IN]][%{{.*}}] : tensor<16xi32>
  // CHECK:   %[[IDX:.*]] = arith.index_cast %[[ELEM]] : i32 to index
  // CHECK:   %[[GE_ZERO:.*]] = arith.cmpi sge, %[[IDX]], %{{.*}}
  // CHECK:   %[[LT_BINS:.*]] = arith.cmpi slt, %[[IDX]], %{{.*}}
  // CHECK:   %[[IN_BOUNDS:.*]] = arith.andi %[[GE_ZERO]], %[[LT_BINS]] : i1
  // CHECK:   %[[MASK_VAL:.*]] = tensor.extract %[[MASK]][%{{.*}}] : tensor<16xi1>
  // CHECK:   %[[FINAL_VALID:.*]] = arith.andi %[[IN_BOUNDS]], %[[MASK_VAL]] : i1
  // CHECK:   scf.if %[[FINAL_VALID]] -> (tensor<4xi32>) {
  // CHECK:     %[[OLD_VAL:.*]] = tensor.extract %[[ITER]][%[[IDX]]] : tensor<4xi32>
  // CHECK:     %[[NEW_VAL:.*]] = arith.addi %[[OLD_VAL]], %{{.*}} : i32
  // CHECK:     %[[INSERTED:.*]] = tensor.insert %[[NEW_VAL]] into %[[ITER]][%[[IDX]]] : tensor<4xi32>
  // CHECK:     scf.yield %[[INSERTED]] : tensor<4xi32>
  // CHECK:   } else {
  // CHECK:     scf.yield %[[ITER]] : tensor<4xi32>
  // CHECK:   }
  // CHECK: }
  func.func @histogram_mask_1d(%arg0: memref<16xi32>, %arg1: memref<16xi1>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<16xi1>
    // Note: The operator syntax now includes the optional %1 (Mask)
    %2 = hfusion.histogram %0, 4, %1 : tensor<16xi32>, tensor<16xi1> -> tensor<4xi32>
    return %2 : tensor<4xi32>
  }

  // ==========================================
  // Test 5: 2D tensor histogram WITH Mask (Verify both are collapsed)
  // ==========================================
  // CHECK-LABEL: func.func @histogram_mask_2d
  // CHECK: %[[IN:.*]] = bufferization.to_tensor %arg0
  // CHECK: %[[MASK:.*]] = bufferization.to_tensor %arg1
  // CHECK-DAG: %[[FLAT_IN:.*]] = tensor.collapse_shape %[[IN]] {{\[\[}}0, 1{{\]\]}} : tensor<2x16xi32> into tensor<32xi32>
  // CHECK-DAG: %[[FLAT_MASK:.*]] = tensor.collapse_shape %[[MASK]] {{\[\[}}0, 1{{\]\]}} : tensor<2x16xi1> into tensor<32xi1>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK:   %[[MASK_VAL:.*]] = tensor.extract %[[FLAT_MASK]][%{{.*}}] : tensor<32xi1>
  // CHECK:   arith.andi %{{.*}}, %[[MASK_VAL]] : i1
  func.func @histogram_mask_2d(%arg0: memref<2x16xi32>, %arg1: memref<2x16xi1>) -> tensor<4xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x16xi32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x16xi1>
    %2 = hfusion.histogram %0, 4, %1 : tensor<2x16xi32>, tensor<2x16xi1> -> tensor<4xi32>
    return %2 : tensor<4xi32>
  }
}