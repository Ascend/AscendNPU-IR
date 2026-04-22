// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" %s | FileCheck %s
// bishengir-opt test_sort.mlir --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" --convert-linalg-to-loops > test_sort_lowered.mlir
module {
  // 1D tensor, ascending sort (descending = false, sort_axis = 0)
  func.func @sort_kernel_1d_asc(%arg0: memref<6xf32>) -> tensor<6xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<6xf32>
    %1 = hfusion.sort ins(%0 : tensor<6xf32>) descending = false sort_axis = 0 -> tensor<6xf32>
    return %1 : tensor<6xf32>
  }

  // 2D tensor, ascending sort (descending = false, sort_axis = 1)
  func.func @sort_kernel_2d_asc(%arg0: memref<4x4xf32>) -> tensor<4x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x4xf32>
    %1 = hfusion.sort ins(%0 : tensor<4x4xf32>) descending = false sort_axis = 1 -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // 3D tensor, descending sort (descending = true, sort_axis = 2)
  func.func @sort_kernel_3d_desc(%arg0: memref<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x3x4xf32>
    %1 = hfusion.sort ins(%0 : tensor<2x3x4xf32>) descending = true sort_axis = 2 -> tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }
}

// ===== 1D ascending =====

// CHECK-LABEL: func.func @sort_kernel_1d_asc
// CHECK-NOT: hfusion.sort
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.cmpf ugt
// CHECK: tensor.insert
// CHECK: tensor.insert
// CHECK: return

// ===== 2D ascending, sort_axis = 1 =====

// CHECK-LABEL: func.func @sort_kernel_2d_asc
// CHECK-NOT: hfusion.sort
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: arith.cmpf ugt
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: return

// ===== 3D descending, sort_axis = 2 =====

// CHECK-LABEL: func.func @sort_kernel_3d_desc
// CHECK-NOT: hfusion.sort
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: arith.cmpf ult
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: return