// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" %s | FileCheck %s

module {
  // 1D tensor, forward cumsum.
  func.func @cumsum_kernel_1d_i32(%arg0: memref<6xi32>) -> tensor<6xi32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<6xi32>
    %1 = hfusion.cumsum %0 : tensor<6xi32> cum_dims = [0] reverse = false -> tensor<6xi32>
    return %1 : tensor<6xi32>
  }

  // 2D tensor, cumsum on the last axis.
  func.func @cumsum_kernel_2d_f32(%arg0: memref<4x4xf32>) -> tensor<4x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x4xf32>
    %1 = hfusion.cumsum %0 : tensor<4x4xf32> cum_dims = [1] reverse = false -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // 3D tensor, reverse cumsum on a middle axis.
  func.func @cumsum_kernel_3d_reverse(%arg0: memref<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x3x4xf32>
    %1 = hfusion.cumsum %0 : tensor<2x3x4xf32> cum_dims = [1] reverse = true -> tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }
}

// ===== 1D i32 forward =====

// CHECK-LABEL: func.func @cumsum_kernel_1d_i32
// CHECK-NOT: hfusion.cumsum
// CHECK: tensor.empty
// CHECK: scf.for
// CHECK: tensor.extract %{{.+}}[%{{.+}}] : tensor<6xi32>
// CHECK: arith.addi
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}] : tensor<6xi32>
// CHECK: return

// ===== 2D f32 forward =====

// CHECK-LABEL: func.func @cumsum_kernel_2d_f32
// CHECK-NOT: hfusion.cumsum
// CHECK: tensor.empty
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: arith.addf
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}] : tensor<4x4xf32>
// CHECK: return

// ===== 3D f32 reverse =====

// CHECK-LABEL: func.func @cumsum_kernel_3d_reverse
// CHECK-NOT: hfusion.cumsum
// CHECK: tensor.empty
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: tensor.extract %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: arith.addf
// CHECK: tensor.insert %{{.+}} into %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : tensor<2x3x4xf32>
// CHECK: return
