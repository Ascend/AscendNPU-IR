// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func @test_pre_vectorize_collapse_broadcast(
// CHECK:       tensor.extract_slice
// CHECK-NOT:   tensor.collapse_shape
// CHECK:       linalg.generic
func.func @test_pre_vectorize_collapse_broadcast(%arg0: tensor<256x64x256xf32>) -> tensor<128x32x128xf32> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [128, 1, 128] [1, 1, 1] : tensor<256x64x256xf32> to tensor<128x1x128xf32>
  %0 = tensor.empty() : tensor<128x32x128xf32>
  %collapsed = tensor.collapse_shape %extracted_slice [[0], [1, 2]] : tensor<128x1x128xf32> into tensor<128x128xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<128x128xf32>) outs(%0 : tensor<128x32x128xf32>) dimensions = [1] 
  return %broadcasted : tensor<128x32x128xf32>
}