// RUN: bishengir-opt --hfusion-flatten-ops="flatten-mode=tidy register-based=true" --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @extract_head_unit_dim_stride_01(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0] [15] [1] : tensor<15xf32> to tensor<15xf32>
func.func @extract_head_unit_dim_stride_01(%arg0: tensor<1x3x5xf32>) -> tensor<1x3x5xf32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, 3, 5] [1, 1, 1]
      : tensor<1x3x5xf32> to tensor<1x3x5xf32>
  return %0 : tensor<1x3x5xf32>
}

// -----

// CHECK-LABEL: func.func @extract_tail_unit_dim_stride_02(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0] [24] [5] : tensor<120xf32> to tensor<24xf32>
func.func @extract_tail_unit_dim_stride_02(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x1xf32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [2, 3, 4, 1] [1, 1, 1, 1]
      : tensor<2x3x4x5xf32> to tensor<2x3x4x1xf32>
  return %0 : tensor<2x3x4x1xf32>
}

// -----

// CHECK-LABEL: func.func @extract_tail_unit_dim_stride_03(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0] [6] [4] : tensor<24xf32> to tensor<6xf32>
func.func @extract_tail_unit_dim_stride_03(%arg0: tensor<2x3x4x1xf32>) -> tensor<2x3x1x1xf32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [2, 3, 1, 1] [1, 1, 1, 1]
      : tensor<2x3x4x1xf32> to tensor<2x3x1x1xf32>
  return %0 : tensor<2x3x1x1xf32>
}

// -----

// CHECK-LABEL: func.func @extract_tail_unit_dim_stride_04(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0] [6] [1] : tensor<6xf32> to tensor<6xf32>
func.func @extract_tail_unit_dim_stride_04(%arg0: tensor<2x3x1xf32>) -> tensor<2x3x1xf32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0] [2, 3, 1] [1, 1, 1]
      : tensor<2x3x1xf32> to tensor<2x3x1xf32>
  return %0 : tensor<2x3x1xf32>
}

// -----

// CHECK-LABEL: func.func @extract_middle_unit_dim_stride_01(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0] [8, 16] [1, 64] : tensor<16x2048xf32> to tensor<8x16xf32>
func.func @extract_middle_unit_dim_stride_01(%arg0: tensor<16x32x64xf32>) -> tensor<8x16x1xf32> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [8, 16, 1] [1, 1, 1] : tensor<16x32x64xf32> to tensor<8x16xf32>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [8, 16, 1] : tensor<8x16xf32> into tensor<8x16x1xf32>
  return %expanded : tensor<8x16x1xf32>
}

// -----

// CHECK-LABEL: func.func @extract_middle_unit_dim_stride_02(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0] [2, 4] [1, 2048] : tensor<4x16384xf32> to tensor<2x4xf32>
func.func @extract_middle_unit_dim_stride_02(%arg0: tensor<4x8x32x64xf32>) -> tensor<2x4x1x1xf32> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [2, 4, 1, 1] [1, 1, 1, 1] : tensor<4x8x32x64xf32> to tensor<2x4xf32>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2, 3]] output_shape [2, 4, 1, 1] : tensor<2x4xf32> into tensor<2x4x1x1xf32>
  return %expanded : tensor<2x4x1x1xf32>
}

// -----

// CHECK-LABEL: func.func @insert_middle_unit_dim_stride_01(
// CHECK: tensor.insert_slice {{.*}}[0, 0] [2, 4] [1, 2048] : tensor<2x4xf32> into tensor<4x16384xf32>
func.func @insert_middle_unit_dim_stride_01(%arg0: tensor<2x4x1x1xf32>, %arg1: tensor<4x8x32x64xf32>) -> tensor<4x8x32x64xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<2x4x1x1xf32> into tensor<2x4xf32>
  %inserted_slice = tensor.insert_slice %collapsed into %arg1[0, 0, 0, 0] [2, 4, 1, 1] [1, 1, 1, 1] : tensor<2x4xf32> into tensor<4x8x32x64xf32>
  return %inserted_slice : tensor<4x8x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @insert_middle_unit_dim_stride_02(
// CHECK: tensor.insert_slice {{.*}}[0, 0] [8, 16] [1, 64] : tensor<8x16xf32> into tensor<16x2048xf32>
func.func @insert_middle_unit_dim_stride_02(%arg0: tensor<8x16x1xf32>, %arg1: tensor<16x32x64xf32>) -> tensor<16x32x64xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<8x16x1xf32> into tensor<8x16xf32>
  %inserted_slice = tensor.insert_slice %collapsed into %arg1[0, 0, 0] [8, 16, 1] [1, 1, 1] : tensor<8x16xf32> into tensor<16x32x64xf32>
  return %inserted_slice : tensor<16x32x64xf32>
}