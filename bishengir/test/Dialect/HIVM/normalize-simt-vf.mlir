// RUN: bishengir-opt %s --normalize-simt-vf --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @rank_zero_to_rank_two
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[SCALAR:.*]] = tensor.extract %{{.*}}[] : tensor<i32>
// CHECK: tensor.insert %[[SCALAR]] into %{{.*}}[%{{.*}}, %{{.*}}] : tensor<8x16xi32>
func.func @rank_zero_to_rank_two(
    %src: tensor<i32>, %dest: tensor<8x16xi32>, %row: index, %col: index)
    -> tensor<8x16xi32> {
  %inserted = tensor.insert_slice %src into %dest[%row, %col] [1, 1] [1, 1]
      : tensor<i32> into tensor<8x16xi32>
  return %inserted : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: func.func @rank_zero_to_rank_one
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[SCALAR:.*]] = tensor.extract %{{.*}}[] : tensor<f32>
// CHECK: tensor.insert %[[SCALAR]] into %{{.*}}[%[[C3]]] : tensor<8xf32>
func.func @rank_zero_to_rank_one(%src: tensor<f32>, %dest: tensor<8xf32>)
    -> tensor<8xf32> {
  %inserted = tensor.insert_slice %src into %dest[3] [1] [1]
      : tensor<f32> into tensor<8xf32>
  return %inserted : tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @same_rank_is_unchanged
// CHECK: tensor.insert_slice
func.func @same_rank_is_unchanged(%src: tensor<4xf32>,
                                  %dest: tensor<8xf32>) -> tensor<8xf32> {
  %inserted = tensor.insert_slice %src into %dest[0] [4] [1]
      : tensor<4xf32> into tensor<8xf32>
  return %inserted : tensor<8xf32>
}

// -----

func.func @unsupported_rank_reduction(%src: tensor<4xf32>,
                                      %dest: tensor<8x4xf32>)
    -> tensor<8x4xf32> {
  // expected-error@+1 {{'tensor.insert_slice' op cannot normalize rank-reduced insertion with source rank 1 and destination rank 2; only rank-0 sources are currently supported}}
  %inserted = tensor.insert_slice %src into %dest[0, 0] [1, 4] [1, 1]
      : tensor<4xf32> into tensor<8x4xf32>
  return %inserted : tensor<8x4xf32>
}
