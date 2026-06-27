// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" %s -split-input-file -verify-diagnostics

// -----
// CHECK-LABEL histogram_same_num_bins
func.func @histogram_same_num_bins(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  // expected-error@+1 {{'hfusion.histogram' op output length (4) must equal num_bins (6)}}
  %res = hfusion.histogram %arg0, 6, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}
// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" %s -split-input-file -verify-diagnostics

// -----
// CHECK-LABEL histogram_1d_output
func.func @histogram_1d_output(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<2x2xi32> {
  // expected-error@+1 {{'hfusion.histogram' op output must be rank-1}}
  %res = hfusion.histogram %arg0, 6, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----
// CHECK-LABEL histogram_mask_i1
func.func @histogram_1d_output(%arg0: tensor<8xi32>, %mask: tensor<8xi32>)
    -> tensor<4xi32> {
  // expected-error@+1 {{'hfusion.histogram' op operand #1 must be ranked tensor of 1-bit signless integer values, but got 'tensor<8xi32>'}}
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
// CHECK-LABEL: func.func @deinterleave_rank0
func.func @deinterleave_rank0(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{'hfusion.deinterleave' op requires input rank to be at least 1}}
  %res = hfusion.deinterleave %arg0 channel<0> : tensor<f32> -> tensor<f32>
  return %res : tensor<f32>
}
