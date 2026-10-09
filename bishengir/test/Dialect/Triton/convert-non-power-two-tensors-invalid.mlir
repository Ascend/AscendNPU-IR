// RUN: bishengir-opt -split-input-file -convert-non-power-two-tensors -verify-diagnostics %s

// expected-error @+1 {{Unsupported non power of two tensor operations found}}
tt.func @unsupportedSplit(%arg: tensor<3x2xf32>) {
  // expected-error @+1 {{Non power of two tensor operation is currently unsupported}}
  %lhs, %rhs = tt.split %arg : tensor<3x2xf32> -> tensor<3xf32>
  tt.return
}

// -----

// expected-error @+1 {{Unsupported non power of two tensor operations found}}
tt.func @nonSplatTensorConstant() {
  // expected-error @+2 {{Non splat tensor constant ops must satisfy num elements == threads per warp}}
  // expected-error @+1 {{Non power of two tensor operation is currently unsupported}}
  %cst = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  tt.return
}

// -----

// expected-error @+1 {{Unsupported non power of two tensor operations found}}
tt.func @unsupportedSliceStride(%arg: tensor<5xf32>) {
  // expected-error @+2 {{strides must all be 1; got non-unit stride at axis 0}}
  // expected-error @+1 {{Non power of two tensor operation is currently unsupported}}
  %slice = tensor.extract_slice %arg[0] [3] [2] : tensor<5xf32> to tensor<3xf32>
  tt.return
}

// -----

// expected-error @+1 {{Unsupported non power of two tensor operations found}}
tt.func @unsupportedTensorPad(%arg: tensor<1xf32>) {
  %zero = arith.constant 0.0 : f32
  // expected-error @+1 {{Non power of two tensor operation is currently unsupported}}
  %padded = tensor.pad %arg low[0] high[2] {
  ^bb0(%index: index):
    tensor.yield %zero : f32
  } : tensor<1xf32> to tensor<3xf32>
  tt.return
}
