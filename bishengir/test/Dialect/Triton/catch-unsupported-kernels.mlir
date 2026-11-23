// RUN: bishengir-opt -catch-unsupported-kernels -verify-diagnostics -split-input-file %s

// -----
// expected-error @+1 {{f64 types not supported in SIMT kernels!}}
tt.func @f64Type(%scalar: i64) -> f32 {
  %0 = arith.sitofp %scalar : i64 to f64
  %1 = arith.truncf %0 : f64 to f32
  tt.return %1 : f32
}

// -----
// expected-error @+1 {{f64 types not supported in SIMT kernels!}}
tt.func @tensorF64Type(%tensor: tensor<4xi64>) -> f32 {
  %0 = arith.sitofp %tensor : tensor<4xi64> to tensor<4xf64>
  %1 = arith.truncf %0 : tensor<4xf64> to tensor<4xf32>
  %2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
    %res = arith.addf %arg0, %arg1 : f32
    "tt.reduce.return"(%res) : (f32) -> ()
  }) : (tensor<4xf32>) -> f32
  tt.return %2 : f32
}

// -----
// expected-error @+1 {{f64 types not supported in SIMT kernels!}}
tt.func @scalarF64PtrType(%ptr: !tt.ptr<f64>) {
  tt.return
}

// -----
// expected-error @+1 {{f64 types not supported in SIMT kernels!}}
tt.func @tensorF64PtrType(%ptr: tensor<4x!tt.ptr<f64>>) {
  tt.return
}
