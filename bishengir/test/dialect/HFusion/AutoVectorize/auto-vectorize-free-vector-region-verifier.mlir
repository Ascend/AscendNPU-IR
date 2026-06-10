// RUN: bishengir-opt %s --hfusion-auto-vectorize-verifier="verify-free-vector-region=true verify-free-vector-func=false" -verify-diagnostics -split-input-file

module {
  func.func @vector_under_outlined_loop_is_valid() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c16 step %c4 {
      %0 = vector.broadcast %cst : f32 to vector<4xf32>
      scf.yield
    } {"outlined-loop-target-0"}

    return
  }
}

// -----

module {
  func.func @free_vector_broadcast_is_rejected() {
    %cst = arith.constant 0.0 : f32

    // expected-error @+1 {{unexpected vector operation outside outlined vector region}}
    %0 = vector.broadcast %cst : f32 to vector<4xf32>

    return
  }
}

// -----

module {
  func.func @loop_without_outline_target_is_rejected() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c16 step %c4 {
      // expected-error @+1 {{unexpected vector operation outside outlined vector region}}
      %0 = vector.broadcast %cst : f32 to vector<4xf32>
      scf.yield
    }

    return
  }
}

// -----

module {
  func.func @vector_function_attr_does_not_bypass_region_check() attributes {hivm.vector_function} {
    %cst = arith.constant 0.0 : f32

    // expected-error @+1 {{unexpected vector operation outside outlined vector region}}
    %0 = vector.broadcast %cst : f32 to vector<4xf32>

    return
  }
}

// -----

module {
  func.func @target_hfusion_under_outlined_loop_is_valid(%arg0: tensor<8xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index

    scf.for %i = %c0 to %c4 step %c64 {
      %0 = hfusion.deinterleave %arg0 channel<0> {"hfusion-auto-vectorize-target-0"} : tensor<8xf32> -> tensor<4xf32>
      scf.yield
    } {"outlined-loop-target-0"}

    return
  }
}

// -----

module {
  func.func @free_target_hfusion_deinterleave(%arg0: tensor<8xf32>) {
    // expected-error @+1 {{unexpected hfusion/linalg operation outside outlined vector region}}
    %0 = hfusion.deinterleave %arg0 channel<0> {"hfusion-auto-vectorize-target-0"} : tensor<8xf32> -> tensor<4xf32>

    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>

module {
  func.func @free_target_linalg_generic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tensor.empty() : tensor<4xf32>

    // expected-error @+1 {{unexpected hfusion/linalg operation outside outlined vector region}}
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) attrs = {"hfusion-auto-vectorize-target-0"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4xf32>

    return %1 : tensor<4xf32>
  }
}
