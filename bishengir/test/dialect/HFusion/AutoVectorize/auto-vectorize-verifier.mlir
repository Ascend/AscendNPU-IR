// RUN: bishengir-opt %s --hfusion-auto-vectorize-verifier -verify-diagnostics -split-input-file

module {
  func.func @free_vector_transfer_read(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32

    // expected-error @+1 {{unexpected vector operation outside vector function}}
    %0 = vector.transfer_read %arg0[%c0], %cst
        : memref<16xf32>, vector<4xf32>

    return
  }
}

// -----

module {
  func.func @free_vector_broadcast() {
    %cst = arith.constant 0.0 : f32

    // expected-error @+1 {{unexpected vector operation outside vector function}}
    %0 = vector.broadcast %cst : f32 to vector<4xf32>

    return
  }
}

// -----

module {
  func.func @free_vector_transfer_write(%arg0: vector<4xf32>,
                                        %arg1: memref<16xf32>) {
    %c0 = arith.constant 0 : index

    // expected-error @+1 {{unexpected vector operation outside vector function}}
    vector.transfer_write %arg0, %arg1[%c0]
        : vector<4xf32>, memref<16xf32>

    return
  }
}