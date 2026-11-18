// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file \
// RUN:   | FileCheck %s

// Elementwise dtype matrix. 32-bit-or-narrower values use 64-lane rows; i64
// uses the 32-lane register-pair ABI. Integer and floating operations keep
// their native arithmetic form after lifting and vectorization.

module {
  func.func @add_f16(%a: tensor<64xf16>, %b: tensor<64xf16>,
                     %init: tensor<64xf16>) -> tensor<64xf16> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xf16>, tensor<64xf16>)
        outs(%init : tensor<64xf16>) -> tensor<64xf16>
    return %0 : tensor<64xf16>
  }
}

// CHECK-LABEL: func.func private @add_f16_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.addf {{.*}} : vector<64xf16>
// CHECK-LABEL: func.func @add_f16(
// CHECK: call @add_f16_fused_0

// -----

module {
  func.func @add_f32(%a: tensor<64xf32>, %b: tensor<64xf32>,
                     %init: tensor<64xf32>) -> tensor<64xf32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @add_f32_fused_0
// CHECK: arith.addf {{.*}} : vector<64xf32>

// -----

module {
  func.func @add_i8(%a: tensor<64xi8>, %b: tensor<64xi8>,
                    %init: tensor<64xi8>) -> tensor<64xi8> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xi8>, tensor<64xi8>)
        outs(%init : tensor<64xi8>) -> tensor<64xi8>
    return %0 : tensor<64xi8>
  }
}

// CHECK-LABEL: func.func private @add_i8_fused_0
// CHECK: arith.addi {{.*}} : vector<64xi8>

// -----

module {
  func.func @add_i16(%a: tensor<64xi16>, %b: tensor<64xi16>,
                     %init: tensor<64xi16>) -> tensor<64xi16> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xi16>, tensor<64xi16>)
        outs(%init : tensor<64xi16>) -> tensor<64xi16>
    return %0 : tensor<64xi16>
  }
}

// CHECK-LABEL: func.func private @add_i16_fused_0
// CHECK: arith.addi {{.*}} : vector<64xi16>

// -----

module {
  func.func @add_i32(%a: tensor<64xi32>, %b: tensor<64xi32>,
                     %init: tensor<64xi32>) -> tensor<64xi32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %0 : tensor<64xi32>
  }
}

// CHECK-LABEL: func.func private @add_i32_fused_0
// CHECK: arith.addi {{.*}} : vector<64xi32>

// -----

module {
  func.func @add_i64(%a: tensor<64xi64>, %b: tensor<64xi64>,
                     %init: tensor<64xi64>) -> tensor<64xi64> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<64xi64>, tensor<64xi64>)
        outs(%init : tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK-LABEL: func.func private @add_i64_fused_0
// CHECK: arith.constant 32 : index
// CHECK: scf.for {{.*}} step %{{.*}}32
// CHECK: arith.addi {{.*}} : vector<32xi64>

// -----

// Floating arithmetic and activation forms.

module {
  func.func @f32ops(%a: tensor<64xf32>, %b: tensor<64xf32>,
                    %init: tensor<64xf32>) -> tensor<64xf32> {
    %0 = hivm.hir.vsub ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %1 = hivm.hir.vdiv ins(%0, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %2 = hivm.hir.vmax ins(%1, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %3 = hivm.hir.vmin ins(%2, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %4 = hivm.hir.vabs ins(%3 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %5 = hivm.hir.vrelu ins(%4 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %5 : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @f32ops_fused_0
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
// CHECK: math.absf
// CHECK: arith.maximumf

// -----

// Bitwise and shift forms.

module {
  func.func @i32ops(%a: tensor<64xi32>, %b: tensor<64xi32>,
                    %init: tensor<64xi32>) -> tensor<64xi32> {
    %0 = hivm.hir.vand ins(%a, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %1 = hivm.hir.vor ins(%0, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %2 = hivm.hir.vxor ins(%1, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %3 = hivm.hir.vshl ins(%2, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %4 = hivm.hir.vshr ins(%3, %b : tensor<64xi32>, tensor<64xi32>)
        outs(%init : tensor<64xi32>) round : false -> tensor<64xi32>
    return %4 : tensor<64xi32>
  }
}

// CHECK-LABEL: func.func private @i32ops_fused_0
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.xori
// CHECK: arith.shli
// CHECK: arith.shrsi

// -----

// Remainder with a scalar divisor on the i64 pair ABI.

module {
  func.func @mod(%a: tensor<32xi64>, %init: tensor<32xi64>)
      -> tensor<32xi64> {
    %d = arith.constant 7 : i64
    %0 = hivm.hir.vmod ins(%a, %d : tensor<32xi64>, i64)
        outs(%init : tensor<32xi64>) -> tensor<32xi64>
    %1 = hivm.hir.vmodui ins(%0, %d : tensor<32xi64>, i64)
        outs(%init : tensor<32xi64>) -> tensor<32xi64>
    return %1 : tensor<32xi64>
  }
}

// CHECK-LABEL: func.func private @mod_fused_0
// CHECK: arith.remsi {{.*}} : vector<32xi64>
// CHECK: arith.remui {{.*}} : vector<32xi64>
