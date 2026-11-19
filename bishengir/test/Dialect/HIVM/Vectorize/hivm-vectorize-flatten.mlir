// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s --implicit-check-not=builtin.unrealized_conversion_cast
// RUN: bishengir-opt %s --hivm-vectorize --verify-each --hivm-vectorize --verify-each | FileCheck %s

module {
  // CHECK-LABEL: func.func private @leading_units_fused_0
  // CHECK: vector.transfer_read
  // CHECK: vector.shape_cast {{.*}} : vector<1x1x4xf16> to vector<4xf16>
  // CHECK: arith.extf {{.*}} {round_mode = #hfusion.round_mode<rint>} : vector<4xf16> to vector<4xf32>
  // CHECK: vector.shape_cast {{.*}} : vector<4xf32> to vector<1x1x4xf32>
  // CHECK: vector.transfer_write
  func.func @leading_units(%input: tensor<1x1x4xf16>,
                           %init: tensor<1x1x4xf32>) -> tensor<1x1x4xf32> {
    %result = hivm.hir.vcast ins(%input : tensor<1x1x4xf16>)
        outs(%init : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    return %result : tensor<1x1x4xf32>
  }

  // CHECK-LABEL: func.func private @matrix_fused_0
  // CHECK: vector.shape_cast {{.*}} : vector<4x16xf16> to vector<64xf16>
  // CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
  // CHECK: vector.shape_cast {{.*}} : vector<64xf32> to vector<4x16xf32>
  func.func @matrix(%input: tensor<4x16xf16>,
                    %init: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %result = hivm.hir.vcast ins(%input : tensor<4x16xf16>)
        outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %result : tensor<4x16xf32>
  }

  // CHECK-LABEL: func.func private @narrow_fused_0
  // CHECK: vector.shape_cast {{.*}} : vector<4x16xf32> to vector<64xf32>
  // CHECK: arith.truncf {{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xf16>
  // CHECK: vector.shape_cast {{.*}} : vector<64xf16> to vector<4x16xf16>
  func.func @narrow(%input: tensor<4x16xf32>,
                    %init: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %result = hivm.hir.vcast {enable_saturate = true, enable_overflow = true}
        ins(%input : tensor<4x16xf32>) outs(%init : tensor<4x16xf16>)
        -> tensor<4x16xf16>
    return %result : tensor<4x16xf16>
  }

  // CHECK-LABEL: func.func private @unit_reduction_fused_0
  // CHECK: vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
  // CHECK: arith.addf {{.*}} : vector<16xf32>
  // CHECK-NOT: arith.addf {{.*}} : vector<1x16xf32>
  // CHECK: func.func @unit_reduction(
  func.func @unit_reduction(%input: tensor<16x4x16xf32>,
                            %init: tensor<1x4x16xf32>) -> tensor<1x4x16xf32> {
    %result = hivm.hir.vreduce<sum> ins(%input : tensor<16x4x16xf32>)
        outs(%init : tensor<1x4x16xf32>) reduce_dims = [0]
        -> tensor<1x4x16xf32>
    return %result : tensor<1x4x16xf32>
  }
}
