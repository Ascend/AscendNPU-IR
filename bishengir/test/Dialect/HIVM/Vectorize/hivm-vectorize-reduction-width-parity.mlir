// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize --verify-each | FileCheck %s --check-prefix=REGBASE
// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s --check-prefix=CPU

module {
  // REGBASE-LABEL: func.func private @row_sum_fused_0
  // REGBASE: vector.constant_mask [1, 8] : vector<1x64xi1>
  // REGBASE: vector.constant_mask [8] : vector<64xi1>
  // REGBASE: vector.mask {{.*}}vector.reduction <add>
  // REGBASE-SAME: vector<64xf32> into f32
  // CPU-LABEL: func.func private @row_sum_fused_0
  // CPU: vector.reduction <add>
  // CPU-SAME: vector<8xf32> into f32
  func.func @row_sum(%input: tensor<2x8xf32>, %init: tensor<2x1xf32>)
      -> tensor<2x1xf32> {
    %result = hivm.hir.vreduce<sum> ins(%input : tensor<2x8xf32>)
        outs(%init : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }

  // REGBASE-LABEL: func.func private @unsigned_min_fused_0
  // REGBASE: vector.mask {{.*}}vector.reduction <minui>
  // REGBASE-SAME: vector<64xi32> into i32
  // CPU-LABEL: func.func private @unsigned_min_fused_0
  // CPU: vector.reduction <minui>
  // CPU-SAME: vector<8xi32> into i32
  func.func @unsigned_min(%input: tensor<8xi32>, %init: tensor<1xi32>)
      -> tensor<1xi32> {
    %result = hivm.hir.vreduce<min> ins(%input : tensor<8xi32>)
        outs(%init : tensor<1xi32>) unsigned_src = true reduce_dims = [0]
        -> tensor<1xi32>
    return %result : tensor<1xi32>
  }

  // REGBASE-LABEL: func.func private @row_product_fused_0
  // REGBASE-NOT: vector.reduction <mul>
  // REGBASE: scf.for
  // REGBASE: arith.mulf
  // CPU-LABEL: func.func private @row_product_fused_0
  // CPU-NOT: vector.reduction <mul>
  // CPU: scf.for
  // CPU: arith.mulf
  func.func @row_product(%input: tensor<8xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %result = hivm.hir.vreduce<prod> ins(%input : tensor<8xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %result : tensor<1xf32>
  }

  // REGBASE-LABEL: func.func private @row_maximum_fused_0
  // REGBASE: arith.constant dense<0xFF800000> : vector<64xf32>
  // REGBASE: arith.select {{.*}} : vector<64xi1>, vector<64xf32>
  // REGBASE-NOT: vector.mask
  // REGBASE: vector.reduction <maximumf>
  // REGBASE-SAME: vector<64xf32> into f32
  // CPU-LABEL: func.func private @row_maximum_fused_0
  // CPU: vector.reduction <maximumf>
  // CPU-SAME: vector<8xf32> into f32
  func.func @row_maximum(%input: tensor<2x8xf32>, %init: tensor<2x1xf32>)
      -> tensor<2x1xf32> {
    %result = hivm.hir.vreduce<max> ins(%input : tensor<2x8xf32>)
        outs(%init : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }

  // REGBASE-LABEL: func.func private @row_minimum_fused_0
  // REGBASE: arith.constant dense<0x7F800000> : vector<64xf32>
  // REGBASE: arith.select {{.*}} : vector<64xi1>, vector<64xf32>
  // REGBASE-NOT: vector.mask
  // REGBASE: vector.reduction <minimumf>
  // REGBASE-SAME: vector<64xf32> into f32
  // CPU-LABEL: func.func private @row_minimum_fused_0
  // CPU: vector.reduction <minimumf>
  // CPU-SAME: vector<8xf32> into f32
  func.func @row_minimum(%input: tensor<2x8xf32>, %init: tensor<2x1xf32>)
      -> tensor<2x1xf32> {
    %result = hivm.hir.vreduce<min> ins(%input : tensor<2x8xf32>)
        outs(%init : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }

  // REGBASE-LABEL: func.func private @prime_sum_fused_0
  // REGBASE-NOT: vector.reduction
  // REGBASE: return
  // CPU-LABEL: func.func private @prime_sum_fused_0
  // CPU-NOT: vector.reduction
  // CPU: return
  func.func @prime_sum(%input: tensor<67xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %result = hivm.hir.vreduce<sum> ins(%input : tensor<67xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %result : tensor<1xf32>
  }

  // REGBASE-LABEL: func.func private @unit_sum_fused_0
  // REGBASE-NOT: vector.reduction
  // REGBASE: return
  // CPU-LABEL: func.func private @unit_sum_fused_0
  // CPU-NOT: vector.reduction
  // CPU: return
  func.func @unit_sum(%input: tensor<2x1xf32>, %init: tensor<2x1xf32>)
      -> tensor<2x1xf32> {
    %result = hivm.hir.vreduce<sum> ins(%input : tensor<2x1xf32>)
        outs(%init : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }
}
