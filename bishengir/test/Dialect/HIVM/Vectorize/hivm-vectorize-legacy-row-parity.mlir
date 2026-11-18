// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize --verify-each | FileCheck %s --check-prefixes=WIDTH,SEPARATE
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize --hfusion-merge-vf=merge-level=1 --verify-each | FileCheck %s --check-prefix=MERGED

module {
  // WIDTH-LABEL: func.func private @group_norm_rows_fused_0
  // WIDTH: vector.constant_mask [1, 8] : vector<1x64xi1>
  // WIDTH: vector.transfer_read
  // WIDTH-SAME: vector<1x64xf32>
  // WIDTH: arith.mulf {{.*}} : vector<64xf32>
  // WIDTH: vector.transfer_write
  func.func @group_norm_rows(%lhs: tensor<2x8xf32>,
                             %rhs: tensor<2x8xf32>,
                             %init: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %result = hivm.hir.vmul ins(%lhs, %rhs : tensor<2x8xf32>, tensor<2x8xf32>)
        outs(%init : tensor<2x8xf32>) -> tensor<2x8xf32>
    return %result : tensor<2x8xf32>
  }

  // WIDTH-LABEL: func.func private @group_norm_rng_fused_0
  // WIDTH: vector.constant_mask [1, 8] : vector<1x64xi1>
  // WIDTH: arith.xori {{.*}} : vector<64xi32>
  func.func @group_norm_rng(%lhs: tensor<2x8xi32>,
                            %rhs: tensor<2x8xi32>,
                            %init: tensor<2x8xi32>) -> tensor<2x8xi32> {
    %result = hivm.hir.vxor ins(%lhs, %rhs : tensor<2x8xi32>, tensor<2x8xi32>)
        outs(%init : tensor<2x8xi32>) -> tensor<2x8xi32>
    return %result : tensor<2x8xi32>
  }

  // WIDTH-LABEL: func.func private @group_norm_reduce_fused_0
  // WIDTH: vector.constant_mask
  // WIDTH: vector.reduction <add>
  // WIDTH-SAME: vector<64xf32> into f32
  func.func @group_norm_reduce(%input: tensor<2x8xf32>,
                               %init: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %result = hivm.hir.vreduce<sum> ins(%input : tensor<2x8xf32>)
        outs(%init : tensor<2x1xf32>) reduce_dims = [1] -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }

  // SEPARATE-LABEL: func.func @independent_rows(
  // SEPARATE: call @independent_rows_fused_{{[0-9]+}}
  // SEPARATE: arith.addi
  // SEPARATE: call @independent_rows_fused_{{[0-9]+}}
  // SEPARATE: return
  // MERGED-LABEL: func.func @independent_rows(
  // MERGED-COUNT-1: call @{{.*}}merged{{.*}}
  // MERGED-NOT: call @
  // MERGED: return
  func.func @independent_rows(%lhs: tensor<2x8xf32>,
                              %rhs: tensor<2x8xf32>,
                              %init0: tensor<2x8xf32>,
                              %init1: tensor<2x8xf32>, %index: index)
      -> (tensor<2x8xf32>, tensor<2x8xf32>, index) {
    %sum = hivm.hir.vadd ins(%lhs, %rhs : tensor<2x8xf32>, tensor<2x8xf32>)
        outs(%init0 : tensor<2x8xf32>) -> tensor<2x8xf32>
    %next = arith.addi %index, %index : index
    %product = hivm.hir.vmul ins(%lhs, %rhs : tensor<2x8xf32>, tensor<2x8xf32>)
        outs(%init1 : tensor<2x8xf32>) -> tensor<2x8xf32>
    return %sum, %product, %next : tensor<2x8xf32>, tensor<2x8xf32>, index
  }
}
