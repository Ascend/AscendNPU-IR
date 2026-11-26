// RUN: bishengir-opt %s -hfusion-normalize-ops | FileCheck %s --check-prefix=NORM --implicit-check-not='<minui>' --implicit-check-not='<maxui>'
// RUN: bishengir-opt %s -hfusion-normalize-ops -convert-hfusion-to-hivm | FileCheck %s --check-prefix=HIVM

module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  // NORM-LABEL: func.func @bool_argmin
  // NORM-SAME: %{{[^,]+}}: tensor<4x8xi1>, %[[IDX:[^:]+]]: tensor<4x8xi32>
  // NORM: hfusion.reduce_with_index {{.*}}tie_break_left = true{{.*}} <min> ins(%{{[^,]+}}, %[[IDX]] : tensor<4x8xf16>, tensor<4x8xi32>)
  // NORM: return {{.*}} : tensor<4xi1>, tensor<4xi32>
  // HIVM-LABEL: func.func @bool_argmin
  // HIVM: hivm.hir.vreduce <min_with_index_left>
  func.func @bool_argmin(%v: tensor<4x8xi1>, %idx: tensor<4x8xi32>, %vo: tensor<4xi1>, %io: tensor<4xi32>) -> (tensor<4xi1>, tensor<4xi32>) {
    %r:2 = hfusion.reduce_with_index {tie_break_left = true} <minui> ins(%v, %idx : tensor<4x8xi1>, tensor<4x8xi32>) outs(%vo, %io : tensor<4xi1>, tensor<4xi32>) dimensions = [1] -> tensor<4xi1>, tensor<4xi32>
    return %r#0, %r#1 : tensor<4xi1>, tensor<4xi32>
  }

  // NORM-LABEL: func.func @bool_argmax
  // NORM-SAME: %{{[^,]+}}: tensor<4x8xi1>, %[[IDX:[^:]+]]: tensor<4x8xi32>
  // NORM: hfusion.reduce_with_index {{.*}}tie_break_left = false{{.*}} <max> ins(%{{[^,]+}}, %[[IDX]] : tensor<4x8xf16>, tensor<4x8xi32>)
  // NORM: return {{.*}} : tensor<4xi1>, tensor<4xi32>
  // HIVM-LABEL: func.func @bool_argmax
  // HIVM: hivm.hir.vreduce <max_with_index_right>
  func.func @bool_argmax(%v: tensor<4x8xi1>, %idx: tensor<4x8xi32>, %vo: tensor<4xi1>, %io: tensor<4xi32>) -> (tensor<4xi1>, tensor<4xi32>) {
    %r:2 = hfusion.reduce_with_index {tie_break_left = false} <maxui> ins(%v, %idx : tensor<4x8xi1>, tensor<4x8xi32>) outs(%vo, %io : tensor<4xi1>, tensor<4xi32>) dimensions = [1] -> tensor<4xi1>, tensor<4xi32>
    return %r#0, %r#1 : tensor<4xi1>, tensor<4xi32>
  }
}
