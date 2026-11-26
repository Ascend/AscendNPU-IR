// RUN: bishengir-opt %s --hivm-propagate-convert-layout | FileCheck %s

#fractal = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>
#nd = #hivm.data_layout<ND>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// The incoming accumulator feeds a channel-split Fixpipe and the else yield.
// CHECK-LABEL: func.func @shared_accumulator(
// CHECK-SAME:    %[[FLAG:.*]]: i1, %[[ACC:.*]]: tensor<1x1x16x16xf32>
// CHECK:       %[[ND:.*]] = hivm.hir.convert_layout %[[ACC]] output_shape [16, 16]
// CHECK:       %[[L1:.*]] = hivm.hir.fixpipe {{.*}}ins(%[[ND]] : tensor<16x16xf32>)
// CHECK:       %[[MERGED:.*]] = scf.if %[[FLAG]] -> (tensor<1x1x16x16xf32>) {
// CHECK:         %[[UPDATED:.*]] = hivm.hir.mmadL1 ins(%[[L1]], %[[L1]],
// CHECK:         scf.yield %[[UPDATED]] : tensor<1x1x16x16xf32>
// CHECK:       } else {
// CHECK:         scf.yield %[[ACC]] : tensor<1x1x16x16xf32>
// CHECK:       %[[OUT_ND:.*]] = hivm.hir.convert_layout %[[MERGED]] output_shape [16, 16]
// CHECK:       hivm.hir.fixpipe {{.*}}ins(%[[OUT_ND]] : tensor<16x16xf32>)
  func.func @shared_accumulator(%flag: i1, %acc: tensor<1x1x16x16xf32>, %out: memref<16x16xf32>) {
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %nd = hivm.hir.convert_layout %acc output_shape [16, 16]
        {srcLayout = #fractal, dstLayout = #nd} : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
    %empty = tensor.empty() : tensor<2x1x16x8xf32>
    %l1 = hivm.hir.fixpipe {channel_split = true} ins(%nd : tensor<16x16xf32>)
        outs(%empty : tensor<2x1x16x8xf32>) -> tensor<2x1x16x8xf32>
    %result = scf.if %flag -> (tensor<16x16xf32>) {
      %updated = hivm.hir.mmadL1 ins(%l1, %l1, %false, %c16, %c16, %c16 :
          tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index)
          outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %then_nd = hivm.hir.convert_layout %updated output_shape [16, 16]
          {srcLayout = #fractal, dstLayout = #nd} : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
      scf.yield %then_nd : tensor<16x16xf32>
    } else {
      scf.yield %nd : tensor<16x16xf32>
    } {hivm.remain_in_l0c = [0 : i32], normalized_in_L0C = [0 : i32]}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%result : tensor<16x16xf32>)
        outs(%out : memref<16x16xf32>)
    return
  }

// Keep the selected extent when the two conversions have different shapes.
// CHECK-LABEL: func.func @different_output_extents(
// CHECK-SAME:    %[[N:[^:]+]]: index, %[[M:[^:]+]]: index
// CHECK-DAG:   %[[ELSE:.*]] = hivm.hir.convert_layout {{.*}} output_shape [%[[N]], 16]
// CHECK-DAG:   %[[THEN:.*]] = hivm.hir.convert_layout {{.*}} output_shape [%[[M]], 16]
// CHECK:       %[[RESULT:.*]] = scf.if {{.*}} -> (tensor<?x16xf32>)
// CHECK:         scf.yield %[[THEN]] : tensor<?x16xf32>
// CHECK:       } else {
// CHECK:         scf.yield %[[ELSE]] : tensor<?x16xf32>
// CHECK:       return %[[RESULT]], %[[ELSE]]
  func.func @different_output_extents(%flag: i1, %acc: tensor<1x?x16x16xf32>,
      %n: index, %m: index) -> (tensor<?x16xf32>, tensor<?x16xf32>) {
    %fallback = hivm.hir.convert_layout %acc output_shape [%n, 16]
        {srcLayout = #fractal, dstLayout = #nd} : (tensor<1x?x16x16xf32>) -> tensor<?x16xf32>
    %result = scf.if %flag -> (tensor<?x16xf32>) {
      %value = hivm.hir.convert_layout %acc output_shape [%m, 16]
          {srcLayout = #fractal, dstLayout = #nd} : (tensor<1x?x16x16xf32>) -> tensor<?x16xf32>
      scf.yield %value : tensor<?x16xf32>
    } else {
      scf.yield %fallback : tensor<?x16xf32>
    }
    return %result, %fallback : tensor<?x16xf32>, tensor<?x16xf32>
  }
}
