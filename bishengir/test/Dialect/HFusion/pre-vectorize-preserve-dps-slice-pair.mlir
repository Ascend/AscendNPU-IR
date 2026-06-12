// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func.func @preserve_slice_pair_after_pre_vectorization_fusion
func.func @preserve_slice_pair_after_pre_vectorization_fusion(
    %arg0: tensor<32x64xf32>,
    %arg1: tensor<32x64xf32>,
    %arg2: index) -> tensor<2x32x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.empty() : tensor<2x32x64xf32>

  %1 = scf.for %arg3 = %c0 to %c2 step %c1
      iter_args(%arg4 = %0) -> (tensor<2x32x64xf32>) {
    %slice = tensor.extract_slice %arg4[%arg2, 0, 0]
      [1, 32, 64] [1, 1, 1]
      : tensor<2x32x64xf32> to tensor<32x64xf32>

    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>)
      outs(%slice : tensor<32x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.mulf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<32x64xf32>

    %inserted_slice = tensor.insert_slice %res into %arg4[%arg2, 0, 0]
      [1, 32, 64] [1, 1, 1]
      : tensor<32x64xf32> into tensor<2x32x64xf32>

    scf.yield %inserted_slice : tensor<2x32x64xf32>
  }

  return %1 : tensor<2x32x64xf32>
}

// CHECK: %[[SLICE:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0, 0] [1, 32, 64] [1, 1, 1] : tensor<2x32x64xf32> to tensor<32x64xf32>
// CHECK: %[[RES:.*]] = linalg.generic
// CHECK-SAME: outs(%[[SLICE]] : tensor<32x64xf32>)
// CHECK: tensor.insert_slice %[[RES]] into %{{.*}}[%{{.*}}, 0, 0] [1, 32, 64] [1, 1, 1] : tensor<32x64xf32> into tensor<2x32x64xf32>