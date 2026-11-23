// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -split-input-file | FileCheck %s

// A buffer token between two elementwise ops must stop fusion: the consumer
// may not be pulled before the token is acquired.

// CHECK-LABEL: func.func @no_fuse_across_get_buf
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: hivm.hir.get_buf
// CHECK: linalg.generic
// CHECK: math.log

#map = affine_map<(d0) -> (d0)>

func.func @no_fuse_across_get_buf(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<64xf32>
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%init = %empty) -> (tensor<64xf32>) {
    %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<64xf32>) outs(%empty : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    hivm.hir.get_buf[<PIPE_V>, 0, false]
    %consumer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%producer : tensor<64xf32>) outs(%init : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.log %in : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>
    scf.yield %consumer : tensor<64xf32>
  }
  return %result : tensor<64xf32>
}
