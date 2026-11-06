// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2='tree-reduce=true' --normalize-tensor-ops --outline-vector-function --hivm-inline-vf-scope --hfusion-auto-vectorize-verifier | FileCheck %s --implicit-check-not='tensor<1x8x1x?xf32>'

// Five tensor inputs split the reduction into eight partials. The short inner
// dimension initially becomes dynamic through tiling. Normalize it before VF
// outlining so that the scratch argument retains its static shape.
// CHECK: tensor<1x8x1x17xf32>
// CHECK: hivm.vector_function
func.func @computed_reduce(%a: tensor<2x16x3x17xf32>,
    %b: tensor<2x16x3x17xf32>, %c: tensor<2x16x3x17xf32>,
    %d: tensor<2x16x3x17xf32>, %e: tensor<2x16x3x17xf32>)
    -> tensor<2x3x17xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x3x17xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3x17xf32>)
      -> tensor<2x3x17xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(i, r, j, k) -> (i, r, j, k)>,
                      affine_map<(i, r, j, k) -> (i, r, j, k)>,
                      affine_map<(i, r, j, k) -> (i, r, j, k)>,
                      affine_map<(i, r, j, k) -> (i, r, j, k)>,
                      affine_map<(i, r, j, k) -> (i, r, j, k)>,
                      affine_map<(i, r, j, k) -> (i, j, k)>],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]
    } ins(%a, %b, %c, %d, %e : tensor<2x16x3x17xf32>,
        tensor<2x16x3x17xf32>, tensor<2x16x3x17xf32>,
        tensor<2x16x3x17xf32>, tensor<2x16x3x17xf32>)
      outs(%init : tensor<2x3x17xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32, %u: f32, %v: f32, %acc: f32):
    %xy = arith.mulf %x, %y : f32
    %zu = arith.mulf %z, %u : f32
    %sum = arith.addf %xy, %zu : f32
    %value = arith.addf %sum, %v : f32
    %next = arith.addf %value, %acc : f32
    linalg.yield %next : f32
  } -> tensor<2x3x17xf32>
  return %result : tensor<2x3x17xf32>
}
