// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2='tree-reduce=true' --normalize-tensor-ops --outline-vector-function --hivm-inline-vf-scope --hfusion-auto-vectorize-verifier | FileCheck %s --implicit-check-not='tensor<1x8x1x?xf32>'

// Five tensor inputs split the reduction into eight partials. The short inner
// dimension initially becomes dynamic through tiling. Cleanup and normalization
// recover its static size before VF outlining. Check that the scratch is created
// in the caller and retains its static shape as a VF argument and loop iter_arg.
// CHECK-LABEL: func.func @computed_reduce_outlined_vf_1(
// CHECK-SAME: %[[SCRATCH_ARG:[^ ,:]+]]: tensor<1x8x1x17xf32>
// CHECK-SAME: hivm.vector_function
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[SCRATCH_ARG]]) -> (tensor<1x8x1x17xf32>)
// CHECK: %[[UPDATED:.*]] = tensor.insert_slice {{.*}} into %[[ACC]][{{.*}}] {{.*}} : tensor<1x1x1x17xf32> into tensor<1x8x1x17xf32>
// CHECK-NEXT: scf.yield %[[UPDATED]] : tensor<1x8x1x17xf32>
// CHECK-LABEL: func.func @computed_reduce(
// CHECK: %[[SCRATCH:.*]] = tensor.empty() : tensor<1x8x1x17xf32>
// CHECK: func.call @computed_reduce_outlined_vf_1({{[^)]*}}, %[[SCRATCH]], {{[^)]*}})
// CHECK-SAME: tensor<1x8x1x17xf32>
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
