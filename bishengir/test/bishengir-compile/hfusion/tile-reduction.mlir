// RUN: bishengir-opt %s -hfusion-auto-vectorize -canonicalize -cse -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s
 
func.func @only_reduction(%arg0 : tensor<300x128xf32>) -> tensor<300xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = tensor.empty() : tensor<300xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<300x128xf32>) outs(%1 : tensor<300xf32>) {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.addf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    return %2 : tensor<300xf32>
}
 
// CHECK: only_reduction_outlined_vf_0
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c300 = arith.constant 300 : index
// CHECK: %c64 = arith.constant 64 : index
// CHECK: %c128 = arith.constant 128 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: scf.for {{.*}} %c0 to %c300 step %c1
// CHECK:   vector.transfer_write
// CHECK:   scf.for {{.*}} %c0 to %c128 step %c64
// CHECK:     arith.addf
// CHECK:   vector.multi_reduction
// CHECK:   vector.transfer_write
 
// -----

func.func @only_reduction_has_tail_block(%arg0 : tensor<300x150xf32>) -> tensor<300xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = tensor.empty() : tensor<300xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<300x150xf32>) outs(%1 : tensor<300xf32>) {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.addf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    return %2 : tensor<300xf32>
}
 
// CHECK: only_reduction_has_tail_block_outlined_vf_0
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c300 = arith.constant 300 : index
// CHECK: %c64 = arith.constant 64 : index
// CHECK: %c150 = arith.constant 150 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: scf.for {{.*}} %c0 to %c300 step %c1
// CHECK:   vector.transfer_write
// CHECK:   scf.for {{.*}} %c0 to %c150 step %c64
// CHECK:     arith.addf
// CHECK:   vector.multi_reduction
// CHECK:   vector.transfer_write
 
// -----

func.func @only_reduction_one_dim(%arg0 : tensor<128xf32>) -> tensor<f32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%arg0 : tensor<128xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.addf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<f32>
    return %2 : tensor<f32>
}
 
// CHECK: only_reduction_one_dim_outlined_vf_0
// CHECK: %c64 = arith.constant 64 : index
// CHECK: %c128 = arith.constant 128 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: vector.transfer_write
// CHECK: scf.for {{.*}} %c0 to %c128 step %c64
// CHECK:   arith.addf
// CHECK: vector.multi_reduction
// CHECK: vector.transfer_write
 
// -----
 
func.func @elewise_fuse_into_reduction(%arg0 : tensor<300x128xf32>) -> (tensor<300xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = tensor.empty() : tensor<300x128xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0: tensor<300x128xf32>, tensor<300x128xf32>) outs(%1 : tensor<300x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %100 = arith.subf %in, %in_1 : f32
      linalg.yield %100 : f32
    } -> tensor<300x128xf32>
    %3 = tensor.empty() : tensor<300xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<300x128xf32>) outs(%3 : tensor<300xf32>) {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.addf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    return %4 : tensor<300xf32>
}
 
// CHECK: elewise_fuse_into_reduction_outlined_vf_0
// CHECK: %c300 = arith.constant 300 : index
// CHECK: %c128 = arith.constant 128 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c64 = arith.constant 64 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: scf.for {{.*}} %c0 to %c300 step %c1
// CHECK:   vector.transfer_write
// CHECK:   scf.for {{.*}} %c0 to %c128 step %c64
// CHECK:     arith.subf
// CHECK:     arith.addf
// CHECK:   vector.multi_reduction
// CHECK:   vector.transfer_write
 
// -----
 
func.func @reduction_fuse_into_elewise(%arg0 : tensor<300x128xf32>, %arg1 : tensor<300xf32>) -> (tensor<300xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = tensor.empty() : tensor<300xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = [ "parallel", "reduction"]} ins(%arg0 : tensor<300x128xf32>) outs(%1 : tensor<300xf32>) {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.addf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    %3 = tensor.empty() : tensor<300xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %arg1: tensor<300xf32>, tensor<300xf32>) outs(%3 : tensor<300xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %100 = arith.subf %in, %in_1 : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    return %4 : tensor<300xf32>
}
 
// CHECK: reduction_fuse_into_elewise_outlined_vf_0
// CHECK: %c300 = arith.constant 300 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c64 = arith.constant 64 : index
// CHECK: %c128 = arith.constant 128 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: scf.for {{.*}} %c0 to %c300 step %c64
// CHECK:   scf.for {{.*}} %c0 to %{{.*}} step %c1
// CHECK:     vector.transfer_write
// CHECK:     scf.for {{.*}} %c0 to %c128 step %c64
// CHECK:       arith.addf
// CHECK:     vector.multi_reduction
// CHECK:     vector.transfer_write
// CHECK:   arith.subf