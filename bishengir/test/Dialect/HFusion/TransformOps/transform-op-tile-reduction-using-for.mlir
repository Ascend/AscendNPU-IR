// RUN: bishengir-opt %s -transform-interpreter -canonicalize-ext -cse -verify-diagnostics -split-input-file | FileCheck %s

func.func @reduction_tile(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) dimensions = [1]
  return %reduced : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//     CHECK: func @reduction_tile(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D2]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[D1]] step %[[C5]] iter_args(%[[ARG3:.*]] = %[[F]]) -> (tensor<?x5xf32>) {
//     CHECK:     %[[PS:.*]] = affine.min #[[MAP0]](%[[K]])[%[[D1]]]
//     CHECK:     %[[EXT2:.*]] = tensor.extract_slice %[[ARG0]][0, %[[K:.*]]] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x5xf32> to tensor<?x?xf32>
//     CHECK:     %[[PR:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXT2]], %[[EXT]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[EXT]] : tensor<?x?xf32>) {
//     CHECK:       arith.addf
//     CHECK:       linalg.yield
//     CHECK:     } -> tensor<?x?xf32>
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[PR]] into %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> into tensor<?x5xf32>
//     CHECK:     scf.yield %[[INS]] : tensor<?x5xf32>
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) dimensions = [1]
//     CHECK:   {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   }
//     CHECK:   return %[[R]] : tensor<?xf32>

// -----

func.func @reduction_tile_multiple_results(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2, %arg3 : tensor<?xf32>, tensor<?xf32>) dimensions = [1]
    (%in: f32, %in_0: f32, %init: f32, %init_1: f32) {
      %0 = arith.addf %in, %init : f32
      %1 = arith.addf %in_0, %init_1 : f32
      linalg.yield %0, %1 : f32, f32
    }
  return %reduced#0, %reduced#1 : tensor<?xf32>, tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op:2, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: func @reduction_tile_multiple_results
// CHECK-DAG:   %[[INIT0:.+]] = linalg.fill
// CHECK-DAG:   %[[INIT1:.+]] = linalg.fill
// CHECK:       %[[OUT:.+]]:2 = scf.for
// CHECK-SAME:            iter_args(%[[ARG0:.+]] = %[[INIT0]], %[[ARG1:.+]] = %[[INIT1]])
// CHECK:         %[[UPDATED:.*]]:2 = linalg.generic
// CHECK:         arith.addf
// CHECK:         arith.addf
// CHECK:         linalg.yield
// CHECK:       %[[INSERT1:.+]] = tensor.insert_slice %[[UPDATED]]#0 into %[[ARG0]]
// CHECK:       %[[INSERT2:.+]] = tensor.insert_slice %[[UPDATED]]#1 into %[[ARG1]]
// CHECK:       scf.yield %[[INSERT1]], %[[INSERT2]]
// CHECK:       linalg.reduce ins(%[[OUT]]#0, %[[OUT]]#1 : tensor<?x5xf32>, tensor<?x5xf32>)
// CHECK:         arith.addf 
// CHECK:         arith.addf

// -----

// CHECK: @reduction_tile(
// CHECK-SAME: {{.*}}: tensor<?x?xf32>, {{.*}}: tensor<?xf32>, [[TILING:.*]]: index
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step [[TILING]]
func.func @reduction_tile(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: index) -> tensor<?xf32> {
  %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) dimensions = [1]
  return %reduced : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.func.get_func_argument %1[2] : (!transform.any_op) -> !transform.any_value
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, %2] : (!transform.any_op, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @test_tile_reduction_result_order
// CHECK-SAME: %[[arg2:.*]]: i64, %[[arg3:.*]]: i64
// CHECK: %[[idx0:.*]] = arith.index_cast %arg2 : i64 to index
// CHECK: %[[idx1:.*]] = arith.index_cast %arg3 : i64 to index
// CHECK: %[[affine0:.*]] = affine.min #{{.*}}{{\[}}%[[idx0]]]
// CHECK: %[[affine1:.*]] = affine.min #{{.*}}{{\[}}%[[idx1]]]
// CHECK: %[[tiled_reduce:.*]] = linalg.generic
// CHECK: tensor.insert_slice %[[tiled_reduce]] into {{.*}}{{\[}}0, 0] {{\[}}%[[affine0]], %[[affine1]]] {{\[}}1, 1]
func.func @test_tile_reduction_result_order(%arg0: tensor<16x256xf32>, %arg1: tensor<256xf32>, %arg2: i64, %arg3: i64) -> tensor<256xf32> {
  %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<16x256xf32>) outs(%arg1 : tensor<256xf32>) dimensions = [0]
  return %reduced : tensor<256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.func.get_func_argument %0[2] : (!transform.any_op) -> !transform.any_value
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.func.get_func_argument %2[3] : (!transform.any_op) -> !transform.any_value
    %4 = transform.structured.match ops{["linalg.reduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %4 by tile_sizes = [%1, %3] : (!transform.any_op, !transform.any_value, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @reduction_tile(%arg0: tensor<16x32xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x32xf32>
  // CHECK: linalg.fill ins({{.*}}) outs(%[[EMPTY]] : tensor<2x32xf32>) -> tensor<2x32xf32>
  // CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} -> (tensor<2x32xf32>)
  // CHECK-NOT: scf.for
  %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<16x32xf32>) outs(%arg1 : tensor<f32>) dimensions = [0, 1]
  return %reduced : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [2, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
