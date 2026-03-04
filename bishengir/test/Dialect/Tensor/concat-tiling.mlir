// RUN: bishengir-opt -transform-interpreter -cse -split-input-file %s | FileCheck %s


// CHECK-LABEL: func @tile_non_concat_dim_0(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<8x32xf32>,
//  CHECK-SAME:     %[[ARG1:.*]]: tensor<8x16xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C8]] step %[[C2]]
//       CHECK:     %[[C48:.*]] = arith.constant 48 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[C48]] step %[[C48]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SLICE0:.*]] = tensor.extract_slice %[[ARG0]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x32xf32> to tensor<?x?xf32>
//       CHECK:       %[[SLICE1:.*]] = tensor.extract_slice %[[ARG1]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x16xf32> to tensor<?x?xf32>
//       CHECK:       %[[CONCAT:.*]] = tensor.concat dim(1) %[[SLICE0]], %[[SLICE1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<2x48xf32>
//       CHECK:       tensor.insert_slice %[[CONCAT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]
func.func @tile_non_concat_dim_0(%arg0: tensor<8x32xf32>,
                                 %arg1: tensor<8x16xf32>) -> tensor<8x48xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<8x32xf32>, tensor<8x16xf32>) -> tensor<8x48xf32>
  return %0 : tensor<8x48xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %concat tile_sizes [2, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_non_concat_dim_1(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<32x8xf32>,
//  CHECK-SAME:     %[[ARG1:.*]]: tensor<16x8xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[C48:.*]] = arith.constant 48 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C48]] step %[[C48]]
//   CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[C8]] step %[[C2]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SLICE0:.*]] = tensor.extract_slice %[[ARG0]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<32x8xf32> to tensor<?x?xf32>
//       CHECK:       %[[SLICE1:.*]] = tensor.extract_slice %[[ARG1]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<16x8xf32> to tensor<?x?xf32>
//       CHECK:       %[[CONCAT:.*]] = tensor.concat dim(0) %[[SLICE0]], %[[SLICE1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<48x2xf32>
//       CHECK:       tensor.insert_slice %[[CONCAT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]
func.func @tile_non_concat_dim_1(%arg0: tensor<32x8xf32>,
                                 %arg1: tensor<16x8xf32>) -> tensor<48x8xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<32x8xf32>, tensor<16x8xf32>) -> tensor<48x8xf32>
  return %0 : tensor<48x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %concat tile_sizes [48, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @test_tile_concat_axis_1(
//   CHECK-SAME: %[[ARG0:.*]]: tensor<136x2048xf32>, %[[ARG1:.*]]: tensor<136x2048xf32>, %[[ARG2:.*]]: tensor<136x2048xf32>, %[[ARG3:.*]]: tensor<136x2048xf32>, %[[ARG4:.*]]: tensor<136x8192xf32>)
//       CHECK:   %[[ELEMWISE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]]
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C136:.*]] = arith.constant 136 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C136]] step %[[C1]]
//   CHECK-DAG:   %[[C8192:.*]] = arith.constant 8192 : index
//   CHECK-DAG:   %[[C256:.*]] = arith.constant 256 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[C8192]] step %[[C256]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SLICE0:.*]] = tensor.extract_slice %[[ELEMWISE]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<136x2048xf32> to tensor<1x?xf32>
//       CHECK:       %[[SLICE1:.*]] = tensor.extract_slice %[[ARG1]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<136x2048xf32> to tensor<1x?xf32>
//       CHECK:       %[[SLICE2:.*]] = tensor.extract_slice %[[ARG2]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<136x2048xf32> to tensor<1x?xf32>
//       CHECK:       %[[SLICE3:.*]] = tensor.extract_slice %[[ARG3]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<136x2048xf32> to tensor<1x?xf32>
//       CHECK:       %[[CONCAT:.*]] = tensor.concat dim(1) %[[SLICE0]], %[[SLICE1]], %[[SLICE2]], %[[SLICE3]]
//       CHECK:       tensor.insert_slice %[[CONCAT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<1x256xf32> into tensor<136x8192xf32>
//       CHECK:   return %[[RESULT]]
func.func @test_tile_concat_axis_1(%arg0: tensor<136x2048xf32>, %arg1: tensor<136x2048xf32>, %arg2: tensor<136x2048xf32>, 
                                   %arg3: tensor<136x2048xf32>, %arg4: tensor<136x8192xf32>) -> (tensor<136x8192xf32>, tensor<136x8192xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<136x2048xf32>
  %1 = tensor.empty() : tensor<136x8192xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<136x2048xf32>) -> tensor<136x2048xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %2 : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%0 : tensor<136x2048xf32>) -> tensor<136x2048xf32>
  %concat = tensor.concat dim(1) %3, %arg1, %arg2, %arg3 : (tensor<136x2048xf32>, tensor<136x2048xf32>, tensor<136x2048xf32>, tensor<136x2048xf32>) -> tensor<136x8192xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%concat, %arg4 : tensor<136x8192xf32>, tensor<136x8192xf32>) outs(%1 : tensor<136x8192xf32>) -> tensor<136x8192xf32>
  return %concat, %4 : tensor<136x8192xf32>, tensor<136x8192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %concat tile_sizes [1, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @tile_dyn_concat_dim_0(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<8x?xf32>, %[[ARG1:.*]]: tensor<8x?xf32>)
//       CHECK:   %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[DIM1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[ADD:.*]] = affine.apply #map(){{\[}}%[[DIM0]], %[[DIM1]]]
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C8]] step %[[C2]]
//   CHECK-DAG:   %[[C48:.*]] = arith.constant 48 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[ADD]] step %[[C48]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SLICE0:.*]] = tensor.extract_slice %[[ARG0]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[SLICE1:.*]] = tensor.extract_slice %[[ARG1]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[CONCAT:.*]] = tensor.concat dim(1) %[[SLICE0]], %[[SLICE1]]
//       CHECK:       tensor.insert_slice %[[CONCAT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<2x?xf32> into tensor<8x?xf32>
//       CHECK:   return %[[RESULT]]
func.func @tile_dyn_concat_dim_0(%arg0: tensor<8x?xf32>,
                                 %arg1: tensor<8x?xf32>) -> tensor<8x?xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<8x?xf32>, tensor<8x?xf32>) -> tensor<8x?xf32>
  return %0 : tensor<8x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %concat tile_sizes [2, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @tile_dyn_non_concat_dim_0(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<8x?xf32>, %[[ARG1:.*]]: tensor<8x?xf32>)
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C2]]
//   CHECK-DAG:   %[[C48:.*]] = arith.constant 48 : index
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[DIM]] step %[[C48]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SLICE0:.*]] = tensor.extract_slice %[[ARG0]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[SLICE1:.*]] = tensor.extract_slice %[[ARG1]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<8x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[CONCAT:.*]] = tensor.concat dim(0) %[[SLICE0]], %[[SLICE1]]
//       CHECK:       tensor.insert_slice %[[CONCAT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1] : tensor<2x?xf32> into tensor<16x?xf32>
//       CHECK:   return %[[RESULT]]
func.func @tile_dyn_non_concat_dim_0(%arg0: tensor<8x?xf32>,
                                 %arg1: tensor<8x?xf32>) -> tensor<16x?xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<8x?xf32>, tensor<8x?xf32>) -> tensor<16x?xf32>
  return %0 : tensor<16x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %concat tile_sizes [2, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}