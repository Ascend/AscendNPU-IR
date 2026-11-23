// RUN: bishengir-opt %s --normalize-tensor-ops --split-input-file | FileCheck %s

// Multiple slice consumers must retain the static shape of the accumulator.
// CHECK-LABEL: func.func @multiple_users_rank_four
// CHECK-NOT: tensor.cast
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} -> (tensor<2x3x1x17xf32>)
// CHECK: tensor.insert_slice {{.*}} into {{.*}} : tensor<1x1x1x17xf32> into tensor<2x3x1x17xf32>
// CHECK: %[[WIDTH:.*]] = arith.constant 17 : index
// CHECK: tensor.extract_slice %[[RESULT]]{{.*}} : tensor<2x3x1x17xf32> to tensor<1x1x1x17xf32>
// CHECK: tensor.extract_slice %[[RESULT]]{{.*}} : tensor<2x3x1x17xf32> to tensor<1x1x1x17xf32>
// CHECK: return {{.*}}, %[[WIDTH]] : tensor<1x1x1x17xf32>, tensor<1x1x1x17xf32>, index
func.func @multiple_users_rank_four(%init: tensor<2x3x1x17xf32>,
    %update: tensor<1x1x1x17xf32>)
    -> (tensor<1x1x1x17xf32>, tensor<1x1x1x17xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cast = tensor.cast %init : tensor<2x3x1x17xf32> to tensor<2x3x1x?xf32>
  %result = scf.for %i = %c0 to %c3 step %c1
      iter_args(%acc = %cast) -> tensor<2x3x1x?xf32> {
    %next = tensor.insert_slice %update into %acc[0, %i, 0, 0]
        [1, 1, 1, 17] [1, 1, 1, 1]
        : tensor<1x1x1x17xf32> into tensor<2x3x1x?xf32>
    scf.yield %next : tensor<2x3x1x?xf32>
  }
  %width = tensor.dim %result, %c3 : tensor<2x3x1x?xf32>
  %first = tensor.extract_slice %result[0, 0, 0, 0] [1, 1, 1, %width]
      [1, 1, 1, 1] : tensor<2x3x1x?xf32> to tensor<1x1x1x?xf32>
  %second = tensor.extract_slice %result[1, 0, 0, 0] [1, 1, 1, %width]
      [1, 1, 1, 1] : tensor<2x3x1x?xf32> to tensor<1x1x1x?xf32>
  %first_static = tensor.cast %first : tensor<1x1x1x?xf32> to tensor<1x1x1x17xf32>
  %second_static = tensor.cast %second : tensor<1x1x1x?xf32> to tensor<1x1x1x17xf32>
  return %first_static, %second_static, %width
      : tensor<1x1x1x17xf32>, tensor<1x1x1x17xf32>, index
}

// -----

// Refine only known dimensions, including non-first iter_args and nested loops.
// CHECK-LABEL: func.func @nested_partial_shape
// CHECK: scf.for {{.*}} -> (index, tensor<3x?xi16>)
// CHECK: scf.for {{.*}} -> (tensor<3x?xi16>)
// CHECK: tensor.insert {{.*}} : tensor<?x?xi16>
// CHECK: tensor.dim {{.*}} : tensor<3x?xi16>
func.func @nested_partial_shape(%init: tensor<3x?xi16>, %value: i16,
    %bound: index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cast = tensor.cast %init : tensor<3x?xi16> to tensor<?x?xi16>
  %result:2 = scf.for %i = %c0 to %bound step %c1
      iter_args(%count = %c0, %acc = %cast) -> (index, tensor<?x?xi16>) {
    %inner = scf.for %j = %c0 to %c3 step %c1
        iter_args(%inner_acc = %acc) -> tensor<?x?xi16> {
      %next = tensor.insert %value into %inner_acc[%j, %c0] : tensor<?x?xi16>
      scf.yield %next : tensor<?x?xi16>
    }
    %next_count = arith.addi %count, %c1 : index
    scf.yield %next_count, %inner : index, tensor<?x?xi16>
  }
  %height = tensor.dim %result#1, %c0 : tensor<?x?xi16>
  %width = tensor.dim %result#1, %c1 : tensor<?x?xi16>
  return %result#0, %height, %width : index, index, index
}

// -----

// A dynamically typed loop may change shape; extracting a smaller slice does.
// CHECK-LABEL: func.func @changing_shape
// CHECK: tensor.cast {{.*}} : tensor<8x17xf32> to tensor<?x17xf32>
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} -> (tensor<?x17xf32>)
// CHECK: tensor.extract_slice {{.*}} : tensor<?x17xf32> to tensor<1x17xf32>
// CHECK: scf.yield {{.*}} : tensor<?x17xf32>
// CHECK: tensor.dim %[[RESULT]]{{.*}} : tensor<?x17xf32>
func.func @changing_shape(%init: tensor<8x17xf32>, %bound: index)
    -> (tensor<?x17xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cast = tensor.cast %init : tensor<8x17xf32> to tensor<?x17xf32>
  %result = scf.for %i = %c0 to %bound step %c1
      iter_args(%acc = %cast) -> tensor<?x17xf32> {
    %slice = tensor.extract_slice %acc[0, 0] [1, 17] [1, 1]
        : tensor<?x17xf32> to tensor<1x17xf32>
    %next = tensor.cast %slice : tensor<1x17xf32> to tensor<?x17xf32>
    scf.yield %next : tensor<?x17xf32>
  }
  %height = tensor.dim %result, %c0 : tensor<?x17xf32>
  return %result, %height : tensor<?x17xf32>, index
}

// -----

// An unrelated iter_arg need not have the same runtime shape.
// CHECK-LABEL: func.func @swapped_iter_args
// CHECK: tensor.cast {{.*}} : tensor<3x7xi16> to tensor<?x?xi16>
// CHECK: scf.for {{.*}} -> (tensor<?x?xi16>, tensor<?x?xi16>)
// CHECK: scf.yield {{.*}} : tensor<?x?xi16>, tensor<?x?xi16>
func.func @swapped_iter_args(%init: tensor<3x7xi16>, %other: tensor<?x?xi16>,
    %bound: index) -> (tensor<?x?xi16>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cast = tensor.cast %init : tensor<3x7xi16> to tensor<?x?xi16>
  %result:2 = scf.for %i = %c0 to %bound step %c1
      iter_args(%first = %cast, %second = %other)
      -> (tensor<?x?xi16>, tensor<?x?xi16>) {
    scf.yield %second, %first : tensor<?x?xi16>, tensor<?x?xi16>
  }
  %height = tensor.dim %result#0, %c0 : tensor<?x?xi16>
  return %result#0, %height : tensor<?x?xi16>, index
}
