// RUN: bishengir-opt -hivm-normalize-tmp-buffer-in-vf --split-input-file %s | FileCheck %s

// The slice written by the inner loop is promoted into a new iter_arg of the
// parent loop, which gets a matching tensor.insert_slice.
// CHECK-LABEL: func.func @vf_single(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x8xf32>, %[[ARG1:.*]]: tensor<4x8xf32>
// CHECK: %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ARG0]], %[[TMP:.*]] = %[[ARG1]])
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[TMP]]
// CHECK: %[[INNER:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[SLICE]])
// CHECK: tensor.insert_slice %[[INNER]] into %[[ACC]]
// CHECK: %[[NEW_TMP:.*]] = tensor.insert_slice %[[INNER]] into %[[TMP]]
// CHECK: scf.yield {{.*}}, %[[NEW_TMP]]
// CHECK: return %[[FOR]]#0
module {
  func.func @vf_single(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.0 : f32
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg0) -> tensor<4x8xf32> {
      %s = tensor.extract_slice %arg1[%i, 0] [1, 8] [1, 1] : tensor<4x8xf32> to tensor<1x8xf32>
      %f = scf.for %k = %c0 to %c8 step %c1 iter_args(%t = %s) -> tensor<1x8xf32> {
        %ins = tensor.insert %cst into %t[%c0, %k] : tensor<1x8xf32>
        scf.yield %ins : tensor<1x8xf32>
      }
      %o = tensor.insert_slice %f into %a[%i, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<4x8xf32>
      scf.yield %o : tensor<4x8xf32>
    }
    return %r : tensor<4x8xf32>
  }
}

// -----

// With a two-level loop nest the new iter_arg is threaded through every
// enclosing loop, one level per match.
// CHECK-LABEL: func.func @vf_nested(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x8xf32>, %[[ARG1:.*]]: tensor<4x8xf32>
// CHECK: %[[OUTER:.*]]:2 = scf.for {{.*}} iter_args(%[[OACC:.*]] = %[[ARG0]], %[[OTMP:.*]] = %[[ARG1]])
// CHECK: %[[MID:.*]]:2 = scf.for {{.*}} iter_args(%[[MACC:.*]] = %[[OACC]], %[[MTMP:.*]] = %[[OTMP]])
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[MTMP]]
// CHECK: %[[INNER:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[SLICE]])
// CHECK: tensor.insert_slice %[[INNER]] into %[[MACC]]
// CHECK: %[[NEW_TMP:.*]] = tensor.insert_slice %[[INNER]] into %[[MTMP]]
// CHECK: scf.yield {{.*}}, %[[NEW_TMP]]
// CHECK: scf.yield %[[MID]]#0, %[[MID]]#1
// CHECK: return %[[OUTER]]#0
module {
  func.func @vf_nested(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.0 : f32
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg0) -> tensor<4x8xf32> {
      %r2 = scf.for %j = %c0 to %c8 step %c1 iter_args(%b = %a) -> tensor<4x8xf32> {
        %s = tensor.extract_slice %arg1[%i, %j] [1, 1] [1, 1] : tensor<4x8xf32> to tensor<1x1xf32>
        %f = scf.for %k = %c0 to %c1 step %c1 iter_args(%t = %s) -> tensor<1x1xf32> {
          %ins = tensor.insert %cst into %t[%c0, %c0] : tensor<1x1xf32>
          scf.yield %ins : tensor<1x1xf32>
        }
        %o = tensor.insert_slice %f into %b[%i, %j] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x8xf32>
        scf.yield %o : tensor<4x8xf32>
      }
      scf.yield %r2 : tensor<4x8xf32>
    }
    return %r : tensor<4x8xf32>
  }
}

// -----

// Functions that are not vector functions are left untouched.
// CHECK-LABEL: func.func @not_vf(
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<4x8xf32>)
// CHECK-NOT: -> (tensor<4x8xf32>, tensor<4x8xf32>)
module {
  func.func @not_vf(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.0 : f32
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg0) -> tensor<4x8xf32> {
      %s = tensor.extract_slice %arg1[%i, 0] [1, 8] [1, 1] : tensor<4x8xf32> to tensor<1x8xf32>
      %f = scf.for %k = %c0 to %c8 step %c1 iter_args(%t = %s) -> tensor<1x8xf32> {
        %ins = tensor.insert %cst into %t[%c0, %k] : tensor<1x8xf32>
        scf.yield %ins : tensor<1x8xf32>
      }
      %o = tensor.insert_slice %f into %a[%i, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<4x8xf32>
      scf.yield %o : tensor<4x8xf32>
    }
    return %r : tensor<4x8xf32>
  }
}

// -----

// Non-tensor iter_args (here a vector) are never promoted.
// CHECK-LABEL: func.func @vf_vector_iter_arg(
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>)
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>)
// CHECK-NOT: -> (vector<8xf32>, vector<8xf32>)
module {
  func.func @vf_vector_iter_arg(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<1.0> : vector<8xf32>
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg0) -> vector<8xf32> {
      %f = scf.for %k = %c0 to %c4 step %c1 iter_args(%t = %arg1) -> vector<8xf32> {
        %add = arith.addf %t, %cst : vector<8xf32>
        scf.yield %add : vector<8xf32>
      }
      scf.yield %f : vector<8xf32>
    }
    return %r : vector<8xf32>
  }
}

// -----

// Without a tensor.extract_slice, the outer tensor itself becomes the new
// iter_arg of the parent loop and the inner loop result is yielded as is.
// CHECK-LABEL: func.func @vf_no_extract_slice(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x8xf32>, %[[ARG1:.*]]: tensor<4x8xf32>
// CHECK: %[[FOR:.*]]:2 = scf.for {{.*}} iter_args({{.*}} = %[[ARG0]], %[[TMP:.*]] = %[[ARG1]])
// CHECK: %[[INNER:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[TMP]])
// CHECK-NOT: tensor.insert_slice
// CHECK: scf.yield %[[INNER]], %[[INNER]]
// CHECK: return %[[FOR]]#0
module {
  func.func @vf_no_extract_slice(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.0 : f32
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg0) -> tensor<4x8xf32> {
      %f = scf.for %k = %c0 to %c4 step %c1 iter_args(%t = %arg1) -> tensor<4x8xf32> {
        %ins = tensor.insert %cst into %t[%i, %k] : tensor<4x8xf32>
        scf.yield %ins : tensor<4x8xf32>
      }
      scf.yield %f : tensor<4x8xf32>
    }
    return %r : tensor<4x8xf32>
  }
}
