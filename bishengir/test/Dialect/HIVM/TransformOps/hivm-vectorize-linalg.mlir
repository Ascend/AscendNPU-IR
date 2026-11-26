// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK-LABEL: func.func private @unary
// CHECK: vector.transfer_read
// CHECK: math.exp2
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @unary(%arg: tensor<8xf32>, %init: tensor<8xf32>)
      -> tensor<8xf32> attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%arg : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp2 %in : f32
      linalg.yield %1 : f32
    } -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @chain
// CHECK-COUNT-1: scf.for
// CHECK: vector.transfer_read
// CHECK: arith.addf
// CHECK: math.exp2
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @chain(%lhs: tensor<8x16xf32>, %rhs: tensor<8x16xf32>,
                           %init: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<8x16xf32>)
        outs(%init : tensor<8x16xf32>) {
    ^bb0(%lhsIn: f32, %rhsIn: f32, %out: f32):
      %1 = arith.addf %lhsIn, %rhsIn : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
    %2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = math.exp2 %in : f32
      linalg.yield %3 : f32
    } -> tensor<8x16xf32>
    return %2 : tensor<8x16xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @broadcast
// CHECK: vector.transfer_read {{.*}}permutation_map = #{{.*}}
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @broadcast(%lhs: tensor<8x1xf32>, %rhs: tensor<8x16xf32>,
                               %init: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<8x1xf32>, tensor<8x16xf32>)
        outs(%init : tensor<8x16xf32>) {
    ^bb0(%lhsIn: f32, %rhsIn: f32, %out: f32):
      %1 = arith.addf %lhsIn, %rhsIn : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @transpose
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @transpose(%arg: tensor<4x8xf32>, %init: tensor<8x4xf32>)
      -> tensor<8x4xf32> attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg : tensor<4x8xf32>) outs(%init : tensor<8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp2 %in : f32
      linalg.yield %1 : f32
    } -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @mulext
// CHECK: arith.mulsi_extended
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @mulext(%lhs: tensor<8xi32>, %rhs: tensor<8xi32>,
                            %low: tensor<8xi32>, %high: tensor<8xi32>)
      -> (tensor<8xi32>, tensor<8xi32>)
      attributes {hivm.vector_function, no_inline} {
    %0:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%low, %high : tensor<8xi32>, tensor<8xi32>) {
    ^bb0(%lhsIn: i32, %rhsIn: i32, %lowOut: i32, %highOut: i32):
      %1:2 = arith.mulsi_extended %lhsIn, %rhsIn : i32
      linalg.yield %1#0, %1#1 : i32, i32
    } -> (tensor<8xi32>, tensor<8xi32>)
    return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @predicates
// CHECK: arith.cmpf
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @predicates(%arg: tensor<8xf32>, %scalar: f32,
                                %init: tensor<8xi1>) -> tensor<8xi1>
      attributes {hivm.vector_function, no_inline} {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> ()>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%arg, %scalar : tensor<8xf32>, f32)
        outs(%init : tensor<8xi1>) {
    ^bb0(%in: f32, %scalarIn: f32, %out: i1):
      %inf = arith.constant 0x7F800000 : f32
      %abs = math.absf %in : f32
      %cmp = arith.cmpf oeq, %abs, %inf : f32
      linalg.yield %cmp : i1
    } -> tensor<8xi1>
    return %0 : tensor<8xi1>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @mixed
// CHECK: hivm.hir.vconcat
// CHECK-NOT: linalg.
module attributes {transform.target_tag = "payload"} {
  func.func private @mixed(%lhs: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                           %init: tensor<4x2xf32>,
                           %concatRhs: tensor<4x2xf32>,
                           %concatInit: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x2xf32>) {
    ^bb0(%lhsIn: f32, %rhsIn: f32, %out: f32):
      %1 = arith.addf %lhsIn, %rhsIn : f32
      linalg.yield %1 : f32
    } -> tensor<4x2xf32>
    %2 = hivm.hir.vconcat dim(1)
        ins(%0, %concatRhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%concatInit : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @empty
// CHECK: return %arg0 : tensor<8xf32>
// CHECK-NOT: vector.
module attributes {transform.target_tag = "payload"} {
  func.func @empty(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match attributes {not_a_vector_function} in %root
        : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.hivm.vectorize %funcs
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %vectorized
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
