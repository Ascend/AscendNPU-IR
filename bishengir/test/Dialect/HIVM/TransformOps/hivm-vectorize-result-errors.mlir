// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: expected only function operations
module attributes {transform.target_tag = "payload"} {
  func.func @non_function(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %vectorized = transform.hivm.vectorize %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: expected only function operations
module attributes {transform.target_tag = "payload"} {
  func.func @non_function(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %cleaned = transform.hivm.cleanup %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: is not marked hivm.vector_function
module attributes {transform.target_tag = "payload"} {
  func.func @plain(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: linalg operation left in vector function 'vf'
module attributes {transform.target_tag = "payload"} {
  func.func private @vf(%arg: tensor<8xf32>, %init: tensor<8xf32>)
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
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: vectorizable operation 'hivm.hir.vadd' left in vector function 'vf'
module attributes {transform.target_tag = "payload"} {
  func.func private @vf(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: HIVM operation 'hivm.hir.vconcat' must remain in the caller of vector function 'vf'
module attributes {transform.target_tag = "payload"} {
  func.func private @vf(%lhs: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                        %init: tensor<4x4xf32>) -> tensor<4x4xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = hivm.hir.vconcat dim(1)
        ins(%lhs, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}
