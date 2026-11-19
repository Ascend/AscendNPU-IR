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
    %lifted = transform.hivm.lift_to_linalg %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: cannot lift 'hivm.hir.vadd': requires tensor results
module attributes {transform.target_tag = "payload"} {
  func.func @memref(%lhs: memref<8xf32>, %rhs: memref<8xf32>,
                    %init: memref<8xf32>) {
    hivm.hir.vadd ins(%lhs, %rhs : memref<8xf32>, memref<8xf32>)
        outs(%init : memref<8xf32>)
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: cannot lift 'hivm.hir.vadd': broadcast dim must be 1 or match the output
module attributes {transform.target_tag = "payload"} {
  func.func @broadcast(%lhs: tensor<8x2xf32>, %rhs: tensor<8x8xf32>,
                       %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x2xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) broadcast = [1] -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: cannot lift 'hivm.hir.vadd': transpose shapes do not match
module attributes {transform.target_tag = "payload"} {
  func.func @transpose(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>,
                       %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x4xf32>) transpose = [1, 0] -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: cannot lift 'hivm.hir.vadd': unexpected extra operands
module attributes {transform.target_tag = "payload"} {
  func.func @temp(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                  %init: tensor<8xf32>, %temp: memref<8xf32>)
      -> tensor<8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) temp_buffer(%temp : memref<8xf32>)
        -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
