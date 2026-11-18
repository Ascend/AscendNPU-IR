// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   2>&1 | FileCheck %s

// The lift stage translates supported HIVM operations into linalg plus scalar
// arithmetic and math. Elementwise arithmetic and unary math are covered here.

module attributes {transform.target_tag = "payload"} {
  func.func @vf(%a: tensor<8xf32>, %b: tensor<8xf32>,
                %init: tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.vector_function} {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %abs = hivm.hir.vabs ins(%sum : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %abs : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %vfs = transform.structured.match ops{["func.func"]}
        attributes {hivm.vector_function} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %vfs
        : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %lifted, "lifted" : !transform.any_op
    transform.yield
  }
}

// CHECK: remark: lifted
// CHECK-LABEL: func.func @vf(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>,
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: math.absf
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: hivm.hir.vabs
