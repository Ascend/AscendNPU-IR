// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" | FileCheck %s

// CHECK-LABEL: func.func @payload
// CHECK-NOT: arith.addi
module attributes {transform.target_tag = "payload"} {
func.func @payload() {
  %zero = arith.constant 0 : i32
  %sum = arith.addi %zero, %zero : i32
  return
}
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @canonicalize(
      %root: !transform.any_op {transform.readonly}
  ) {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}
  ) {
    transform.hivm.include @canonicalize failures(propagate) (%root)
        : (!transform.any_op) -> ()
    transform.yield
  }
}
