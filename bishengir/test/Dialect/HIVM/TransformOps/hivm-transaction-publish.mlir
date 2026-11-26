// RUN: (bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file || true) | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s --check-prefix=STALE


// CHECK-LABEL: module attributes {{.*published.*}}
// CHECK-NOT: arith.addi
// CHECK: func.func @publish
module attributes {transform.target_tag = "payload"} {
  func.func @publish() {
    %c0 = arith.constant 0 : i32
    %sum = arith.addi %c0, %c0 : i32
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_publish(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.annotate %root "hivm.published" : !transform.any_op
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @cost(
      %root:     !transform.any_op     {transform.readonly},
      %result:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.yield %result : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_publish(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    %fn = transform.structured.match ops{["func.func"]} in %new_root
        : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %fn, "new root is usable" : !transform.any_op
    transform.yield
  }
}

// -----


// STALE-LABEL: module attributes {{.*live_changed.*}}
// STALE-NOT: published
// STALE: func.func @stale
module attributes {transform.target_tag = "payload"} {
  func.func @stale() {
    %c0 = arith.constant 0 : i32
    %sum = arith.addi %c0, %c0 : i32
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_stale(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.annotate %root "hivm.published" : !transform.any_op
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @cost(
      %root:     !transform.any_op     {transform.readonly},
      %result:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.yield %result : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_stale(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    transform.annotate %root "hivm.live_changed" : !transform.any_op
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.yield
  }
}

// -----


// CHECK-LABEL: module attributes {transform.target_tag = "payload"}
// CHECK-NOT: hivm.attempt_mutated
// CHECK: func.func @isolated
module attributes {transform.target_tag = "payload"} {
  func.func @isolated(%arg0: index, %arg1: index) -> index {
    %sum = arith.addi %arg0, %arg1 : index
    return %sum : index
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_mutate_then_fail(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.annotate %root "hivm.attempt_mutated" : !transform.any_op
    %addi = transform.structured.match ops{["arith.addi"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %missing = transform.get_result %addi[1]
        : (!transform.any_op) -> !transform.any_value
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @cost(
      %root:     !transform.any_op     {transform.readonly},
      %result:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.yield %result : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_mutate_then_fail(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.yield
  }
}

// -----


// STALE: invalidated by this transform op that consumes its operand #0
module attributes {transform.target_tag = "payload"} {
  func.func @old_handle() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_old(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @cost(
      %root:     !transform.any_op     {transform.readonly},
      %result:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.yield %result : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_old(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_remark_at %fn, "stale" : !transform.any_op
    transform.yield
  }
}
