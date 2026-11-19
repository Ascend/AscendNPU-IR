// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s



// CHECK: remark: selected transaction ticket 1 with cost 3.5
// CHECK: remark: result #hivm_vectorize.transaction_result<session = {{[0-9]+}}, ticket = 1, outcome = succeeded, options = {}, result = 3.5
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = true, ticket = 1, cost = 3.5
// CHECK-SAME: verdicts =
// CHECK-SAME: transaction_verdict<ticket = 1, state = applied, cost = 3.5
// CHECK: remark: new root
module attributes {transform.target_tag = "payload"} {
  func.func @happy() {
    %c0 = arith.constant 0 : i32
    %sum = arith.addi %c0, %c0 : i32
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_happy(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %score = transform.param.constant 3.5 : f64 -> !transform.any_param
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
    %ticket = transform.hivm.transaction @attempt_happy(%root, %options) {sync}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %result, "result" : !transform.any_param
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.debug.emit_remark_at %new_root, "new root" : !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: remark: selected transaction ticket 2 with cost 9
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = true, ticket = 2, cost = 9
// CHECK-SAME: transaction_verdict<ticket = 1, state = eligible, cost = 2
// CHECK-SAME: transaction_verdict<ticket = 2, state = applied, cost = 9
// CHECK-SAME: transaction_verdict<ticket = 3, state = eligible, cost = 5
module attributes {transform.target_tag = "payload"} {
  func.func @ranking() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @score_two(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 2.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @score_nine(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 9.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @score_five(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 5.0 : f64 -> !transform.any_param
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
    %t0 = transform.hivm.transaction @score_two(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %t1 = transform.hivm.transaction @score_nine(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %t2 = transform.hivm.transaction @score_five(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %r0 = transform.hivm.transaction.await %t0
        : (!transform.any_param) -> !transform.any_param
    %r1 = transform.hivm.transaction.await %t1
        : (!transform.any_param) -> !transform.any_param
    %r2 = transform.hivm.transaction.await %t2
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %r0, %r1, %r2)
        : (!transform.any_op, !transform.any_param, !transform.any_param,
           !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: selected transaction ticket 1 with cost 4
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = true, ticket = 1, cost = 4
// CHECK-SAME: transaction_verdict<ticket = 1, state = applied, cost = 4
module attributes {transform.target_tag = "payload"} {
  func.func @tie() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @score_four_a(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 4.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @score_four_b(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 4.0 : f64 -> !transform.any_param
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
    %t0 = transform.hivm.transaction @score_four_a(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %t1 = transform.hivm.transaction @score_four_b(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %r0 = transform.hivm.transaction.await %t0
        : (!transform.any_param) -> !transform.any_param
    %r1 = transform.hivm.transaction.await %t1
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %r0, %r1)
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: no transaction candidate published
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = false,
// CHECK-SAME: transaction_verdict<ticket = 1, state = rejected
module attributes {transform.target_tag = "payload"} {
  func.func @rejected(%arg0: index, %arg1: index) -> index {
    %sum = arith.addi %arg0, %arg1 : index
    return %sum : index
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_rejected(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
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
    %ticket = transform.hivm.transaction @attempt_rejected(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: warning: transaction ticket 1 failed:
// CHECK: remark: no transaction candidate published
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = false,
// CHECK-SAME: transaction_verdict<ticket = 1, state = error
module attributes {transform.target_tag = "payload"} {
  func.func @attempt_error(%arg0: index) -> index {
    return %arg0 : index
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @fail_map(
      %target: !transform.any_op {transform.consumed}
  ) {
    %idx = transform.hivm.map_forall_to_blocks %target
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @attempt_error(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.hivm.include @fail_map failures(propagate) (%fn)
        : (!transform.any_op) -> ()
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
    %ticket = transform.hivm.transaction @attempt_error(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: no transaction candidate published
// CHECK: remark: selection #hivm_vectorize.transaction_selection<applied = false,
module attributes {transform.target_tag = "payload"} {
  func.func @empty_selection() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @cost(
      %root:     !transform.any_op     {transform.readonly},
      %result:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    transform.yield %result : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %new_root, %selection = transform.hivm.transaction.select @cost(%root)
        : (!transform.any_op) -> (!transform.any_op, !transform.any_param)
    transform.debug.emit_param_as_remark %selection, "selection" : !transform.any_param
    transform.yield
  }
}
