// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s

// CHECK: attempt must have type
module attributes {transform.target_tag = "payload"} {
  func.func @bad_attempt_signature() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @bad(
      %root: !transform.any_op {transform.readonly}
  ) {
    transform.yield
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @bad(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: transaction-family operation is not reachable from a transaction callback
module attributes {transform.target_tag = "payload"} {
  func.func @nested_transaction() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_nested(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %nested = transform.hivm.transaction.await %options
        : (!transform.any_param) -> !transform.any_param
    transform.yield %options : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_nested(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: expected a payload module root
module attributes {transform.target_tag = "payload"} {
  func.func @non_module_root() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt(%fn, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: transaction library must be separate from the payload module
module attributes {transform.with_named_sequence, transform.target_tag = "payload"} {
  func.func @payload_inside_library() {
    return
  }

  transform.named_sequence @attempt_same(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt_same(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: transaction ticket belongs to another session
module attributes {transform.target_tag = "payload"} {
  func.func @foreign_ticket() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %forged = transform.param.constant #hivm_vectorize.transaction_ticket<session = 999, ticket = 1>
        -> !transform.any_param
    %result = transform.hivm.transaction.await %forged
        : (!transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: transaction ticket has been claimed
module attributes {transform.target_tag = "payload"} {
  func.func @claimed_ticket() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
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
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result)
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    %again = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: duplicate transaction result
module attributes {transform.target_tag = "payload"} {
  func.func @duplicate_results() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
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
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result, %result)
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.yield
  }
}

// -----

// CHECK: cost callback must yield a FloatAttr or an IntegerAttr
module attributes {transform.target_tag = "payload"} {
  func.func @non_numeric_cost() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt_text(
      %root:      !transform.any_op     {transform.readonly},
      %options:   !transform.any_param  {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant "not a number" -> !transform.any_param
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
    %ticket = transform.hivm.transaction @attempt_text(%root, %options)
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

// CHECK: expected a transaction result
module attributes {transform.target_tag = "payload"} {
  func.func @malformed_result() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @attempt(
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
    %options = transform.param.constant {} -> !transform.any_param
    %ticket = transform.hivm.transaction @attempt(%root, %options)
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %result = transform.hivm.transaction.await %ticket
        : (!transform.any_param) -> !transform.any_param
    %new_root, %selection = transform.hivm.transaction.select @cost(%root, %result, %options)
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> (!transform.any_op, !transform.any_param)
    transform.yield
  }
}

// -----

// CHECK: cost must have type
module attributes {transform.target_tag = "payload"} {
  func.func @bad_cost_signature() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @bad_cost(
      %root: !transform.any_op {transform.readonly}
  ) -> !transform.any_param {
    %score = transform.param.constant 1.0 : f64 -> !transform.any_param
    transform.yield %score : !transform.any_param
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %new_root, %selection = transform.hivm.transaction.select @bad_cost(%root)
        : (!transform.any_op) -> (!transform.any_op, !transform.any_param)
    transform.yield
  }
}
