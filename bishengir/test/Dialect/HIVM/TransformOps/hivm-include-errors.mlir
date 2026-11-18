// RUN: split-file %s %t
// RUN: not bishengir-opt %t/operand-arity.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=OPERAND-ARITY
// RUN: not bishengir-opt %t/result-arity.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=RESULT-ARITY
// RUN: not bishengir-opt %t/operand-kind.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=OPERAND-KIND
// RUN: not bishengir-opt %t/result-kind.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=RESULT-KIND
// RUN: not bishengir-opt %t/missing-target.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=MISSING-TARGET
// RUN: not bishengir-opt %t/extra-operand.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=EXTRA-OPERAND
// RUN: not bishengir-opt %t/non-transform-body.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=NON-TRANSFORM
// RUN: not bishengir-opt %t/self-recursion.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=RECURSION
// RUN: not bishengir-opt %t/mutual-recursion.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=RECURSION
// RUN: not bishengir-opt %t/silenceable.mlir -transform-interpreter 2>&1 | FileCheck %s --check-prefix=SILENCEABLE
// RUN: not bishengir-opt %t/definite.mlir -transform-interpreter -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=DEFINITE


// OPERAND-ARITY: number of operands (1) does not match the number of inputs (2)
// RESULT-ARITY: number of results (1) does not match the number of outputs (0)
// OPERAND-KIND: operand #0 does not match the expected type
// RESULT-KIND: result #0 does not match the expected type
// MISSING-TARGET: target symbol '@missing' not found
// EXTRA-OPERAND: number of operands (2) does not match the number of inputs (1)
// NON-TRANSFORM: expected an executable Transform operation
// RECURSION: recursive HIVM include
// SILENCEABLE: position overflow 1 (updated from 1) for maximum 1
// SILENCEABLE: while considering positions of this payload operation
// DEFINITE: expect an scf.forall

//--- operand-arity.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @needs_two(%a: !transform.any_op {transform.readonly},
                                      %b: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @needs_two failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- result-arity.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @no_results(%root: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %unused = transform.hivm.include @no_results failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//--- operand-kind.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @needs_param(%count: !transform.param<i64> {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @needs_param failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- result-kind.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @returns_op(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %unused = transform.hivm.include @returns_op failures(propagate) (%root) : (!transform.any_op) -> !transform.param<i64>
    transform.yield
  }
}

//--- missing-target.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @missing failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- extra-operand.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @one(%root: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @one failures(propagate) (%root, %root) : (!transform.any_op, !transform.any_op) -> ()
    transform.yield
  }
}

//--- non-transform-body.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @helper(%root: !transform.any_op {transform.readonly}) {
    arith.constant 0 : i32
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @helper failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- self-recursion.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @recurse(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @recurse failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @recurse failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- mutual-recursion.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @ping(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @pong failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @pong(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @ping failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @ping failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- silenceable.mlir

func.func @one_result(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @out_of_range(%root: !transform.any_op {transform.readonly}) {
    %addi = transform.structured.match ops{["arith.addi"]} in %root : (!transform.any_op) -> !transform.any_op
    %missing = transform.get_result %addi[1] : (!transform.any_op) -> !transform.any_value
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.hivm.include @out_of_range failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}

//--- definite.mlir

func.func @not_a_forall() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @map_func(%target: !transform.any_op {transform.consumed}) {
    %idx = transform.hivm.map_forall_to_blocks %target : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.hivm.include @map_func failures(propagate) (%fn) : (!transform.any_op) -> ()
    transform.yield
  }
}
