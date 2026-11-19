// RUN: bishengir-opt %s -transform-interpreter -split-input-file \
// RUN:     -allow-unregistered-dialect -verify-diagnostics | FileCheck %s


// expected-remark @below {{first}}
// expected-remark @below {{second}}
func.func @reentry() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @mark(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %first = transform.hivm.include @mark failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %first, "first" : !transform.any_op
    %second = transform.hivm.include @mark failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %second, "second" : !transform.any_op
    transform.yield
  }
}

// -----


// expected-remark @below {{from-left}}
// expected-remark @below {{from-right}}
func.func @diamond() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @leaf(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @left(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.hivm.include @leaf failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @right(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.hivm.include @leaf failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %from_left = transform.hivm.include @left failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %from_left, "from-left" : !transform.any_op
    %from_right = transform.hivm.include @right failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %from_right, "from-right" : !transform.any_op
    transform.yield
  }
}

// -----


// expected-remark @below {{forwarded}}
func.func @forward_op() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @inner(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @outer(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %fn = transform.hivm.include @inner failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.yield %fn : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fn = transform.hivm.include @outer failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %fn, "forwarded" : !transform.any_op
    transform.yield
  }
}

// -----

func.func @forward_value(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{addi result}}
  // expected-note @below {{value handle points to an op result #0}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @addi_result(%root: !transform.any_op {transform.readonly}) -> !transform.any_value {
    %addi = transform.structured.match ops{["arith.addi"]} in %root : (!transform.any_op) -> !transform.any_op
    %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
    transform.yield %result : !transform.any_value
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %result = transform.hivm.include @addi_result failures(propagate) (%root) : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %result, "addi result" : !transform.any_value
    transform.yield
  }
}

// -----


func.func @forward_param() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @echo(%root: !transform.any_op {transform.readonly},
                                 %count: !transform.param<i64> {transform.readonly}) -> !transform.param<i64> {
    transform.yield %count : !transform.param<i64>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fn = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %count = transform.num_associations %fn : (!transform.any_op) -> !transform.param<i64>
    %echoed = transform.hivm.include @echo failures(propagate) (%root, %count) : (!transform.any_op, !transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{1 : i64}}
    transform.debug.emit_param_as_remark %count : !transform.param<i64>
    // expected-remark @below {{1 : i64}}
    transform.debug.emit_param_as_remark %echoed : !transform.param<i64>
    transform.yield
  }
}

// -----


func.func @empty_selection() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @select_addi(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %none = transform.structured.match ops{["arith.addi"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %none : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %none = transform.hivm.include @select_addi failures(propagate) (%root) : (!transform.any_op) -> !transform.any_op
    %count = transform.num_associations %none : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{0 : i64}}
    transform.debug.emit_param_as_remark %count : !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @map_forall
// CHECK: hivm.hir.get_block_idx
// CHECK: "some_use"
func.func @map_forall() {
  scf.forall (%arg0) in (100) {
    "some_use"(%arg0) : (index) -> ()
  } {mapping = [#hivm.block]}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @map_blocks(%forall: !transform.any_op {transform.consumed}) {
    %idx = transform.hivm.map_forall_to_blocks %forall : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %forall = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.hivm.include @map_blocks failures(propagate) (%forall) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----


func.func @use_after_consume() {
  // expected-note @below {{ancestor payload op}}
  // expected-note @below {{nested payload op}}
  scf.forall (%arg0) in (100) {
    "some_use"(%arg0) : (index) -> ()
  } {mapping = [#hivm.block]}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @map_blocks(%forall: !transform.any_op {transform.consumed}) {
    %idx = transform.hivm.map_forall_to_blocks %forall : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // expected-note @below {{handle to invalidated ops}}
    %forall = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.hivm.include @map_blocks failures(propagate) (%forall) : (!transform.any_op) -> ()
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.debug.emit_remark_at %forall, "stale" : !transform.any_op
    transform.yield
  }
}
