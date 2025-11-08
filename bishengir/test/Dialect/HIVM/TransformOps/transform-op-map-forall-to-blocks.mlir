// RUN: bishengir-opt -transform-interpreter -split-input-file \
// RUN:     -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: trivial_test
  func.func @trivial_test() {
    // CHECK: %[[BLOCK_X:.*]] = hivm.hir.get_block_idx -> i64
    // CHECK: %[[CAST_X:.*]] = arith.index_cast %[[BLOCK_X]] : i64 to index
    // CHECK: %[[DELINEAR:.*]] = affine.delinearize_index %[[CAST_X]]
    // CHECK: "some_use"(%[[DELINEAR]]
    // CHECK-NOT: scf.forall
    scf.forall (%arg0) in (100) {
      "some_use"(%arg0) : (index) -> ()
    } {mapping = [#hivm.block]}
    return
  }

  // CHECK-LABEL: __transform_main
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.hivm.map_forall_to_blocks %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @no_block_attr() {
    scf.forall (%arg0) in (100) {
      "some_use"(%arg0) : (index) -> ()
    }
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{scf.forall op requires a mapping attribute}}
    %1 = transform.hivm.map_forall_to_blocks %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @unkown_block_attr() {
    scf.forall (%arg0) in (100) {
      "some_use"(%arg0) : (index) -> ()
    } {mapping = [#gpu.block<x>]}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{only support hivm block/sub_block attr}}
    %1 = transform.hivm.map_forall_to_blocks %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @duplicate_block_attr() {
    scf.forall (%arg0, %arg1) in (100, 100) {
      "some_use"(%arg0, %arg1) : (index, index) -> ()
    } {mapping = [#hivm.block, #hivm.block]}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{duplicate attribute, cannot map different loops to the same mapping id}}
    %1 = transform.hivm.map_forall_to_blocks %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @non_bufferized_ir() {
    %cst = arith.constant dense<[1]> : tensor<1xi32>
    %out = tensor.empty() : tensor<2xi32>

    %result = scf.forall (%arg0) = (0) to (2) step (1)
      shared_outs (%tmp = %out) -> tensor<2xi32> {

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cst into %tmp[%arg0] [1] [1]
          : tensor<1xi32> into tensor<2xi32>
      }
    } {mapping = [#hivm.block]}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{only bufferized scf.forall can be mapped}}
    %1 = transform.hivm.map_forall_to_blocks %0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}
