// RUN: bishengir-opt %s -create-preload -allow-unregistered-dialect -split-input-file | FileCheck %s

// A private counter, including its nested loop, moves into its only consumer.
// CHECK-LABEL: func.func @private_counter
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[BUFFER:.*]] = memref.alloca
// CHECK-NEXT: memref.store
// CHECK-NEXT: scf.for
// CHECK: memref.load %[[BUFFER]]
// CHECK: memref.store {{.*}}, %[[BUFFER]]
// CHECK: memref.load %[[BUFFER]]
// CHECK-NOT: test.source
// CHECK: test.consumer
func.func @private_counter(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<index> {
      %buffer = memref.alloca() {normalize_matmul_counter} : memref<index>
      memref.store %c0, %buffer[] : memref<index>
      scf.for %j = %c0 to %n step %c1 {
        %old = memref.load %buffer[] : memref<index>
        %next = arith.addi %old, %c1 : index
        memref.store %next, %buffer[] : memref<index>
      }
      scope.return %buffer : memref<index>
    } {test.source, no_inline}
    scope.scope : () -> () {
      %value = memref.load %counter[] : memref<index>
      "test.consume"(%value) : (index) -> ()
      scope.return
    } {test.consumer, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

// Clone only the counter update into each consumer, dropping unrelated loop
// results, users of the counter load, and accesses to another buffer.
// The source scope remains because it contains unrelated side effects.
// CHECK-LABEL: func.func @filter_counter_loop
// CHECK: scf.for
// CHECK: "test.work"
// CHECK: test.source
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[COUNTER:.*]] = memref.alloca
// CHECK-NEXT: memref.store {{.*}}, %[[COUNTER]][]
// CHECK-NEXT: scf.for {{.*}} {
// CHECK-NEXT: %[[OLD:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[OLD]],
// CHECK-NEXT: memref.store %[[NEXT]], %[[COUNTER]][]
// CHECK-NEXT: }
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: "test.consume"(%[[VALUE]])
// CHECK: test.consumer_one
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[COUNTER2:.*]] = memref.alloca
// CHECK-NEXT: memref.store {{.*}}, %[[COUNTER2]][]
// CHECK-NEXT: scf.for {{.*}} {
// CHECK-NEXT: %[[OLD2:.*]] = memref.load %[[COUNTER2]][]
// CHECK-NEXT: %[[NEXT2:.*]] = arith.addi %[[OLD2]],
// CHECK-NEXT: memref.store %[[NEXT2]], %[[COUNTER2]][]
// CHECK-NEXT: }
// CHECK-NEXT: %[[VALUE2:.*]] = memref.load %[[COUNTER2]][]
// CHECK-NEXT: "test.consume"(%[[VALUE2]])
// CHECK: test.consumer_two
func.func @filter_counter_loop(%n: index, %other: memref<index>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<index> {
      %buffer = memref.alloca() {normalize_matmul_counter} : memref<index>
      memref.store %c0, %buffer[] : memref<index>
      %unused = scf.for %j = %c0 to %n step %c1 iter_args(%arg = %c0) -> index {
        %old = memref.load %buffer[] : memref<index>
        %unrelated = "test.work"(%arg, %old) : (index, index) -> index
        %other_old = memref.load %other[] : memref<index>
        %other_next = arith.addi %other_old, %c1 : index
        memref.store %other_next, %other[] : memref<index>
        %next = arith.addi %old, %c1 : index
        memref.store %next, %buffer[] : memref<index>
        scf.yield %unrelated : index
      }
      scope.return %buffer : memref<index>
    } {no_inline, test.source}
    scope.scope : () -> () {
      %value = memref.load %counter[] : memref<index>
      "test.consume"(%value) : (index) -> ()
      scope.return
    } {no_inline, test.consumer_one}
    scope.scope : () -> () {
      %value = memref.load %counter[] : memref<index>
      "test.consume"(%value) : (index) -> ()
      scope.return
    } {no_inline, test.consumer_two}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Retain nested loop bounds and remap induction variables while filtering both
// loop bodies. Dependencies outside the loops must also be cloned.
// Check the consumer after the source, which retains the unknown side effects.
// CHECK-LABEL: func.func @nested_counter_loop
// CHECK: scf.for
// CHECK: "test.outer_work"
// CHECK: "test.inner_work"
// CHECK: test.source
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %[[COUNTER:.*]] = memref.alloca
// CHECK-NEXT: memref.store {{.*}}, %[[COUNTER]][]
// CHECK-NEXT: %[[BOUND:.*]] = arith.addi
// CHECK-NEXT: scf.for %[[J:.*]] = {{.*}} to %[[BOUND]] {{.*}} {
// CHECK-NEXT: scf.for {{.*}} to %[[J]] {{.*}} {
// CHECK-NEXT: %[[OLD:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[OLD]],
// CHECK-NEXT: memref.store %[[NEXT]], %[[COUNTER]][]
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: "test.consume"(%[[VALUE]])
// CHECK: test.consumer
func.func @nested_counter_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<index> {
      %buffer = memref.alloca() {normalize_matmul_counter} : memref<index>
      memref.store %c0, %buffer[] : memref<index>
      %bound = arith.addi %n, %c1 : index
      scf.for %j = %c0 to %bound step %c1 {
        "test.outer_work"(%j) : (index) -> ()
        scf.for %k = %c0 to %j step %c1 {
          %old = memref.load %buffer[] : memref<index>
          "test.inner_work"(%old) : (index) -> ()
          %next = arith.addi %old, %c1 : index
          memref.store %next, %buffer[] : memref<index>
        }
      }
      scope.return %buffer : memref<index>
    } {no_inline, test.source}
    scope.scope : () -> () {
      %value = memref.load %counter[] : memref<index>
      "test.consume"(%value) : (index) -> ()
      scope.return
    } {no_inline, test.consumer}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Do not drop a loop-carried value that is needed by the counter update.
// CHECK-LABEL: func.func @counter_uses_iter_arg
// CHECK: %[[COUNTER:.*]] = scope.scope
// CHECK: scf.for {{.*}} iter_args
// CHECK: test.source
// CHECK: scope.scope : () -> () {
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: "test.consume"(%[[VALUE]])
func.func @counter_uses_iter_arg(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<index> {
      %buffer = memref.alloca() {normalize_matmul_counter} : memref<index>
      memref.store %c0, %buffer[] : memref<index>
      %unused = scf.for %j = %c0 to %n step %c1 iter_args(%arg = %c1) -> index {
        %old = memref.load %buffer[] : memref<index>
        %next = arith.addi %old, %arg : index
        memref.store %next, %buffer[] : memref<index>
        %next_arg = arith.addi %arg, %c1 : index
        scf.yield %next_arg : index
      }
      scope.return %buffer : memref<index>
    } {no_inline, test.source}
    scope.scope : () -> () {
      %value = memref.load %counter[] : memref<index>
      "test.consume"(%value) : (index) -> ()
      scope.return
    } {no_inline, test.consumer}
  } {hivm.cv_pipelined_loop}
  return
}
