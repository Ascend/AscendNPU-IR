// RUN: bishengir-opt %s -create-preload -allow-unregistered-dialect -split-input-file | FileCheck %s

// A counter used only by an empty annotation is not live across stages.
// The real work must survive without exporting the private alloca.
// CHECK-LABEL: func.func @annotated_private_counter
// CHECK: scf.for
// CHECK: "test.work"
// CHECK-NOT: annotation.mark {{.*}} : memref<i32>
// CHECK: return
func.func @annotated_private_counter(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<i32> {
      "test.work"() : () -> ()
      %buffer = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %zero, %buffer[] : memref<i32>
      scope.return %buffer : memref<i32>
    } {hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
    annotation.mark %counter : memref<i32>
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A counter can share a scope result tuple with a live buffer. Its conditional
// update must be cloned with the condition and used by the consuming stage.
// CHECK-LABEL: func.func @conditional_counter_with_buffer
// CHECK: %[[SOURCE:.*]] = scope.scope : () -> memref<4xf32>
// CHECK: "test.branch_work"
// CHECK: test.source
// CHECK: scope.scope : () -> () {
// CHECK: %[[COUNTER:.*]] = memref.alloca
// CHECK: memref.store {{.*}}, %[[COUNTER]][]
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: %[[OLD:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[OLD]],
// CHECK-NEXT: memref.store %[[NEXT]], %[[COUNTER]][]
// CHECK: } else {
// CHECK: memref.store {{.*}}, %[[COUNTER]][]
// CHECK: }
// CHECK-NOT: "test.branch_work"
// CHECK: %[[VALUE:.*]] = memref.load %[[COUNTER]][]
// CHECK: "test.consume"(%[[VALUE]], %[[SOURCE]])
// CHECK: test.consumer
// CHECK: scope.scope : () -> () {
// CHECK-NEXT: "test.buffer_only"(%[[SOURCE]])
func.func @conditional_counter_with_buffer(%n: index, %cond: i1, %data: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 {
    %results:2 = scope.scope : () -> (memref<4xf32>, memref<i32>) {
      "test.work"() : () -> ()
      %buffer = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %zero, %buffer[] : memref<i32>
      scf.if %cond {
        "test.branch_work"() : () -> ()
        %old = memref.load %buffer[] : memref<i32>
        %next = arith.addi %old, %one : i32
        memref.store %next, %buffer[] : memref<i32>
      } else {
        memref.store %zero, %buffer[] : memref<i32>
      }
      scope.return %data, %buffer : memref<4xf32>, memref<i32>
    } {no_inline, test.source}
    scope.scope : () -> () {
      %value = memref.load %results#1[] : memref<i32>
      "test.consume"(%value, %results#0) : (i32, memref<4xf32>) -> ()
      scope.return
    } {no_inline, test.consumer}
    scope.scope : () -> () {
      "test.buffer_only"(%results#0) : (memref<4xf32>) -> ()
      scope.return
    } {no_inline}
  } {hivm.cv_pipelined_loop}
  return
}


// -----

// A nonempty annotation is meaningful metadata and must not be discarded.
// CHECK-LABEL: func.func @preserve_counter_metadata
// CHECK: %[[COUNTER:.*]] = scope.scope : () -> memref<i32>
// CHECK: annotation.mark %[[COUNTER]] {test.metadata} : memref<i32>
func.func @preserve_counter_metadata(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<i32> {
      %buffer = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      scope.return %buffer : memref<i32>
    } {no_inline}
    annotation.mark %counter {test.metadata} : memref<i32>
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// A read/write mark must not be treated as an empty default-write mark.
// CHECK-LABEL: func.func @preserve_counter_read_effect
// CHECK: %[[COUNTER:.*]] = scope.scope : () -> memref<i32>
// CHECK: annotation.mark %[[COUNTER]] {effects = ["read", "write"]} : memref<i32>
func.func @preserve_counter_read_effect(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %counter = scope.scope : () -> memref<i32> {
      %buffer = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      scope.return %buffer : memref<i32>
    } {no_inline}
    annotation.mark %counter {effects = ["read", "write"]} : memref<i32>
  } {hivm.cv_pipelined_loop}
  return
}
