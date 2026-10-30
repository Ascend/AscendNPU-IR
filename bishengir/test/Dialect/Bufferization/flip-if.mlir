// RUN: bishengir-opt %s -bufferization-flip-if | FileCheck %s --check-prefix=FLIP
// RUN: bishengir-opt %s -bufferization-flip-if -bufferization-flip-if | FileCheck %s --check-prefix=FLIP
// RUN: bishengir-opt %s -bufferization-flip-if -bufferization-flip-if=restore-only | FileCheck %s --check-prefix=ORIGINAL --implicit-check-not=flip_if.restore_state
// RUN: bishengir-opt %s -bufferization-flip-if=restore-only | FileCheck %s --check-prefix=ORIGINAL
// RUN: bishengir-opt %s -bufferization-flip-if -hivm-decompose-alloc-tensor-copy -tensor-copy-insertion="analysis-heuristic=top-down allow-return-allocs-from-loops bufferize-function-boundaries" -hivm-decompose-alloc-tensor-copy -one-shot-bufferize="analysis-heuristic=top-down allow-return-allocs-from-loops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -bufferization-flip-if=restore-only | FileCheck %s --check-prefix=BUFFER --implicit-check-not=flip_if.restore_state --implicit-check-not=test.saved

func.func @update(%old: tensor<4xf32>, %input: tensor<4xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
  %new = linalg.max ins(%old, %input : tensor<4xf32>, tensor<4xf32>) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %new : tensor<4xf32>
}

func.func @read_old(%old: tensor<4xf32>, %new: tensor<4xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
  %alpha = linalg.sub ins(%old, %new : tensor<4xf32>, tensor<4xf32>) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %alpha : tensor<4xf32>
}

// Include a scalar result before the tensor to check that the entire if is
// flipped while result ordering stays unchanged.
// FLIP-LABEL: func.func @state_update(
// FLIP-SAME: %[[COND:.*]]: i1,
// FLIP: %[[INV:.*]] = arith.cmpi eq, %[[COND]], %false {flip_if.restore_state} : i1
// FLIP: scf.if %[[INV]] -> (i32, tensor<4xf32>) {
// FLIP-NEXT: scf.yield
// FLIP-NEXT: } else {
// FLIP: call @update
// FLIP: call @read_old
// ORIGINAL-LABEL: func.func @state_update(
// ORIGINAL-SAME: %[[COND:.*]]: i1,
// ORIGINAL: scf.if %[[COND]] -> (i32, tensor<4xf32>) {
// ORIGINAL: call @update
// ORIGINAL: call @read_old
// ORIGINAL: } else {
// ORIGINAL-NEXT: scf.yield
// ORIGINAL-NEXT: } {test.saved}
// BUFFER-LABEL: func.func @state_update(
// BUFFER-SAME: %[[COND:.*]]: i1,
// BUFFER: scf.for {{.*}} iter_args(%[[OLD:[^ ]+]] = %{{[^,]+}},
// BUFFER: scf.if %[[COND]]
// BUFFER: %[[DEST:.*]] = memref.alloc()
// BUFFER-NEXT: %[[NEW:.*]] = func.call @update(%[[OLD]], %{{.*}}, %[[DEST]])
// BUFFER: func.call @read_old(%[[OLD]], %[[NEW]],
// BUFFER: hivm.hir.copy ins(%[[NEW]]
// BUFFER: } else {
// BUFFER-NEXT: scf.yield %{{.*}}, %[[OLD]] : i32, memref<4xf32>
// BUFFER-NEXT: }
// BUFFER-NEXT: scf.yield
func.func @state_update(%fire: i1, %n: index, %initial: tensor<4xf32>, %input: tensor<4xf32>, %sink: memref<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%old = %initial, %progress = %zero) -> (tensor<4xf32>, i32) {
    %next:2 = scf.if %fire -> (i32, tensor<4xf32>) {
      %empty = tensor.empty() : tensor<4xf32>
      %new = func.call @update(%old, %input, %empty) {hivm.vector_function} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %empty2 = tensor.empty() : tensor<4xf32>
      %alpha = func.call @read_old(%old, %new, %empty2) {hivm.vector_function} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %value = tensor.extract %alpha[%c0] : tensor<4xf32>
      memref.store %value, %sink[] : memref<f32>
      %inc = arith.addi %progress, %one : i32
      scf.yield %inc, %new : i32, tensor<4xf32>
    } else {
      scf.yield %progress, %old : i32, tensor<4xf32>
    } {test.saved}
    scf.yield %next#1, %next#0 : tensor<4xf32>, i32
  }
  return %result#0 : tensor<4xf32>
}

// Forwarding a loop iter_arg in else is sufficient, even without late uses.
// FLIP-LABEL: func.func @no_later_use(
// FLIP: %[[INV:.*]] = arith.cmpi eq, %arg0, %false {flip_if.restore_state} : i1
// FLIP: scf.if %[[INV]] -> (tensor<4xf32>) {
// FLIP-NEXT: scf.yield
// FLIP-NEXT: } else {
// FLIP: call @update
// ORIGINAL-LABEL: func.func @no_later_use(
// ORIGINAL: scf.if %arg0 -> (tensor<4xf32>) {
// ORIGINAL: call @update
// ORIGINAL: } else {
// ORIGINAL-NEXT: scf.yield
func.func @no_later_use(%fire: i1, %n: index, %initial: tensor<4xf32>, %input: tensor<4xf32>, %sink: memref<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %iv = %c0 to %n step %c1 iter_args(%old = %initial) -> tensor<4xf32> {
    %next = scf.if %fire -> tensor<4xf32> {
      %value = tensor.extract %old[%c0] : tensor<4xf32>
      memref.store %value, %sink[] : memref<f32>
      %empty = tensor.empty() : tensor<4xf32>
      %new = func.call @update(%old, %input, %empty) {hivm.vector_function} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    }
    scf.yield %next : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// The target can be nested in another if, use ordinary tensor operations,
// and have a result that is not forwarded by the for yield.
// FLIP-LABEL: func.func @nested_if(
// FLIP: scf.if %arg0 {
// FLIP: %[[INV:.*]] = arith.cmpi eq, %arg1, %false {flip_if.restore_state} : i1
// FLIP: scf.if %[[INV]] -> (tensor<4xf32>) {
// FLIP-NEXT: scf.yield
// FLIP-NEXT: } else {
// FLIP: tensor.insert
// FLIP: tensor.extract
// ORIGINAL-LABEL: func.func @nested_if(
// ORIGINAL: scf.if %arg0 {
// ORIGINAL: scf.if %arg1 -> (tensor<4xf32>) {
// ORIGINAL: tensor.insert
// ORIGINAL: tensor.extract
func.func @nested_if(%outer: i1, %fire: i1, %n: index, %initial: tensor<4xf32>, %value: f32, %sink: memref<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %iv = %c0 to %n step %c1 iter_args(%old = %initial) -> tensor<4xf32> {
    scf.if %outer {
      %next = scf.if %fire -> tensor<4xf32> {
        %new = tensor.insert %value into %old[%c0] : tensor<4xf32>
        %late = tensor.extract %old[%c0] : tensor<4xf32>
        memref.store %late, %sink[] : memref<f32>
        scf.yield %new : tensor<4xf32>
      } else {
        scf.yield %old : tensor<4xf32>
      }
      %selected = tensor.extract %next[%c1] : tensor<4xf32>
      memref.store %selected, %sink[] : memref<f32>
    }
    %advance = tensor.insert %value into %old[%c1] : tensor<4xf32>
    scf.yield %advance : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// A while do-region argument is also a loop-carried tensor state.
// FLIP-LABEL: func.func @while_state_update(
// FLIP-SAME: %[[FIRE:[^:]+]]: i1,
// FLIP: scf.while
// FLIP: } do {
// FLIP: %[[INV:.*]] = arith.cmpi eq, %[[FIRE]], %false {flip_if.restore_state} : i1
// FLIP: scf.if %[[INV]] -> (tensor<4xf32>) {
// FLIP-NEXT: scf.yield
// FLIP-NEXT: } else {
// FLIP: call @update
// FLIP: call @read_old
// ORIGINAL-LABEL: func.func @while_state_update(
// ORIGINAL-SAME: %[[FIRE:[^:]+]]: i1,
// ORIGINAL: scf.while
// ORIGINAL: scf.if %[[FIRE]] -> (tensor<4xf32>) {
// ORIGINAL: call @update
// ORIGINAL: call @read_old
// ORIGINAL: } else {
// ORIGINAL-NEXT: scf.yield
// BUFFER-LABEL: func.func @while_state_update(
// BUFFER-SAME: %[[FIRE:[^:]+]]: i1,
// BUFFER: scf.while
// BUFFER: scf.if %[[FIRE]]
// BUFFER: call @update
// BUFFER: call @read_old
func.func @while_state_update(%fire: i1, %n: index, %initial: tensor<4xf32>, %input: tensor<4xf32>, %sink: memref<f32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.while (%old = %initial, %iv = %c0) : (tensor<4xf32>, index) -> (tensor<4xf32>, index) {
    %continue = arith.cmpi slt, %iv, %n : index
    scf.condition(%continue) %old, %iv : tensor<4xf32>, index
  } do {
  ^bb0(%old: tensor<4xf32>, %iv: index):
    %next = scf.if %fire -> tensor<4xf32> {
      %empty = tensor.empty() : tensor<4xf32>
      %new = func.call @update(%old, %input, %empty) {hivm.vector_function} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %empty2 = tensor.empty() : tensor<4xf32>
      %alpha = func.call @read_old(%old, %new, %empty2) {hivm.vector_function} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %value = tensor.extract %alpha[%c0] : tensor<4xf32>
      memref.store %value, %sink[] : memref<f32>
      scf.yield %new : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    }
    %inc = arith.addi %iv, %c1 : index
    scf.yield %next, %inc : tensor<4xf32>, index
  }
  return %result#0 : tensor<4xf32>
}
