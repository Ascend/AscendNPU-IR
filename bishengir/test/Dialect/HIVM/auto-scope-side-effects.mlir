// RUN: bishengir-opt --auto-scope --verify-each --split-input-file %s | FileCheck %s

// A tensor result does not make a region pure. Cloning these loops would run
// the cursor update twice and scatter using the second set of old values.
// CHECK-LABEL: func.func @atomic_loop_indices
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK: scf.for
// CHECK: memref.atomic_rmw addi
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scope.scope
// CHECK-NOT: scf.for
// CHECK-NOT: memref.atomic_rmw
// CHECK: hivm.hir.scatter_store ins(%[[RESULT]] : tensor<8xi64>
// CHECK-NOT: memref.atomic_rmw
// CHECK: return
func.func @atomic_loop_indices(%cursor: memref<8xi64>, %base: memref<?xi32>, %values: tensor<8xi32>, %init: tensor<8xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %one = arith.constant 1 : i64
  %stride = arith.constant 1 : i32
  %indices = scf.for %outer = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<8xi64>) {
    %row = scf.for %i = %c0 to %c8 step %c1 iter_args(%inner = %acc) -> (tensor<8xi64>) {
      %old = memref.atomic_rmw addi %one, %cursor[%i] : (i64, memref<8xi64>) -> i64
      %next = tensor.insert %old into %inner[%i] : tensor<8xi64>
      scf.yield %next : tensor<8xi64>
    }
    scf.yield %row : tensor<8xi64>
  }
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}

// -----

// Ordinary writes also remain outside the scope. It is not sufficient to
// check for atomics or the region's tensor result types alone.
// CHECK-LABEL: func.func @store_if_indices
// CHECK: %[[RESULT:.*]] = scf.if
// CHECK: memref.store
// CHECK: scf.yield
// CHECK: else
// CHECK: scf.yield
// CHECK: scope.scope
// CHECK-NOT: scf.if
// CHECK-NOT: memref.store
// CHECK: hivm.hir.scatter_store ins(%[[RESULT]] : tensor<8xi64>
// CHECK-NOT: memref.store
// CHECK: return
func.func @store_if_indices(%cond: i1, %cursor: memref<1xi64>, %base: memref<?xi32>, %values: tensor<8xi32>, %lhs: tensor<8xi64>, %rhs: tensor<8xi64>) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i64
  %stride = arith.constant 1 : i32
  %indices = scf.if %cond -> (tensor<8xi64>) {
    memref.store %one, %cursor[%c0] : memref<1xi64>
    scf.yield %lhs : tensor<8xi64>
  } else {
    scf.yield %rhs : tensor<8xi64>
  }
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}

// -----

// A call without a memory-effect contract is also a conservative boundary.
// CHECK-LABEL: func.func @unknown_call_if_indices
// CHECK: %[[RESULT:.*]] = scf.if
// CHECK: func.call @unknown_effect
// CHECK: scope.scope
// CHECK-NOT: scf.if
// CHECK-NOT: func.call
// CHECK: hivm.hir.scatter_store ins(%[[RESULT]] : tensor<8xi64>
// CHECK-NOT: func.call
// CHECK: return
func.func @unknown_call_if_indices(%cond: i1, %base: memref<?xi32>, %values: tensor<8xi32>, %lhs: tensor<8xi64>, %rhs: tensor<8xi64>) {
  %stride = arith.constant 1 : i32
  %indices = scf.if %cond -> (tensor<8xi64>) {
    func.call @unknown_effect() : () -> ()
    scf.yield %lhs : tensor<8xi64>
  } else {
    scf.yield %rhs : tensor<8xi64>
  }
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}
func.func private @unknown_effect()

// -----

// Cover the actual HIVM atomic-store form without relying on an allocation or
// another memory operation to make the enclosing loop effectful.
// CHECK-LABEL: func.func @hivm_atomic_store_indices
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK: hivm.hir.store
// CHECK-SAME: atomic = <add>
// CHECK: scf.yield
// CHECK: scope.scope
// CHECK-NOT: scf.for
// CHECK-NOT: hivm.hir.store
// CHECK: hivm.hir.scatter_store ins(%[[RESULT]] : tensor<8xi64>
// CHECK-NOT: hivm.hir.store
// CHECK: return
func.func @hivm_atomic_store_indices(%cursor: memref<1xi32>, %update: tensor<1xi32>, %base: memref<?xi32>, %values: tensor<8xi32>, %init: tensor<8xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %stride = arith.constant 1 : i32
  %indices = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (tensor<8xi64>) {
    hivm.hir.store ins(%update : tensor<1xi32>) outs(%cursor : memref<1xi32>) atomic = <add>
    scf.yield %acc : tensor<8xi64>
  }
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}

// -----

// A scalar read is a dependency even though its result is not a memref. Moving
// a cloned read past the intervening write would change the scatter indices.
// CHECK-LABEL: func.func @leaf_read_before_write
// CHECK: %[[OLD:.*]] = memref.load
// CHECK: memref.store
// CHECK: scope.scope
// CHECK-NOT: memref.load
// CHECK: %[[INDICES:.*]] = tensor.splat %[[OLD]] : tensor<8xi64>
// CHECK-NOT: memref.load
// CHECK: hivm.hir.scatter_store ins(%[[INDICES]] : tensor<8xi64>
// CHECK-NOT: memref.load
// CHECK: return
func.func @leaf_read_before_write(%cursor: memref<1xi64>, %base: memref<?xi32>, %values: tensor<8xi32>, %new: i64) {
  %c0 = arith.constant 0 : index
  %old = memref.load %cursor[%c0] : memref<1xi64>
  %indices = tensor.splat %old : tensor<8xi64>
  memref.store %new, %cursor[%c0] : memref<1xi64>
  %stride = arith.constant 1 : i32
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}

// -----

// Leaf atomics must also execute exactly once, not just atomics nested in SCF.
// CHECK-LABEL: func.func @leaf_atomic_indices
// CHECK: %[[OLD:.*]] = memref.atomic_rmw addi
// CHECK: scope.scope
// CHECK-NOT: memref.atomic_rmw
// CHECK: %[[INDICES:.*]] = tensor.splat %[[OLD]] : tensor<8xi64>
// CHECK-NOT: memref.atomic_rmw
// CHECK: hivm.hir.scatter_store ins(%[[INDICES]] : tensor<8xi64>
// CHECK-NOT: memref.atomic_rmw
// CHECK: return
func.func @leaf_atomic_indices(%cursor: memref<1xi64>, %base: memref<?xi32>, %values: tensor<8xi32>) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i64
  %old = memref.atomic_rmw addi %one, %cursor[%c0] : (i64, memref<1xi64>) -> i64
  %indices = tensor.splat %old : tensor<8xi64>
  %stride = arith.constant 1 : i32
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}

// -----

// Unknown effects are a boundary even without a containing region.
// CHECK-LABEL: func.func @leaf_unknown_call_indices
// CHECK: %[[INDICES:.*]] = {{(func[.])?}}call @unknown_indices
// CHECK: scope.scope
// CHECK-NOT: call @unknown_indices
// CHECK: hivm.hir.scatter_store ins(%[[INDICES]] : tensor<8xi64>
// CHECK-NOT: call @unknown_indices
// CHECK: return
func.func @leaf_unknown_call_indices(%base: memref<?xi32>, %values: tensor<8xi32>) {
  %indices = func.call @unknown_indices() : () -> tensor<8xi64>
  %stride = arith.constant 1 : i32
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}
func.func private @unknown_indices() -> tensor<8xi64>

// -----

// The memory-effect boundary must not disable cloning pure SSA computations.
// CHECK-LABEL: func.func @pure_leaf_indices
// CHECK: scope.scope
// CHECK: %[[INDICES:.*]] = arith.addi
// CHECK: hivm.hir.scatter_store ins(%[[INDICES]] : tensor<8xi64>
// CHECK: return
func.func @pure_leaf_indices(%lhs: tensor<8xi64>, %rhs: tensor<8xi64>, %base: memref<?xi32>, %values: tensor<8xi32>) {
  %indices = arith.addi %lhs, %rhs : tensor<8xi64>
  %stride = arith.constant 1 : i32
  hivm.hir.scatter_store ins(%indices : tensor<8xi64>, %values : tensor<8xi32>, %stride : i32) outs(%base : memref<?xi32>)
  return
}
