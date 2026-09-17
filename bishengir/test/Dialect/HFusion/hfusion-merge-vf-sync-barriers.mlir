// RUN: bishengir-opt %s --hfusion-merge-vf="merge-level=1" --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-merge-vf="merge-level=2" --split-input-file | FileCheck %s

// The Cube peer waits for input-ready (flag 2), computes, then signals
// output-ready (flag 4). Moving the wait before the input copy/set deadlocks.
// Tensor SSA dependencies alone do not describe this cross-core dependency.
// CHECK-LABEL: func.func @handshake(
// CHECK: call @producer_vf_0
// CHECK: scf.if
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.sync_block_set
// CHECK-SAME: flag = 2
// CHECK: scf.if
// CHECK: hivm.hir.sync_block_wait
// CHECK-SAME: flag = 4
// CHECK: call @consumer_vf_0
// CHECK: return
func.func @producer_vf_0(%input: tensor<1xf32>) -> tensor<1xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %input : tensor<1xf32>
}
func.func @consumer_vf_0(%input: tensor<1xf32>) -> tensor<1xf32>
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return %input : tensor<1xf32>
}
func.func @handshake(%active: i1, %input: tensor<1xf32>,
                     %cubeResult: memref<1xf32>) -> tensor<1xf32> {
  %cubeInput = memref.alloc() : memref<1xf32>
  %produced = func.call @producer_vf_0(%input)
      {hivm.vector_function, no_inline} : (tensor<1xf32>) -> tensor<1xf32>
  scf.if %active {
    hivm.hir.copy ins(%produced : tensor<1xf32>) outs(%cubeInput : memref<1xf32>)
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  }
  %ready = scf.if %active -> (tensor<1xf32>) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
    %view = bufferization.to_tensor %cubeResult restrict writable : memref<1xf32>
    scf.yield %view : tensor<1xf32>
  } else {
    %unused = tensor.empty() : tensor<1xf32>
    scf.yield %unused : tensor<1xf32>
  }
  %result = func.call @consumer_vf_0(%ready)
      {hivm.vector_function, no_inline} : (tensor<1xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @block_set(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.sync_block_set
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @block_set(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @block_wait(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.sync_block_wait
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @block_wait(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @create_lock(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.create_sync_block_lock
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @create_lock(%active: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @pipe_barrier(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.pipe_barrier
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @pipe_barrier(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      hivm.hir.pipe_barrier[<PIPE_ALL>]
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @block_sync(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.sync_block
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @block_sync(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      hivm.hir.sync_block[<BARRIER_VECTOR>]
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @lock(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.sync_block_lock
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @lock(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// Check this barrier independently so another barrier cannot mask an omission.
// CHECK-LABEL: func.func @unlock(
// CHECK: call @first_vf_0
// CHECK: hivm.hir.sync_block_unlock
// CHECK: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @unlock(%active: i1, %n: index, %lock: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @first_vf_0() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    scf.if %active {
      hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    }
  }
  func.call @second_vf_0() : () -> ()
  return
}

// -----

// A region without synchronization must not become a blanket merge barrier.
// CHECK-LABEL: func.func @first{{_+}}merged_vf_0
// CHECK-LABEL: func.func @no_sync(
// CHECK-NOT: call @first_vf_0
// CHECK-NOT: call @second_vf_0
// CHECK: call @first{{_+}}merged_vf_0
// CHECK-NOT: call @first_vf_0
// CHECK-NOT: call @second_vf_0
// CHECK: return
func.func @first_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @second_vf_0() attributes {
    hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}
func.func @no_sync(%active: i1) {
  func.call @first_vf_0() : () -> ()
  scf.if %active {
    %c0 = arith.constant 0 : index
  }
  func.call @second_vf_0() : () -> ()
  return
}
