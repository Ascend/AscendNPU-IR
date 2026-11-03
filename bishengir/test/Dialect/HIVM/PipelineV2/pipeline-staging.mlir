// RUN: bishengir-opt %s -pipeline-staging -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @copy_to_l1
// CHECK:       scf.for
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
// CHECK-NEXT:    annotation.mark %[[ALLOC]] {hivm.multi_buffer = 2 : i32}
// CHECK-NEXT:    %[[SCOPE:.*]] = scope.scope : () -> i32 {
// CHECK:           hivm.hir.copy ins(%{{.*}} : memref<16xf32>) outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<cbuf>>)
// CHECK:           %[[SEVEN:.*]] = arith.constant 7 : i32
// CHECK:           scope.return %[[SEVEN]] : i32
// CHECK-NEXT:    } {{.*}}
// CHECK-NEXT:    %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<16xf32, #hivm.address_space<cbuf>>
// CHECK:         scope.scope : () -> () {
// CHECK:           tensor.dim %[[TENSOR]]
// CHECK:           arith.addi %[[SCOPE]], %[[SCOPE]]
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @copy_to_l1(%src: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %tensor, %scalar = scope.scope : () -> (tensor<16xf32>, i32) {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%src : memref<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
      %t = bufferization.to_tensor %alloc restrict writable : memref<16xf32, #hivm.address_space<cbuf>>
      %v = arith.constant 7 : i32
      scope.return %t, %v : tensor<16xf32>, i32
    }
    scope.scope : () -> () {
      %dim = tensor.dim %tensor, %c0 : tensor<16xf32>
      %sum = arith.addi %scalar, %scalar : i32
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @fixpipe_to_ub
// CHECK:       scf.for
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}
// CHECK-NEXT:    %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
// CHECK-NEXT:    %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16xf32>
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           tensor.dim %[[TENSOR]]
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

func.func @fixpipe_to_ub(%src: tensor<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %tensor = scope.scope : () -> tensor<16xf32> {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%src : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
      %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
      scope.return %t : tensor<16xf32>
    }
    scope.scope : () -> () {
      %dim = tensor.dim %tensor, %c0 : tensor<16xf32>
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// The last scope depends on the first. Staging moves the first scope into the
// same stage as the third scope, leaving the second scope with a larger preload
// number. Keep the first and third scopes in their original relative order.
// CHECK-LABEL: func.func @stable_preload_order
// CHECK:       scf.for
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.constant 20 : i32
// CHECK:         } {{.*}}hivm.preload_num = 3 : i32
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.constant 10 : i32
// CHECK:         } {{.*}}hivm.preload_num = 2 : i32
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.constant 30 : i32
// CHECK:         } {{.*}}hivm.preload_num = 2 : i32
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.constant 40 : i32
// CHECK:         } {{.*}}hivm.preload_num = 1 : i32
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.addi
// CHECK:         } {{.*}}hivm.preload_num = 0 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @stable_preload_order() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %a = scope.scope : () -> i32 {
        %v = arith.constant 10 : i32
        scope.return %v : i32
      }
      %b = scope.scope : () -> i32 {
        %v = arith.constant 20 : i32
        scope.return %v : i32
      }
      %c = scope.scope : () -> i32 {
        %v = arith.constant 30 : i32
        scope.return %v : i32
      }
      %d = scope.scope : () -> i32 {
        %v = arith.constant 40 : i32
        scope.return %v : i32
      }
      %e = scope.scope : () -> i32 {
        %v = arith.addi %a, %a : i32
        scope.return %v : i32
      }
    } {hivm.cv_pipelined_loop}
    return
  }
}

// -----

// CHECK-LABEL: func.func @hoist_sunk_with_scf_if
// CHECK:       scf.for
// CHECK:         %[[L1_ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
// CHECK-NEXT:    annotation.mark %[[L1_ALLOC]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           scf.if
// CHECK:             hivm.hir.copy ins(%{{.*}} : memref<16xf32>) outs(%[[L1_ALLOC]] : memref<16xf32, #hivm.address_space<cbuf>>)
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}
// CHECK-NEXT:    %[[L1_TENSOR:.*]] = bufferization.to_tensor %[[L1_ALLOC]] restrict writable : memref<16xf32, #hivm.address_space<cbuf>>
// CHECK-NEXT:    %[[UB_ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    annotation.mark %[[UB_ALLOC]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           scf.if
// CHECK:             hivm.hir.fixpipe ins(%[[L1_TENSOR]] : tensor<16xf32>) outs(%[[UB_ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}
// CHECK-NEXT:    %[[CAST:.*]] = memref.memory_space_cast %[[UB_ALLOC]] : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
// CHECK-NEXT:    %[[UB_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16xf32>
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           tensor.dim %[[UB_TENSOR]]
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @hoist_sunk_with_scf_if(%src: memref<16xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %l1_tensor = scope.scope : () -> tensor<16xf32> {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<cbuf>>
      %t = bufferization.to_tensor %alloc restrict writable : memref<16xf32, #hivm.address_space<cbuf>>
      scf.if %cond {
        hivm.hir.copy ins(%src : memref<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
      }
      scope.return %t : tensor<16xf32>
    }
    %ub_tensor = scope.scope : () -> tensor<16xf32> {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
      scf.if %cond {
        hivm.hir.fixpipe ins(%l1_tensor : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
      }
      scope.return %t : tensor<16xf32>
    }
    scope.scope : () -> () {
      %dim = tensor.dim %ub_tensor, %c0 : tensor<16xf32>
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}
}


