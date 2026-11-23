// RUN: bishengir-opt %s -pipeline-staging="max-pipeline-multibuffer=3" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -pipeline-staging="max-pipeline-multibuffer=3" -wrap-cv-preload-scopes -split-input-file | FileCheck %s --check-prefix=WRAP

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

// CHECK-LABEL: func.func @fixpipe_to_ub_return_and_inside_use
// CHECK:       scf.for
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
// CHECK-DAG:     %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
// CHECK-DAG:     scope.scope : () -> tensor<16xf32> {
// CHECK-DAG:       hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:           scf.if {{.*}} -> (tensor<16xf32>) {
// CHECK:             scf.yield %[[TENSOR]] : tensor<16xf32>
// CHECK:           } {{.*}} {
// CHECK:             scf.yield %[[TENSOR]] : tensor<16xf32>
// CHECK:           }
// CHECK:           scope.return %[[TENSOR]] : tensor<16xf32>
// CHECK-NEXT:    } {{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

func.func @fixpipe_to_ub_return_and_inside_use(%src: tensor<16xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %tensor = scope.scope : () -> tensor<16xf32> {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%src : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
      %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
      scf.if %cond -> (tensor<16xf32>) {
        scf.yield %t : tensor<16xf32>
      } else {
        scf.yield %t : tensor<16xf32>
      }
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
// WRAP-LABEL: func.func @stable_preload_order
// WRAP: } {{.*}}hivm.preload_num = 3 : i32
// WRAP-NEXT: %[[STAGE:.*]] = scope.scope : () -> i32 {
// WRAP-NEXT: %[[FIRST:.*]] = scope.scope : () -> i32 {
// WRAP: arith.constant 10 : i32
// WRAP: } {no_inline}
// WRAP-NEXT: %{{.*}} = scope.scope : () -> i32 {
// WRAP: arith.constant 30 : i32
// WRAP: } {no_inline}
// WRAP-NEXT: scope.return %[[FIRST]] : i32
// WRAP-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
// WRAP: arith.addi %[[STAGE]], %[[STAGE]] : i32
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

// -----

// CHECK-LABEL: func.func @store_to_workspace
// CHECK:       scf.for
// CHECK:         %[[WS:.*]] = memref_ext.alloc_workspace() from %{{.*}} : from memref<?xi8> to memref<1xi32>
// CHECK-NEXT:    annotation.mark %[[WS]] {hivm.multi_buffer = 2 : i32}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           hivm.hir.store ins(%{{.*}} : tensor<1xi32>) outs(%[[WS]] : memref<1xi32>)
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}
// CHECK-NEXT:    %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<1xi32>
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:           tensor.extract %[[WST]]
// CHECK:           scope.return
// CHECK-NEXT:    } {{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @store_to_workspace(%ws_arg: memref<?xi8>, %src: tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %wst = scope.scope : () -> tensor<1xi32> {
      %ws = memref_ext.alloc_workspace() from %ws_arg : from memref<?xi8> to memref<1xi32>
      annotation.mark %ws {hivm.multi_buffer = 2 : i32} : memref<1xi32>
      hivm.hir.store ins(%src : tensor<1xi32>) outs(%ws : memref<1xi32>) {"hivm.inserted-store"}
      %t = bufferization.to_tensor %ws restrict writable : memref<1xi32>
      scope.return %t : tensor<1xi32>
    }
    scope.scope : () -> () {
      %v = tensor.extract %wst[%c0] : tensor<1xi32>
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @mark_vector_scope_ifs
// CHECK:       scf.for
// CHECK:         scope.scope : () -> i32 {
// CHECK:           scf.if %{{.*}} -> (tensor<16xf32>) {
// CHECK:           } else {
// CHECK:           } {hivm.vec_only}
// CHECK:           scf.if %{{.*}} {
// CHECK:             memref.store
// CHECK-NEXT:      }{{ *$}}
// CHECK:           %[[ESC:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           } else {
// CHECK:           }{{ *$}}
// CHECK:           scope.return %[[ESC]] : i32
// CHECK-NEXT:    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @mark_vector_scope_ifs(%cond: i1, %init: tensor<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cnt = memref.alloca() : memref<i32>
  scf.for %i = %c0 to %c1 step %c1 {
    %res = scope.scope : () -> i32 {
      %t = scf.if %cond -> (tensor<16xf32>) {
        %out = tensor.empty() : tensor<16xf32>
        %add = hivm.hir.vadd ins(%init, %init : tensor<16xf32>, tensor<16xf32>) outs(%out : tensor<16xf32>) -> tensor<16xf32>
        scf.yield %add : tensor<16xf32>
      } else {
        scf.yield %init : tensor<16xf32>
      }
      scf.if %cond {
        memref.store %c1_i32, %cnt[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
      }
      %esc = scf.if %cond -> (i32) {
        scf.yield %c1_i32 : i32
      } else {
        scf.yield %c0_i32 : i32
      }
      scope.return %esc : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %sum = arith.addi %res, %c1_i32 : i32
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @hoist_tcb_returned_via_scf_if
// CHECK:       scf.for
// CHECK:         %[[INIT:.*]] = scope.scope : () -> tensor<16xf32> {
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK-NEXT:    %[[TCB:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    annotation.mark %[[TCB]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// CHECK-NEXT:    %[[CUBE:.*]] = scope.scope : () -> tensor<16xf32> {
// CHECK:           scf.for
// CHECK:             %[[LOCAL:.*]] = memref.alloc() : memref<16xf32>
// CHECK:             memref.copy {{.*}}, %[[LOCAL]]
// CHECK:           %[[CAST:.*]] = memref.memory_space_cast %[[TCB]]
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
// CHECK:           %[[IF:.*]] = scf.if %{{.*}} -> (tensor<16xf32>) {
// CHECK:             scf.yield %[[INIT]]
// CHECK:           } else {
// CHECK:             hivm.hir.fixpipe {{.*}} outs(%[[TCB]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:             scf.yield %[[TENSOR]]
// CHECK:           }
// CHECK:           scope.return %[[IF]]
// CHECK-NEXT:    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @hoist_tcb_returned_via_scf_if(%src: tensor<16xf32>, %gm: memref<16xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %init = scope.scope : () -> tensor<16xf32> {
      %out = tensor.empty() : tensor<16xf32>
      %v = hivm.hir.vadd ins(%src, %src : tensor<16xf32>, tensor<16xf32>) outs(%out : tensor<16xf32>) -> tensor<16xf32>
      scope.return %v : tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %cube = scope.scope : () -> tensor<16xf32> {
      scf.for %k = %c0 to %c1 step %c1 {
        %local = memref.alloc() : memref<16xf32>
        memref.copy %gm, %local : memref<16xf32> to memref<16xf32>
      }
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
      %res = scf.if %cond -> (tensor<16xf32>) {
        scf.yield %init : tensor<16xf32>
      } else {
        hivm.hir.fixpipe ins(%src : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
        scf.yield %t : tensor<16xf32>
      }
      scope.return %res : tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %dim = tensor.dim %cube, %c0 : tensor<16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @hoist_tcb_from_nested_merged_scope
// CHECK:       scf.for
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    annotation.mark %[[ALLOC]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// CHECK-NEXT:    %[[MERGED:.*]] = scope.scope : () -> tensor<16xf32> {
// CHECK:           scope.scope : () -> () {
// CHECK:             hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:           %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
// CHECK:           %[[T:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
// CHECK:           %[[VEC:.*]] = scope.scope : () -> tensor<16xf32> {
// CHECK:             hivm.hir.vadd ins(%[[T]], %[[T]]
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:           scope.return %[[VEC]]
// CHECK-NEXT:    } {cv_merged_scope, hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @hoist_tcb_from_nested_merged_scope(%src: tensor<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %merged = scope.scope : () -> tensor<16xf32> {
      %inner_cube = scope.scope : () -> tensor<16xf32> {
        %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe ins(%src : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
        %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
        %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
        scope.return %t : tensor<16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %inner_vec = scope.scope : () -> tensor<16xf32> {
        %empty = tensor.empty() : tensor<16xf32>
        %add = hivm.hir.vadd ins(%inner_cube, %inner_cube : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
        scope.return %add : tensor<16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %inner_vec : tensor<16xf32>
    } {cv_merged_scope, hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
    scope.scope : () -> () {
      %dim = tensor.dim %merged, %c0 : tensor<16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @hoist_tcb_before_multi_scope_stage_wrapper
// CHECK:       scf.for
// CHECK:         scope.scope : () -> i32 {
// CHECK:           arith.constant 20 : i32
// CHECK:         } {{.*}}hivm.preload_num = 3 : i32
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NEXT:    annotation.mark %[[ALLOC]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// CHECK-NEXT:    %[[FIRST:.*]] = scope.scope : () -> i32 {
// CHECK:           arith.constant 10 : i32
// CHECK:         } {{.*}}hivm.preload_num = 2 : i32
// CHECK-NEXT:    scope.scope : () -> tensor<16xf32> {
// CHECK:           scf.if
// CHECK:             hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// CHECK:         } {{.*}}hivm.preload_num = 2 : i32
//
// WRAP-LABEL: func.func @hoist_tcb_before_multi_scope_stage_wrapper
// WRAP:        scf.for
// WRAP:          scope.scope : () -> i32 {
// WRAP:            arith.constant 20 : i32
// WRAP:          } {{.*}}hivm.preload_num = 3 : i32
// WRAP-NEXT:     %[[ALLOC:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// WRAP-NEXT:     annotation.mark %[[ALLOC]] {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
// WRAP-NEXT:     %[[STAGE:.*]] = scope.scope : () -> i32 {
// WRAP-NEXT:       %[[INNER_A:.*]] = scope.scope : () -> i32 {
// WRAP:            } {{.*}}no_inline
// WRAP-NEXT:       scope.scope : () -> tensor<16xf32> {
// WRAP:              hivm.hir.fixpipe {{.*}} outs(%[[ALLOC]] : memref<16xf32, #hivm.address_space<ub>>)
// WRAP:            } {{.*}}no_inline
// WRAP-NEXT:       scope.return %[[INNER_A]] : i32
// WRAP-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @hoist_tcb_before_multi_scope_stage_wrapper(%src: tensor<16xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %a = scope.scope : () -> i32 {
      %v = arith.constant 10 : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %b = scope.scope : () -> i32 {
      %v = arith.constant 20 : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %c = scope.scope : () -> tensor<16xf32> {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<16xf32, #hivm.address_space<ub>> to memref<16xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<16xf32>
      %res = scf.if %cond -> (tensor<16xf32>) {
        scf.yield %src : tensor<16xf32>
      } else {
        hivm.hir.fixpipe ins(%src : tensor<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
        scf.yield %t : tensor<16xf32>
      }
      scope.return %res : tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %d = scope.scope : () -> i32 {
      %v = arith.constant 40 : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %e = scope.scope : () -> i32 {
      %v = arith.addi %a, %a : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}
}
