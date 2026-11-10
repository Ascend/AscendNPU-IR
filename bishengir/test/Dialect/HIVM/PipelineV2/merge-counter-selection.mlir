// RUN: bishengir-opt %s -allow-unregistered-dialect -merge-cv-scopes -canonicalize -verify-each -o %t
// RUN: FileCheck %s --check-prefix=MERGE < %t
// RUN: bishengir-opt %s -allow-unregistered-dialect -merge-cv-scopes -canonicalize -pipeline-staging -wrap-cv-preload-scopes -create-preload -verify-each | FileCheck %s --check-prefix=PRELOAD

// Keep initialization before the nested counter update. The conditional buffer
// selection must stay with its consumer so CreatePreload can give that stage a
// private counter instead of rematerializing a load in the inactive branch.
// MERGE-LABEL: func.func @counter_selection
// MERGE: scf.for
// MERGE-NEXT: %[[COUNTER:.*]] = scope.scope : () -> memref<index> {
// MERGE-NEXT: %[[ALLOC:.*]] = memref.alloca
// MERGE-NEXT: memref.store {{.*}}, %[[ALLOC]][]
// MERGE-NEXT: scf.for
// MERGE: "test.cube"
// MERGE: scope.return %[[ALLOC]] : memref<index>
// MERGE-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// MERGE-NEXT: scope.scope : () -> () {
// MERGE-NEXT: %[[VALUE:.*]] = memref.load %[[COUNTER]][]
// MERGE-NEXT: %[[EMPTY:.*]] = arith.cmpi eq, %[[VALUE]],
// MERGE-NEXT: %[[SELECTED:.*]] = arith.select %[[EMPTY]],
// MERGE-NEXT: hivm.hir.store ins(%[[SELECTED]]
// MERGE: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

// PRELOAD-LABEL: func.func @counter_selection
// PRELOAD: scf.for
// PRELOAD: scf.if
// PRELOAD: %[[CUBE_COUNTER:.*]] = memref.alloca
// PRELOAD-NEXT: memref.store {{.*}}, %[[CUBE_COUNTER]][]
// PRELOAD-NEXT: scf.for
// PRELOAD: "test.cube"
// PRELOAD: scf.if
// PRELOAD-NEXT: %[[VECTOR_COUNTER:.*]] = memref.alloca
// PRELOAD-NEXT: memref.store {{.*}}, %[[VECTOR_COUNTER]][]
// PRELOAD-NEXT: scf.for
// PRELOAD-NEXT: %[[OLD:.*]] = memref.load %[[VECTOR_COUNTER]][]
// PRELOAD-NEXT: %[[NEXT:.*]] = arith.addi %[[OLD]],
// PRELOAD-NEXT: memref.store %[[NEXT]], %[[VECTOR_COUNTER]][]
// PRELOAD-NEXT: }
// PRELOAD-NEXT: %[[COUNT:.*]] = memref.load %[[VECTOR_COUNTER]][]
// PRELOAD-NEXT: %[[COND:.*]] = arith.cmpi eq, %[[COUNT]],
// PRELOAD-NEXT: %[[BUFFER:.*]] = arith.select %[[COND]],
// PRELOAD-NEXT: hivm.hir.store ins(%[[BUFFER]]

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @counter_selection(%n: index, %a: memref<16xf32, #hivm.address_space<ub>>, %b: memref<16xf32, #hivm.address_space<ub>>, %out: memref<16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %counter = scope.scope : () -> memref<index> {
        %alloca = memref.alloca() {normalize_matmul_counter} : memref<index>
        scope.return %alloca : memref<index>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
      scope.scope : () -> () {
        memref.store %c0, %counter[] : memref<index>
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
      scope.scope : () -> () {
        scf.for %j = %c0 to %n step %c1 {
          %old = memref.load %counter[] : memref<index>
          %next = arith.addi %old, %c1 : index
          memref.store %next, %counter[] : memref<index>
          "test.cube"(%old) : (index) -> ()
        }
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %count = scope.scope : () -> index {
        %value = memref.load %counter[] : memref<index>
        scope.return %value : index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
      %selected = scope.scope : () -> memref<16xf32, #hivm.address_space<ub>> {
        %empty = arith.cmpi eq, %count, %c0 : index
        %result = scf.if %empty -> memref<16xf32, #hivm.address_space<ub>> {
          scf.yield %a : memref<16xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %b : memref<16xf32, #hivm.address_space<ub>>
        }
        scope.return %result : memref<16xf32, #hivm.address_space<ub>>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
      scope.scope : () -> () {
        hivm.hir.store ins(%selected : memref<16xf32, #hivm.address_space<ub>>) outs(%out : memref<16xf32, #hivm.address_space<gm>>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    } {hivm.cv_pipelined_loop}
    return
  }
}
