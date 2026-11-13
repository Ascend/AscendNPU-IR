// RUN: bishengir-opt %s --hivm-v2-split-kernel --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --pass-pipeline='builtin.module(hivm-v2-split-kernel,hivm-v2-split-kernel)' --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @shared_results_mix_aic(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: hivm.part_of_mix
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[COUNTER:.*]] = {{.*}}) -> (index) {
// CHECK: %[[SCOPE:.*]] = scope.scope : () -> index {
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<8x64xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[BUFFER]] {{.*}}hivm.tightly_coupled_buffer
// CHECK: hivm.hir.fixpipe {{.*}} outs(%[[BUFFER]] : memref<8x64xf32, #hivm.address_space<ub>>)
// CHECK: %[[NEXT:.*]] = arith.addi %[[COUNTER]], {{.*}} : index
// CHECK: scope.return %[[NEXT]] : index
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NOT: call @consume
// CHECK: scf.yield %[[SCOPE]] : index
// CHECK: memref.store %[[LOOP]],
// CHECK-LABEL: func.func @shared_results_mix_aiv(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.part_of_mix
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[COUNTER:.*]] = {{.*}}) -> (index) {
// CHECK: %[[SCOPE:.*]]:2 = scope.scope : () -> (memref<8x64xf32, #hivm.address_space<ub>>, index) {
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<8x64xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[BUFFER]] {{.*}}hivm.tightly_coupled_buffer
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: %[[NEXT:.*]] = arith.addi %[[COUNTER]], {{.*}} : index
// CHECK: scope.return %[[BUFFER]], %[[NEXT]]
// CHECK-NEXT: }
// CHECK: call @consume(%[[SCOPE]]#0)
// CHECK: scf.yield %[[SCOPE]]#1 : index
// CHECK: memref.store %[[LOOP]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @consume(memref<8x64xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function}
  func.func @shared_results(
      %src: memref<4x1x16x16xf32, #hivm.address_space<cc>>,
      %count: memref<1xindex>, %n: index)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%counter = %c0) -> (index) {
      %buffer, %next = scope.scope : () -> (memref<8x64xf32, #hivm.address_space<ub>>, index) {
        %alloc = memref.alloc() : memref<8x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
            : memref<8x64xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
            ins(%src : memref<4x1x16x16xf32, #hivm.address_space<cc>>)
            outs(%alloc : memref<8x64xf32, #hivm.address_space<ub>>)
            dual_dst_mode = <ROW_SPLIT>
        %next = arith.addi %counter, %c1 : index
        scope.return %alloc, %next : memref<8x64xf32, #hivm.address_space<ub>>, index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      func.call @consume(%buffer) : (memref<8x64xf32, #hivm.address_space<ub>>) -> ()
      scf.yield %next : index
    }
    memref.store %result, %count[%c0] : memref<1xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @scalar_scopes_mix_aic(
// CHECK-SAME: %[[CUBE:[^:]+]]: memref<1xi32>,
// CHECK: memref.store %{{.*}}, %[[CUBE]][%{{.*}}]
// CHECK-NOT: memref.store
// CHECK: {{^ *}}return
// CHECK-LABEL: func.func @scalar_scopes_mix_aiv(
// CHECK-SAME: %{{[^:]+}}: memref<1xi32>, %[[VECTOR:[^:]+]]: memref<1xi32>,
// CHECK-NOT: memref.store
// CHECK: scf.if
// CHECK: memref.store %{{.*}}, %[[VECTOR]][%{{.*}}]
// CHECK-NOT: memref.store
// CHECK: {{^ *}}return
module {
  func.func @scalar_scopes(%cube: memref<1xi32>, %vector: memref<1xi32>, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    scope.scope : () -> () {
      memref.store %c1, %cube[%c0] : memref<1xi32>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      scf.if %cond {
        memref.store %c1, %vector[%c0] : memref<1xi32>
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// -----

// CHECK-LABEL: func.func private @result_chain(
// CHECK-SAME: hacc.mix_entry
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
// CHECK-LABEL: func.func @result_chain_mix_aic(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
// CHECK-NOT: @vector_index
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.subview
// CHECK-NOT: memref.store
// CHECK-NEXT: return
// CHECK-LABEL: func.func @result_chain_mix_aiv(
// CHECK: %[[INDEX:.*]] = call @vector_index()
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: %[[VIEW:.*]] = memref.subview %[[ALLOC]]
// CHECK: memref.store %[[INDEX]], %[[VIEW]]
// CHECK-LABEL: func.func @host(
// CHECK: call @result_chain()
module {
  func.func private @vector_index() -> index
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
  func.func @result_chain() attributes {
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.core_ratio = #hivm.core_ratio<1, 2>,
      hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %idx = func.call @vector_index() : () -> index
    %alloc = memref.alloc() : memref<16xindex>
    %view = memref.subview %alloc[0] [1] [1] : memref<16xindex> to memref<1xindex>
    memref.store %idx, %view[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xindex>
    return
  }
  func.func @host() attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    func.call @result_chain() : () -> ()
    return
  }
}

// -----

// CHECK-LABEL: func.func @nested_sync_mix_aic(
// CHECK: annotation.mark {{.*}} {logical_block_num}
// CHECK: scf.if
// CHECK: scope.scope
// CHECK-NOT: <VECTOR>
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
// CHECK-NOT: <VECTOR>
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK-NOT: <VECTOR>
// CHECK: return
// CHECK-LABEL: func.func @nested_sync_mix_aiv(
// CHECK: annotation.mark {{.*}} {logical_block_num}
// CHECK: scf.if
// CHECK: scope.scope
// CHECK-NOT: <CUBE>
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
// CHECK-NOT: <CUBE>
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK-NOT: <CUBE>
// CHECK-NOT: hivm.cube_only
// CHECK: return
module {
  func.func @nested_sync(%cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c2 = arith.constant 2 : i32
    annotation.mark %c2 {logical_block_num} : i32
    scf.if %cond {
      scope.scope : () -> () {
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    } {hivm.cube_only}
    return
  }
}

// -----

// CHECK-LABEL: func.func @inline_vf_mix_aic(
// CHECK-NEXT: return
// CHECK-LABEL: func.func @inline_vf_mix_aiv(
// CHECK: scope.scope
// CHECK: ave.hir.vload
// CHECK: ave.hir.masked_store
// CHECK: hivm.vector_function
module {
  func.func @inline_vf(%buffer: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      %mask = ave.hir.pge <ALL> : vector<64xi1>
      %value = ave.hir.vload <NORM> %buffer[%c0]
          : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
      ave.hir.masked_store <NORM_B32> %buffer[%c0], %mask, %value
          : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      scope.return
    } {hivm.vector_function, no_inline, outline}
    return
  }
}

// -----

// CHECK-LABEL: func.func @unused_scope_result_mix_aic(
// CHECK-NOT: @vector_index
// CHECK-NEXT: return
// CHECK-LABEL: func.func @unused_scope_result_mix_aiv(
// CHECK: %[[VALUE:.*]] = {{(func\.)?}}call @vector_index()
// CHECK: scope.return %[[VALUE]] : index
// CHECK: memref.store

module {
  func.func private @vector_index() -> index attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
  func.func @unused_scope_result(%out: memref<1xindex>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %idx = scope.scope : () -> index {
      %v = func.call @vector_index() : () -> index
      scope.return %v : index
    }
    memref.store %idx, %out[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @cfg_result_mix_aic(
// CHECK-NOT: @vector_index
// CHECK-NOT: memref.store
// CHECK: {{^ *}}return
// CHECK-LABEL: func.func @cfg_result_mix_aiv(
// CHECK: %[[VALUE:.*]] = {{(func\.)?}}call @vector_index()
// CHECK: memref.store %[[VALUE]],

module {
  func.func private @vector_index() -> index attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
  func.func @cfg_result(%out: memref<1xindex>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %idx = func.call @vector_index() : () -> index
    cf.br ^bb1
  ^bb1:
    memref.store %idx, %out[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @inferred_load_mix_aic(
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.debug
// CHECK-NOT: memref.alloc
// CHECK-NEXT: return
// CHECK-LABEL: func.func @inferred_load_mix_aiv(
// CHECK: hivm.hir.load
// CHECK: hivm.hir.debug
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.store

module {
  func.func @inferred_load(%src: memref<16xf32>, %dst: memref<16xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %in = memref.alloc() : memref<16xf32>
    %out = memref.alloc() : memref<16xf32>
    hivm.hir.load ins(%src : memref<16xf32>) outs(%in : memref<16xf32>)
    hivm.hir.debug {debugtype = "print", hex = false, prefix = "input"} %in : memref<16xf32>
    hivm.hir.vexp ins(%in : memref<16xf32>) outs(%out : memref<16xf32>)
    hivm.hir.store ins(%out : memref<16xf32>) outs(%dst : memref<16xf32>)
    return
  }
}
