// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v1 -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v2 -split-input-file %s | FileCheck %s

// Contract for hoist-simt-scalar-calls-to-simd: a PIPE_S store into the SIMT
// VF's shared buffer, plus hivm.memory_effect on the declaration argument, must
// make GSS emit one PIPE_S -> PIPE_V flag pair before the call.

// CHECK-LABEL: func.func @straight_line
// CHECK:         memref.store
// CHECK:         hivm.hir.set_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    call @straight_line_scope_0
func.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
func.func private @straight_line_scope_0(memref<20xi8, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.shared_memory}, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}
func.func @straight_line(%d: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %shared = memref.alloc() {hivm.shared_memory} : memref<20xi8, #hivm.address_space<ub>>
  %sh = func.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%d) : (i32) -> i32
  %v0 = memref.view %shared[%c0][] : memref<20xi8, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
  memref.store %sh, %v0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xi32, #hivm.address_space<ub>>
  call @straight_line_scope_0(%shared, %d) : (memref<20xi8, #hivm.address_space<ub>>, i32) -> ()
  return
}

// -----

// Negative control: identical IR with no hivm.memory_effect on the shared
// argument. GSS must emit nothing. This is the regression guard for the exact
// bug being fixed.

// CHECK-LABEL: func.func @no_effect_attr
// CHECK-NOT:     <PIPE_S>, <PIPE_V>
func.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
func.func private @no_effect_attr_scope_0(memref<20xi8, #hivm.address_space<ub>> {hivm.shared_memory}, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}
func.func @no_effect_attr(%d: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %shared = memref.alloc() {hivm.shared_memory} : memref<20xi8, #hivm.address_space<ub>>
  %sh = func.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%d) : (i32) -> i32
  %v0 = memref.view %shared[%c0][] : memref<20xi8, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
  memref.store %sh, %v0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xi32, #hivm.address_space<ub>>
  call @no_effect_attr_scope_0(%shared, %d) : (memref<20xi8, #hivm.address_space<ub>>, i32) -> ()
  return
}

// -----

// Scalar calls hoisted above the loop that holds the call: one pair before the
// loop, none inside it.

// CHECK-LABEL: func.func @loop_hoisted
// CHECK:         memref.store
// CHECK-NEXT:    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    scf.for
// CHECK-NOT:     <PIPE_S>, <PIPE_V>
// CHECK:         func.call @loop_hoisted_scope_0
// CHECK-NOT:     <PIPE_S>, <PIPE_V>
func.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
func.func private @loop_hoisted_scope_0(memref<20xi8, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.shared_memory}, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}
func.func @loop_hoisted(%d: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc() {hivm.shared_memory} : memref<20xi8, #hivm.address_space<ub>>
  %sh = func.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%d) : (i32) -> i32
  %v0 = memref.view %shared[%c0][] : memref<20xi8, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
  memref.store %sh, %v0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xi32, #hivm.address_space<ub>>
  scf.for %i = %c0 to %c4 step %c1 {
    func.call @loop_hoisted_scope_0(%shared, %d) : (memref<20xi8, #hivm.address_space<ub>>, i32) -> ()
  }
  return
}

// -----

// Scalar calls inside the loop: the RAW pair before the call, and the V -> S
// anti-dependency pair that lets the next iteration overwrite the slot.

// CHECK-LABEL: func.func @loop_inside
// CHECK:         hivm.hir.set_flag[<PIPE_V>, <PIPE_S>
// CHECK-NEXT:    scf.for
// CHECK:         hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>
// CHECK-NEXT:    memref.store
// CHECK-NEXT:    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>
// CHECK-NEXT:    func.call @loop_inside_scope_0
// CHECK-NEXT:    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>
// CHECK:         hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>
func.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
func.func private @loop_inside_scope_0(memref<20xi8, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.shared_memory}, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}
func.func @loop_inside(%d: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc() {hivm.shared_memory} : memref<20xi8, #hivm.address_space<ub>>
  scf.for %i = %c0 to %c4 step %c1 {
    %sh = func.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%d) : (i32) -> i32
    %v0 = memref.view %shared[%c0][] : memref<20xi8, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
    memref.store %sh, %v0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<1xi32, #hivm.address_space<ub>>
    func.call @loop_inside_scope_0(%shared, %d) : (memref<20xi8, #hivm.address_space<ub>>, i32) -> ()
  }
  return
}
