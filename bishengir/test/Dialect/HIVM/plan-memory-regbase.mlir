// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory -split-input-file -verify-diagnostics | FileCheck %s

// expected-error@below {{ub overflow, requires 1966080 bits while 1769472 bits available!}}
func.func @invalid_alloc_for_mix(%arg0: memref<61440xf16, #hivm.address_space<gm>>, %arg1: memref<61440xf16, #hivm.address_space<gm>>) -> f16 attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<61440xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<61440xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<61440xf16, #hivm.address_space<gm>>) outs(%alloc : memref<61440xf16, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<61440xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<61440xf16, #hivm.address_space<ub>>)
  %0 = memref.load %alloc[%c0] : memref<61440xf16, #hivm.address_space<ub>>
  %1 = memref.load %alloc_0[%c0] : memref<61440xf16, #hivm.address_space<ub>>
  %2 = arith.mulf %0, %1 : f16
  return %2 : f16
}

// -----

// CHECK: hivm.hir.pointer_cast({{.*}})
func.func @valid_alloc_for_simd(%arg0: memref<61440xf16, #hivm.address_space<gm>>, %arg1: memref<61440xf16, #hivm.address_space<gm>>) -> f16 attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<61440xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<61440xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<61440xf16, #hivm.address_space<gm>>) outs(%alloc : memref<61440xf16, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<61440xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<61440xf16, #hivm.address_space<ub>>)
  %0 = memref.load %alloc[%c0] : memref<61440xf16, #hivm.address_space<ub>>
  %1 = memref.load %alloc_0[%c0] : memref<61440xf16, #hivm.address_space<ub>>
  %2 = arith.mulf %0, %1 : f16
  return %2 : f16
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_950z">} {
  func.func @test_tightly_coupled_buffer_in_AIC(%arg0: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK: %[[ARG1:.*]] = hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16x16xf32, #hivm.address_space<cbuf>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%arg1 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16x16xf32, #hivm.address_space<cbuf>>)
    // CHECK: %[[ARG2:.*]] = hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<16x16xf16, #hivm.address_space<cbuf>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[ARG3:.*]] = hivm.hir.pointer_cast(%[[CONST2:.*]]) : memref<16x16xf32, #hivm.address_space<cc>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
    hivm.hir.load ins(%arg1 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%alloc_3 : memref<16x16xf32, #hivm.address_space<cc>>)
    // CHECK: %[[ARG4:.*]] = hivm.hir.pointer_cast(%[[CONST3:.*]]) : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {enable_nz2nd, l0c_to_ub} ins(%alloc_3 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc_4 : memref<16x16xf32, #hivm.address_space<ub>>)
    return
  }
  func.func @test_tightly_coupled_buffer_in_AIV(%arg0: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}  {
    // CHECK: %[[ARG5:.*]] = hivm.hir.pointer_cast(%[[CONST4:.*]]) : memref<16x16xf16, #hivm.address_space<ub>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
    // CHECK: %[[ARG6:.*]] = hivm.hir.pointer_cast(%[[CONST1]]) : memref<16x16xf16, #hivm.address_space<cbuf>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%alloc : memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_2 : memref<16x16xf16, #hivm.address_space<cbuf>>) {ub_to_l1}
    // CHECK: %[[ARG7:.*]] = hivm.hir.pointer_cast(%[[CONST3]]) : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf32, #hivm.address_space<ub>>
    return
  }
}