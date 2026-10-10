// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase -split-input-file -verify-diagnostics | FileCheck %s

// SIMD-only coverage for the "no option" case: the `shared-mem-dynamic-size`
// option defaults to -1 when unset, and SIMT/MIX functions are then rejected
// by the range check before planning, so this file only contains SIMD cases.
// The explicit option-value coverage (SIMT/MIX/SIMD) lives in
// plan-memory-shared-mem-dynamic-size.mlir.
//
// SIMD ignores the option entirely, so the effective UB capacity stays the
// full device size (Ascend950PR_950z: 2031616 bits) whether or not the
// option is passed.

// -----
// SIMD planning succeeds without the option.

// CHECK-LABEL: func.func @simd_no_option_success
// CHECK: hivm.hir.pointer_cast
func.func @simd_no_option_success(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// SIMD keeps the full UB capacity (2031616 bits) without the option.
// Pinned exactly, no regex.

// expected-error@below {{ub overflow, requires 3932160 bits while 2031616 bits available!}}
func.func @simd_no_option_full_ub_capacity(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc : memref<61440xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<61440xf32, #hivm.address_space<ub>>)
  %0 = memref.load %alloc[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %1 = memref.load %alloc_0[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %2 = arith.mulf %0, %1 : f32
  return
}
