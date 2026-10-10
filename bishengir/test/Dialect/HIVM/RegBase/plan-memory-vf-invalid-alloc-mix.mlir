// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase=shared-mem-dynamic-size=221184 -split-input-file -verify-diagnostics

// MIX ub-overflow with an explicit in-range `shared-mem-dynamic-size`: the
// option gives 221184 bytes, i.e. 221184 * 8 = 1769472 bits of UB, and two
// live 61440xf32 buffers require 3932160 bits, which overflows.
//
// This case lives in its own file so it only runs with the explicit option.
// Without the option the pass defaults shared-mem-dynamic-size to -1 and the
// MIX range check rejects the function before planning (see
// plan-memory-shared-mem-dynamic-size.mlir for the option-value matrix and
// plan-memory-shared-mem-dynamic-size-default.mlir for unset-option SIMD
// coverage).

// expected-error@below {{ub overflow, requires 3932160 bits while 1769472 bits available!}}
func.func @invalid_alloc_for_mix(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
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
