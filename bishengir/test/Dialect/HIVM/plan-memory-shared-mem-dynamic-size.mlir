// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase=shared-mem-dynamic-size=122880 -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase=shared-mem-dynamic-size=221184 -split-input-file -verify-diagnostics | FileCheck %s

// Verifies that the effective UB capacity used by the RegBase PlanMemory
// planner follows the `shared-mem-dynamic-size` option for SIMT/MIX vector
// functions, while SIMD keeps the full UB size.
//
// Device spec (Ascend950PR_950z):
//   UbSize            = 2031616 bits (253952 bytes)
//   MinimalDCacheSize =  262144 bits ( 32768 bytes)
//   MaximumDCacheSize = 1048576 bits (131072 bytes)
//   Valid option range: [122880, 221184] bytes
//
// RUN 1 (value 122880, inclusive lower bound): capacity =
//       122880 * 8 = 983040 bits.
// RUN 2 (value 221184, inclusive upper bound): capacity =
//       221184 * 8 = 1769472 bits.
//
// There is deliberately no "no option" RUN here: SIMT/MIX functions are
// rejected by the range check when the option is unset (it defaults to
// -1). The unset-option SIMD-only coverage lives in
// plan-memory-shared-mem-dynamic-size-default.mlir.
//
// The capacity is only observable through the "available" number in the
// ub-overflow diagnostic, so the overflow tests below match the capacity of
// whichever RUN is executing with a regex. Diagnostic directives
// (expected-error*) apply under every RUN regardless of the FileCheck
// prefix.

// -----
// SIMT planning succeeds with in-range option values under both RUNs.

// CHECK-LABEL: func.func @simt_in_range_success
// CHECK: hivm.hir.pointer_cast
func.func @simt_in_range_success(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// MIX planning succeeds with in-range option values under all RUNs.

// CHECK-LABEL: func.func @mix_in_range_success
// CHECK: hivm.hir.pointer_cast
func.func @mix_in_range_success(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// SIMD planning succeeds with in-range option values under all RUNs.

// CHECK-LABEL: func.func @simd_in_range_success
// CHECK: hivm.hir.pointer_cast
func.func @simd_in_range_success(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// SIMT: two live 61440xf32 buffers require 3932160 bits, which exceeds the
// effective capacity under both RUNs. The "available" number in the overflow
// message must be the adjusted capacity (1769472 or 983040 bits), proving the
// planner actually used sharedMemDynamicSize as the SIMT UB size.

// expected-error-re@below {{ub overflow, requires 3932160 bits while {{(1769472|983040)}} bits available!}}
func.func @simt_effective_ub_capacity(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
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

// -----
// MIX: same capacity adjustment as SIMT.

// expected-error-re@below {{ub overflow, requires 3932160 bits while {{(1769472|983040)}} bits available!}}
func.func @mix_effective_ub_capacity(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
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

// -----
// SIMD: the option is ignored, so the capacity stays the full UB size
// (2031616 bits) under all RUNs. Pinned exactly, no regex.

// expected-error@below {{ub overflow, requires 3932160 bits while 2031616 bits available!}}
func.func @simd_full_ub_capacity(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
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

// -----
// MIX (f16 variant): two live 61440xf16 buffers require 1966080 bits, which
// exceeds the effective capacity under all RUNs (1769472 bits without the
// option and with the explicit 221184 RUN, 983040 bits for the 122880 RUN).

// expected-error-re@below {{ub overflow, requires 1966080 bits while {{(1769472|983040)}} bits available!}}
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
// SIMD (f16 variant): 1966080 bits fits the full UB size (2031616 bits),
// and the option is ignored, so planning succeeds under every RUN. This
// mirrors @invalid_alloc_for_mix above: the same buffers overflow for MIX
// but fit for SIMD.

// CHECK-LABEL: func.func @valid_alloc_for_simd
// CHECK: hivm.hir.pointer_cast
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
