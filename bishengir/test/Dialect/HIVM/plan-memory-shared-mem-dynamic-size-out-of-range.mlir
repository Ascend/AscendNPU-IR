// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase=shared-mem-dynamic-size=100000 -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase=shared-mem-dynamic-size=300000 -split-input-file -verify-diagnostics | FileCheck %s

// Out-of-range `shared-mem-dynamic-size` values must reject SIMT and MIX
// vector functions before any buffer planning with:
//   PlanMemory Fail : SharedMemDynamicSize have to in range
//   [122880, 221184], but got <value>!
//
// 100000 is below the lower bound (122880), 300000 is above the upper bound
// (221184). Every section runs under both RUN lines, so the expected-error
// regex accepts either "but got" value. If an out-of-range value were
// wrongly accepted, the corresponding function would plan successfully and
// the unfulfilled expected-error would fail verification.

// -----
// SIMT is rejected for out-of-range option values.

// expected-error-re@below {{PlanMemory Fail : SharedMemDynamicSize have to in range [122880, 221184], but got {{(100000|300000)}}!}}
func.func @simt_out_of_range_rejected(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// MIX is rejected for out-of-range option values.

// expected-error-re@below {{PlanMemory Fail : SharedMemDynamicSize have to in range [122880, 221184], but got {{(100000|300000)}}!}}
func.func @mix_out_of_range_rejected(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}

// -----
// SIMD ignores the option entirely: no range diagnostic is emitted even for
// out-of-range values, and planning succeeds normally.

// CHECK-LABEL: func.func @simd_ignores_out_of_range
// CHECK: hivm.hir.pointer_cast
func.func @simd_ignores_out_of_range(%arg0: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
  return
}
