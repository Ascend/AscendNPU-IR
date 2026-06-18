// RUN: bishengir-opt %s -hivm-mark-tightly-coupled-buffer -split-input-file | FileCheck %s

// Every L1/UB alloc gets a fresh tightly_coupled_buffer id; the gmem alloc is
// skipped.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_ub_and_l1
  func.func @mark_ub_and_l1() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %alloc_ub = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    // CHECK: %[[UB:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[UB]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
    %alloc_l1 = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    // CHECK: %[[L1:.*]] = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark %[[L1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
    %alloc_gm = memref.alloc() : memref<16xf32>
    // CHECK: memref.alloc() : memref<16xf32>
    // CHECK-NOT: tightly_coupled_buffer
    return
  }
}

// -----

// Allocs already carrying an id keep it; new allocs avoid colliding with it.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_preserves_existing
  func.func @mark_preserves_existing() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %alloc0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: %[[A0:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[A0]] {{.*}}tightly_coupled_buffer<0>
    %alloc1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: %[[A1:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[A1]] {{.*}}tightly_coupled_buffer<1>
    return
  }
}

// -----

// Without an Ascend950 target the pass is a no-op (no hacc.target -> not 950).
module {
  // CHECK-LABEL: func.func @no_mark_non_950
  func.func @no_mark_non_950() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    // CHECK: memref.alloc()
    // CHECK-NOT: tightly_coupled_buffer
    return
  }
}
