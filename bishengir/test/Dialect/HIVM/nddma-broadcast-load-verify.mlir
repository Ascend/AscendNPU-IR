// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
// RUN: bishengir-opt %s -split-input-file -verify-diagnostics -hivm-aggregated-decompose-op | FileCheck %s --check-prefix=DECOMPOSE

// `broadcast_axes` on a load is a descriptor property: the source axis has
// extent 1, the destination axis is larger, and every other axis agrees. Only
// A6 GM -> UB transfer is defined; rank-3/dynamic cases have dedicated tests.

// A6, axis 0: accepted and preserved through decomposition.
// DECOMPOSE-LABEL: func.func @brc_axis0(
// DECOMPOSE: hivm.hir.load {{.*}}broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_axis0(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}


}
// -----

// A6, axis 1: accepted.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_axis1(%src: memref<4x1xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x1xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}


}
// -----

// Unknown target must not opt into the A6-only semantic contract.
func.func @brc_no_target(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  // expected-error @+1 {{broadcast_axes is only supported on A6 (Ascend960/dav-920r1)}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}

// -----

// Any other target keeps the unfused path, so the attribute is rejected there.
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @brc_not_a6(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast_axes is only supported on A6 (Ascend960/dav-920r1)}}
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    return
  }
}

// -----

// The axis that is not broadcast must keep the same extent on both sides.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_axis_mismatch(%src: memref<1x16xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  // expected-error @+1 {{non-broadcast axis 1 must have the same extent on src and dst, got src 16 and dst 32}}
  hivm.hir.load ins(%src : memref<1x16xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}


}
// -----

// The broadcast axis needs a unit source and a growing destination.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_not_unit(%src: memref<4x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<8x32xf32, #hivm.address_space<ub>>
  // expected-error @+1 {{broadcast axis 0 needs src extent 1 and dst extent > 1, got src 4 and dst 8}}
  hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<8x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}


}
// -----

// Rank 3 requires every expanded axis to be declared.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_rank3(%src: memref<1x1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x4x32xf32, #hivm.address_space<ub>>
  // expected-error @+1 {{non-broadcast axis 1 must have the same extent on src and dst}}
  hivm.hir.load ins(%src : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}


}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @unknown_space(%s: memref<1x32xf32>, %d: memref<4x32xf32>) {
  // expected-error @+1 {{broadcast_axes only supports a GM -> UB transfer}}
  hivm.hir.load ins(%s : memref<1x32xf32>) outs(%d : memref<4x32xf32>) broadcast_axes = [0]
  return
}
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @unsupported_i64(%s: memref<1x32xi64, #hivm.address_space<gm>>, %d: memref<4x32xi64, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast_axes has an unsupported element type}}
  hivm.hir.load ins(%s : memref<1x32xi64, #hivm.address_space<gm>>) outs(%d : memref<4x32xi64, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}

// -----

// Initialization has no defined meaning for the new broadcast load contract.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_with_init(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast_axes cannot be combined with initialization yet}}
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) init_out_buffer = true broadcast_axes = [0]
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_with_conditional_init(%src: memref<1x32xf32, #hivm.address_space<gm>>, %cond: i1) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast_axes cannot be combined with initialization yet}}
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) init_out_buffer = true init_condition = %cond : i1 broadcast_axes = [0]
    return
  }
}

// -----

// Explicit false is the default, not an initialization request. A narrow row
// remains valid IR; 128-byte row width is only a template selection policy.
// DECOMPOSE-LABEL: func.func @brc_false_init_narrow(
// DECOMPOSE: hivm.hir.load {{.*}}broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_false_init_narrow(%src: memref<1x16xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x16xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x16xf32, #hivm.address_space<ub>>) init_out_buffer = false broadcast_axes = [0]
    return
  }
}

// -----

// Ordinary loads must remain outside the broadcast-only initialization guard.
// DECOMPOSE-LABEL: func.func @ordinary_no_init(
// DECOMPOSE: hivm.hir.load
// DECOMPOSE-NOT: broadcast_axes
// DECOMPOSE: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @ordinary_no_init(%src: memref<4x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>)
    return
  }
}


// -----
// An empty axis list carries no broadcast information.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_empty_axes(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast_axes expects at least one axis}}
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = []
    return
  }
}

// -----
// A repeated axis would zero the same descriptor level twice.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_duplicate_axes(%src: memref<1x1x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast axes must be unique}}
    hivm.hir.load ins(%src : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 0]
    return
  }
}

// -----
// Axis numbers index the destination rank.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_axis_out_of_range(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast axis 2 is out of range for rank 2}}
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [2]
    return
  }
}

// -----
// broadcast_axes is defined for rank 2 and 3 only.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @brc_rank1(%src: memref<32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    // expected-error @+1 {{broadcast_axes only supports rank 2 or 3 loads}}
    hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%ub : memref<32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @rank4_broadcast(%src: memref<1x2x2x8xf32, #hivm.address_space<gm>>, %dst: memref<3x2x2x8xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast_axes only supports rank 2 or 3 loads}}
  hivm.hir.load ins(%src : memref<1x2x2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x2x2x8xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @negative_axis(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<3x8xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast axis -1 is out of range for rank 2}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x8xf32, #hivm.address_space<ub>>) broadcast_axes = [-1]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @broadcast_dst_not_larger(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x8xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast axis 0 needs src extent 1 and dst extent > 1, got src 1 and dst 1}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x8xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @tensor_broadcast(%src: tensor<1x8xf32>, %dst: tensor<3x8xf32>) {
  // expected-error @+1 {{broadcast_axes only supports a GM -> UB transfer}}
  %r = hivm.hir.load ins(%src : tensor<1x8xf32>) outs(%dst : tensor<3x8xf32>) broadcast_axes = [0] -> tensor<3x8xf32>
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @broadcast_with_transpose(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast_axes does not support the implicit transpose with the last axis}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) may_implicit_transpose_with_last_axis = true broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @pad_null_broadcast(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadNull> broadcast_axes = [0]
  return
 }
}
