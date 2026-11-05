// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s

// Every reg-base GM->UB load of rank two or more resolves to the NDDMA
// template of its own rank, capped at the five loop levels the descriptor
// drives; a higher rank keeps five and MultiDimOpToLibraryCallPattern peels the
// rest into surrounding loops. Nothing here inspects the strides.
//
// The rank does not pick the hardware mechanism. load_gm_to_ubuf_nd_core tries
// load_gm_to_ubuf_nd_by_block_dma first and reaches the descriptor only when no
// burst applies, so a monotonic tile moves by block DMA whichever template it
// lands in, and the choice of template is settled with the real strides in
// hand rather than guessed from the types here.
//
// These cases run the conversion on its own, so a shape hivm-flatten-ops would
// have collapsed still arrives at full rank. Most of the tiles below do not
// survive that pass: a contiguous 2x4x8x16 tile reaches the real pipeline as a
// single 1D load. What Triton leaves at these ranks is a permuting load whose
// inversion sits in the innermost axes - a 3D permute lowers to
// load_gm_to_ubuf_3d and a 4D permute to load_gm_to_ubuf_4d, with or without a
// mask.

// A rank-4 permuting load: the UB view has an inverted adjacent stride pair
// (32 < 256), so one 4D descriptor covers it.
// CHECK-LABEL: @load_rank4_permuting
// CHECK: call @load_gm_to_ubuf_4d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank4_permuting(%src: memref<4x16x8x32xf16, #hivm.address_space<gm>>,
                                  %dst: memref<4x16x8x32xf16, strided<[4096, 32, 256, 1]>, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<4x16x8x32xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<4x16x8x32xf16, strided<[4096, 32, 256, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A rank-5 permuting load: every adjacent pair inverted, all five loop
// levels in use.
// CHECK-LABEL: @load_rank5_permuting
// CHECK: call @load_gm_to_ubuf_5d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank5_permuting(%src: memref<2x3x4x5x6xf16, #hivm.address_space<gm>>,
                                  %dst: memref<2x3x4x5x6xf16, strided<[1, 2, 6, 24, 120]>, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<2x3x4x5x6xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x3x4x5x6xf16, strided<[1, 2, 6, 24, 120]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A rank-4 load with no inverted pair needs no descriptor, so it keeps the 3D
// template and the outer axis becomes a loop.
// CHECK-LABEL: @load_rank4_not_permuting
// CHECK: call @load_gm_to_ubuf_4d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank4_not_permuting(%src: memref<4x16x8x32xf16, #hivm.address_space<gm>>,
                                      %dst: memref<4x16x8x32xf16, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<4x16x8x32xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<4x16x8x32xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A rank-6 permuting load exceeds the five descriptor loops, so one axis is
// peeled into a loop and the body is still a five-loop descriptor. That beats
// peeling all the way down to three.
// CHECK-LABEL: @load_rank6_permuting
// CHECK: scf.for
// CHECK: call @load_gm_to_ubuf_5d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank6_permuting(%src: memref<2x2x2x2x2x4xf16, #hivm.address_space<gm>>,
                                  %dst: memref<2x2x2x2x2x4xf16, strided<[1, 2, 4, 8, 32, 16]>, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<2x2x2x2x2x4xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x2x2x2x2x4xf16, strided<[1, 2, 4, 8, 32, 16]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// The inversion is between axes 0 and 1 only. Those axes are peeled into
// loops, and what remains is a plain block-DMA shape, so the descriptor is
// not needed and the 3D templates are the right choice.
// CHECK-LABEL: @load_rank6_outer_inversion_only
// CHECK: scf.for
// CHECK: call @load_gm_to_ubuf_5d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank6_outer_inversion_only(%src: memref<2x2x2x2x2x4xf16, #hivm.address_space<gm>>,
                                             %dst: memref<2x2x2x2x2x4xf16, strided<[32, 64, 16, 8, 4, 1]>, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<2x2x2x2x2x4xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x2x2x2x2x4xf16, strided<[32, 64, 16, 8, 4, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Strides known only at run time need no special handling, because nothing is
// inferred from them. Kept as a guard: an inspection reintroduced here would
// have to answer for these, and could not.
// CHECK-LABEL: @load_rank4_dynamic_strides
// CHECK: call @load_gm_to_ubuf_4d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank4_dynamic_strides(%src: memref<4x16x8x32xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>,
                                        %dst: memref<4x16x8x32xf16, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<4x16x8x32xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>)
                  outs(%dst : memref<4x16x8x32xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A dynamic stride only on the UB side is equally unprovable.
// CHECK-LABEL: @load_rank5_dynamic_dst_strides
// CHECK: call @load_gm_to_ubuf_5d_half
// CHECK-NOT: call @load_gm_to_ubuf_3d_half
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @load_rank5_dynamic_dst_strides(%src: memref<2x4x16x8x32xf16, #hivm.address_space<gm>>,
                                            %dst: memref<2x4x16x8x32xf16, strided<[?, ?, ?, ?, ?], offset: ?>, #hivm.address_space<ub>>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.load ins(%src : memref<2x4x16x8x32xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x4x16x8x32xf16, strided<[?, ?, ?, ?, ?], offset: ?>, #hivm.address_space<ub>>)
    return
  }
}
