
// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// IndirectLoad is a gather-style op: `src` is indexed by the runtime
// `offsets` tensor rather than a linear stride pattern, so its
// dst/offsets/mask/other operands do not benefit from stride alignment.
// MarkStrideAlign must skip IndirectLoadOp entirely to avoid unnecessary
// UB alloc padding in the subsequent EnableStrideAlign pass.

// -----

// Even when dst / offsets / mask / other are UB memrefs whose innermost
// dim (3 * f32 = 12 bytes) is not naturally 32-byte aligned, no mark
// should be produced for any operand of the IndirectLoadOp.
// CHECK-LABEL: @test_indirect_load_full_operands_not_marked
// CHECK-NOT: hivm.stride_align_dims
// CHECK-NOT: hivm.stride_align_value_in_byte
func.func @test_indirect_load_full_operands_not_marked(
    %src : memref<?xf32, #hivm.address_space<gm>>) {
  %offsets = memref.alloc() : memref<4x3xi32, #hivm.address_space<ub>>
  %mask    = memref.alloc() : memref<4x3xi1,  #hivm.address_space<ub>>
  %other   = memref.alloc() : memref<4x3xf32, #hivm.address_space<ub>>
  %dst     = memref.alloc() : memref<4x3xf32, #hivm.address_space<ub>>
  hivm.hir.indirect_load
      ins(%src : memref<?xf32, #hivm.address_space<gm>>,
          %offsets : memref<4x3xi32, #hivm.address_space<ub>>,
          %mask    : memref<4x3xi1,  #hivm.address_space<ub>>,
          %other   : memref<4x3xf32, #hivm.address_space<ub>>)
      outs(%dst : memref<4x3xf32, #hivm.address_space<ub>>)
  return
}

// -----

// Minimal form (no mask, no other) — still must be skipped.
// CHECK-LABEL: @test_indirect_load_minimal_operands_not_marked
// CHECK-NOT: hivm.stride_align_dims
// CHECK-NOT: hivm.stride_align_value_in_byte
func.func @test_indirect_load_minimal_operands_not_marked(
    %src : memref<?xf32, #hivm.address_space<gm>>) {
  %offsets = memref.alloc() : memref<4x3xi32, #hivm.address_space<ub>>
  %dst     = memref.alloc() : memref<4x3xf32, #hivm.address_space<ub>>
  hivm.hir.indirect_load
      ins(%src : memref<?xf32, #hivm.address_space<gm>>,
          %offsets : memref<4x3xi32, #hivm.address_space<ub>>)
      outs(%dst : memref<4x3xf32, #hivm.address_space<ub>>)
  return
}

// -----

// Sanity check: a regular DMA hivm.hir.load on a UB buffer with a
// discontiguous inner dim (last dim size 4 but stride 4 with GM source
// strided<[1, 1]>) *does* still produce a mark. This proves the skip is
// specific to IndirectLoadOp and does not accidentally disable marking
// for other ops sharing the same code path.
// CHECK-LABEL: @test_hir_load_still_gets_marked
// CHECK: annotation.mark %{{.*}} {hivm.stride_align_dims = {{.*}}, hivm.stride_align_value_in_byte = {{.*}}}
func.func @test_hir_load_still_gets_marked(
    %gm : memref<8x4xf32, strided<[1, 1]>, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<8x4xf32, #hivm.address_space<ub>>
  hivm.hir.load
      ins(%gm : memref<8x4xf32, strided<[1, 1]>, #hivm.address_space<gm>>)
      outs(%ub : memref<8x4xf32, #hivm.address_space<ub>>)
  return
}

// -----

// Mixed IR: an IndirectLoadOp and a regular hivm.hir.load in the same
// function. Only the hivm.hir.load's UB output should be marked; the
// IndirectLoad's dst must not.
// CHECK-LABEL: @test_indirect_load_mixed_with_hir_load
// CHECK: %[[DST_INDIRECT:.*]] = memref.alloc() : memref<4x3xf32, #hivm.address_space<ub>>
// CHECK-NOT: annotation.mark %[[DST_INDIRECT]]
// CHECK: %[[UB:.*]] = memref.alloc() : memref<8x4xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[UB]] {hivm.stride_align_dims = {{.*}}, hivm.stride_align_value_in_byte = {{.*}}}
func.func @test_indirect_load_mixed_with_hir_load(
    %src : memref<?xf32, #hivm.address_space<gm>>,
    %gm  : memref<8x4xf32, strided<[1, 1]>, #hivm.address_space<gm>>) {
  %offsets = memref.alloc() : memref<4x3xi32, #hivm.address_space<ub>>
  %dst     = memref.alloc() : memref<4x3xf32, #hivm.address_space<ub>>
  hivm.hir.indirect_load
      ins(%src : memref<?xf32, #hivm.address_space<gm>>,
          %offsets : memref<4x3xi32, #hivm.address_space<ub>>)
      outs(%dst : memref<4x3xf32, #hivm.address_space<ub>>)
  %ub = memref.alloc() : memref<8x4xf32, #hivm.address_space<ub>>
  hivm.hir.load
      ins(%gm : memref<8x4xf32, strided<[1, 1]>, #hivm.address_space<gm>>)
      outs(%ub : memref<8x4xf32, #hivm.address_space<ub>>)
  return
}
