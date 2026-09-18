// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// Regbase (Ascend950) variant of the IndirectLoad skip test.
// IndirectLoad performs gather-style access: the source is indexed by the
// runtime `offsets` tensor rather than a linear stride pattern. Its
// dst/offsets/mask/other operands do not benefit from stride alignment;
// marking them would only trigger unnecessary UB alloc padding in
// EnableStrideAlign. MarkStrideAlign must skip IndirectLoadOp entirely.

// -----

// CHECK-LABEL: @test_indirect_load_regbase_not_marked
// CHECK-NOT: hivm.stride_align_dims
// CHECK-NOT: hivm.stride_align_value_in_byte
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_indirect_load_regbase_not_marked(
      %src : memref<?xf32, #hivm.address_space<gm>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
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
}

// -----

// Sanity check on the regbase path: a regular hivm.hir.vexp on a UB
// buffer whose innermost dim (3 * f32 = 12 bytes) is not naturally
// 32-byte aligned *does* still produce a mark. This proves the skip is
// specific to IndirectLoadOp and does not accidentally disable marking
// on the regbase code path.
// CHECK-LABEL: @test_vexp_regbase_still_gets_marked
// CHECK: annotation.mark %{{.*}} {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_vexp_regbase_still_gets_marked()
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %a = memref.alloc() : memref<4x8x3xf32, #hivm.address_space<ub>>
    hivm.hir.vexp
        ins(%a : memref<4x8x3xf32, #hivm.address_space<ub>>)
        outs(%a : memref<4x8x3xf32, #hivm.address_space<ub>>)
    return
  }
}
