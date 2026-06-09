// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file | FileCheck %s
//
// Minimal testcase set for hivm-bind-sub-block.
// Inspired by tile-and-bind-sub-block.mlir.
//
// Tile-and-bind behavior guide (for future debugging/agents):
// - Success path:
//   - `hivm.hir.store` gets `{tiled_op}` and loop gets
//     `{map_for_to_forall, mapping = [#hivm.sub_block<x>]}`.
// - Fallback path:
//   - `scf.if ... {limit_sub_block_id0}` is inserted to run only sub-block 0
//     when safe tiling cannot be proven.
// - `annotation.mark`: generic metadata carrier op for later passes.
// - `hivm.tightly_coupled_buffer`: marks UB/L1 buffers tracked for mix tiling.
// - `hivm.tiling_dim`: selected axis metadata attached on marked buffers.
// - `tiling_dim_mapping`: reshape/expand helper metadata to propagate tiling axis.
// - Minimal testcase:
//   1) start with `scf.for` + `hivm.hir.store`,
//   2) feed store from `tensor.extract_slice`,
//   3) keep shapes small/static,
//   4) keep attrs: `hacc.function_kind`, `hivm.func_core_type`,
//      `hivm.part_of_mix`, `mix_mode`.

#off = affine_map<()[s0] -> (s0 * 128)>
module {
  // CHECK-LABEL: func.func @mini_bind_sub_block
  // CHECK: scf.for
  // CHECK: hivm.hir.store{{.*}} {tiled_op}
  // CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  func.func @mini_bind_sub_block(%arg0: tensor<256xf32>, %arg1: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off()[%i]
      %slice = tensor.extract_slice %arg0[%offset] [128] [1] {to_be_bubbled_slice} : tensor<256xf32> to tensor<128xf32>
      hivm.hir.store ins(%slice : tensor<128xf32>) outs(%arg1 : memref<128xf32>)
    }
    return
  }
}

// -----

#off = affine_map<()[s0] -> (s0 * 8)>
module {
  // Fourth variant: explicitly surface hivm.tiling_dim on annotation.mark.
  // This chunk is meant to observe the attr in output even when bind falls back.
  // CHECK-LABEL: func.func @mini_bind_sub_block_with_tiling_dim
  // CHECK: annotation.mark
  // CHECK: hivm.tiling_dim =
  // CHECK: hivm.hir.store
  // CHECK: limit_sub_block_id0
  func.func @mini_bind_sub_block_with_tiling_dim(%arg0: tensor<16x16xf32>, %arg1: memref<16x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<16x16xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off()[%i]
      %src = tensor.extract_slice %arg0[%offset, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      %dst = memref.subview %cast[%offset, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      hivm.hir.store ins(%src : tensor<8x16xf32>) outs(%dst : memref<8x16xf32, strided<[16, 1], offset: ?>>)
    }
    return
  }
}

// -----

#off = affine_map<()[s0] -> (s0 * 8)>
module {
  // Fifth variant: group-tiling from the tiling_dim case.
  // Two independent groups, each writing to its own tightly-coupled buffer.
  // CHECK-LABEL: func.func @mini_group_tiling_with_tiling_dim
  // CHECK: annotation.mark
  // CHECK-DAG: hivm.tiling_dim = 1 : index
  // CHECK-DAG: hivm.tiling_dim = 0 : index
  // CHECK: hivm.hir.vadd
  // CHECK: hivm.hir.vmul
  // CHECK-COUNT-2: hivm.hir.store
  // CHECK: limit_sub_block_id0
  func.func @mini_group_tiling_with_tiling_dim(%arg0: tensor<1x16x14x5xf32>, %arg1: tensor<1x16x14x5xf32>, %arg2: tensor<16x1x14x5xf32>, %arg3: tensor<16x1x14x5xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %allocA = memref.alloc() : memref<1x16x14x5xf32, #hivm.address_space<ub>>
    %allocB = memref.alloc() : memref<16x1x14x5xf32, #hivm.address_space<ub>>
    annotation.mark %allocA {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index} : memref<1x16x14x5xf32, #hivm.address_space<ub>>
    annotation.mark %allocB {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index} : memref<16x1x14x5xf32, #hivm.address_space<ub>>
    %castA = memref.memory_space_cast %allocA : memref<1x16x14x5xf32, #hivm.address_space<ub>> to memref<1x16x14x5xf32>
    %castB = memref.memory_space_cast %allocB : memref<16x1x14x5xf32, #hivm.address_space<ub>> to memref<16x1x14x5xf32>
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off()[%i]
      %a0 = tensor.extract_slice %arg0[0, %offset, 0, 0] [1, 8, 14, 5] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<1x16x14x5xf32> to tensor<1x8x14x5xf32>
      %a1 = tensor.extract_slice %arg1[0, %offset, 0, 0] [1, 8, 14, 5] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<1x16x14x5xf32> to tensor<1x8x14x5xf32>
      %b0 = tensor.extract_slice %arg2[%offset, 0, 0, 0] [8, 1, 14, 5] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x1x14x5xf32> to tensor<8x1x14x5xf32>
      %b1 = tensor.extract_slice %arg3[%offset, 0, 0, 0] [8, 1, 14, 5] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x1x14x5xf32> to tensor<8x1x14x5xf32>
      %bufA = tensor.empty() : tensor<1x8x14x5xf32>
      %bufB = tensor.empty() : tensor<8x1x14x5xf32>
      %groupA = hivm.hir.vadd ins(%a0, %a1 : tensor<1x8x14x5xf32>, tensor<1x8x14x5xf32>) outs(%bufA : tensor<1x8x14x5xf32>) -> tensor<1x8x14x5xf32>
      %groupB = hivm.hir.vmul ins(%b0, %b1 : tensor<8x1x14x5xf32>, tensor<8x1x14x5xf32>) outs(%bufB : tensor<8x1x14x5xf32>) -> tensor<8x1x14x5xf32>
      %dstA = memref.subview %castA[0, %offset, 0, 0] [1, 8, 14, 5] [1, 1, 1, 1] : memref<1x16x14x5xf32> to memref<1x8x14x5xf32, strided<[1120, 70, 5, 1], offset: ?>>
      %dstB = memref.subview %castB[%offset, 0, 0, 0] [8, 1, 14, 5] [1, 1, 1, 1] : memref<16x1x14x5xf32> to memref<8x1x14x5xf32, strided<[70, 70, 5, 1], offset: ?>>
      hivm.hir.store ins(%groupA : tensor<1x8x14x5xf32>) outs(%dstA : memref<1x8x14x5xf32, strided<[1120, 70, 5, 1], offset: ?>>)
      hivm.hir.store ins(%groupB : tensor<8x1x14x5xf32>) outs(%dstB : memref<8x1x14x5xf32, strided<[70, 70, 5, 1], offset: ?>>)
    }
    return
  }
}

// -----

#off = affine_map<()[s0] -> (s0 * 128)>
module {
  // CHECK-LABEL: func.func @mini_bind_sub_block_with_ops
  // CHECK: hivm.hir.vadd
  // CHECK: hivm.hir.vmul
  // CHECK: hivm.hir.vexp
  // CHECK: hivm.hir.store{{.*}} {tiled_op}
  // CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  func.func @mini_bind_sub_block_with_ops(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: memref<128xf32>, %arg3: f32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off()[%i]
      %lhs = tensor.extract_slice %arg0[%offset] [128] [1] {to_be_bubbled_slice} : tensor<256xf32> to tensor<128xf32>
      %rhs = tensor.extract_slice %arg1[%offset] [128] [1] {to_be_bubbled_slice} : tensor<256xf32> to tensor<128xf32>
      %tmp0 = tensor.empty() : tensor<128xf32>
      %sum = hivm.hir.vadd ins(%lhs, %rhs : tensor<128xf32>, tensor<128xf32>) outs(%tmp0 : tensor<128xf32>) -> tensor<128xf32>
      %tmp1 = tensor.empty() : tensor<128xf32>
      %scaled = hivm.hir.vmul ins(%sum, %arg3 : tensor<128xf32>, f32) outs(%tmp1 : tensor<128xf32>) -> tensor<128xf32>
      %tmp2 = tensor.empty() : tensor<128xf32>
      %out = hivm.hir.vexp ins(%scaled : tensor<128xf32>) outs(%tmp2 : tensor<128xf32>) -> tensor<128xf32>
      hivm.hir.store ins(%out : tensor<128xf32>) outs(%arg2 : memref<128xf32>)
    }
    return
  }
}

// -----

#off = affine_map<()[s0] -> (s0 * 8)>
module {
  // explicitly keep two independent value roots/groups.
  // Group A: arg0/arg1 -> vadd -> store to arg4
  // Group B: arg2/arg3 -> vmul -> store to arg5
  // This variant is rank-2 to exercise multi-dimension slicing/tiling.
  // CHECK-LABEL: func.func @mini_bind_sub_block_two_groups
  // CHECK: scf.for
  // CHECK-COUNT-2: hivm.hir.store{{.*}} {tiled_op}
  // CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  func.func @mini_bind_sub_block_two_groups(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>, %arg3: tensor<16x16xf32>, %arg4: memref<8x16xf32>, %arg5: memref<8x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %offset = affine.apply #off()[%i]
      %a0 = tensor.extract_slice %arg0[%offset, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      %a1 = tensor.extract_slice %arg1[%offset, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      %bufA = tensor.empty() : tensor<8x16xf32>
      %groupA = hivm.hir.vadd ins(%a0, %a1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%bufA : tensor<8x16xf32>) -> tensor<8x16xf32>
      hivm.hir.store ins(%groupA : tensor<8x16xf32>) outs(%arg4 : memref<8x16xf32>)
      %b0 = tensor.extract_slice %arg2[%offset, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      %b1 = tensor.extract_slice %arg3[%offset, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      %bufB = tensor.empty() : tensor<8x16xf32>
      %groupB = hivm.hir.vmul ins(%b0, %b1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%bufB : tensor<8x16xf32>) -> tensor<8x16xf32>
      hivm.hir.store ins(%groupB : tensor<8x16xf32>) outs(%arg5 : memref<8x16xf32>)
    }
    return
  }
}
