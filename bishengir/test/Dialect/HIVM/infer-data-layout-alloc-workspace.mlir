// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout -split-input-file | FileCheck %s

// Regression test for AllocWorkspaceOp fractal seeding.
// A CV-staged cube operand is staged through a rank>=4 GM workspace buffer
// that is already fractal-shaped (1x1x16x16, pre-arranged by the AIV side).
// The buffer must be seeded Fractal like ViewLikeOpInterface results;
// otherwise it inherits ND from the raw workspace arg and the GM->cbuf
// inserted-load of an mmad operand is misread as an ND->zN conversion, whose
// rank invariant (dst.rank == src.rank + 2) cannot hold for an
// already-fractal rank-4 source, aborting the pass with
// "Unsupported operand shape when convert copy to ND2NZ".
// With the seeding both sides are fractal and the load stays a direct copy.
// CHECK-LABEL: func.func @test_alloc_workspace_fractal_seed_a3
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @test_alloc_workspace_fractal_seed_a3(%ws: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %stage = memref_ext.alloc_workspace() from %ws offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<1x1x16x16xf16, #hivm.address_space<gm>>
    %a_l1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.load
    // CHECK-NOT: hivm.hir.nd2nz
    // CHECK: hivm.hir.mmadL1
    hivm.hir.load ins(%stage : memref<1x1x16x16xf16, #hivm.address_space<gm>>) outs(%b_l1 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-load"} core_type = <CUBE>
    %acc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c16, %c16, %c16 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, memref<1x1x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----

// Rank<4 workspace results are unaffected by the seeding: a rank-2 GM
// staging buffer keeps the inherited ND layout and its GM->cbuf load still
// folds to nd2nz (rank-2 -> rank-4 satisfies dst.rank == src.rank + 2).
// CHECK-LABEL: func.func @test_alloc_workspace_rank2_nd2nz_a3
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @test_alloc_workspace_rank2_nd2nz_a3(%ws: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %stage = memref_ext.alloc_workspace() from %ws offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf16, #hivm.address_space<gm>>
    %a_l1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    // CHECK: hivm.hir.mmadL1
    hivm.hir.load ins(%stage : memref<16x16xf16, #hivm.address_space<gm>>) outs(%b_l1 : memref<16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-load"} core_type = <CUBE>
    %acc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c16, %c16, %c16 : memref<16x16xf16, #hivm.address_space<cbuf>>, memref<16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
    return
  }
}
