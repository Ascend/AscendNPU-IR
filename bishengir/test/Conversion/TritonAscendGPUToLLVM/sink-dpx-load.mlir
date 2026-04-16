// RUN: bishengir-opt --sink-dpx-load %s | FileCheck %s
//
// Verify that the SinkDPXLoad pass interleaves load-compute-store chains.
//
// Before the pass, the IR has the pattern:
//   all loads for A → all loads for B → all divs → all stores
// After the pass (SROA + bottom-up scheduling), each store's dependency tree
// is emitted contiguously:
//   load_a[i], load_b[i], div, div, store  (repeated per store)

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_sink_loads
  llvm.func @test_sink_loads(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64

    // Address computation for A
    %pa0 = llvm.getelementptr %arg0[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pa1 = llvm.getelementptr %arg0[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // Address computation for B
    %pb0 = llvm.getelementptr %arg1[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pb1 = llvm.getelementptr %arg1[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // Address computation for C (output)
    %pc0 = llvm.getelementptr %arg2[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pc1 = llvm.getelementptr %arg2[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // --- All loads for A (before pass: grouped together) ---
    %va0 = ascend_dpx.load %pa0 : (!llvm.ptr<1>) -> vector<2xi64>
    %va1 = ascend_dpx.load %pa1 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- All loads for B (before pass: grouped together) ---
    %vb0 = ascend_dpx.load %pb0 : (!llvm.ptr<1>) -> vector<2xi64>
    %vb1 = ascend_dpx.load %pb1 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- Unpack loaded values ---
    %a0 = llvm.extractelement %va0[%c0 : i32] : vector<2xi64>
    %a1 = llvm.extractelement %va0[%c1 : i32] : vector<2xi64>
    %a2 = llvm.extractelement %va1[%c0 : i32] : vector<2xi64>
    %a3 = llvm.extractelement %va1[%c1 : i32] : vector<2xi64>

    %b0 = llvm.extractelement %vb0[%c0 : i32] : vector<2xi64>
    %b1 = llvm.extractelement %vb0[%c1 : i32] : vector<2xi64>
    %b2 = llvm.extractelement %vb1[%c0 : i32] : vector<2xi64>
    %b3 = llvm.extractelement %vb1[%c1 : i32] : vector<2xi64>

    // --- Pack into struct (the dependency bottleneck) ---
    %sa = llvm.mlir.undef : !llvm.struct<(i64, i64, i64, i64)>
    %sa0 = llvm.insertvalue %a0, %sa[0] : !llvm.struct<(i64, i64, i64, i64)>
    %sa1 = llvm.insertvalue %a1, %sa0[1] : !llvm.struct<(i64, i64, i64, i64)>
    %sa2 = llvm.insertvalue %a2, %sa1[2] : !llvm.struct<(i64, i64, i64, i64)>
    %sa3 = llvm.insertvalue %a3, %sa2[3] : !llvm.struct<(i64, i64, i64, i64)>

    %sb = llvm.mlir.undef : !llvm.struct<(i64, i64, i64, i64)>
    %sb0 = llvm.insertvalue %b0, %sb[0] : !llvm.struct<(i64, i64, i64, i64)>
    %sb1 = llvm.insertvalue %b1, %sb0[1] : !llvm.struct<(i64, i64, i64, i64)>
    %sb2 = llvm.insertvalue %b2, %sb1[2] : !llvm.struct<(i64, i64, i64, i64)>
    %sb3 = llvm.insertvalue %b3, %sb2[3] : !llvm.struct<(i64, i64, i64, i64)>

    // --- Unpack from struct (SROA will bypass these) ---
    %xa0 = llvm.extractvalue %sa3[0] : !llvm.struct<(i64, i64, i64, i64)>
    %xa1 = llvm.extractvalue %sa3[1] : !llvm.struct<(i64, i64, i64, i64)>
    %xa2 = llvm.extractvalue %sa3[2] : !llvm.struct<(i64, i64, i64, i64)>
    %xa3 = llvm.extractvalue %sa3[3] : !llvm.struct<(i64, i64, i64, i64)>

    %xb0 = llvm.extractvalue %sb3[0] : !llvm.struct<(i64, i64, i64, i64)>
    %xb1 = llvm.extractvalue %sb3[1] : !llvm.struct<(i64, i64, i64, i64)>
    %xb2 = llvm.extractvalue %sb3[2] : !llvm.struct<(i64, i64, i64, i64)>
    %xb3 = llvm.extractvalue %sb3[3] : !llvm.struct<(i64, i64, i64, i64)>

    // --- All divs (before pass: grouped together) ---
    %d0 = ascend_dpx.div %xa0, %xb0 : (i64, i64) -> i64
    %d1 = ascend_dpx.div %xa1, %xb1 : (i64, i64) -> i64
    %d2 = ascend_dpx.div %xa2, %xb2 : (i64, i64) -> i64
    %d3 = ascend_dpx.div %xa3, %xb3 : (i64, i64) -> i64

    // --- Pack div results, then unpack for stores ---
    %sd = llvm.mlir.undef : !llvm.struct<(i64, i64, i64, i64)>
    %sd0 = llvm.insertvalue %d0, %sd[0] : !llvm.struct<(i64, i64, i64, i64)>
    %sd1 = llvm.insertvalue %d1, %sd0[1] : !llvm.struct<(i64, i64, i64, i64)>
    %sd2 = llvm.insertvalue %d2, %sd1[2] : !llvm.struct<(i64, i64, i64, i64)>
    %sd3 = llvm.insertvalue %d3, %sd2[3] : !llvm.struct<(i64, i64, i64, i64)>

    %xd0 = llvm.extractvalue %sd3[0] : !llvm.struct<(i64, i64, i64, i64)>
    %xd1 = llvm.extractvalue %sd3[1] : !llvm.struct<(i64, i64, i64, i64)>
    %xd2 = llvm.extractvalue %sd3[2] : !llvm.struct<(i64, i64, i64, i64)>
    %xd3 = llvm.extractvalue %sd3[3] : !llvm.struct<(i64, i64, i64, i64)>

    // --- Repack for vector stores ---
    %vc0 = llvm.mlir.undef : vector<2xi64>
    %vc0a = llvm.insertelement %xd0, %vc0[%c0 : i32] : vector<2xi64>
    %vc0b = llvm.insertelement %xd1, %vc0a[%c1 : i32] : vector<2xi64>

    %vc1 = llvm.mlir.undef : vector<2xi64>
    %vc1a = llvm.insertelement %xd2, %vc1[%c0 : i32] : vector<2xi64>
    %vc1b = llvm.insertelement %xd3, %vc1a[%c1 : i32] : vector<2xi64>

    // --- All stores (before pass: grouped together) ---
    ascend_dpx.store %pc0, %vc0b : <1>, vector<2xi64>
    ascend_dpx.store %pc1, %vc1b : <1>, vector<2xi64>

    llvm.return
  }
}

// After SinkDPXLoad, each store's dependency tree is emitted contiguously.
// The key invariant: loads for the second store appear AFTER the first store,
// not before it. We capture SSA names to verify the data-flow connections
// (load → extract → div → insert → store) are wired correctly after SROA
// and scheduling.

// --- First store's dependency tree (offset 8) ---
//
// Load A[0..1] and B[0..1] from offset-8 addresses, compute divs, store.
// CHECK:       [[VA0:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       [[A0:%[0-9]+]] = llvm.extractelement [[VA0]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[VB0:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       [[B0:%[0-9]+]] = llvm.extractelement [[VB0]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[D0:%[0-9]+]] = ascend_dpx.div [[A0]], [[B0]]
// CHECK:       [[A1:%[0-9]+]] = llvm.extractelement [[VA0]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[B1:%[0-9]+]] = llvm.extractelement [[VB0]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[D1:%[0-9]+]] = ascend_dpx.div [[A1]], [[B1]]
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<2xi64>

// --- Second store's dependency tree (offset 16) ---
//
// Loads for the second store appear AFTER the first store — not grouped
// with the first store's loads.
// CHECK:       [[VA1:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       [[A2:%[0-9]+]] = llvm.extractelement [[VA1]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[VB1:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       [[B2:%[0-9]+]] = llvm.extractelement [[VB1]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[D2:%[0-9]+]] = ascend_dpx.div [[A2]], [[B2]]
// CHECK:       [[A3:%[0-9]+]] = llvm.extractelement [[VA1]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[B3:%[0-9]+]] = llvm.extractelement [[VB1]][{{%[0-9]+}} : i32] : vector<2xi64>
// CHECK:       [[D3:%[0-9]+]] = ascend_dpx.div [[A3]], [[B3]]
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<2xi64>

// --- No extractvalue ops remain (SROA eliminated them) ---
// CHECK-NOT:   llvm.extractvalue
