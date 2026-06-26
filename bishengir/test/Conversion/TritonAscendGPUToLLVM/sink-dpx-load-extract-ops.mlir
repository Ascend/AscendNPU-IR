// RUN: bishengir-opt --sink-dpx-load %s | FileCheck %s
//
// Verify that the SinkDPXLoad pass correctly identifies
// when it can and cannot reorder based on extract ops
// and block arguments

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_follow_extract_ops
  llvm.func @test_follow_extract_ops(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c24 = llvm.mlir.constant(24 : i64) : i64

    // Address computation for A
    %pa0 = llvm.getelementptr %arg0[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pa1 = llvm.getelementptr %arg0[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pa2 = llvm.getelementptr %arg0[%c24] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // Address computation for B
    %pb0 = llvm.getelementptr %arg1[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pb1 = llvm.getelementptr %arg1[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pb2 = llvm.getelementptr %arg1[%c24] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // --- Pack into struct (the dependency bottleneck) ---
    %sa = llvm.mlir.undef : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sa0 = llvm.insertvalue %arg0, %sa[0, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sa1 = llvm.insertvalue %pa0, %sa0[0, 1] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sa2 = llvm.insertvalue %pa1, %sa1[1, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sa3 = llvm.insertvalue %pa2, %sa2[1, 1] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>

    %sb = llvm.mlir.undef : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sb0 = llvm.insertvalue %arg1, %sb[0, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sb1 = llvm.insertvalue %pb0, %sb0[0, 1] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sb2 = llvm.insertvalue %pb1, %sb1[1, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %sb3 = llvm.insertvalue %pb2, %sb2[1, 1] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>

    %p0 = llvm.extractvalue %sa3[0, 1] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %p1 = llvm.extractvalue %sb3[1, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>
    %p2 = llvm.extractvalue %sb3[0, 0] : !llvm.struct<(struct<(ptr<1>, ptr<1>)>, struct<(ptr<1>, ptr<1>)>)>

    // --- Section A : Follow through to different func args ---
    %ndl0 = ascend_dpx.load %p0 : (!llvm.ptr<1>) -> i32

    ascend_dpx.store %p1, %c1 : <1>, i32
    ascend_dpx.store %p0, %ndl0 : <1>, i32

    // --- Section B : Follow through to same func arg, non-aliasing memory ---
    %ndl1 = ascend_dpx.load %p1 : (!llvm.ptr<1>) -> i32

    ascend_dpx.store %p2, %c1 : <1>, i32
    ascend_dpx.store %p1, %ndl1 : <1>, i32

    // --- Section C : Follow through to same func arg, aliasing memory ---
    %dl0 = ascend_dpx.load %p1 : (!llvm.ptr<1>) -> i32

    ascend_dpx.store %p1, %c1 : <1>, i32
    ascend_dpx.store %p1, %dl0 : <1>, i32

    llvm.return
  }
}

// After SinkDPXLoad, the Section A and Bloads/stores should be
// reordered, but the Section C load/stores shouldn't since they
// are from aliasing memory.

// --- Section A ---
//
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32
// CHECK:       [[NDL0:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> i32
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32

// --- Section B ---
//
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32
// CHECK:       [[NDL1:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> i32
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32

// --- Section C ---
//
// CHECK:       [[DL0:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> i32
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, i32
