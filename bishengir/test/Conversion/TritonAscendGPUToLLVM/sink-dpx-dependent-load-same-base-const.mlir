// RUN: bishengir-opt --sink-dpx-load %s | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_dependent_loads
  llvm.func @test_dependent_loads(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c256 = llvm.mlir.constant(256 : i64) : i64

    // --- Make the vector to store ---
    %v = llvm.mlir.undef : vector<4xi64>
    %va = llvm.insertelement %c8, %v[%c0 : i32] : vector<4xi64>
    %vb = llvm.insertelement %c16, %va[%c1 : i32] : vector<4xi64>
    %vc = llvm.insertelement %c16, %vb[%c2 : i32] : vector<4xi64>
    %vd = llvm.insertelement %c8, %vc[%c3 : i32] : vector<4xi64>

    // --- Section A: ---
    // --- 1. Get one ptr to %arg0 + 8 bytes and another to %arg0 + 16 bytes ---
    %pa0 = llvm.getelementptr %arg0[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pa1 = llvm.getelementptr %arg0[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // --- 2. Load 16 bytes from %arg0 + 16 bytes ---
    %dependent_load = ascend_dpx.load %pa1 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- 3. Store 32 bytes at %arg0 + 8 bytes and then 16 bytes at %arg0 + 16 bytes ---
    ascend_dpx.store %pa0, %vd : <1>, vector<4xi64>
    ascend_dpx.store %pa1, %dependent_load : <1>, vector<2xi64>

    // --- Section B: ---
    // --- 1. Get a ptr to %arg0 + 256 bytes ---
    %pb0 = llvm.getelementptr %arg0[%c256] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // --- 2. Load 32 bytes from %arg0 + 256 bytes ---
    %non_dependent_load = ascend_dpx.load %pb0 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- 3. Store 32 bytes at %arg0 + 8 bytes and then 16 bytes at %arg0 + 256 bytes ---
    ascend_dpx.store %pa0, %vd : <1>, vector<4xi64>
    ascend_dpx.store %pb0, %non_dependent_load : <1>, vector<2xi64>

    llvm.return
  }
}

// The dependent load should NOT be reordered after the first store,
// since the first store will change the data at the location
// being loaded from, hence changing program behaviour.

// CHECK:       [[DEPENDENT_LOAD:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<4xi64>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<2xi64>

// The non dependent load SHOULD be reordered after the 3rd store,
// since the third store does not affect the same memory region

// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<4xi64>
// CHECK:       [[NON_DEPENDENT_LOAD:%[0-9]+]] = ascend_dpx.load {{%[0-9]+}} : (!llvm.ptr<1>) -> vector<2xi64>
// CHECK:       ascend_dpx.store {{%[0-9]+}}, {{%[0-9]+}} : <1>, vector<2xi64>
