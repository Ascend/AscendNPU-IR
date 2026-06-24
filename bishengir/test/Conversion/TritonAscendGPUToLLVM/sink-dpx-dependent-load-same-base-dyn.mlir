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
    %c256 = llvm.mlir.constant(256 : i32) : i64

    // --- Make the vector to store ---
    %v = llvm.mlir.undef : vector<4xi64>
    %va = llvm.insertelement %c8, %v[%c0 : i32] : vector<4xi64>
    %vb = llvm.insertelement %c16, %va[%c1 : i32] : vector<4xi64>
    %vc = llvm.insertelement %c16, %vb[%c2 : i32] : vector<4xi64>
    %vd = llvm.insertelement %c8, %vc[%c3 : i32] : vector<4xi64>

    // --- Get a dynamic value and an intermediate value using it ---
    %dynamic_int = ascend_dpx.thread_id_x
    %intermediate_val = llvm.add %c3, %dynamic_int : i32

    // --- Section A: ---
    // --- 1. Get pointers to overlapping memory regions in two ways ---
    %pa0 = llvm.getelementptr %arg0[%intermediate_val] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    %pa1 = llvm.getelementptr %arg0[%c2] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    %pa2 = llvm.getelementptr %pa1[%dynamic_int] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8

    // --- 2. Load from one of the locations ---
    %dependent_load = ascend_dpx.load %pa2 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- 3. Store to a dependent memory region first, then to same memory address ---
    ascend_dpx.store %pa0, %vd : <1>, vector<4xi64>
    ascend_dpx.store %pa2, %dependent_load : <1>, vector<2xi64>

    // --- Section B: ---
    // --- 1. Get a ptr to memory region not overlapping with the one from intermediate value ---
    %pb1 = llvm.getelementptr %arg0[%c256] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %pb2 = llvm.getelementptr %pb1[%dynamic_int] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8

    // --- 2. Load from new ptr ---
    %non_dependent_load = ascend_dpx.load %pb2 : (!llvm.ptr<1>) -> vector<2xi64>

    // --- 3. Store first to non overlapping region, then to same address ---
    ascend_dpx.store %pa0, %vd : <1>, vector<4xi64>
    ascend_dpx.store %pb2, %non_dependent_load : <1>, vector<2xi64>

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
