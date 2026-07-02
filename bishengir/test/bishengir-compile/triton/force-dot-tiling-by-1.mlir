// This test will force a k-tiling with tile size 1
// in a dot of size 32x32x32 which is within budget
// and would not be tiled at all otherwise.

// RUN: bishengir-compile %s \
// RUN:   --target=Ascend910_9589 \
// RUN:   --enable-hivm-compile=false \
// RUN:   --enable-triton-ir-compile \
// RUN:   --pure-simt \
// RUN:   --num-warps=8 \
// RUN:   --threads-per-warp=32 \
// RUN:   --shared-mem-dynamic-size=122880 \
// RUN:   --k-tile-size=1 \
// RUN:   --debug-only=tile-dot-loads \
// RUN:   --mlir-print-ir-after=tile-dot-loads,enable-ascend-dpx-mma \
// RUN:   2>&1 | FileCheck %s

// CHECK-LABEL: force_dot_tiling
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @force_dot_tiling(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // --- Define constants ---
    %cst = arith.constant dense<5.000000e-01> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_2 = arith.constant dense<32> : tensor<32xi32>

    // --- Load 32x32xf32 tensors from pointers provided in %arg0 and %arg1 ---
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_2 : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<32x1xi32>
    %4 = arith.muli %2, %3 : tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<32x1xi32> -> tensor<32x32xi32>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %8 = arith.addi %6, %7 : tensor<32x32xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %11 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %13 = tt.broadcast %11 : tensor<32x1xi1> -> tensor<32x32xi1>
    %14 = tt.broadcast %12 : tensor<1x32xi1> -> tensor<32x32xi1>
    %15 = arith.andi %13, %14 : tensor<32x32xi1>
    %16 = tt.load %10, %15, %cst_1 : tensor<32x32x!tt.ptr<f32>>
    %17 = tt.splat %arg4 : i32 -> tensor<32x1xi32>
    %18 = arith.muli %2, %17 : tensor<32x1xi32>
    %19 = tt.broadcast %18 : tensor<32x1xi32> -> tensor<32x32xi32>
    %20 = arith.addi %19, %7 : tensor<32x32xi32>
    %21 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %22 = tt.addptr %21, %20 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %23 = tt.load %22, %15, %cst_1 : tensor<32x32x!tt.ptr<f32>>

    // --- Carry out arithmetic on loaded tensors to force them to be staged ---
    %24 = arith.addf %16, %cst_0 : tensor<32x32xf32>
    %25 = arith.subf %23, %cst : tensor<32x32xf32>

    // --- Do one dot that must be staged and one that can be tiled without staging ---
    %26 = tt.dot %24, %25, %cst_1 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
    %27 = tt.dot %16, %23, %cst_1 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>

    // --- Store the result from both dots to the pointer at %arg2 ---
    // --- (to prevent the dots from being optimized away) ---
    %28 = tt.splat %arg5 : i32 -> tensor<32x1xi32>
    %29 = arith.muli %2, %28 : tensor<32x1xi32>
    %30 = tt.broadcast %29 : tensor<32x1xi32> -> tensor<32x32xi32>
    %31 = arith.addi %30, %7 : tensor<32x32xi32>
    %32 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %33 = tt.addptr %32, %31 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %33, %26, %15 : tensor<32x32x!tt.ptr<f32>>
    tt.store %33, %27, %15 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// First, we check that the dot to tile without staging was
// tiled successfully with a tile-size of 1
// 
// CHECK: [TileDotLoads]   -> K-tiling tileSize=1 numTiles=32
// CHECK: [TileDotLoads]   -> tiling succeeded

// Next, we check that the dot to tile with staging was tiled
// with a tile-size of 1 (despite the cost model saying it
// shouldn't be tiled)
// 
// CHECK: [StageNonLoadOperand] examining tt.dot [M=32 K=32 N=32] MKN=32768 budget=32768  kTile=1 numTiles=32 A=stage B=stage
// CHECK: -> cost model says staging is not profitable
// CHECK: (numTiles=32, kTile=1)

// Next, we check to make sure that two scf.for loops, each
// with a tt.dot of dims 32x1x32, have been introduced to the
// IR, to ensure that tiling has actually occured.
// 
// CHECK: IR Dump After TileDotLoads
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: {bishengir.dot.tiled} : tensor<32x1xf32> * tensor<1x32xf32> -> tensor<32x32xf32>
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: {bishengir.dot.tiled} : tensor<32x1xf32> * tensor<1x32xf32> -> tensor<32x32xf32>

// A final check that all the above happened in the TileDotLoads
// pass, by checking the existend of the pass right after it
// 
// CHECK: IR Dump After EnableAscendDPXMMA
