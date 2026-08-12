// RUN: bishengir-opt '--tile-dot-loads=k-tile-size=2' %s | FileCheck %s

// Reduced from dot3.mlir.  This is deliberately an untiled dot: the pass must
// create the K loop and rebuild both tensor-of-pointer loads at K=2.

// CHECK-LABEL: tt.func @dot3_tiled
// CHECK: scf.for
// CHECK: tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK: tt.load {{.*}}, {{.*}} : tensor<64x2x!tt.ptr<f32>>
// CHECK: tt.load {{.*}}, {{.*}} : tensor<2x16x!tt.ptr<f32>>
// CHECK: tt.dot {{.*}} {bishengir.dot.tiled} : tensor<64x2xf32> * tensor<2x16xf32> -> tensor<64x16xf32>
module {
  tt.func @dot3_tiled(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>, %m: i32, %n: i32, %k: i32) {
    %zero = arith.constant dense<0.0> : tensor<64x16xf32>
    %zeroA = arith.constant dense<0.0> : tensor<64x64xf32>
    %zeroB = arith.constant dense<0.0> : tensor<64x16xf32>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %rK = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %rN = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %rM = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %mS = tt.splat %m : i32 -> tensor<64xi32>
    %nS = tt.splat %n : i32 -> tensor<1x16xi32>
    %kS = tt.splat %k : i32 -> tensor<1x64xi32>
    %kSB = tt.splat %k : i32 -> tensor<64x1xi32>
    %mMask = arith.cmpi slt, %rM, %mS : tensor<64xi32>
    %pa = tt.splat %a : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %pb = tt.splat %b : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    %mIdx = tt.expand_dims %rM {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %nIdx = tt.expand_dims %rN {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %nMask = arith.cmpi slt, %nIdx, %nS : tensor<1x16xi32>
    %kA = tt.expand_dims %rK {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %kB = tt.expand_dims %rK {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %idxA = tt.broadcast %kA : tensor<1x64xi32> -> tensor<64x64xi32>
    %idxB = tt.broadcast %kB : tensor<64x1xi32> -> tensor<64x16xi32>
    %idxM = tt.broadcast %mIdx : tensor<64x1xi32> -> tensor<64x64xi32>
    %idxN = tt.broadcast %nIdx : tensor<1x16xi32> -> tensor<64x16xi32>
    %mMask1 = tt.expand_dims %mMask {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mMask2 = tt.broadcast %mMask1 : tensor<64x1xi1> -> tensor<64x64xi1>
    %kMaskA0 = arith.cmpi slt, %kA, %kS : tensor<1x64xi32>
    %kMaskA = tt.broadcast %kMaskA0 : tensor<1x64xi1> -> tensor<64x64xi1>
    %maskA = arith.andi %mMask2, %kMaskA : tensor<64x64xi1>
    %kMaskB0 = arith.cmpi slt, %kB, %kSB : tensor<64x1xi32>
    %kMaskB = tt.broadcast %kMaskB0 : tensor<64x1xi1> -> tensor<64x16xi1>
    %nMask2 = tt.broadcast %nMask : tensor<1x16xi1> -> tensor<64x16xi1>
    %maskB = arith.andi %kMaskB, %nMask2 : tensor<64x16xi1>
    %offA = arith.addi %idxM, %idxA : tensor<64x64xi32>
    %offB = arith.addi %idxB, %idxN : tensor<64x16xi32>
    %ptrA = tt.addptr %pa, %offA : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %ptrB = tt.addptr %pb, %offB : tensor<64x16x!tt.ptr<f32>>, tensor<64x16xi32>
    %la = tt.load %ptrA, %maskA, %zeroA : tensor<64x64x!tt.ptr<f32>>
    %lb = tt.load %ptrB, %maskB, %zeroB : tensor<64x16x!tt.ptr<f32>>
    %d = tt.dot %la, %lb, %zero : tensor<64x64xf32> * tensor<64x16xf32> -> tensor<64x16xf32>
    %outPtrs = tt.splat %out : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>>
    tt.store %outPtrs, %d : tensor<64x16x!tt.ptr<f32>>
    tt.return
  }
}
