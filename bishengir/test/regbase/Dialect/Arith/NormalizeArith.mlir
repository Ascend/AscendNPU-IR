// RUN: bishengir-opt -normalize-arith %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: cast_i1_to_f16
func.func @cast_i1_to_f16(%arg0: memref<2x8x1024xf32, #hivm.address_space<ub>>, %arg1: memref<2x8x1024xf32, #hivm.address_space<ub>>, %arg2: memref<2x8x1024xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: arith.constant dense<1.000000e+00> : vector<1x1x64xf16>
  // CHECK: arith.constant dense<0.000000e+00> : vector<1x1x64xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c1024 step %c64 {
        %subview = memref.subview %arg0[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
        %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
        %2 = arith.cmpf ogt, %0, %1 : vector<1x1x64xf32>
        // CHECK-NO: arith.uitofp
        // CHECK: arith.select
        %3 = arith.uitofp %2 : vector<1x1x64xi1> to vector<1x1x64xf16>
        %4 = arith.extf %3 : vector<1x1x64xf16> to vector<1x1x64xf32>
        vector.transfer_write %4, %subview_1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x64xf32>, memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: cast_i1_to_bf16
func.func @cast_i1_to_bf16(%arg0: memref<2x8x1024xbf16, #hivm.address_space<ub>>, %arg1: memref<2x8x1024xbf16, #hivm.address_space<ub>>, %arg2: memref<2x8x1024xbf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: arith.constant dense<1.000000e+00> : vector<1x1x64xf32>
  // CHECK-NEXT: arith.constant dense<0.000000e+00> : vector<1x1x64xf32>
  %cst = arith.constant 0.000000e+00 : bf16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c1024 step %c64 {
        %subview = memref.subview %arg0[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xbf16, #hivm.address_space<ub>> to memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xbf16, #hivm.address_space<ub>> to memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4, %arg5] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xbf16, #hivm.address_space<ub>> to memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xbf16>
        %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xbf16>
        %2 = arith.extf %1 : vector<1x1x64xbf16> to vector<1x1x64xf32>
        %3 = arith.extf %0 : vector<1x1x64xbf16> to vector<1x1x64xf32>
        %4 = arith.cmpf olt, %3, %2 : vector<1x1x64xf32>
        // CHECK-NO: arith.uitofp
        // CHECK: arith.select {{.*}} : vector<1x1x64xi1>, vector<1x1x64xf32>
        // CHECK-NEXT: arith.truncf {{.*}} : vector<1x1x64xf32> to vector<1x1x64xbf16>
        %5 = arith.uitofp %4 : vector<1x1x64xi1> to vector<1x1x64xbf16>
        vector.transfer_write %5, %subview_1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x64xbf16>, memref<1x1x64xbf16, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
  }
  return
}