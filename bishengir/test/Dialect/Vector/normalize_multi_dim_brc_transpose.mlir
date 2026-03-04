// RUN: bishengir-opt %s -normalize-vector | FileCheck %s

//....Case from #3021

func.func @triton_softmax_outlined_vf_2(%arg0: memref<13x5x31xf32, strided<[160, 32, 1]>, #hivm.address_space<ub>>, %arg1: memref<13x31xf32, #hivm.address_space<ub>>, %arg2: memref<13x5x31xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c13 = arith.constant 13 : index
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %0 = vector.constant_mask [1, 1, 31] : vector<1x1x64xi1>
  %1 = vector.constant_mask [1, 31] : vector<1x64xi1>
  %2 = vector.shape_cast %0 : vector<1x1x64xi1> to vector<1x64xi1>
  scf.for %arg3 = %c0 to %c13 step %c1 {
    %subview = memref.subview %arg0[%arg3, 0, 0] [1, 5, 31] [1, 1, 1] : memref<13x5x31xf32, strided<[160, 32, 1]>, #hivm.address_space<ub>> to memref<1x5x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3, 0] [1, 31] [1, 1] : memref<13x31xf32, #hivm.address_space<ub>> to memref<1x31xf32, strided<[31, 1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg4 = %c0 to %c5 step %c1 {
      %subview_1 = memref.subview %arg0[%arg3, %arg4, 0] [1, 1, 31] [1, 1, 1] : memref<13x5x31xf32, strided<[160, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %3 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x31xf32, strided<[31, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %4 = vector.shape_cast %3 : vector<1x64xf32> to vector<64xf32>
      %5 = scf.for %arg5 = %c0 to %c5 step %c1 iter_args(%arg6 = %4) -> (vector<64xf32>) {
        %12 = vector.shape_cast %arg6 : vector<64xf32> to vector<1x64xf32>
        %subview_3 = memref.subview %subview[0, %arg5, 0] [1, 1, 31] [1, 1, 1] : memref<1x5x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x1x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %13 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true]} : memref<1x1x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
        %14 = vector.shape_cast %13 : vector<1x1x64xf32> to vector<1x64xf32>
        %15 = arith.maxnumf %12, %14 : vector<1x64xf32>
        %16 = arith.select %2, %15, %14 : vector<1x64xi1>, vector<1x64xf32>
        %17 = vector.shape_cast %16 : vector<1x64xf32> to vector<64xf32>
        scf.yield %17 : vector<64xf32>
      }
      %6 = vector.shape_cast %5 : vector<64xf32> to vector<1x64xf32>
      %subview_2 = memref.subview %arg2[%arg3, %arg4, 0] [1, 1, 31] [1, 1, 1] : memref<13x5x31xf32, #hivm.address_space<ub>> to memref<1x1x31xf32, strided<[155, 31, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true]} : memref<1x1x31xf32, strided<[160, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
      %8 = vector.broadcast %6 : vector<1x64xf32> to vector<1x1x64xf32>
      %9 = vector.transpose %8, [1, 0, 2] : vector<1x1x64xf32> to vector<1x1x64xf32>
      // CHECK-NOT: vector::broadcast
      // CHECK-NOT: vector::transpose
      %10 = arith.subf %7, %9 : vector<1x1x64xf32>
      %11 = math.exp %10 : vector<1x1x64xf32>
      vector.transfer_write %11, %subview_2[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xf32>, memref<1x1x31xf32, strided<[155, 31, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}
