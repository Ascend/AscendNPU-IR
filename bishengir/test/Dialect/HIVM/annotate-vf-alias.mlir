// RUN: bishengir-opt -hivm-annotate-vf-alias %s | FileCheck %s

// CHECK-LABEL: func.func @test_vf
func.func @test_vf(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>,%arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: index) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mask = vector.create_mask %c1 : vector<64xi1>
  %0 = vector.transfer_read %arg1[%c0], %cst, %mask {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %1 = vector.transfer_read %arg0[%c0], %cst, %mask {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg2[%c0], %cst, %mask {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  vector.transfer_write %1, %arg1[%c0],%mask {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  vector.transfer_write %0, %arg0[%c0],%mask {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}   