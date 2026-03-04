// RUN: bishengir-opt -lift-arith-indexcast %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: lift_indexcast_i32_1024
func.func @lift_indexcast_i32_1024(%arg0: memref<1024xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xi32>
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xindex>
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg1 = %c0 to %c1024 step %c64 {
    %subview = memref.subview %arg0[%arg1] [64] [1] : memref<1024xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: [[R:%.*]] = arith.index_cast %arg1 : index to i32
    // CHECK: [[R:%.*]] = vector.broadcast [[V0:%.*]] : i32 to vector<64xi32>
    // CHECK: [[R:%.*]] = arith.addi [[V0:%.*]], [[V1:%.*]] : vector<64xi32>
    %0 = vector.broadcast %arg1 : index to vector<64xindex>
    %1 = arith.addi %0, %cst : vector<64xindex>
    %2 = arith.index_cast %1 : vector<64xindex> to vector<64xi32>
    vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<64xi32>, memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: lift_vector_broadcast
// CHECK-NOT: arith.index_cast
func.func @lift_vector_broadcast() -> vector<1x2xi32> attributes {hivm.vector_function} {
  %cst = arith.constant dense<[0, 1]> : vector<2xindex>
  %1 = vector.broadcast %cst : vector<2xindex> to vector<1x2xindex>
  %2 = arith.index_cast %1 : vector<1x2xindex> to vector<1x2xi32>
  return %2 : vector<1x2xi32>
}
