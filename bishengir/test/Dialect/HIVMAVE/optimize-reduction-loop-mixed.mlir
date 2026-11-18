// RUN: bishengir-opt -optimize-reduction-loop -split-input-file %s | FileCheck %s

// Four running values, only one operation is tagged as reductionOp.
// CHECK-LABEL: func.func @partially_tagged_reductions(
// CHECK-NOT: splitDepth
// CHECK: %[[RESULT:.*]]:4 = scf.for %{{.*}} = %c64 to %c256 step %c64
// CHECK-NOT: splitDepth
// CHECK: return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2, %[[RESULT]]#3
func.func @partially_tagged_reductions(%input: memref<256xi32>) -> (i32, i32, i32, vector<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %zero = arith.constant 0 : i32
  %vector_zero = arith.constant dense<0> : vector<64xi32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result:4 = scf.for %i = %c0 to %c256 step %c64
      iter_args(%sum = %zero, %max = %zero, %xor = %zero, %vector_sum = %vector_zero)
      -> (i32, i32, i32, vector<64xi32>) {
    %value = memref.load %input[%i] : memref<256xi32>
    %vector_value = ave.hir.broadcast %value, %mask : i32, vector<64xi1> -> vector<64xi32>
    %next_vector_sum = ave.hir.vadd %vector_value, %vector_sum, %mask {reductionOp} : vector<64xi32>, vector<64xi1>
    %next_sum = arith.addi %sum, %value : i32
    %next_max = arith.maxui %max, %value : i32
    %next_xor = arith.xori %xor, %value : i32
    scf.yield %next_sum, %next_max, %next_xor, %next_vector_sum : i32, i32, i32, vector<64xi32>
  } {reductionLoop}
  return %result#0, %result#1, %result#2, %result#3 : i32, i32, i32, vector<64xi32>
}

// -----

// Match combiners to yielded results, not the order of operations in the body.
// The max is computed before the min, but the min is the first loop result.
// CHECK-LABEL: func.func @reordered_reductions(
// CHECK: %[[SPLIT:.*]]:4 = scf.for %{{.*}} = %c64 to %c256 step %c64
// CHECK: } {splitDepth = 1 : i64}
// CHECK: %[[MIN:.*]] = ave.hir.vumin %[[SPLIT]]#0, %[[SPLIT]]#1
// CHECK: %[[MAX:.*]] = ave.hir.vumax %[[SPLIT]]#2, %[[SPLIT]]#3
// CHECK: return %[[MIN]], %[[MAX]]
func.func @reordered_reductions(%input: memref<512xi32, #hivm.address_space<ub>>) -> (vector<64xi32>, vector<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %min_init = arith.constant dense<-1> : vector<64xi32>
  %max_init = arith.constant dense<0> : vector<64xi32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result:2 = scf.for %i = %c0 to %c512 step %c64
      iter_args(%min = %min_init, %max = %max_init) -> (vector<64xi32>, vector<64xi32>) {
    %value = ave.hir.vload <NORM> %input[%i] : memref<512xi32, #hivm.address_space<ub>> into vector<64xi32>
    %next_max = ave.hir.vumax %value, %max, %mask {reductionOp} : vector<64xi32>, vector<64xi1>
    %next_min = ave.hir.vumin %value, %min, %mask {reductionOp} : vector<64xi32>, vector<64xi1>
    scf.yield %next_min, %next_max : vector<64xi32>, vector<64xi32>
  } {reductionLoop}
  return %result#0, %result#1 : vector<64xi32>, vector<64xi32>
}
