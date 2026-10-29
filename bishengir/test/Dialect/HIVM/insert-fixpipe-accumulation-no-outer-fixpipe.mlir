// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
// CHECK-LABEL: func.func @test_mmad_accumulation_remain_in_l0c_no_outer_fixpipe
// When an accumulation loop stays in L0C and its result feeds another mmad
// loop, do not insert an outer fixpipe on the first loop's result.
func.func @test_mmad_accumulation_remain_in_l0c_no_outer_fixpipe(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %C_init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  // CHECK: %[[ACC_RES:.*]] = scf.for
  %acc_res = scf.for %i = %c0 to %c2 step %c1 iter_args(%C_curr = %C_init) -> (tensor<64x64xf32>) {
    // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
    %mmad = hivm.hir.mmadL1 ins(%A, %B, %true, %c64, %c64, %c64
      : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
      outs(%C_curr : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK: scf.yield %[[MMAD]]
    scf.yield %mmad : tensor<64x64xf32>
  } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}
  // CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[ACC_RES]]
  // CHECK: %[[INNER_RES:.*]] = scf.for
  // CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[ACC_RES]]
  %inner_res = scf.for %j = %c0 to %c2 step %c1 iter_args(%C_next = %acc_res) -> (tensor<64x64xf32>) {
    %mmad2 = hivm.hir.mmadL1 ins(%A, %B, %true, %c64, %c64, %c64
      : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
      outs(%C_next : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %mmad2 : tensor<64x64xf32>
  }
  // CHECK: %[[OUTER_FIX:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[INNER_RES]]
  // CHECK: return %[[OUTER_FIX]]
  return %inner_res : tensor<64x64xf32>
}

}
// -----
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
// CHECK-LABEL: func.func @test_mmad_accumulation_scf_if_no_outer_fixpipe
// Regression from flex-attention: mmad inside scf.if still accumulates via
// scf.yield to iter_arg; InsertFixpipe must not add an outer tensor fixpipe
// on the enclosing loop result when that result feeds another mmad loop.
func.func @test_mmad_accumulation_scf_if_no_outer_fixpipe(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %C_init: tensor<64x64xf32>, %cond: i1) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  // CHECK: %[[ACC_RES:.*]] = scf.for
  %acc_res = scf.for %i = %c0 to %c2 step %c1 iter_args(%C_curr = %C_init) -> (tensor<64x64xf32>) {
    // CHECK: scf.if
    %if_res = scf.if %cond -> (tensor<64x64xf32>) {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
      %mmad = hivm.hir.mmadL1 ins(%A, %B, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%C_curr : tensor<64x64xf32>) -> tensor<64x64xf32>
      // CHECK: scf.yield %[[MMAD]]
      scf.yield %mmad : tensor<64x64xf32>
    } else {
      scf.yield %C_curr : tensor<64x64xf32>
    }
    // CHECK: scf.yield %{{.*}}
    scf.yield %if_res : tensor<64x64xf32>
  } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}
  // CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[ACC_RES]]
  // CHECK: %[[INNER_RES:.*]] = scf.for
  %inner_res = scf.for %j = %c0 to %c2 step %c1 iter_args(%C_next = %acc_res) -> (tensor<64x64xf32>) {
    %mmad2 = hivm.hir.mmadL1 ins(%A, %B, %true, %c64, %c64, %c64
      : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
      outs(%C_next : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %mmad2 : tensor<64x64xf32>
  }
  // CHECK: %[[OUTER_FIX:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[INNER_RES]]
  // CHECK: return %[[OUTER_FIX]]
  return %inner_res : tensor<64x64xf32>
}

}
