// RUN: bishengir-opt -hivm-insert-fixpipe %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
// CHECK-LABEL: func.func @test_mmad_accumulation_while_no_outer_fixpipe
// A remain-in-L0C scf.while result that initializes a later MMAD scf.while
// must stay in L0C instead of being materialized through an outer fixpipe.
func.func @test_mmad_accumulation_while_no_outer_fixpipe(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %C_init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[ACC_RES:.*]]:2 = scf.while
  // CHECK-NOT: hivm.hir.fixpipe
  %acc_res:2 = scf.while (%counter = %c0_i32, %C_curr = %C_init) : (i32, tensor<64x64xf32>) -> (i32, tensor<64x64xf32>) {
    scf.condition(%true) %counter, %C_curr : i32, tensor<64x64xf32>
  } do {
  ^bb0(%counter: i32, %C_curr: tensor<64x64xf32>):
    %if_res = scf.if %true -> (tensor<64x64xf32>) {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
      %mmad = hivm.hir.mmadL1 {hivm.remain_in_l0c = [0 : i32], normalized_in_L0C}
        ins(%A, %B, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%C_curr : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %mmad : tensor<64x64xf32>
    } else {
      scf.yield %C_curr : tensor<64x64xf32>
    } {hivm.matmul_limited_in_cube, hivm.remain_in_l0c = [0 : i32]}
    // CHECK-NOT: hivm.hir.fixpipe
    scf.yield %counter, %if_res : i32, tensor<64x64xf32>
  } attributes {hivm.remain_in_l0c = [1 : i32], normalized_in_L0C = [1 : i32]}
  // CHECK: %[[INNER_RES:.*]]:2 = scf.while ({{[^,]+}}, {{[^=]+}} = %[[ACC_RES]]#1)
  // CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[ACC_RES]]#1
  %inner_res:2 = scf.while (%counter = %c0_i32, %C_next = %acc_res#1) : (i32, tensor<64x64xf32>) -> (i32, tensor<64x64xf32>) {
    scf.condition(%true) %counter, %C_next : i32, tensor<64x64xf32>
  } do {
  ^bb0(%counter: i32, %C_next: tensor<64x64xf32>):
    %mmad2 = hivm.hir.mmadL1 ins(%A, %B, %true, %c64, %c64, %c64
      : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
      outs(%C_next : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %counter, %mmad2 : i32, tensor<64x64xf32>
  }
  // CHECK: %[[OUT:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[INNER_RES]]#1
  // CHECK: return %[[OUT]]
  return %inner_res#1 : tensor<64x64xf32>
}
}
