// RUN: bishengir-opt -ave-process-membar %s | FileCheck %s

func.func @test_membar(%arg0: memref<4xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i64 = arith.constant 0 : i64
  // CHECK: ave.hir.membar
  annotation.mark %c0_i64 {SYNC_IN_VF = "VV_ALL"} : i64
  return
}

