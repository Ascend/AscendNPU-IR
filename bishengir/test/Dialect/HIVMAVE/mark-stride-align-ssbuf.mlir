// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_ssbuf_unaffected
  // CHECK-SAME: (%[[ARG0:.*]]: i64)
  func.func @test_ssbuf_unaffected(%arg0: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8

    // CHECK: %[[PTR:.*]] = hivm.hir.pointer_cast(%[[ARG0]]) : memref<i8, #hivm.address_space<ssbuf>>
    %0 = hivm.hir.pointer_cast(%arg0) : memref<i8, #hivm.address_space<ssbuf>>

    // CHECK: %[[VAL:.*]] = memref.load %[[PTR]][] : memref<i8, #hivm.address_space<ssbuf>>
    %1 = memref.load %0[] : memref<i8, #hivm.address_space<ssbuf>>

    %2 = arith.cmpi sgt, %1, %c0_i8 : i8
    scf.if %2 {
      %3 = arith.subi %1, %c1_i8 : i8
      // CHECK: memref.store {{.*}}, %[[PTR]][] : memref<i8, #hivm.address_space<ssbuf>>
      memref.store %3, %0[] : memref<i8, #hivm.address_space<ssbuf>>
    }
    return
  }
}