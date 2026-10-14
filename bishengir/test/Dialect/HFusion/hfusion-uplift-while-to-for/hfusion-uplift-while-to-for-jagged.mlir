// Registered jagged-DK kernel enables the 3160 sign-extend normalize without
// --bypass-shape-registry.
// RUN: bishengir-opt %s -hfusion-uplift-while-to-for | FileCheck %s

// CHECK-LABEL: func.func @_jagged_dense_flash_attention_bwd_dk_kernel
//       CHECK: %[[MIN:.*]] = arith.constant -2147483648 : i64
//       CHECK: %[[SATURATED:.*]] = arith.maxsi %{{.*}}, %[[MIN]] : i64
//       CHECK: %[[END:.*]] = arith.trunci %[[SATURATED]] : i64 to i32
//   CHECK-NOT: scf.while
//       CHECK: scf.for %{{.*}} = %{{.*}} to %[[END]] step %{{.*}} : i32
func.func @_jagged_dense_flash_attention_bwd_dk_kernel(%begin: i32, %wide: i64,
                                                       %cap: i32, %step: i32)
    -> i32 {
  %wide_cap = arith.extsi %cap : i32 to i64
  %end = arith.minsi %wide, %wide_cap : i64
  %0 = scf.while (%iv = %begin) : (i32) -> i32 {
    %wide_iv = arith.extsi %iv : i32 to i64
    %condition = arith.cmpi slt, %wide_iv, %end : i64
    scf.condition(%condition) %iv : i32
  } do {
  ^bb0(%iv: i32):
    "test.body"(%iv) : (i32) -> ()
    %next = arith.addi %iv, %step : i32
    scf.yield %next : i32
  }
  return %0 : i32
}
