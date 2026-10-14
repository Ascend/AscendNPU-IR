// Off-registry kernels must not normalize the generated sign-extended bound.
// RUN: bishengir-opt %s -hfusion-uplift-while-to-for | FileCheck %s

// CHECK-LABEL: func.func @uplift_sign_extended_iv
//       CHECK: scf.while
//   CHECK-NOT: scf.for
func.func @uplift_sign_extended_iv(%begin: i32, %wide: i64, %cap: i32,
                                   %step: i32) -> i32 {
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
