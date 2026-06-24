// RUN: bishengir-opt %s -hfusion-simplify-ops 2>&1 | FileCheck %s
//
// Regression: When computing iterarg index for a hfusion.cast whose output
// is defined by tensor.empty and input is a non-first iter_arg of an scf.for,
// for the Nth iterarg it yields -N instead of N, which wraps incorrectly when
// cast to unsigned and trips the assert failure, or an out-of-bounds access
// in a no-assert build.
//
// CHECK-LABEL: func.func @looped_cast_second_iterarg
// CHECK: scf.for
// CHECK: hfusion.cast
// CHECK: linalg.matmul
// CHECK: scf.yield

func.func @looped_cast_second_iterarg(
    %init0: tensor<16x16xf32>,
    %init1: tensor<16x16xf32>,
    %w: tensor<16x16xf16>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %e16 = tensor.empty() : tensor<16x16xf16>
  %r:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%a = %init0, %b = %init1)
      -> (tensor<16x16xf32>, tensor<16x16xf32>) : i32 {
    %cb = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                        round_mode = #hfusion.round_mode<rint>}
        ins(%b : tensor<16x16xf32>) outs(%e16 : tensor<16x16xf16>)
        -> tensor<16x16xf16>
    %mm = linalg.matmul {input_precision = "ieee"}
        ins(%cb, %w : tensor<16x16xf16>, tensor<16x16xf16>)
        outs(%a : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mm, %mm : tensor<16x16xf32>, tensor<16x16xf32>
  }
  return %r#0, %r#1 : tensor<16x16xf32>, tensor<16x16xf32>
}

