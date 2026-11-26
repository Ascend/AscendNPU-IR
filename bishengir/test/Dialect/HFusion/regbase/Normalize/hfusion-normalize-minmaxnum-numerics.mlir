// REQUIRES: execution-engine
// RUN: bishengir-opt %s --hfusion-normalize-ops="use-regbase=true" \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm --finalize-memref-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: sed -e "s/ {round_mode = #hfusion.round_mode<rint>}//g" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=i32 | FileCheck %s
// CHECK: {{^0$}}

// The NaN predicate casts only 0/1 to f32, so rounding is immaterial. Drop the
// round-mode annotation retained on llvm.sitofp for the upstream runner.

// Compare all pairs of positive/negative NaNs, infinities, finite values and
// signed zeros against scalar maxnumf/minnumf. In particular, two NaNs must
// produce a NaN, and a single NaN must select the other operand in either order.
func.func @minmaxnum(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %init = tensor.empty() : tensor<8x8xf32>
  %max = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>}
    ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
    outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
  %min = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>}
    ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
    outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %max, %min : tensor<8x8xf32>, tensor<8x8xf32>
}

func.func @main() -> i32 {
  %values = arith.constant dense<[0x7FC00000, 0xFFC00001,
    0x7F800000, 0xFF800000, 3.0, -2.0, 0.0, -0.0]> : tensor<8xf32>
  %init = tensor.empty() : tensor<8x8xf32>
  %lhs = linalg.broadcast ins(%values : tensor<8xf32>)
    outs(%init : tensor<8x8xf32>) dimensions = [1]
  %rhs = linalg.broadcast ins(%values : tensor<8xf32>)
    outs(%init : tensor<8x8xf32>) dimensions = [0]
  %max, %min = func.call @minmaxnum(%lhs, %rhs)
    : (tensor<8x8xf32>, tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  %errors = scf.for %i = %c0 to %c8 step %c1 iter_args(%outer = %zero) -> i32 {
    %row = scf.for %j = %c0 to %c8 step %c1 iter_args(%inner = %outer) -> i32 {
      %x = tensor.extract %values[%i] : tensor<8xf32>
      %y = tensor.extract %values[%j] : tensor<8xf32>
      %expected_max = arith.maxnumf %x, %y : f32
      %expected_min = arith.minnumf %x, %y : f32
      %actual_max = tensor.extract %max[%i, %j] : tensor<8x8xf32>
      %actual_min = tensor.extract %min[%i, %j] : tensor<8x8xf32>
      %max_equal = arith.cmpf oeq, %actual_max, %expected_max : f32
      %min_equal = arith.cmpf oeq, %actual_min, %expected_min : f32
      %max_nan = arith.cmpf uno, %actual_max, %actual_max : f32
      %min_nan = arith.cmpf uno, %actual_min, %actual_min : f32
      %expected_nan = arith.cmpf uno, %expected_max, %expected_max : f32
      %max_ok = arith.select %expected_nan, %max_nan, %max_equal : i1
      %min_ok = arith.select %expected_nan, %min_nan, %min_equal : i1
      %ok = arith.andi %max_ok, %min_ok : i1
      %bad = arith.xori %ok, %true : i1
      %error = arith.extui %bad : i1 to i32
      %next = arith.addi %inner, %error : i32
      scf.yield %next : i32
    }
    scf.yield %row : i32
  }
  return %errors : i32
}
