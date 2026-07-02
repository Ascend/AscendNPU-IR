// RUN: bishengir-opt %s -hivm-sink-op-to-consumer-in-loop -split-input-file | FileCheck %s

// CHECK: hivm.hir.vbrc
// CHECK: scf.for
// CHECK: hivm.hir.vbrc
// CHECK: scf.for
func.func @test_sink_op_to_consumer_in_loop(%arg0: i32, %arg1: tensor<256xf16>, %arg2: tensor<256xf16>) -> tensor<256xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = tensor.empty() : tensor<256xf16>
  %3 = hivm.hir.vbrc ins(%cst : f16) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
  %4 = scf.for %arg3 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg4 = %1) -> (tensor<256xf16>)  : i32 {
    %5 = scf.for %arg5 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg6 = %3) -> (tensor<256xf16>)  : i32 {
      %8 = tensor.empty() : tensor<256xf16>
      %9 = hivm.hir.vadd ins(%arg6, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%8 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %9 : tensor<256xf16>
    }
    %6 = tensor.empty() : tensor<256xf16>
    %7 = hivm.hir.vadd ins(%5, %arg4 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %7 : tensor<256xf16>
  }
  return %4 : tensor<256xf16>
}

//-----

// CHECK: hivm.hir.vbrc
// CHECK: scf.for
// CHECK: hivm.hir.vbrc
func.func @test_sink_op_to_consumer(%arg0: i32, %arg1: tensor<256xf16>) -> tensor<256xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = tensor.empty() : tensor<256xf16>
  %3 = hivm.hir.vbrc ins(%cst : f16) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
  %4 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg3 = %1) -> (tensor<256xf16>)  : i32 {
    %5 = tensor.empty() : tensor<256xf16>
    %6 = hivm.hir.vadd ins(%arg3, %3 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %6 : tensor<256xf16>
  }
  return %4 : tensor<256xf16>
}

//-----

// CHECK: hivm.hir.vbrc
// CHECK: scf.for
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: scf.for
func.func @test_sink_op_to_consumer_in_loop_and_if(%arg0: i32, %arg1: tensor<256xf16>, %arg2: tensor<256xf16>) -> tensor<256xf16> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = tensor.empty() : tensor<256xf16>
  %3 = hivm.hir.vbrc ins(%cst : f16) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
  %4 = scf.for %arg3 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg4 = %1) -> (tensor<256xf16>)  : i32 {
    %7 = arith.cmpi eq, %arg3, %c4_i32 : i32
    %8 = scf.if %7 -> (tensor<256xf16>) {
      %5 = scf.for %arg5 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg6 = %3) -> (tensor<256xf16>)  : i32 {
        %8 = tensor.empty() : tensor<256xf16>
        %9 = hivm.hir.vadd ins(%arg6, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%8 : tensor<256xf16>) -> tensor<256xf16>
        scf.yield %9 : tensor<256xf16>
      }
      scf.yield %5 : tensor<256xf16>
    } else {
      scf.yield %arg4 : tensor<256xf16>
    }
    scf.yield %8 : tensor<256xf16>
  }
  return %4 : tensor<256xf16>
}

//-----

// CHECK: hivm.hir.vbrc
// CHECK: scf.for
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vbrc
func.func @test_sink_op_to_consumer_with_multi_users(%arg0: i32, %arg1: tensor<256xf16>) -> tensor<256xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = tensor.empty() : tensor<256xf16>
  %3 = hivm.hir.vbrc ins(%cst : f16) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
  %4 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c4_i32 iter_args(%arg3 = %1) -> (tensor<256xf16>)  : i32 {
    %5 = tensor.empty() : tensor<256xf16>
    %6 = hivm.hir.vadd ins(%arg3, %3 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
    %7 = tensor.empty() : tensor<256xf16>
    %8 = hivm.hir.vsub ins(%6, %3 : tensor<256xf16>, tensor<256xf16>) outs(%7 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %8 : tensor<256xf16>
  }
  return %4 : tensor<256xf16>
}
