// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-PAD

// -----

// CHECK-LABEL: @test_hfusion_store_pad(
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.pad
// CHECK-PAD-LABEL: @test_hfusion_store_pad(
// CHECK-PAD: %[[padded:.*]] = tensor.pad
// CHECK-PAD: hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%[[padded]] : tensor<4093xf32>)
module {
  func.func @test_hfusion_store_pad(%arg0: tensor<1x1x2047xf32>) -> tensor<4093xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<6140xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6140xf32>) -> tensor<6140xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[2046] [2047] [1] : tensor<2047xf32> into tensor<6140xf32>
    %padded = tensor.pad %inserted_slice low[0] high[-2047] {
    ^bb0(%arg1: index):
      tensor.yield %cst : f32
    } : tensor<6140xf32> to tensor<4093xf32>
    return %padded : tensor<4093xf32>
  }
}
