// RUN: bishengir-opt %s --vf-fusion="fusion-mode=max-parallel enable-ra=true enable-new-tree-reduce-policy=true" | FileCheck %s

// CHECK: func.func private @jagged_softmax_kernel_snippet_fused_{{[0-9]+}}(%[[INPUT:.*]]: tensor<128x64xf32>, %[[INIT:.*]]: tensor<64xf32>)
// CHECK-NEXT: %[[SUM:.*]] = linalg.reduce ins(%[[INPUT]] : tensor<128x64xf32>) outs(%[[INIT]] : tensor<64xf32>) dimensions = [0]
// CHECK: return %[[SUM]] : tensor<64xf32>

module {
  func.func @jagged_softmax_kernel_snippet(%data: tensor<128x64xbf16>,
      %limit: i64, %max_values: tensor<64xf32>, %initial_sum: tensor<64xf32>)
      -> (tensor<64xi1>, tensor<128x64xf32>, tensor<64xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant -1.0 : f16
    %cst_0 = arith.constant 0.0 : f16
    %cst_2 = arith.constant 0.0 : f32
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<64xi32>
    %4 = tensor.empty() : tensor<64xi1>
    %5 = tensor.empty() : tensor<128x64xf32>
    %13 = hfusion.arange offset[%c0] strides[%c1] outs(%0 : tensor<64xi32>) -> tensor<64xi32>
    %14 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%13, %c64_i32 : tensor<64xi32>, i32) outs(%4 : tensor<64xi1>) -> tensor<64xi1>
    %15 = tensor.empty() : tensor<128xi32>
    %16 = hfusion.arange offset[%c0] strides[%c1] outs(%15 : tensor<128xi32>) -> tensor<128xi32>
    %17 = tensor.empty() : tensor<128xi64>
    %18 = tensor.empty() : tensor<128x64xi1>
    %19 = tensor.empty() : tensor<64xf16>
    %21 = tensor.empty() : tensor<128x64xf16>
    %outer_mask = linalg.broadcast ins(%14 : tensor<64xi1>) outs(%18 : tensor<128x64xi1>) dimensions = [0]
    %result:3 = scf.for %arg13 = %c0_i32 to %c1024_i32 step %c128_i32
        iter_args(%last_mask = %4, %last_data = %5, %sum = %initial_sum)
        -> (tensor<64xi1>, tensor<128x64xf32>, tensor<64xf32>) : i32 {
      %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg13, %16 : i32, tensor<128xi32>) outs(%15 : tensor<128xi32>) -> tensor<128xi32>
      %28 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%27 : tensor<128xi32>) outs(%17 : tensor<128xi64>) -> tensor<128xi64>
      %29 = tensor.empty() : tensor<128xi1>
      %30 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%28, %limit : tensor<128xi64>, i64) outs(%29 : tensor<128xi1>) -> tensor<128xi1>
      %31 = tensor.empty() : tensor<128xf16>
      %32 = hfusion.select ins(%30, %cst, %cst_0 : tensor<128xi1>, f16, f16) outs(%31 : tensor<128xf16>) -> tensor<128xf16>
      %broadcasted_6 = linalg.broadcast ins(%32 : tensor<128xf16>) outs(%21 : tensor<128x64xf16>) dimensions = [1]
      %33 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_6, %cst_0 : tensor<128x64xf16>, f16) outs(%18 : tensor<128x64xi1>) -> tensor<128x64xi1>
      %34 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%33, %outer_mask : tensor<128x64xi1>, tensor<128x64xi1>) outs(%18 : tensor<128x64xi1>) -> tensor<128x64xi1>
      %49 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%data : tensor<128x64xbf16>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %76 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%max_values, %cst_3 : tensor<64xf32>, f32) outs(%4 : tensor<64xi1>) -> tensor<64xi1>
      %80 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%76 : tensor<64xi1>) outs(%76 : tensor<64xi1>) -> tensor<64xi1>
      %81 = hfusion.select ins(%80, %cst, %cst_0 : tensor<64xi1>, f16, f16) outs(%19 : tensor<64xf16>) -> tensor<64xf16>
      %broadcasted_9 = linalg.broadcast ins(%81 : tensor<64xf16>) outs(%21 : tensor<128x64xf16>) dimensions = [0]
      %82 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_9, %cst_0 : tensor<128x64xf16>, f16) outs(%18 : tensor<128x64xi1>) -> tensor<128x64xi1>
      %83 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%34, %82 : tensor<128x64xi1>, tensor<128x64xi1>) outs(%18 : tensor<128x64xi1>) -> tensor<128x64xi1>
      %broadcasted_10 = linalg.broadcast ins(%max_values : tensor<64xf32>) outs(%5 : tensor<128x64xf32>) dimensions = [0]
      %84 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%49, %broadcasted_10 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %85 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%84 : tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %86 = hfusion.select ins(%83, %85, %cst_2 : tensor<128x64xi1>, tensor<128x64xf32>, f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %reduced_11 = linalg.reduce ins(%86 : tensor<128x64xf32>) outs(%sum : tensor<64xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %89 = arith.addf %in, %init : f32
          linalg.yield %89 : f32
        }
      scf.yield %76, %49, %reduced_11 : tensor<64xi1>, tensor<128x64xf32>, tensor<64xf32>
    }
    return %result#0, %result#1, %result#2 : tensor<64xi1>, tensor<128x64xf32>, tensor<64xf32>
  }
}
