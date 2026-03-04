// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_deinterleave_tile_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<4096x32x2xf16>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[arg0]]
// CHECK: %[[load:.*]] = hfusion.load {__intermediate_producer__} ins(%[[slice]]
// CHECK: hfusion.deinterleave %[[load]] channel<1> {__intermediate_producer__}
module {
  func.func @test_deinterleave_tile_0(%arg0: tensor<4096x32x2xf16>) -> tensor<4096x32x1xf16> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4096x32x1xf16>
    %1 = hfusion.deinterleave %arg0 channel<1> : tensor<4096x32x2xf16> -> tensor<4096x32x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
                                ins(%1, %cst : tensor<4096x32x1xf16>, f16) 
                                outs(%0 : tensor<4096x32x1xf16>) -> tensor<4096x32x1xf16>
    return %2 : tensor<4096x32x1xf16>
  }
}

// -----

// CHECK-LABEL: func.func @test_deinterleave_tile_reduction_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<4096x32x2xf16>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[arg0]]
// CHECK: %[[load:.*]] = hfusion.load {__intermediate_producer__, __reduction0_axis0_fusible_producer__} ins(%[[slice]]
// CHECK: hfusion.deinterleave %[[load]] channel<1> {__intermediate_producer__, __reduction0_axis0_fusible_producer__}
module {
  func.func @test_deinterleave_tile_reduction_0(%arg0: tensor<4096x32x2xf16>) -> tensor<32x1xf16> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<32x1xf16>
    %1 = hfusion.deinterleave %arg0 channel<1> : tensor<4096x32x2xf16> -> tensor<4096x32x1xf16>
    %2 = linalg.reduce ins(%1 : tensor<4096x32x1xf16>) outs(%0 : tensor<32x1xf16>) dimensions = [0] 
      (%in: f16, %init: f16) {
        %31 = arith.addf %in, %init : f16
        linalg.yield %31 : f16
      }
    return %2 : tensor<32x1xf16>
  }
}
