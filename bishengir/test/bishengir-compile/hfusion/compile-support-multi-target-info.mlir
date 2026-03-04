// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 %s | FileCheck %s

// CHECK-DAG: #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>
// CHECK-DAG: #hacc.target
func.func @rms_norm_f32(%arg0: tensor<8x4194304xf32>, %arg1: tensor<4194304xf32>, %arg2: tensor<8x4194304xf32>, %arg3: tensor<8xf32>) -> (tensor<8x4194304xf32>, tensor<8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 2.38418579E-7 : f32
  %cst_0 = arith.constant 1.000000e-01 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x4194304xf32>
  %1 = tensor.empty() : tensor<8xf32>
  %2 = linalg.elemwise_binary {__1__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<8x4194304xf32>, tensor<8x4194304xf32>) outs(%0 : tensor<8x4194304xf32>) -> tensor<8x4194304xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<4194304xf32>) outs(%0 : tensor<8x4194304xf32>) dimensions = [0]  {__9__}
  %reduced = linalg.reduce { arith.addf } ins(%2 : tensor<8x4194304xf32>) outs(%1 : tensor<8xf32>) dimensions = [1]  {__2__}
  %3 = linalg.elemwise_binary {__3__, fun = #linalg.binary_fn<mul>} ins(%reduced, %cst : tensor<8xf32>, f32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  %4 = linalg.elemwise_binary {__4__, fun = #linalg.binary_fn<add>} ins(%3, %cst_0 : tensor<8xf32>, f32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  %5 = hfusion.elemwise_unary {__5__, fun = #hfusion.unary_fn<sqrt>} ins(%4 : tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  %6 = hfusion.elemwise_unary {__6__, fun = #hfusion.unary_fn<rec>} ins(%5 : tensor<8xf32>) outs(%arg3 : tensor<8xf32>) -> tensor<8xf32>
  %broadcasted_2 = linalg.broadcast ins(%6 : tensor<8xf32>) outs(%0 : tensor<8x4194304xf32>) dimensions = [1]  {__7__}
  %7 = linalg.elemwise_binary {__8__, fun = #linalg.binary_fn<mul>} ins(%broadcasted_2, %arg0 : tensor<8x4194304xf32>, tensor<8x4194304xf32>) outs(%0 : tensor<8x4194304xf32>) -> tensor<8x4194304xf32>
  %8 = linalg.elemwise_binary {__10__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %7 : tensor<8x4194304xf32>, tensor<8x4194304xf32>) outs(%arg2 : tensor<8x4194304xf32>) -> tensor<8x4194304xf32>
  return %8, %6 : tensor<8x4194304xf32>, tensor<8xf32>
}