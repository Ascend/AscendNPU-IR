// RUN: bishengir-opt --hfusion-fuse-ops="multi-kernel=true" %s --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK: SHALLOW_CV
// CHECK: SHALLOW_CV
// CHECK: SHALLOW_CV
// CHECK-LABEL: func.func @main(
// CHECK: return
module {
  func.func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<2x128xi64>, %arg2: tensor<1x2x128x2x128xf32>, %arg3: tensor<1x2x128x2x128xf32>, %arg4: tensor<2xi32>, %arg5: tensor<2xi32>, %arg6: tensor<2xi32>) -> tensor<2x128x3200xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>}{
    %cst = arith.constant dense<0> : tensor<1xi64>
    %c3_i64 = arith.constant 3 : i64
    %c-1_i64 = arith.constant -1 : i64
    %cst_0 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32_1>"() : () -> tensor<3200x256xf32>
    %cst_1 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_2>"() : () -> tensor<256xf32>
    %cst_2 = "stablehlo.constant dense_resource<torch_tensor_256_688_torch.float32>"() : () -> tensor<256x688xf32>
    %cst_3 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32_1>"() : () -> tensor<688x256xf32>
    %cst_4 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32>"() : () -> tensor<688x256xf32>
    %cst_5 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_1>"() : () -> tensor<256xf32>
    %cst_6 = "stablehlo.constant dense_resource<torch_tensor_256_256_torch.float32>"() : () -> tensor<256x256xf32>
    %cst_7 = "stablehlo.constant dense_resource<torch_tensor_2048_2048_torch.float32>"() : () -> tensor<2048x2048xf32>
    %cst_8 = "stablehlo.constant dense_resource<torch_tensor_768_256_torch.float32>"() : () -> tensor<768x256xf32>
    %cst_9 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32>"() : () -> tensor<256xf32>
    %cst_10 = "stablehlo.constant dense_resource<torch_tensor_2048_64_2_torch.float32>"() : () -> tensor<2048x64x2xf32>
    %cst_11 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32>"() : () -> tensor<3200x256xf32>
    %c0_i64 = arith.constant 0 : i64
    %0 = "compose.gather"(%cst_11, %arg0, %c0_i64) : (tensor<3200x256xf32>, tensor<2x128xi64>, i64) -> tensor<2x128x256xf32>
    %expanded = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [2, 128, 1] : tensor<2x128xi64> into tensor<2x128x1xi64>
    %c0_i64_12 = arith.constant 0 : i64
    %1 = "compose.gather"(%cst_10, %expanded, %c0_i64_12) : (tensor<2048x64x2xf32>, tensor<2x128x1xi64>, i64) -> tensor<2x128x64x2xf32>
    %2 = "compose.rms_norm"(%0, %cst_9) <{eps = 9.99999974E-6 : f32, inputScale = 1.000000e+00 : f32}> : (tensor<2x128x256xf32>, tensor<256xf32>) -> tensor<2x128x256xf32>
    %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x128x256xf32> into tensor<256x256xf32>
    %3 = tensor.empty() : tensor<256x768xf32>
    %4 = linalg.matmul_transpose_b ins(%collapsed, %cst_8 : tensor<256x256xf32>, tensor<768x256xf32>) outs(%3 : tensor<256x768xf32>) -> tensor<256x768xf32>
    %expanded_13 = tensor.expand_shape %4 [[0, 1], [2]] output_shape [2, 128, 768] : tensor<256x768xf32> into tensor<2x128x768xf32>
    %5:3 = "compose.split"(%expanded_13, %c3_i64, %c-1_i64) : (tensor<2x128x768xf32>, i64, i64) -> (tensor<2x128x256xf32>, tensor<2x128x256xf32>, tensor<2x128x256xf32>)
    %expanded_14 = tensor.expand_shape %5#0 [[0], [1], [2, 3]] output_shape [2, 128, 2, 128] : tensor<2x128x256xf32> into tensor<2x128x2x128xf32>
    %expanded_15 = tensor.expand_shape %5#1 [[0], [1], [2, 3]] output_shape [2, 128, 2, 128] : tensor<2x128x256xf32> into tensor<2x128x2x128xf32>
    %expanded_16 = tensor.expand_shape %5#2 [[0], [1], [2, 3]] output_shape [2, 128, 2, 128] : tensor<2x128x256xf32> into tensor<2x128x2x128xf32>
    %expanded_17 = tensor.expand_shape %1 [[0], [1], [2, 3], [4]] output_shape [2, 128, 1, 64, 2] : tensor<2x128x64x2xf32> into tensor<2x128x1x64x2xf32>
    %extracted_slice = tensor.extract_slice %expanded_17[0, 0, 0, 0, 0] [2, 128, 1, 64, 1] [1, 1, 1, 1, 1] : tensor<2x128x1x64x2xf32> to tensor<2x128x1x64x1xf32>
    %collapsed_18 = tensor.collapse_shape %extracted_slice [[0], [1], [2], [3, 4]] : tensor<2x128x1x64x1xf32> into tensor<2x128x1x64xf32>
    %extracted_slice_19 = tensor.extract_slice %expanded_17[0, 0, 0, 0, 1] [2, 128, 1, 64, 1] [1, 1, 1, 1, 1] : tensor<2x128x1x64x2xf32> to tensor<2x128x1x64x1xf32>
    %collapsed_20 = tensor.collapse_shape %extracted_slice_19 [[0], [1], [2], [3, 4]] : tensor<2x128x1x64x1xf32> into tensor<2x128x1x64xf32>
    %q_out, %k_out = "compose.rope"(%expanded_14, %expanded_15, %collapsed_18, %collapsed_20) <{rotaryCoeff = 2 : i64}> : (tensor<2x128x2x128xf32>, tensor<2x128x2x128xf32>, tensor<2x128x1x64xf32>, tensor<2x128x1x64xf32>) -> (tensor<2x128x2x128xf32>, tensor<2x128x2x128xf32>)
    %6 = "compose.kv_cache"(%k_out, %cst, %arg2, %arg5, %arg4) : (tensor<2x128x2x128xf32>, tensor<1xi64>, tensor<1x2x128x2x128xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2x128x2x128xf32>
    %7 = "compose.kv_cache"(%expanded_16, %cst, %arg3, %arg5, %arg4) : (tensor<2x128x2x128xf32>, tensor<1xi64>, tensor<1x2x128x2x128xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2x128x2x128xf32>
    %8 = "compose.flash_attention_prefill"(%q_out, %k_out, %expanded_16, %cst_7, %arg5) <{kv_heads_num = 2 : i64, q_heads_num = 2 : i64, qk_scale = 0.0883883461 : f32}> : (tensor<2x128x2x128xf32>, tensor<2x128x2x128xf32>, tensor<2x128x2x128xf32>, tensor<2048x2048xf32>, tensor<2xi32>) -> tensor<2x128x2x128xf32>
    %collapsed_21 = tensor.collapse_shape %8 [[0, 1], [2, 3]] : tensor<2x128x2x128xf32> into tensor<256x256xf32>
    %9 = tensor.empty() : tensor<256x256xf32>
    %10 = linalg.matmul_transpose_b ins(%collapsed_21, %cst_6 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%9 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %expanded_22 = tensor.expand_shape %10 [[0, 1], [2]] output_shape [2, 128, 256] : tensor<256x256xf32> into tensor<2x128x256xf32>
    %11 = tensor.empty() : tensor<2x128x256xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %expanded_22 : tensor<2x128x256xf32>, tensor<2x128x256xf32>) outs(%11 : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
    %13 = "compose.rms_norm"(%12, %cst_5) <{eps = 9.99999974E-6 : f32, inputScale = 1.000000e+00 : f32}> : (tensor<2x128x256xf32>, tensor<256xf32>) -> tensor<2x128x256xf32>
    %collapsed_23 = tensor.collapse_shape %13 [[0, 1], [2]] : tensor<2x128x256xf32> into tensor<256x256xf32>
    %14 = tensor.empty() : tensor<256x688xf32>
    %15 = linalg.matmul_transpose_b ins(%collapsed_23, %cst_4 : tensor<256x256xf32>, tensor<688x256xf32>) outs(%14 : tensor<256x688xf32>) -> tensor<256x688xf32>
    %expanded_24 = tensor.expand_shape %15 [[0, 1], [2]] output_shape [2, 128, 688] : tensor<256x688xf32> into tensor<2x128x688xf32>
    %16 = tensor.empty() : tensor<2x128x688xf32>
    %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%expanded_24 : tensor<2x128x688xf32>) outs(%16 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %18 = tensor.empty() : tensor<2x128x688xf32>
    %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%17 : tensor<2x128x688xf32>) outs(%18 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %cst_25 = arith.constant 1.000000e+00 : f32
    %splat = tensor.splat %cst_25 : tensor<2x128x688xf32>
    %20 = tensor.empty() : tensor<2x128x688xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%splat, %19 : tensor<2x128x688xf32>, tensor<2x128x688xf32>) outs(%20 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %22 = tensor.empty() : tensor<2x128x688xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%splat, %21 : tensor<2x128x688xf32>, tensor<2x128x688xf32>) outs(%22 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %24 = tensor.empty() : tensor<2x128x688xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %expanded_24 : tensor<2x128x688xf32>, tensor<2x128x688xf32>) outs(%24 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %26 = tensor.empty() : tensor<256x688xf32>
    %27 = linalg.matmul_transpose_b ins(%collapsed_23, %cst_3 : tensor<256x256xf32>, tensor<688x256xf32>) outs(%26 : tensor<256x688xf32>) -> tensor<256x688xf32>
    %expanded_26 = tensor.expand_shape %27 [[0, 1], [2]] output_shape [2, 128, 688] : tensor<256x688xf32> into tensor<2x128x688xf32>
    %28 = tensor.empty() : tensor<2x128x688xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %expanded_26 : tensor<2x128x688xf32>, tensor<2x128x688xf32>) outs(%28 : tensor<2x128x688xf32>) -> tensor<2x128x688xf32>
    %collapsed_27 = tensor.collapse_shape %29 [[0, 1], [2]] : tensor<2x128x688xf32> into tensor<256x688xf32>
    %30 = tensor.empty() : tensor<256x256xf32>
    %31 = linalg.matmul_transpose_b ins(%collapsed_27, %cst_2 : tensor<256x688xf32>, tensor<256x688xf32>) outs(%30 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %expanded_28 = tensor.expand_shape %31 [[0, 1], [2]] output_shape [2, 128, 256] : tensor<256x256xf32> into tensor<2x128x256xf32>
    %32 = tensor.empty() : tensor<2x128x256xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%12, %expanded_28 : tensor<2x128x256xf32>, tensor<2x128x256xf32>) outs(%32 : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
    %34 = "compose.rms_norm"(%33, %cst_1) <{eps = 9.99999974E-6 : f32, inputScale = 1.000000e+00 : f32}> : (tensor<2x128x256xf32>, tensor<256xf32>) -> tensor<2x128x256xf32>
    %collapsed_29 = tensor.collapse_shape %34 [[0, 1], [2]] : tensor<2x128x256xf32> into tensor<256x256xf32>
    %35 = tensor.empty() : tensor<256x3200xf32>
    %36 = linalg.matmul_transpose_b ins(%collapsed_29, %cst_0 : tensor<256x256xf32>, tensor<3200x256xf32>) outs(%35 : tensor<256x3200xf32>) -> tensor<256x3200xf32>
    %expanded_30 = tensor.expand_shape %36 [[0, 1], [2]] output_shape [2, 128, 3200] : tensor<256x3200xf32> into tensor<2x128x3200xf32>
    return %expanded_30 : tensor<2x128x3200xf32>
  }
}

// -----

// SHALLOW-CV-LABEL: func.func @testA(
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV: linalg.elemwise_binary
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV: linalg.matmul
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV-LABEL-NOT: func.func
func.func @testA(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg2 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%7 : tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %13 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%12: tensor<?x?x?xf32>) dimensions = [0]
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %17 = linalg.broadcast ins(%13 : tensor<?x?x?xf32>) outs(%16: tensor<?x?x?x?xf32>) dimensions = [3]
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.transpose ins(%15 : tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) permutation = [0, 1]
  return %arg1, %9, %11, %13, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK: func.func @main_multi_
// CHECK-SAME: LAST_AXIS_PBR
// CHECK: return
// CHECK: func.func @main(
// CHECK-NOT: linalg
// CHECK: return
module {
  func.func @main(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>, %arg2: tensor<32x128x2x128xf32>, %arg3: tensor<32x128x2x128xf32>, %arg4: tensor<1xi32>, %arg5: tensor<?xi32>) -> (tensor<?x1xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<UNKNOWN>} {
    %c-1_i64 = arith.constant -1 : i64
    %c3_i64 = arith.constant 3 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant -1.000000e+00 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 2.560000e+02 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %0 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32_1>"() : () -> tensor<3200x256xf32>
    %1 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_2>"() : () -> tensor<256xf32>
    %2 = "stablehlo.constant dense_resource<torch_tensor_256_688_torch.float32>"() : () -> tensor<256x688xf32>
    %3 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32_1>"() : () -> tensor<688x256xf32>
    %4 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32>"() : () -> tensor<688x256xf32>
    %5 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_1>"() : () -> tensor<256xf32>
    %6 = "stablehlo.constant dense_resource<torch_tensor_256_256_torch.float32>"() : () -> tensor<256x256xf32>
    %7 = "stablehlo.constant dense_resource<torch_tensor_2048_2048_torch.float32>"() : () -> tensor<2048x2048xf32>
    %8 = "stablehlo.constant dense_resource<torch_tensor_768_256_torch.float32>"() : () -> tensor<768x256xf32>
    %9 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32>"() : () -> tensor<256xf32>
    %10 = "stablehlo.constant dense_resource<torch_tensor_2048_128_torch.float32_1>"() : () -> tensor<2048x128xf32>
    %11 = "stablehlo.constant dense_resource<torch_tensor_2048_128_torch.float32>"() : () -> tensor<2048x128xf32>
    %12 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32>"() : () -> tensor<3200x256xf32>
    %13 = "stablehlo.gather"(%12, %arg0) : (tensor<3200x256xf32>, tensor<?xi64>) -> tensor<?x256xf32>
    %dim = tensor.dim %arg1, %c0 : tensor<?xi64>
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [%dim, 1] : tensor<?xi64> into tensor<?x1xi64>
    %14 = "stablehlo.gather"(%11, %expanded) : (tensor<2048x128xf32>, tensor<?x1xi64>) -> tensor<?x128xf32>
    %15 = "stablehlo.gather"(%10, %expanded) : (tensor<2048x128xf32>, tensor<?x1xi64>) -> tensor<?x128xf32>
    %dim_5 = tensor.dim %13, %c0 : tensor<?x256xf32>
    %16 = tensor.empty(%dim_5) : tensor<?x256xf32>
    %17 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%13, %cst_4 : tensor<?x256xf32>, f32) outs(%16 : tensor<?x256xf32>) -> tensor<?x256xf32>
    %18 = tensor.empty(%dim_5) : tensor<?xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?xf32>) -> tensor<?xf32>
    %reduced = linalg.reduce { arith.addf } ins(%17 : tensor<?x256xf32>) outs(%19 : tensor<?xf32>) dimensions = [1]
    %20 = tensor.empty(%dim_5) : tensor<?x1xf32>
    %broadcasted = linalg.broadcast ins(%reduced : tensor<?xf32>) outs(%20 : tensor<?x1xf32>) dimensions = [1]
    return %broadcasted : tensor<?x1xf32>
  }
}

// -----
// CHECK-LABEL: main_last_pbr_multi_LAST_AXIS_PBR_0(
// CHECK: collapse_shape
// CHECK: return
// CHECK-LABEL: main_last_pbr(
// CHECK: return
module {
  func.func @main_last_pbr(%arg0: tensor<?x4096xf16>, %arg1: tensor<?x4096xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?x1xf16>) -> (tensor<?x1xf16>, tensor<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant 1.001360e-05 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x1xf16>
    %1 = tensor.empty(%dim) : tensor<?xf16>
    %2 = tensor.empty(%dim) : tensor<?x1xf16>
    %reduced = linalg.reduce { arith.addf } ins(%arg1 : tensor<?x4096xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<?xf32>) outs(%1 : tensor<?xf16>) -> tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%3 : tensor<?xf16>) outs(%0 : tensor<?x1xf16>) dimensions = [1]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%broadcasted, %arg3 : tensor<?x1xf16>, tensor<?x1xf16>) outs(%0 : tensor<?x1xf16>) -> tensor<?x1xf16>
    %collapsed = tensor.collapse_shape %4 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted_0 = linalg.broadcast ins(%collapsed : tensor<?xf16>) outs(%0 : tensor<?x1xf16>) dimensions = [1]
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted_0, %cst : tensor<?x1xf16>, f16) outs(%0 : tensor<?x1xf16>) -> tensor<?x1xf16>
    %6 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%5 : tensor<?x1xf16>) outs(%2 : tensor<?x1xf16>) -> tensor<?x1xf16>
    %7 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%6 : tensor<?x1xf16>) outs(%0 : tensor<?x1xf16>) -> tensor<?x1xf16>
    %collapsed_1 = tensor.collapse_shape %7 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    return %7, %collapsed_1 : tensor<?x1xf16>, tensor<?xf16>
  }
}

// -----
// CHECK-LABEL: main_mix_cv_multi_MIX_CV_0
// CHECK: linalg.matmul_transpose_b
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: main_mix_cv_multi_MIX_CV_1
// CHECK: hfusion.elemwise_binary
// CHECK: hfusion.cast
// CHECK: return

module{
  func.func @main_mix_cv(%arg0: tensor<?x4096xf16>, %arg1: tensor<?x32x128xf16>, %arg2: tensor<?x4096xf16>, %arg3: tensor<4096x4096xf16>) -> (tensor<?x4096xf16>, tensor<?x4096xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant 2.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x4096xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?x32x128xf16>
    %1 = tensor.empty(%dim_0) : tensor<?x4096xf16>
    %2 = linalg.matmul_transpose_b ins(%arg2, %arg3 : tensor<?x4096xf16>, tensor<4096x4096xf16>) outs(%1 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %2 : tensor<?x4096xf16>, tensor<?x4096xf16>) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%3, %cst : tensor<?x4096xf16>, f16) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %5 = tensor.empty(%dim) : tensor<?x4096xf32>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<?x4096xf16>) outs(%5 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    return %3, %6 : tensor<?x4096xf16>, tensor<?x4096xf32>
  }
}

// -----
// CHECK-LABEL: main_shallow_vv_multi_SHALLOW_VV_0(
// CHECK: linalg.reduce
// CHECK: linalg.broadcast
// CHECK: return
// CHECK-LABEL: main_shallow_vv(
// CHECK: call @main_shallow_vv_multi_SHALLOW_VV_0
// CHECK: return
module {
  func.func @main_shallow_vv(%arg0: tensor<7x4096xf16>, %arg1: tensor<4096xf16>) -> (tensor<4096xf16>, tensor<4096x1xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<4096x1xf16>
    %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<7x4096xf16>) outs(%arg1 : tensor<4096xf16>) dimensions = [0]
    %broadcasted = linalg.broadcast ins(%reduced : tensor<4096xf16>) outs(%0 : tensor<4096x1xf16>) dimensions = [1]
    return %reduced, %broadcasted : tensor<4096xf16>, tensor<4096x1xf16>
  }
}

// -----
// CHECK-LABEL: main_shallow_vv_any_pbr_multi_SHALLOW_VV_0_multi_LAST_AXIS_PBR_0(
// CHECK: linalg.reduce
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_unary
// CHECK: return
// CHECK-LABEL: main_shallow_vv_any_pbr_multi_SHALLOW_VV_0(
// CHECK: call @main_shallow_vv_any_pbr_multi_SHALLOW_VV_0_multi_LAST_AXIS_PBR_0
// CHECK: return
// CHECK-LABEL: main_shallow_vv_any_pbr(
// CHECK: call @main_shallow_vv_any_pbr_multi_SHALLOW_VV_0
module {
  func.func @main_shallow_vv_any_pbr(%arg0: tensor<7x4096xf16>, %arg1: tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<7x1xf16>
    %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<7x4096xf16>) outs(%arg1 : tensor<7xf16>) dimensions = [1]
    %broadcasted = linalg.broadcast ins(%reduced : tensor<7xf16>) outs(%0 : tensor<7x1xf16>) dimensions = [1]
    %1 = tensor.empty() : tensor<7x1xf16>
    %logged = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%broadcasted : tensor<7x1xf16>) outs(%1 : tensor<7x1xf16>) -> tensor<7x1xf16>
    return %reduced, %logged : tensor<7xf16>, tensor<7x1xf16>
  }
}

// -----
// CHECK-LABEL: func.func @multi_end_alias_0_multi_SHALLOW_VV_0_multi_LAST_AXIS_PBR_0(
// CHECK: expand_shape
// CHECK: return
// CHECK-LABEL: func.func @multi_end_alias_0_multi_SHALLOW_VV_0(
// CHECK-LABEL: multi_end_alias_0
// CHECK: call @multi_end_alias_0_multi_SHALLOW_VV_0(
module {
  func.func @multi_end_alias_0(%arg0: tensor<1x512xbf16>, %arg1: tensor<1x512xbf16>, %arg2: tensor<512xbf16>) -> (tensor<512xbf16>, tensor<1x512xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x512xbf16> into tensor<512xbf16>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x512xbf16> into tensor<512xbf16>
    %0 = tensor.empty() : tensor<512xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<512xbf16>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<512xbf16>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<512xf32>) outs(%arg2 : tensor<512xbf16>) -> tensor<512xbf16>
    %expanded = tensor.expand_shape %4 [[0, 1]] output_shape [1, 512] : tensor<512xbf16> into tensor<1x512xbf16>
    return %4, %expanded : tensor<512xbf16>, tensor<1x512xbf16>
  }
  func.func @multi_end_alias(%arg0: tensor<1x512xbf16>, %arg1: tensor<1x512xbf16>, %arg2: tensor<1x512xbf16>, %arg3: tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<512xbf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, mindspore_kernel, process = "aicore"} {
    %0 = tensor.empty() : tensor<512xbf16>
    %1:2 = call @multi_end_alias_0(%arg3, %arg2, %0) : (tensor<1x512xbf16>, tensor<1x512xbf16>, tensor<512xbf16>) -> (tensor<512xbf16>, tensor<1x512xbf16>)
    return %1#1, %1#0 : tensor<1x512xbf16>, tensor<512xbf16>
  }
}

// -----
 
// CHECK-LABEL: func.func @squeeze_returned_operands_on_multi_kernel_0(
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape
// CHECK: return %[[EXPAND]]
// CHECK-LABEL: func.func @squeeze_returned_operands_on_multi_kernel(
// CHECK: %[[CALL:.*]] = call @squeeze_returned_operands_on_multi_kernel_0(
// CHECK: return %[[CALL]], %[[CALL]], %[[CALL]]
#map = affine_map<()[s0] -> (s0 floordiv 2)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 2) * 1280)>
module {
  func.func @squeeze_returned_operands_on_multi_kernel(%arg0: tensor<?x640xf16>, %arg1: i64) -> (tensor<2x?x640xbf16>, tensor<2x?x640xbf16>, tensor<2x?x640xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x640xf16> into tensor<?xf16>
    %dim = tensor.dim %arg0, %c0 : tensor<?x640xf16>
    %0 = affine.apply #map()[%dim]
    %1 = affine.apply #map1()[%dim]
    %2 = tensor.empty(%1) : tensor<?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<?xf16>) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    %4 = affine.apply #map1()[%dim]
    %5 = tensor.empty(%4) : tensor<?xbf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<?xf32>) outs(%5 : tensor<?xbf16>) -> tensor<?xbf16>
    %expanded = tensor.expand_shape %6 [[0, 1, 2]] output_shape [2, %0, 640] : tensor<?xbf16> into tensor<2x?x640xbf16>
    return %expanded, %expanded, %expanded : tensor<2x?x640xbf16>, tensor<2x?x640xbf16>, tensor<2x?x640xbf16>
  }
}