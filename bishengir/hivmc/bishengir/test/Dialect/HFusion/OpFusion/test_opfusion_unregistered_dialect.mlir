// RUN: bishengir-opt  %s --allow-unregistered-dialect --split-input-file --hfusion-fuse-ops="multi-kernel=true" --hfusion-flatten-ops="flatten-mode=tidy" | FileCheck %s

// CHECK-LABEL: func.func @main_multi_SHALLOW_CV_0
// CHECK-LABEL: func.func @main_multi_SHALLOW_CV_1
// CHECK-LABEL: func.func @main_multi_SHALLOW_CV_2
"func.func"() <{function_type = (tensor<128xi64>, tensor<128xi64>, tensor<32x128x2x128xf32>, tensor<32x128x2x128xf32>, tensor<1xi32>, tensor<128xi32>) -> (tensor<128x3200xf32>, tensor<32x128x2x128xf32>, tensor<32x128x2x128xf32>), sym_name = "main"}> ({
^bb0(%arg0: tensor<128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<32x128x2x128xf32>, %arg3: tensor<32x128x2x128xf32>, %arg4: tensor<1xi32>, %arg5: tensor<128xi32>):
  %0 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
  %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
  %2 = "arith.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "arith.constant"() <{value = dense<1.000000e-05> : tensor<1xf64>}> : () -> tensor<1xf64>
  %4 = "arith.constant"() <{value = dense<256> : tensor<1xi64>}> : () -> tensor<1xi64>
  %5 = "arith.constant"() <{value = dense<2> : tensor<1xi64>}> : () -> tensor<1xi64>
  %6 = "arith.constant"() <{value = 3 : i64}> : () -> i64
  %7 = "arith.constant"() <{value = -1 : i64}> : () -> i64
  %8 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32_1>"() : () -> tensor<3200x256xf32>
  %9 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_2>"() : () -> tensor<256xf32>
  %10 = "stablehlo.constant dense_resource<torch_tensor_256_688_torch.float32>"() : () -> tensor<256x688xf32>
  %11 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32_1>"() : () -> tensor<688x256xf32>
  %12 = "stablehlo.constant dense_resource<torch_tensor_688_256_torch.float32>"() : () -> tensor<688x256xf32>
  %13 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32_1>"() : () -> tensor<256xf32>
  %14 = "stablehlo.constant dense_resource<torch_tensor_256_256_torch.float32>"() : () -> tensor<256x256xf32>
  %15 = "stablehlo.constant dense_resource<torch_tensor_2048_2048_torch.float32>"() : () -> tensor<2048x2048xf32>
  %16 = "stablehlo.constant dense_resource<torch_tensor_768_256_torch.float32>"() : () -> tensor<768x256xf32>
  %17 = "stablehlo.constant dense_resource<torch_tensor_256_torch.float32>"() : () -> tensor<256xf32>
  %18 = "stablehlo.constant dense_resource<torch_tensor_2048_128_torch.float32_1>"() : () -> tensor<2048x128xf32>
  %19 = "stablehlo.constant dense_resource<torch_tensor_2048_128_torch.float32>"() : () -> tensor<2048x128xf32>
  %20 = "stablehlo.constant dense_resource<torch_tensor_3200_256_torch.float32>"() : () -> tensor<3200x256xf32>
  %21 = "stablehlo.constant dense<0.000000e+00>"() : () -> tensor<f32>
  %22 = "compose.gather"(%20, %arg0, %2) : (tensor<3200x256xf32>, tensor<128xi64>, i64) -> tensor<128x256xf32>
  %23 = "tensor.expand_shape"(%arg1) <{reassociation = [[0, 1]], static_output_shape = array<i64: 128, 1>}> : (tensor<128xi64>) -> tensor<128x1xi64>
  %24 = "compose.gather"(%19, %23, %2) : (tensor<2048x128xf32>, tensor<128x1xi64>, i64) -> tensor<128x128xf32>
  %25 = "compose.gather"(%18, %23, %2) : (tensor<2048x128xf32>, tensor<128x1xi64>, i64) -> tensor<128x128xf32>
  %26 = "tensor.empty"() : () -> tensor<1xf32>
  %27 = "linalg.map"(%5, %26) ({
  ^bb0(%arg6: i64):
    %93 = "arith.sitofp"(%arg6) : (i64) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<1xi64>, tensor<1xf32>) -> tensor<1xf32>
  %28 = "tensor.collapse_shape"(%27) <{reassociation = []}> : (tensor<1xf32>) -> tensor<f32>
  %29 = "tensor.empty"() : () -> tensor<128x256xf32>
  %30 = "tensor.extract"(%28) : (tensor<f32>) -> f32
  %31 = "hfusion.elemwise_binary"(%22, %30, %29) <{fun = #hfusion.binary_fn<powf>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "math.powf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, f32, tensor<128x256xf32>) -> tensor<128x256xf32>
  %32 = "tensor.empty"() : () -> tensor<128xf32>
  %33 = "linalg.fill"(%1, %32) <{operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (f32, tensor<128xf32>) -> tensor<128xf32>
  %34 = "linalg.reduce"(%31, %33) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.addf"(%arg7, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128xf32>) -> tensor<128xf32>
  %35 = "tensor.expand_shape"(%34) <{reassociation = [[0, 1]], static_output_shape = array<i64: 128, 1>}> : (tensor<128xf32>) -> tensor<128x1xf32>
  %36 = "linalg.map"(%4, %26) ({
  ^bb0(%arg6: i64):
    %93 = "arith.sitofp"(%arg6) : (i64) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<1xi64>, tensor<1xf32>) -> tensor<1xf32>
  %37 = "tensor.collapse_shape"(%36) <{reassociation = []}> : (tensor<1xf32>) -> tensor<f32>
  %38 = "tensor.empty"() : () -> tensor<128x1xf32>
  %39 = "linalg.broadcast"(%37, %38) <{dimensions = array<i64: 0, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<f32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %40 = "linalg.elemwise_binary"(%35, %39, %38) <{fun = #linalg.binary_fn<div>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.divf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %41 = "linalg.map"(%3, %26) ({
  ^bb0(%arg6: f64):
    %93 = "arith.truncf"(%arg6) : (f64) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<1xf64>, tensor<1xf32>) -> tensor<1xf32>
  %42 = "tensor.collapse_shape"(%41) <{reassociation = []}> : (tensor<1xf32>) -> tensor<f32>
  %43 = "tensor.extract"(%42) : (tensor<f32>) -> f32
  %44 = "linalg.elemwise_binary"(%40, %43, %38) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, f32, tensor<128x1xf32>) -> tensor<128x1xf32>
  %45 = "hfusion.elemwise_unary"(%44, %38) <{fun = #hfusion.unary_fn<sqrt>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "math.sqrt"(%arg6) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %46 = "hfusion.elemwise_unary"(%45, %38) <{fun = #hfusion.unary_fn<rec>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.divf"(%0, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %47 = "tensor.collapse_shape"(%46) <{reassociation = [[0, 1]]}> : (tensor<128x1xf32>) -> tensor<128xf32>
  %48 = "linalg.broadcast"(%47, %29) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<128xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %49 = "linalg.elemwise_binary"(%22, %48, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %50 = "linalg.broadcast"(%17, %29) <{dimensions = array<i64: 0>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %51 = "linalg.elemwise_binary"(%49, %50, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %52 = "tensor.empty"() : () -> tensor<128x768xf32>
  %53 = "linalg.matmul_transpose_b"(%51, %16, %52) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %94 = "arith.addf"(%arg8, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%94) : (f32) -> ()
  }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<128x256xf32>, tensor<768x256xf32>, tensor<128x768xf32>) -> tensor<128x768xf32>
  %54:3 = "compose.split"(%53, %6, %7) : (tensor<128x768xf32>, i64, i64) -> (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>)
  %55 = "tensor.expand_shape"(%54#0) <{reassociation = [[0], [1, 2]], static_output_shape = array<i64: 128, 2, 128>}> : (tensor<128x256xf32>) -> tensor<128x2x128xf32>
  %56 = "tensor.expand_shape"(%54#1) <{reassociation = [[0], [1, 2]], static_output_shape = array<i64: 128, 2, 128>}> : (tensor<128x256xf32>) -> tensor<128x2x128xf32>
  %57 = "tensor.expand_shape"(%54#2) <{reassociation = [[0], [1, 2]], static_output_shape = array<i64: 128, 2, 128>}> : (tensor<128x256xf32>) -> tensor<128x2x128xf32>
  %58:2 = "compose.rope"(%55, %56, %24, %25) : (tensor<128x2x128xf32>, tensor<128x2x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> (tensor<128x2x128xf32>, tensor<128x2x128xf32>)
  %59:2 = "compose.reshape_and_cache"(%58#1, %57, %arg2, %arg3, %arg5) : (tensor<128x2x128xf32>, tensor<128x2x128xf32>, tensor<32x128x2x128xf32>, tensor<32x128x2x128xf32>, tensor<128xi32>) -> (tensor<32x128x2x128xf32>, tensor<32x128x2x128xf32>)
  %60 = "compose.unpad_flash_attention_prefill"(%58#0, %58#1, %57, %15, %arg4) : (tensor<128x2x128xf32>, tensor<128x2x128xf32>, tensor<128x2x128xf32>, tensor<2048x2048xf32>, tensor<1xi32>) -> tensor<128x2x128xf32>
  %61 = "tensor.collapse_shape"(%60) <{reassociation = [[0], [1, 2]]}> : (tensor<128x2x128xf32>) -> tensor<128x256xf32>
  %62 = "linalg.matmul_transpose_b"(%61, %14, %29) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %94 = "arith.addf"(%arg8, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%94) : (f32) -> ()
  }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<128x256xf32>, tensor<256x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %63 = "linalg.elemwise_binary"(%22, %62, %29) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %64 = "hfusion.elemwise_binary"(%63, %30, %29) <{fun = #hfusion.binary_fn<powf>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "math.powf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, f32, tensor<128x256xf32>) -> tensor<128x256xf32>
  %65 = "linalg.reduce"(%64, %33) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.addf"(%arg7, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128xf32>) -> tensor<128xf32>
  %66 = "tensor.expand_shape"(%65) <{reassociation = [[0, 1]], static_output_shape = array<i64: 128, 1>}> : (tensor<128xf32>) -> tensor<128x1xf32>
  %67 = "linalg.elemwise_binary"(%66, %39, %38) <{fun = #linalg.binary_fn<div>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.divf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %68 = "linalg.elemwise_binary"(%67, %43, %38) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, f32, tensor<128x1xf32>) -> tensor<128x1xf32>
  %69 = "hfusion.elemwise_unary"(%68, %38) <{fun = #hfusion.unary_fn<sqrt>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "math.sqrt"(%arg6) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %70 = "hfusion.elemwise_unary"(%69, %38) <{fun = #hfusion.unary_fn<rec>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.divf"(%0, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %71 = "tensor.collapse_shape"(%70) <{reassociation = [[0, 1]]}> : (tensor<128x1xf32>) -> tensor<128xf32>
  %72 = "linalg.broadcast"(%71, %29) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<128xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %73 = "linalg.elemwise_binary"(%63, %72, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %74 = "linalg.broadcast"(%13, %29) <{dimensions = array<i64: 0>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %75 = "linalg.elemwise_binary"(%73, %74, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %76 = "compose.swiglu_forward"(%75, %12, %11) : (tensor<128x256xf32>, tensor<688x256xf32>, tensor<688x256xf32>) -> tensor<128x688xf32>
  %77 = "linalg.matmul_transpose_b"(%76, %10, %29) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %94 = "arith.addf"(%arg8, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%94) : (f32) -> ()
  }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<128x688xf32>, tensor<256x688xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %78 = "linalg.elemwise_binary"(%63, %77, %29) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %79 = "hfusion.elemwise_binary"(%78, %30, %29) <{fun = #hfusion.binary_fn<powf>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "math.powf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, f32, tensor<128x256xf32>) -> tensor<128x256xf32>
  %80 = "linalg.reduce"(%79, %33) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.addf"(%arg7, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128xf32>) -> tensor<128xf32>
  %81 = "tensor.expand_shape"(%80) <{reassociation = [[0, 1]], static_output_shape = array<i64: 128, 1>}> : (tensor<128xf32>) -> tensor<128x1xf32>
  %82 = "linalg.elemwise_binary"(%81, %39, %38) <{fun = #linalg.binary_fn<div>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.divf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %83 = "linalg.elemwise_binary"(%82, %43, %38) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, f32, tensor<128x1xf32>) -> tensor<128x1xf32>
  %84 = "hfusion.elemwise_unary"(%83, %38) <{fun = #hfusion.unary_fn<sqrt>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "math.sqrt"(%arg6) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %85 = "hfusion.elemwise_unary"(%84, %38) <{fun = #hfusion.unary_fn<rec>, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    %93 = "arith.divf"(%0, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %86 = "tensor.collapse_shape"(%85) <{reassociation = [[0, 1]]}> : (tensor<128x1xf32>) -> tensor<128xf32>
  %87 = "linalg.broadcast"(%86, %29) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<128xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %88 = "linalg.elemwise_binary"(%78, %87, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %89 = "linalg.broadcast"(%9, %29) <{dimensions = array<i64: 0>}> ({
  ^bb0(%arg6: f32, %arg7: f32):
    "linalg.yield"(%arg6) : (f32) -> ()
  }) : (tensor<256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %90 = "linalg.elemwise_binary"(%88, %89, %29) <{fun = #linalg.binary_fn<mul>, operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%93) : (f32) -> ()
  }) : (tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %91 = "tensor.empty"() : () -> tensor<128x3200xf32>
  %92 = "linalg.matmul_transpose_b"(%90, %8, %91) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %93 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %94 = "arith.addf"(%arg8, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%94) : (f32) -> ()
  }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<128x256xf32>, tensor<3200x256xf32>, tensor<128x3200xf32>) -> tensor<128x3200xf32>
  "func.return"(%92, %59#0, %59#1) : (tensor<128x3200xf32>, tensor<32x128x2x128xf32>, tensor<32x128x2x128xf32>) -> ()
}) {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<UNKNOWN>} : () -> ()
