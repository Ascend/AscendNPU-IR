// RUN: bishengir-opt %s -split-input-file \
// RUN:   -pass-pipeline="builtin.module(  \
// RUN:     canonicalize-ext,                  \
// RUN:     func.func(hfusion-outline-single-op))" | FileCheck %s

// RUN: bishengir-opt --hfusion-fuse-ops="move-out-to-param=true" %s --split-input-file | FileCheck %s --check-prefix=CHECK-FUSE-MOVEPARAM-TRUE
// RUN: bishengir-opt --hfusion-fuse-ops="move-out-to-param=false" %s --split-input-file | FileCheck %s --check-prefix=CHECK-FUSE-MOVEPARAM-FALSE

// CHECK-LABEL: func.func @add_mul_reduce_single_outlined_0
// CHECK-SAME: LAST_AXIS_PBR
// CHECK: linalg.reduce
// CHECK: return
// CHECK-LABEL: func.func @add_mul_reduce(
// CHECK-NOT: linalg
// CHECK: return
func.func @add_mul_reduce_0(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> tensor<?xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg3 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}
func.func @add_mul_reduce(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<f32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %2 = call @add_mul_reduce_0(%arg0, %arg1, %0, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = tensor.empty() : tensor<f32>
  %reduced = linalg.reduce { arith.addf } ins(%2 : tensor<?xf32>) outs(%3 : tensor<f32>) dimensions = [0]
  return %reduced : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @host_elemwise_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: func.func @host_elemwise
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_elemwise_single_outlined_0
// CHECK: return
func.func @host_elemwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @host_matmul_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SINGLE_CUBE>}
// CHECK: linalg.matmul
// CHECK: return
// CHECK-LABEL: func.func @host_matmul
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_matmul_single_outlined_0
// CHECK: return
func.func @host_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func.func @host_reduce_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}
// CHECK: linalg.reduce
// CHECK: return
// CHECK-LABEL: func.func @host_reduce
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_reduce_single_outlined_0
// CHECK: return
func.func @host_reduce(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<f32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %reduced = linalg.reduce { arith.addf } 
    ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>) dimensions = [0]
  return %reduced : tensor<f32>
}

// -----
// CHECK-LABEL: func.func @device_broadcast_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>}
// CHECK: linalg.broadcast
// CHECK: return
// CHECK-LABEL: func.func @device_broadcast
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @device_broadcast_single_outlined_0
// CHECK: return
func.func @device_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = linalg.broadcast 
    ins(%arg0 : tensor<?xf32>) 
    outs(%arg1 : tensor<?x?xf32>) 
    dimensions = [1]
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func.func @host_multi_ops_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: func.func @host_multi_ops_single_outlined_1
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}
// CHECK: linalg.reduce
// CHECK: return
// CHECK-LABEL: func.func @host_multi_ops
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_multi_ops_single_outlined_0
// CHECK: call @host_multi_ops_single_outlined_1
// CHECK: return
func.func @host_multi_ops(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %0 : tensor<?xf32>, %3 : tensor<f32>) -> tensor<f32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.matmul ins(%arg2, %arg3: tensor<?x?xf32>, tensor<?x?xf32>) 
    outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
  %reduced = linalg.reduce { arith.addf } 
    ins(%1 : tensor<?xf32>) outs(%3 : tensor<f32>) dimensions = [0]
  return %reduced : tensor<f32>
}

// -----
// CHECK-LABEL: @fused_with_aux_single_outlined_0
// CHECK: arith.constant
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: fused_with_aux
// CHECK-NOT: linalg
// CHECK: return
func.func @fused_with_aux(%arg0: tensor<4x3072x3072xf32>, %arg1: tensor<4x3072x3072xf32>) -> tensor<4x3072x3072xf32> attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["4", "3072", "3072"], output_0 = ["4", "3072", "3072"]}, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.0441941731 : f32
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<4x3072x3072xf32>, f32) outs(%arg1 : tensor<4x3072x3072xf32>) -> tensor<4x3072x3072xf32>
  return %0 : tensor<4x3072x3072xf32>
}

// -----

// CHECK-FUSE-MOVEPARAM-TRUE-LABEL: @op_fusion_select(
// CHECK-FUSE-MOVEPARAM-TRUE: call @op_fusion_select_0
module {
  func.func @op_fusion_select(%arg0: tensor<2x4x1x1xi1>, %arg1: tensor<2x4x768x152xf32>, %arg2: tensor<2x4x768x152xf32>) -> tensor<2x4x768x152xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<2x4x1x1xi1> into tensor<2x4xi1>
    %0 = tensor.empty() : tensor<2x4x768x152xi1>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x4xi1>) outs(%0 : tensor<2x4x768x152xi1>) dimensions = [2, 3]
    %1 = tensor.empty() : tensor<2x4x768x152xf32>
    %2 = hfusion.select ins(%broadcasted, %arg1, %arg2 : tensor<2x4x768x152xi1>, tensor<2x4x768x152xf32>, tensor<2x4x768x152xf32>)
                        outs(%1 : tensor<2x4x768x152xf32>) -> tensor<2x4x768x152xf32>
    return %2 : tensor<2x4x768x152xf32>
  }
}

// -----

// CHECK: func.func @test_outline_return_only_func({{.*}}: tensor<?xbf16>, {{.*}}: tensor<?x?xbf16>)
// CHECK-SAME: hacc.function_kind<HOST>
// CHECK: %[[call:.*]]:2 = call @test_outline_return_only_func_single_outlined
// CHECK: return %[[call]]#0, %[[call]]#1 : tensor<?xbf16>, tensor<?x?xbf16>

// CHECK-LABEL: func.func @test_outline_return_only_func_single_outlined(
// CHECK-SAME: {{.*}}: tensor<?xbf16>, {{.*}}: tensor<?x?xbf16>)
// CHECK-SAME: hacc.function_kind<DEVICE>
// CHECK: return {{.*}} : tensor<?xbf16>, tensor<?x?xbf16>
module {
  func.func @test_outline_return_only_func(%arg0: tensor<?xbf16>, %arg1: tensor<?x?xbf16>) -> (tensor<?xbf16>, tensor<?x?xbf16>) 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    return %arg0, %arg1 : tensor<?xbf16>, tensor<?x?xbf16>
  }
}

// -----
// CHECK-FUSE-MOVEPARAM-FALSE-LABEL: @main_0_0(
// CHECK-FUSE-MOVEPARAM-FALSE-LABEL: @main_0(
// CHECK-FUSE-MOVEPARAM-FALSE-NEXT: tensor.collapse_shape
// CHECK-FUSE-MOVEPARAM-FALSE-NEXT: call @main_0_0
// CHECK-FUSE-MOVEPARAM-FALSE-NEXT: return

// CHECK-FUSE-MOVEPARAM-TRUE-LABEL: @main_0_0(
// CHECK-FUSE-MOVEPARAM-TRUE: arith.constant
// CHECK-FUSE-MOVEPARAM-TRUE: return
// CHECK-FUSE-MOVEPARAM-TRUE-LABEL: @main_0(
// CHECK-FUSE-MOVEPARAM-TRUE: arith.constant
// CHECK-FUSE-MOVEPARAM-TRUE: return
module {
  func.func @main_0(%arg3: tensor<?x4096xf32>, %arg0: tensor<?x32x128xf32>, %arg1: tensor<4096x4096xf32>, %arg2: tensor<?x4096xf32>) -> tensor<?x4096xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x32x128xf32> into tensor<?x4096xf32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xf32>

    %dim_202 = tensor.dim %arg3, %c0 : tensor<?x4096xf32>
    %8 = tensor.empty(%dim_202) : tensor<?x4096xf32>

    %0 = tensor.empty(%dim) : tensor<?x4096xf32>
    %1 = linalg.matmul_transpose_b ins(%collapsed, %arg1 : tensor<?x4096xf32>, tensor<4096x4096xf32>) outs(%0 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %1 : tensor<?x4096xf32>, tensor<?x4096xf32>) outs(%8 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    return %2 : tensor<?x4096xf32>
  }
}

// -----

// CHECK-FUSE-MOVEPARAM-TRUE-LABEL: @main(
// CHECK-FUSE-MOVEPARAM-TRUE-NEXT: call
// CHECK-FUSE-MOVEPARAM-TRUE-NEXT: return
#map = affine_map<()[s0] -> (s0 * 1152)>
module {
  func.func @main(%arg0: tensor<?x1152xbf16>) -> (tensor<?x1152xf32>, tensor<?x1152xbf16>) attributes {OperatorType = "Broadcast", compute_capability = "", frontend_symbol = {input_0 = ["s631", "1152"], output_0 = ["s631", "1152"], output_1 = ["s631", "1152"]}, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, mindspore_kernel, process = "aicore"} {
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1152xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1152xbf16> into tensor<?xbf16>
    %0 = affine.apply #map()[%dim]
    %1 = tensor.empty(%0) : tensor<?xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<?xbf16>) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %cst : tensor<?xf32>, f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<?xf32>) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %cst_0 : tensor<?xf32>, f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%2, %5 : tensor<?xf32>, tensor<?xf32>) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    %7 = tensor.empty(%0) : tensor<?xbf16>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<?xf32>) outs(%7 : tensor<?xbf16>) -> tensor<?xbf16>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [%dim, 1152] : tensor<?xf32> into tensor<?x1152xf32>
    %expanded_1 = tensor.expand_shape %8 [[0, 1]] output_shape [%dim, 1152] : tensor<?xbf16> into tensor<?x1152xbf16>
    return %expanded, %expanded_1 : tensor<?x1152xf32>, tensor<?x1152xbf16>
  }
}
