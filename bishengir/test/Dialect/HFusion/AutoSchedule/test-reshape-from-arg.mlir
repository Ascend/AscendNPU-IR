// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=1" -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// CHECK-LABEL: @last_axis_pbr_arg_collapse_from_arg_0_tiling_func
// CHECK-DEBUG-DAG: %[[dim0:.*]] = affine.apply affine_map<() -> (4608)>()
// CHECK-DEBUG-DAG: %[[dim1:.*]] = affine.apply affine_map<() -> (192)>()
func.func @last_axis_pbr_arg_collapse_from_arg_0(%arg0: tensor<24x192x192xf32>) -> tensor<4608xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}{
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<24x192x192xf32> into tensor<4608x192xf32>
    %empty = tensor.empty() : tensor<4608xf32>
    %reduced = linalg.reduce ins(%collapsed : tensor<4608x192xf32>) outs(%empty : tensor<4608xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
    }
    return %reduced : tensor<4608xf32>
}

// -----

// CHECK-LABEL: @last_axis_pbr_arg_collapse_from_arg_1_tiling_func
// CHECK-DEBUG: %[[dim0:.*]] = affine.apply affine_map<() -> (16)>()
// CHECK-DEBUG: %[[dim1:.*]] = affine.apply affine_map<() -> (64)>()
func.func @last_axis_pbr_arg_collapse_from_arg_1(%arg0: tensor<16x8x4x2xf32>) -> tensor<16xf32> 
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}{
	%collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<16x8x4x2xf32> into tensor<16x64xf32>
	%empty = tensor.empty() : tensor<16xf32>
	%reduced = linalg.reduce ins(%collapsed : tensor<16x64xf32>) outs(%empty : tensor<16xf32>) dimensions = [1] 
		(%in: f32, %init: f32) {
		%8 = arith.addf %in, %init : f32
		linalg.yield %8 : f32
	}
	return %reduced : tensor<16xf32>
}

// -----

// CHECK-LABEL: @last_axis_pbr_arg_expand_from_arg_0_tiling_func
// CHECK-DEBUG: %[[dim0:.*]] = affine.apply affine_map<() -> (8)>()
// CHECK-DEBUG: %[[dim1:.*]] = affine.apply affine_map<() -> (8)>()
func.func @last_axis_pbr_arg_expand_from_arg_0(%arg0: tensor<64xf32>) -> tensor<8xf32> 
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}{
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [8, 8] : tensor<64xf32> into tensor<8x8xf32>
  %empty = tensor.empty() : tensor<8xf32>
	%reduced = linalg.reduce ins(%expanded : tensor<8x8xf32>) outs(%empty : tensor<8xf32>) dimensions = [1] 
		(%in: f32, %init: f32) {
		%8 = arith.addf %in, %init : f32
		linalg.yield %8 : f32
	}
	return %reduced : tensor<8xf32>
}

// -----

// CHECK-LABEL: @test_reshape_from_arg_dyn_dim_tiling_func
// CHECK-DEBUG: (%[[arg0:.*]]: tensor<?x256xf32> {{.*}}, %[[arg1:.*]]: tensor<256xf32> {{.*}}, %[[arg2:.*]]: tensor<?x256xf32> {{.*}})
// CHECK-DEBUG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DEBUG: %[[dim:.*]] = tensor.dim %[[arg2]], %[[c0]] : tensor<?x256xf32>
// CHECK-DEBUG: tensor.expand_shape %[[arg2]] {{\[}}[0, 1, 2], [3]] output_shape {{\[}}%[[dim]], 1, 1, 256]
func.func @test_reshape_from_arg_dyn_dim(%arg0: tensor<?x256xf32>, %arg1: tensor<256xf32>) -> tensor<?x256xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
  %0 = tensor.empty(%dim) : tensor<?x1x1x256xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<256xf32>) outs(%0 : tensor<?x1x1x256xf32>) dimensions = [0, 1, 2] 
  %collapsed = tensor.collapse_shape %broadcasted [[0, 1, 2], [3]] : tensor<?x1x1x256xf32> into tensor<?x256xf32>
  return %collapsed : tensor<?x256xf32>
}
