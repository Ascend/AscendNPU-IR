// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// CHECK-DEBUG: @model_2
module {
  func.func @model_2(%arg0: tensor<391x1xf16>, %arg1: tensor<1x288xf16>, %arg2: tensor<391x288xf16>) -> tensor<391x288xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<391x288xf16>
    %1 = tensor.empty() : tensor<391x1xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<391x1xf16> into tensor<391xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<391xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [1]
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x288xf16> into tensor<288xf16>
    %broadcasted_1 = linalg.broadcast ins(%collapsed_0 : tensor<288xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_1 : tensor<391x288xf16>, tensor<391x288xf16>) outs(%arg2 : tensor<391x288xf16>) -> tensor<391x288xf16>
    return %2 : tensor<391x288xf16>
  }
}

// -----

// CHECK-DEBUG-LABEL: @any_pb_arg_collapse_from_arg_0_tiling_func
// CHECK-DEBUG-DAG: %[[dim1:.*]] = affine.apply affine_map<() -> (147456)>()
// CHECK-DEBUG-DAG: %[[dim0:.*]] = affine.apply affine_map<() -> (24)>()
func.func @any_pb_arg_collapse_from_arg_0(%arg0: tensor<24xi1>, %arg1: tensor<24x3x256x192xf32>, %arg2: tensor<24x3x256x192xf32>) -> tensor<24x3x256x192xf32> 
	attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
	%collapsed = tensor.collapse_shape %arg2 [[0], [1, 2, 3]] : tensor<24x3x256x192xf32> into tensor<24x147456xf32>
	%collapsed_0 = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<24x3x256x192xf32> into tensor<24x147456xf32>
	%0 = tensor.empty() : tensor<24xf32>
	%1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<24xi1>) outs(%0 : tensor<24xf32>) -> tensor<24xf32>
	%2 = tensor.empty() : tensor<24x147456xf32>
	%broadcasted = linalg.broadcast ins(%1 : tensor<24xf32>) outs(%2 : tensor<24x147456xf32>) dimensions = [1] 
	%3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %collapsed_0 : tensor<24x147456xf32>, tensor<24x147456xf32>) outs(%collapsed : tensor<24x147456xf32>) -> tensor<24x147456xf32>
	%expanded = tensor.expand_shape %3 [[0], [1, 2, 3]] output_shape [24, 3, 256, 192] : tensor<24x147456xf32> into tensor<24x3x256x192xf32>
	return %expanded : tensor<24x3x256x192xf32>
}

// -----

// CHECK-LABEL: @test_multiple_result
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.block
module {
  func.func @test_multiple_result(%arg0: tensor<391x1xf16>, %arg1: tensor<1x288xf16>, %arg2: tensor<391x288xf16>, %arg3: tensor<391x288xf16>) -> (tensor<391x288xf16>, tensor<391x288xf16>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<391x288xf16>
    %1 = tensor.empty() : tensor<391x1xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<391x1xf16> into tensor<391xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<391xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [1]
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x288xf16> into tensor<288xf16>
    %broadcasted_1 = linalg.broadcast ins(%collapsed_0 : tensor<288xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_1 : tensor<391x288xf16>, tensor<391x288xf16>) outs(%arg2 : tensor<391x288xf16>) -> tensor<391x288xf16>
    %3 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<add>} ins(%broadcasted, %broadcasted_1 : tensor<391x288xf16>, tensor<391x288xf16>) outs(%arg3 : tensor<391x288xf16>) -> tensor<391x288xf16>
    return %2, %3 : tensor<391x288xf16>, tensor<391x288xf16>
  }
}

// -----

// CHECK-LABEL: @test_no_computation(
// CHECK: %[[ARG0:.*]]: tensor<32xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}
// CHECK: %[[ARG1:.*]]: tensor<32xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
// CHECK: hfusion.load
// CHECK: hfusion.store
func.func @test_no_computation(%arg0: tensor<32xf32>) -> tensor<32xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  return %arg0 : tensor<32xf32>
}

// -----

// CHECK-DAG: test_dynamic_shape_single_output_0
// CHECK-DAG: test_dynamic_shape_single_output_1
module {
  func.func @test_dynamic_shape_single_output(%arg0: tensor<?x1xf16>,
                                              %arg1: tensor<1x?xf16>,
                                              %arg2: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf16> into tensor<?xf16>
    %broadcasted_2 = linalg.broadcast ins(%collapsed_1 : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}

// -----

// CHECK-DAG: test_dynamic_shape_multiple_output_0
// CHECK-DAG: test_dynamic_shape_multiple_output_1
module {
  func.func @test_dynamic_shape_multiple_output(%arg0: tensor<?x1xf16>,
                                                %arg1: tensor<1x?xf16>,
                                                %arg2: tensor<?x?xf16>,
                                                %arg3: tensor<?x?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf16> into tensor<?xf16>
    %broadcasted_2 = linalg.broadcast ins(%collapsed_1 : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg3 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1, %2 : tensor<?x?xf16>, tensor<?x?xf16>
  }
}

// -----

// CHECK: test_fill
module {
  func.func @test_fill(%arg0: tensor<?x256xf32>) -> (tensor<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
    %2 = tensor.empty(%dim) : tensor<?xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    return %3 : tensor<?xf32>
  }
}

// -----

// CHECK: test_tensor_extract
module {
  func.func @test_tensor_extract(%arg0: tensor<1024xf32>, %arg1: tensor<f32>) -> tensor<1024xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %extracted = tensor.extract %arg1[] : tensor<f32>
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %extracted : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL: test_avoid_mark_index_cast_from_arg_as_producer
// CHECK: scf.for
// CHECK: arith.index_cast
#map = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @test_avoid_mark_index_cast_from_arg_as_producer(%arg0: tensor<64x?x?x?xi8>, %arg1: tensor<64x?x?x?xf32>, %arg2: i64, %arg3: i64, %arg4: i64) -> tensor<64x?x?x?xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<64x?x?x?xi8>
    %dim_0 = tensor.dim %arg0, %c2 : tensor<64x?x?x?xi8>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<64x?x?x?xi8>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<64x?x?x?xi8> into tensor<?x?x?xi8>
    %dim_2 = tensor.dim %arg0, %c1 : tensor<64x?x?x?xi8>
    %0 = affine.apply #map()[%dim_2]
    %dim_3 = tensor.dim %arg0, %c2 : tensor<64x?x?x?xi8>
    %dim_4 = tensor.dim %arg0, %c3 : tensor<64x?x?x?xi8>
    %1 = tensor.empty(%0, %dim_3, %dim_4) : tensor<?x?x?xi1>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<?x?x?xi8>) outs(%1 : tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
    %collapsed_5 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<64x?x?x?xf32> into tensor<?x?x?xf32>
    %3 = arith.index_cast %arg2 : i64 to index
    %4 = affine.apply #map()[%3]
    %5 = arith.index_cast %arg3 : i64 to index
    %6 = arith.index_cast %arg4 : i64 to index
    %7 = tensor.empty(%4, %5, %6) : tensor<?x?x?xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %9 = hfusion.select ins(%2, %8, %collapsed_5 : tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%7 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %expanded = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [64, %dim_1, %dim_0, %dim] : tensor<?x?x?xf32> into tensor<64x?x?x?xf32>
    return %expanded : tensor<64x?x?x?xf32>
  }
}