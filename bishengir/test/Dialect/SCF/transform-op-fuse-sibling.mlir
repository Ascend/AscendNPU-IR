// RUN: bishengir-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: fuse_sibling_when_target_users_not_dominate_source_properly
// CHECK: %[[A:.*]]: tensor<128xf32>, %[[B:.*]]: tensor<128xf32>
// CHECK: %[[for:.*]]:2 = scf.for
// CHECK: %[[expanded:.*]] = tensor.expand_shape %[[for]]#0 {{\[}}[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[expanded]] {{\[}}[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>
// CHECK-NOT: scf.for
// CHECK: %[[expanded_0:.*]] = tensor.expand_shape %[[for]]#1 {{\[}}[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
// CHECK: %[[collapsed_0:.*]] = tensor.collapse_shape %[[expanded_0]] {{\[}}[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>
// CHECK: return %[[collapsed]], %[[collapsed_0]]
func.func @fuse_sibling_when_target_users_not_dominate_source_properly(
    %A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>

  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  %expanded_0 = tensor.expand_shape %dup1 [[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
  %collapsed_0 = tensor.collapse_shape %expanded_0 [[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>

  return %collapsed, %collapsed_0 : tensor<16x8xf32>, tensor<16x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----
// CHECK-LABEL: fuse_sibling_when_source_before_target
// CHECK: %[[A:.*]]: tensor<128xf32>, %[[B:.*]]: tensor<128xf32>
// CHECK: %[[for:.*]]:2 = scf.for
// CHECK: %[[expanded:.*]] = tensor.expand_shape %[[for]]#1 {{\[}}[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[expanded]] {{\[}}[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>
// CHECK-NOT: scf.for
// CHECK: %[[expanded_0:.*]] = tensor.expand_shape %[[for]]#0 {{\[}}[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
// CHECK: %[[collapsed_1:.*]] = tensor.collapse_shape %[[expanded_0]] {{\[}}[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>
// CHECK: return %[[collapsed]], %[[collapsed_1]]
func.func @fuse_sibling_when_source_before_target(
    %A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>
  %c = tensor.empty(): tensor<128xf32>
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %c) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  %expanded_0 = tensor.expand_shape %dup1 [[0, 1, 2]] output_shape [16, 4, 2] : tensor<128xf32> into tensor<16x4x2xf32>
  %collapsed_0 = tensor.collapse_shape %expanded_0 [[0], [1, 2]] : tensor<16x4x2xf32> into tensor<16x8xf32>

  return %collapsed, %collapsed_0 : tensor<16x8xf32>, tensor<16x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----
// CHECK-LABEL: fuse_sibling_when_source_before_target_and_have_different_expanded_size
// CHECK: %[[for:.*]]:2 = scf.for
// CHECK: %[[expanded:.*]] = tensor.expand_shape %[[for]]#1 {{\[}}[0, 1], [2]] output_shape [1, 224, 3072] : tensor<224x3072xbf16> into tensor<1x224x3072xbf16>
// CHECK-NOT: scf.for
// CHECK: %[[expanded_4:.*]] = tensor.expand_shape %[[for]]#0 {{\[}}[0, 1, 2]] output_shape [1, 1, 3072] : tensor<3072xf32> into tensor<1x1x3072xf32>
// CHECK: return %[[expanded_4]], %[[expanded]]
#map = affine_map<(d0)[s0] -> (-d0 + 3072, s0)>
func.func @fuse_sibling_when_source_before_target_and_have_different_expanded_size(
      %arg0: tensor<1x224x3072xbf16>, %arg2: tensor<1x224x3072xbf16>, 
      %arg3: tensor<1x224x3072xbf16>, %arg4: tensor<1x1x3072xf32>,
      %arg5: tensor<1x224x3072xbf16>, %arg9: i64) -> (tensor<1x1x3072xf32>, tensor<1x224x3072xbf16>) {
    %c3072 = arith.constant 3072 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<224x3072xbf16>
    %1 = tensor.empty() : tensor<224x3072xf32>
    %2 = tensor.empty() : tensor<3072xf32>
    %3 = tensor.empty() : tensor<3072xbf16>
    %16 = arith.index_cast %arg9 : i64 to index
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %collapsed_1 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %6 = hfusion.load {__intermediate_producer__, __reduction0_fusible_producer__, __result1_fusible_producer__, __result2_fusible_producer__} 
                  ins(%collapsed_1 : tensor<224x3072xbf16>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %collapsed_2 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %7 = hfusion.load {__intermediate_producer__, __reduction0_fusible_producer__, __result1_fusible_producer__, __result2_fusible_producer__} 
                  ins(%collapsed_2 : tensor<224x3072xbf16>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %collapsed_3 = tensor.collapse_shape %arg4 [[0, 1, 2]] : tensor<1x1x3072xf32> into tensor<3072xf32>
    %collapsed_4 = tensor.collapse_shape %arg5 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %4 = hfusion.load {__intermediate_producer__, __reduction0_fusible_producer__} 
                  ins(%collapsed : tensor<224x3072xbf16>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %8 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, round_mode = #hfusion.round_mode<rint>}
                  ins(%4 : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %10 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, __result1_fusible_producer__, __result2_fusible_producer__, round_mode = #hfusion.round_mode<rint>} 
                  ins(%6 : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %11 = hfusion.cast {__intermediate_producer__, __reduction0_fusible_producer__, __result1_fusible_producer__, __result2_fusible_producer__, round_mode = #hfusion.round_mode<rint>} 
                  ins(%7 : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %12 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, __result1_fusible_producer__, __result2_fusible_producer__, fun = #linalg.binary_fn<add>} 
                  ins(%11, %10 : tensor<224x3072xf32>, tensor<224x3072xf32>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %13 = hfusion.cast {__intermediate_producer__, __result1_fusible_producer__, round_mode = #hfusion.round_mode<rint>}
                  ins(%12 : tensor<224x3072xf32>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %15 = linalg.elemwise_binary {__intermediate_producer__, __reduction0_fusible_producer__, fun = #linalg.binary_fn<mul>}
                  ins(%8, %12 : tensor<224x3072xf32>, tensor<224x3072xf32>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %17 = scf.for %arg11 = %c0 to %c3072 step %16 iter_args(%arg12 = %collapsed_4) -> (tensor<224x3072xbf16>) {
      %21 = affine.min #map(%arg11)[%16]
      %extracted_slice = tensor.extract_slice %13[0, %arg11] [224, %21] [1, 1] : tensor<224x3072xbf16> to tensor<224x?xbf16>
      %extracted_slice_8 = tensor.extract_slice %arg12[0, %arg11] [224, %21] [1, 1] : tensor<224x3072xbf16> to tensor<224x?xbf16>
      %22 = hfusion.store {__result1_fusible_producer__, hfusion.return_operand_num = 1 : i64} ins(%extracted_slice : tensor<224x?xbf16>) outs(%extracted_slice_8 : tensor<224x?xbf16>) -> tensor<224x?xbf16>
      %inserted_slice = tensor.insert_slice %22 into %arg12[0, %arg11] [224, %21] [1, 1] : tensor<224x?xbf16> into tensor<224x3072xbf16>
      scf.yield %inserted_slice : tensor<224x3072xbf16>
    } {__tiled_for___1}
    %expanded = tensor.expand_shape %17 [[0, 1], [2]] output_shape [1, 224, 3072] : tensor<224x3072xbf16> into tensor<1x224x3072xbf16>
    %reduced = linalg.reduce ins(%15 : tensor<224x3072xf32>) outs(%2 : tensor<3072xf32>) dimensions = [0]  {__intermediate_producer__, __reduction0__}
      (%in: f32, %init: f32) {
        %21 = arith.addf %in, %init : f32
        linalg.yield %21 : f32
    }
    %20 = scf.for %arg11 = %c0 to %c3072 step %16 iter_args(%arg12 = %collapsed_3) -> (tensor<3072xf32>) {
      %21 = affine.min #map(%arg11)[%16]
      %extracted_slice = tensor.extract_slice %reduced[%arg11] [%21] [1] : tensor<3072xf32> to tensor<?xf32>
      %extracted_slice_8 = tensor.extract_slice %arg12[%arg11] [%21] [1] : tensor<3072xf32> to tensor<?xf32>
      %22 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_8 : tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %22 into %arg12[%arg11] [%21] [1] : tensor<?xf32> into tensor<3072xf32>
      scf.yield %inserted_slice : tensor<3072xf32>
    } {__tiled_for__}
    %expanded_7 = tensor.expand_shape %20 [[0, 1, 2]] output_shape [1, 1, 3072] : tensor<3072xf32> into tensor<1x1x3072xf32>
    return %expanded_7, %expanded : tensor<1x1x3072xf32>, tensor<1x224x3072xbf16>
  }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----
// CHECK-LABEL: fuse_sibling_for_inner_sibling_loops(
// CHECK:     %1:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %0, %arg6 = %0) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:  %2:2 = scf.for %arg7 = %c0 to %c128 step %c64 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %extracted_slice_1 = tensor.extract_slice %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%extracted_slice_1 : tensor<1x64xf32>) attrs =  {__a__} {
// CHECK-NEXT:    ^bb0(%in: f32, %in_6: f32, %out: f32):
// CHECK-NEXT:      %5 = arith.mulf %in, %in_6 : f32
// CHECK-NEXT:      linalg.yield %5 : f32
// CHECK-NEXT:    } -> tensor<1x64xf32>
// CHECK-NEXT:    %inserted_slice = tensor.insert_slice %3 into %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
// CHECK-NEXT:    %extracted_slice_2 = tensor.extract_slice %arg2[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %extracted_slice_3 = tensor.extract_slice %arg3[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %extracted_slice_4 = tensor.extract_slice %arg9[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%extracted_slice_4 : tensor<1x64xf32>) attrs =  {__b__} {
// CHECK-NEXT:    ^bb0(%in: f32, %in_6: f32, %out: f32):
// CHECK-NEXT:      %5 = arith.mulf %in, %in_6 : f32
// CHECK-NEXT:      linalg.yield %5 : f32
// CHECK-NEXT:    } -> tensor<1x64xf32>
// CHECK-NEXT:    %inserted_slice_5 = tensor.insert_slice %4 into %arg9[%arg4, %arg7] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
// CHECK-NEXT:    scf.yield %inserted_slice, %inserted_slice_5 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.yield %2#0, %2#1 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:}

func.func @fuse_sibling_for_inner_sibling_loops(
    %arg0 : tensor<64x128xf32>, %arg1 : tensor<64x128xf32>, %arg2 : tensor<64x128xf32>, %arg3 : tensor<64x128xf32>) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) attrs = {__a__} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<64x128xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg3 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) attrs = {__b__} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<64x128xf32>
  return %1, %2 : tensor<64x128xf32>, tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:2 = transform.structured.tile_using_for %1 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %2 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %2 : !transform.any_op
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op

    %loops_2 = transform.loop.fuse_sibling %loops#0 into %loops_1#0 {fuse_inner_sibling_loops = true} : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
// CHECK-LABEL: fuse_sibling_for_inner_sibling_loops_with_different_step(
// CHECK:     %1:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %0, %arg6 = %0) -> (tensor<64x128x256xf32>, tensor<64x128x256xf32>) {
// CHECK-NEXT:  %2:2 = scf.for %arg7 = %c0 to %c128 step %c1 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<64x128x256xf32>, tensor<64x128x256xf32>) {
// CHECK-NEXT:    %3 = scf.for %arg10 = %c0 to %c256 step %c64 iter_args(%arg11 = %arg8) -> (tensor<64x128x256xf32>) {
// CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg7, %arg10] [1, 1, 64] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x64xf32>
// CHECK-NEXT:      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, %arg7, %arg10] [1, 1, 64] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x64xf32>
// CHECK-NEXT:      %extracted_slice_1 = tensor.extract_slice %arg11[%arg4, %arg7, %arg10] [1, 1, 64] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x64xf32>
// CHECK-NEXT:      %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x1x64xf32>, tensor<1x1x64xf32>) outs(%extracted_slice_1 : tensor<1x1x64xf32>) attrs =  {__a__} {
// CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:        %6 = arith.mulf %in, %in_2 : f32
// CHECK-NEXT:        linalg.yield %6 : f32
// CHECK-NEXT:      } -> tensor<1x1x64xf32>
// CHECK-NEXT:      %inserted_slice = tensor.insert_slice %5 into %arg11[%arg4, %arg7, %arg10] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<64x128x256xf32>
// CHECK-NEXT:      scf.yield %inserted_slice : tensor<64x128x256xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = scf.for %arg10 = %c0 to %c256 step %c128 iter_args(%arg11 = %arg9) -> (tensor<64x128x256xf32>) {
// CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg2[%arg4, %arg7, %arg10] [1, 1, 128] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x128xf32>
// CHECK-NEXT:      %extracted_slice_0 = tensor.extract_slice %arg3[%arg4, %arg7, %arg10] [1, 1, 128] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x128xf32>
// CHECK-NEXT:      %extracted_slice_1 = tensor.extract_slice %arg11[%arg4, %arg7, %arg10] [1, 1, 128] [1, 1, 1] : tensor<64x128x256xf32> to tensor<1x1x128xf32>
// CHECK-NEXT:      %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%extracted_slice_1 : tensor<1x1x128xf32>) attrs =  {__b__} {
// CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:        %6 = arith.mulf %in, %in_2 : f32
// CHECK-NEXT:        linalg.yield %6 : f32
// CHECK-NEXT:      } -> tensor<1x1x128xf32>
// CHECK-NEXT:      %inserted_slice = tensor.insert_slice %5 into %arg11[%arg4, %arg7, %arg10] [1, 1, 128] [1, 1, 1] : tensor<1x1x128xf32> into tensor<64x128x256xf32>
// CHECK-NEXT:      scf.yield %inserted_slice : tensor<64x128x256xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %3, %4 : tensor<64x128x256xf32>, tensor<64x128x256xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.yield %2#0, %2#1 : tensor<64x128x256xf32>, tensor<64x128x256xf32>
// CHECK-NEXT:}
func.func @fuse_sibling_for_inner_sibling_loops_with_different_step(
    %arg0 : tensor<64x128x256xf32>, %arg1 : tensor<64x128x256xf32>, %arg2 : tensor<64x128x256xf32>, %arg3 : tensor<64x128x256xf32>) -> (tensor<64x128x256xf32>, tensor<64x128x256xf32>) {
  %0 = tensor.empty() : tensor<64x128x256xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128x256xf32>, tensor<64x128x256xf32>) outs(%0 : tensor<64x128x256xf32>) attrs = {__a__} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<64x128x256xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2, %arg3 : tensor<64x128x256xf32>, tensor<64x128x256xf32>) outs(%0 : tensor<64x128x256xf32>) attrs = {__b__} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  } -> tensor<64x128x256xf32>
  return %1, %2 : tensor<64x128x256xf32>, tensor<64x128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:3 = transform.structured.tile_using_for %1 tile_sizes [1, 1, 128] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %2 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %2 : !transform.any_op
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op

    %loops_2 = transform.loop.fuse_sibling %loops#0 into %loops_1#0 {fuse_inner_sibling_loops = true} : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}