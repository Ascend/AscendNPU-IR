// RUN: bishengir-opt -transform-interpreter -verify-diagnostics -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_producer_has_other_users_dominated_by_containing_op(
// CHECK:      %2:2 = scf.for %arg4 = %c0 to %c256 step %c64 iter_args(%arg5 = %1, %arg6 = %arg1) -> (tensor<256xf32>, tensor<256xf32>) {
// CHECK-NEXT:   %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [64, 128] [1, 1] : tensor<256x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:   %extracted_slice_3 = tensor.extract_slice %arg6[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %c0_4 = arith.constant 0 : index
// CHECK-NEXT:   %c64_5 = arith.constant 64 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %4 = scf.for %arg7 = %c0_4 to %c64_5 step %c1 iter_args(%arg8 = %extracted_slice_3) -> (tensor<64xf32>) {
// CHECK-NEXT:     %c0_9 = arith.constant 0 : index
// CHECK-NEXT:     %c128 = arith.constant 128 : index
// CHECK-NEXT:     %c64_10 = arith.constant 64 : index
// CHECK-NEXT:     %6 = scf.for %arg9 = %c0_9 to %c128 step %c64_10 iter_args(%arg10 = %arg8) -> (tensor<64xf32>) {
// CHECK-NEXT:       %extracted_slice_11 = tensor.extract_slice %extracted_slice[%arg7, %arg9] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %arg10[%arg7] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK-NEXT:       %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_11 : tensor<1x64xf32>) outs(%extracted_slice_12 : tensor<1xf32>) attrs =  {__a__} {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %8 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %8 : f32
// CHECK-NEXT:       } -> tensor<1xf32>
// CHECK-NEXT:       %inserted_slice_13 = tensor.insert_slice %7 into %arg10[%arg7] [1] [1] : tensor<1xf32> into tensor<64xf32>
// CHECK-NEXT:       scf.yield %inserted_slice_13 : tensor<64xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %6 : tensor<64xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %extracted_slice_6 = tensor.extract_slice %arg2[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %extracted_slice_7 = tensor.extract_slice %arg5[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %5 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%4, %extracted_slice_6 : tensor<64xf32>, tensor<64xf32>) outs(%extracted_slice_7 : tensor<64xf32>) attrs =  {__b__} {
// CHECK-NEXT:   ^bb0(%in: f32, %in_9: f32, %out: f32):
// CHECK-NEXT:     %6 = arith.mulf %in, %in_9 : f32
// CHECK-NEXT:     linalg.yield %6 : f32
// CHECK-NEXT:   } -> tensor<64xf32>
// CHECK-NEXT:   %inserted_slice = tensor.insert_slice %5 into %arg5[%arg4] [64] [1] : tensor<64xf32> into tensor<256xf32>
// CHECK-NEXT:   %inserted_slice_8 = tensor.insert_slice %4 into %arg6[%arg4] [64] [1] : tensor<64xf32> into tensor<256xf32>
// CHECK-NEXT:   scf.yield %inserted_slice, %inserted_slice_8 : tensor<256xf32>, tensor<256xf32>
// CHECK-NEXT: }
module attributes {transform.with_named_sequence} {
  func.func @test_producer_has_other_users_dominated_by_containing_op(
      %arg0: tensor<256x128xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
        ins(%arg0 : tensor<256x128xf32>) outs(%arg1 : tensor<256xf32>) attrs =  {__a__} {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<256xf32>

    %1 = tensor.empty() : tensor<256xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
        ins(%0, %arg2 : tensor<256xf32>, tensor<256xf32>) outs(%1 : tensor<256xf32>) attrs =  {__b__} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.mulf %in, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<256xf32>

    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
        ins(%0, %arg3 : tensor<256xf32>, tensor<256xf32>) outs(%1 : tensor<256xf32>) attrs =  {__c__} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.mulf %in, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<256xf32>
    return %1, %2, %3 : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loop = transform.structured.tile_using_for %0 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1:2 = transform.structured.tile_using_for %fused_op tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loop_2 = transform.structured.tile_using_for %3 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_producer_has_other_users_and_containing_op_has_nested_for(
// CHECK:      %2:2 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %arg1, %arg4 = %0) -> (tensor<64xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:   %c0_3 = arith.constant 0 : index
// CHECK-NEXT:   %c128 = arith.constant 128 : index
// CHECK-NEXT:   %c64_4 = arith.constant 64 : index
// CHECK-NEXT:   %5:2 = scf.for %arg5 = %c0_3 to %c128 step %c64_4 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<64xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5] [1, 64] [1, 1] : tensor<64x128xf16> to tensor<1x64xf16>
// CHECK-NEXT:     %extracted_slice_5 = tensor.extract_slice %arg7[%arg2, %arg5] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:     %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x64xf16>) outs(%extracted_slice_5 : tensor<1x64xf32>) attrs =  {__a__} {
// CHECK-NEXT:     ^bb0(%in: f16, %out: f32):
// CHECK-NEXT:       %8 = arith.extf %in : f16 to f32
// CHECK-NEXT:       linalg.yield %8 : f32
// CHECK-NEXT:     } -> tensor<1x64xf32>
// CHECK-NEXT:     %extracted_slice_6 = tensor.extract_slice %arg6[%arg2] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK-NEXT:     %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<1x64xf32>) outs(%extracted_slice_6 : tensor<1xf32>) attrs =  {__b__} {
// CHECK-NEXT:     ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:       %8 = arith.mulf %in, %in : f32
// CHECK-NEXT:       %9 = arith.addf %8, %out : f32
// CHECK-NEXT:       linalg.yield %9 : f32
// CHECK-NEXT:     } -> tensor<1xf32>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %7 into %arg6[%arg2] [1] [1] : tensor<1xf32> into tensor<64xf32>
// CHECK-NEXT:     %inserted_slice_7 = tensor.insert_slice %6 into %arg7[%arg2, %arg5] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
// CHECK-NEXT:     scf.yield %inserted_slice, %inserted_slice_7 : tensor<64xf32>, tensor<64x128xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.yield %5#0, %5#1 : tensor<64xf32>, tensor<64x128xf32>
// CHECK-NEXT: }
module attributes {transform.with_named_sequence} {
  func.func @test_producer_has_other_users_and_containing_op_has_nested_for(
      %arg0: tensor<64x128xf16>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64x128xf16>) {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<64x128xf16>) outs(%0 : tensor<64x128xf32>) attrs =  {__a__} {
    ^bb0(%in: f16, %out: f32):
      %5 = arith.extf %in : f16 to f32
      linalg.yield %5 : f32
    } -> tensor<64x128xf32>

    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
        ins(%1 : tensor<64x128xf32>) outs(%arg1 : tensor<64xf32>) attrs =  {__b__} {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.mulf %in, %in : f32
      %7 = arith.addf %6, %out : f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>

    %3 = tensor.empty() : tensor<64x128xf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
        ins(%1 : tensor<64x128xf32>) outs(%3 : tensor<64x128xf16>) attrs =  {__c__} {
    ^bb0(%in: f32, %out: f16):
      %8 = arith.mulf %in, %in : f32
      %9 = arith.truncf %8 {round_mode = #hfusion.round_mode<rint>} : f32 to f16
      linalg.yield %9 : f16
    } -> tensor<64x128xf16>
    return %2, %4 : tensor<64xf32>, tensor<64x128xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %loops#0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:2 = transform.structured.tile_using_for %3 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_producer_has_multi_extract_uses_in_containing_op_1(
// CHECK:     %3:2 = scf.for %arg3 = %c0 to %c300 step %c1 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<300xf32>, tensor<300x150xf32>) {
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg4[%arg3] [1] [1] : tensor<300xf32> to tensor<1xf32>
// CHECK-NEXT:  %4 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK-NEXT:  %5:2 = scf.for %arg6 = %c0 to %c150 step %c64 iter_args(%arg7 = %4, %arg8 = %arg5) -> (tensor<1x64xf32>, tensor<300x150xf32>) {
// CHECK-NEXT:    %6 = affine.min #map(%arg6)
// CHECK-NEXT:    %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, %arg6] [1, %6] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, %arg6] [1, %6] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %extracted_slice_2 = tensor.extract_slice %0[%arg3, %arg6] [1, %6] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_0, %extracted_slice_1 : tensor<1x?xf32>, tensor<1x?xf32>) outs(%extracted_slice_2 : tensor<1x?xf32>) attrs =  {__a__} {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %10 = arith.addf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %10 : f32
// CHECK-NEXT:    } -> tensor<1x?xf32>
// CHECK-NEXT:    %extracted_slice_3 = tensor.extract_slice %arg7[0, 0] [1, %6] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<1x?xf32>) outs(%extracted_slice_3 : tensor<1x?xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %10 = arith.maxnumf %in, %out {reductionOp} : f32
// CHECK-NEXT:      linalg.yield %10 : f32
// CHECK-NEXT:    } -> tensor<1x?xf32>
// CHECK-NEXT:    %inserted_slice_4 = tensor.insert_slice %8 into %arg7[0, 0] [1, %6] [1, 1] : tensor<1x?xf32> into tensor<1x64xf32>
// CHECK-NEXT:    %extracted_slice_5 = tensor.extract_slice %arg2[%arg3, %arg6] [1, %6] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %extracted_slice_6 = tensor.extract_slice %arg8[%arg3, %arg6] [1, %6] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:    %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7, %extracted_slice_5 : tensor<1x?xf32>, tensor<1x?xf32>) outs(%extracted_slice_6 : tensor<1x?xf32>) attrs =  {__c__} {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %10 = arith.subf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %10 : f32
// CHECK-NEXT:    } -> tensor<1x?xf32>
// CHECK-NEXT:    %inserted_slice_7 = tensor.insert_slice %9 into %arg8[%arg3, %arg6] [1, %6] [1, 1] : tensor<1x?xf32> into tensor<300x150xf32>
// CHECK-NEXT:    scf.yield %inserted_slice_4, %inserted_slice_7 : tensor<1x64xf32>, tensor<300x150xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %reduced = linalg.reduce ins(%5#0 : tensor<1x64xf32>) outs(%extracted_slice : tensor<1xf32>) dimensions = [1]
// CHECK-NEXT:    (%in: f32, %init: f32) {
// CHECK-NEXT:      %6 = arith.maxnumf %in, %init {reductionOp, withoutInitMergeOp} : f32
// CHECK-NEXT:      linalg.yield %6 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  %inserted_slice = tensor.insert_slice %reduced into %arg4[%arg3] [1] [1] : tensor<1xf32> into tensor<300xf32>
// CHECK-NEXT:  scf.yield %inserted_slice, %5#1 : tensor<300xf32>, tensor<300x150xf32>
// CHECK-NEXT:}
func.func @test_producer_has_multi_extract_uses_in_containing_op_1(
    %arg0 : tensor<300x150xf32>, %arg1 : tensor<300x150xf32>, %arg2 : tensor<300x150xf32>) ->
    (tensor<300xf32>, tensor<300x150xf32>) {
    %1 = tensor.empty() : tensor<300x150xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
                        ins(%arg0, %arg1: tensor<300x150xf32>, tensor<300x150xf32>)
                        outs(%1 : tensor<300x150xf32>) attrs =  {"__a__"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %100 = arith.addf %in, %in_1 : f32
      linalg.yield %100 : f32
    } -> tensor<300x150xf32>
    %3 = tensor.empty() : tensor<300xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
                        ins(%2 : tensor<300x150xf32>) outs(%3 : tensor<300xf32>) attrs =  {"__b__"} {
    ^bb0(%in: f32, %out: f32):
      %100 = arith.maxnumf %in, %out : f32
      linalg.yield %100 : f32
    } -> tensor<300xf32>
    %5 = tensor.empty() : tensor<300x150xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
                        ins(%2, %arg2: tensor<300x150xf32>, tensor<300x150xf32>)
                        outs(%5 : tensor<300x150xf32>) attrs =  {"__c__"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %100 = arith.subf %in, %in_1 : f32
      linalg.yield %100 : f32
    } -> tensor<300x150xf32>
    return %4, %6 : tensor<300xf32>, tensor<300x150xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {"__b__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %tiled_linalg_op by tile_sizes = [0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {"__c__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:2 = transform.structured.tile_using_for %1 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %2 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %2 : !transform.any_op
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op

    %4 = transform.loop.fuse_sibling %loops into %loops_1#0 {fuse_inner_sibling_loops = true} : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %5 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %5 : !transform.any_op
    %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %6 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op

    %7 = transform.structured.match attributes {"__a__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %7 into %4 {merge_multiple_extract_uses = true} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %8 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %8 : !transform.any_op
    %9 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %9 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op
    transform.yield
  }
}

// -----
// CHECK-LABEL: func.func @test_producer_has_multi_extract_uses_in_containing_op_2(
// CHECK:     %1:2 = scf.for %arg4 = %c0 to %c300 step %c7 iter_args(%arg5 = %0, %arg6 = %0) -> (tensor<300x150xf32>, tensor<300x150xf32>) {
// CHECK-NEXT:  %2 = affine.min #map(%arg4)
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [%2, 150] [1, 1] : tensor<300x150xf32> to tensor<?x150xf32>
// CHECK-NEXT:  %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0] [%2, 150] [1, 1] : tensor<300x150xf32> to tensor<?x150xf32>
// CHECK-NEXT:  %extracted_slice_1 = tensor.extract_slice %0[%arg4, 0] [%2, 150] [1, 1] : tensor<300x150xf32> to tensor<?x150xf32>
// CHECK-NEXT:  %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<?x150xf32>, tensor<?x150xf32>) outs(%extracted_slice_1 : tensor<?x150xf32>) attrs =  {__a__} {
// CHECK-NEXT:  ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:    %6 = arith.addf %in, %in_2 : f32
// CHECK-NEXT:          linalg.yield %6 : f32
// CHECK-NEXT:  } -> tensor<?x150xf32>
// CHECK-NEXT:  %4 = scf.for %arg7 = %c0 to %c150 step %c8 iter_args(%arg8 = %arg5) -> (tensor<300x150xf32>) {
// CHECK-NEXT:    %6 = affine.min #map2(%arg7)
// CHECK-NEXT:    %extracted_slice_2 = tensor.extract_slice %arg8[%arg4, %arg7] [%2, %6] [1, 1] : tensor<300x150xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %7 = tensor.empty(%2, %6) : tensor<?x?x64xf32>
// CHECK-NEXT:    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?x64xf32>) -> tensor<?x?x64xf32>
// CHECK-NEXT:    %extracted_slice_3 = tensor.extract_slice %3[0, %arg7] [%2, %6] [1, 1] : tensor<?x150xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %9 = scf.for %arg9 = %c0 to %c100 step %c64 iter_args(%arg10 = %8) -> (tensor<?x?x64xf32>) {
// CHECK-NEXT:      %10 = affine.min #map3(%arg9)
// CHECK-NEXT:      %extracted_slice_4 = tensor.extract_slice %arg2[%arg4, %arg7, %arg9] [%2, %6, %10] [1, 1, 1] : tensor<300x150x100xf32> to tensor<?x?x?xf32>
// CHECK-NEXT:      %extracted_slice_5 = tensor.extract_slice %arg10[0, 0, 0] [%2, %6, %10] [1, 1, 1] : tensor<?x?x64xf32> to tensor<?x?x?xf32>
// CHECK-NEXT:      %11 = linalg.generic {indexing_maps = [#map4, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?x?xf32>, tensor<?x?x?xf32>) outs(%extracted_slice_5 : tensor<?x?x?xf32>) {
// CHECK-NEXT:      ^bb0(%in: f32, %in_7: f32, %out: f32):
// CHECK-NEXT:        %12 = arith.mulf %in, %in_7 : f32
// CHECK-NEXT:        %13 = arith.maxnumf %12, %out {reductionOp} : f32
// CHECK-NEXT:        linalg.yield %13 : f32
// CHECK-NEXT:      } -> tensor<?x?x?xf32>
// CHECK-NEXT:      %inserted_slice_6 = tensor.insert_slice %11 into %arg10[0, 0, 0] [%2, %6, %10] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x64xf32>
// CHECK-NEXT:      scf.yield %inserted_slice_6 : tensor<?x?x64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %reduced = linalg.reduce ins(%9 : tensor<?x?x64xf32>) outs(%extracted_slice_2 : tensor<?x?xf32>) dimensions = [2]
// CHECK-NEXT:      (%in: f32, %init: f32) {
// CHECK-NEXT:        %10 = arith.maxnumf %in, %init {reductionOp, withoutInitMergeOp} : f32
// CHECK-NEXT:          linalg.yield %10 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %inserted_slice = tensor.insert_slice %reduced into %arg8[%arg4, %arg7] [%2, %6] [1, 1] : tensor<?x?xf32> into tensor<300x150xf32>
// CHECK-NEXT:    scf.yield %inserted_slice : tensor<300x150xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %5 = scf.for %arg7 = %c0 to %c150 step %c64 iter_args(%arg8 = %arg6) -> (tensor<300x150xf32>) {
// CHECK-NEXT:    %6 = affine.min #map6(%arg7)
// CHECK-NEXT:    %extracted_slice_2 = tensor.extract_slice %3[0, %arg7] [%2, %6] [1, 1] : tensor<?x150xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %extracted_slice_3 = tensor.extract_slice %arg3[%arg4, %arg7] [%2, %6] [1, 1] : tensor<300x150xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %extracted_slice_4 = tensor.extract_slice %arg8[%arg4, %arg7] [%2, %6] [1, 1] : tensor<300x150xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_4 : tensor<?x?xf32>) attrs =  {__c__} {
// CHECK-NEXT:    ^bb0(%in: f32, %in_5: f32, %out: f32):
// CHECK-NEXT:      %8 = arith.subf %in, %in_5 : f32
// CHECK-NEXT:      linalg.yield %8 : f32
// CHECK-NEXT:    } -> tensor<?x?xf32>
// CHECK-NEXT:    %inserted_slice = tensor.insert_slice %7 into %arg8[%arg4, %arg7] [%2, %6] [1, 1] : tensor<?x?xf32> into tensor<300x150xf32>
// CHECK-NEXT:    scf.yield %inserted_slice : tensor<300x150xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.yield %4, %5 : tensor<300x150xf32>, tensor<300x150xf32>
// CHECK-NEXT:}
func.func @test_producer_has_multi_extract_uses_in_containing_op_2(
    %arg0 : tensor<300x150xf32>, %arg1 : tensor<300x150xf32>, %arg2 : tensor<300x150x100xf32>, %arg3 : tensor<300x150xf32>) ->
    (tensor<300x150xf32>, tensor<300x150xf32>) {
  %1 = tensor.empty() : tensor<300x150xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"]}
                      ins(%arg0, %arg1: tensor<300x150xf32>, tensor<300x150xf32>)
                      outs(%1 : tensor<300x150xf32>) attrs =  {"__a__"} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %100 = arith.addf %in, %in_1 : f32
    linalg.yield %100 : f32
  } -> tensor<300x150xf32>
  %3 = tensor.empty() : tensor<300x150xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel", "reduction"]}
                      ins(%2, %arg2 : tensor<300x150xf32>, tensor<300x150x100xf32>)
                      outs(%3 : tensor<300x150xf32>) attrs = {"__b__"} {
  ^bb0(%in_1: f32, %in: f32, %out: f32):
    %99 = arith.mulf %in_1, %in : f32
    %100 = arith.maxnumf %99, %out : f32
    linalg.yield %100 : f32
  } -> tensor<300x150xf32>
  %5 = tensor.empty() : tensor<300x150xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"]}
                      ins(%2, %arg3: tensor<300x150xf32>, tensor<300x150xf32>)
                      outs(%5 : tensor<300x150xf32>) attrs =  {"__c__"} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %100 = arith.subf %in, %in_1 : f32
    linalg.yield %100 : f32
  } -> tensor<300x150xf32>
  return %4, %6 : tensor<300x150xf32>, tensor<300x150xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {"__b__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [7, 8, 0] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %tiled_linalg_op by tile_sizes = [0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %1 = transform.structured.match attributes {"__c__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:2 = transform.structured.tile_using_for %1 tile_sizes [7, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %2 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %2 : !transform.any_op
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op
    %4 = transform.loop.fuse_sibling %loops#0 into %loops_1#0 {fuse_inner_sibling_loops = true} : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %5 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %5 : !transform.any_op
    %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %6 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op

    %7 = transform.structured.match attributes {"__a__"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %7 into %4 {merge_multiple_extract_uses = true} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %8 = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %8 : !transform.any_op
    %9 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %9 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op
    transform.yield
  }
}