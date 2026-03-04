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