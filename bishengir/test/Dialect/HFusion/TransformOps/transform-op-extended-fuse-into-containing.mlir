// RUN: bishengir-opt -transform-interpreter -canonicalize --split-input-file %s | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module attributes { transform.with_named_sequence } {
  // CHECK: func.func @fuse_tileable_op_with_multi_consumers(%[[ARG0:.+]]: index, %[[ARG1:.+]]: tensor<?xf32>,
  // CHECK: %[[FOR:.+]] = scf.for
  // CHECK:   %[[EXT:.+]] = tensor.extract_slice %[[ARG1]][%[[OFFSET:.+]]] [%[[SIZE:.+]]] [1]
  // CHECK:   linalg.fill ins(%[[VAL:.+]] : f32) outs(%[[EXT]] : tensor<?xf32>)
  // CHECK-NOT:   linalg.fill ins
  func.func @fuse_tileable_op_with_multi_consumers(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]

    %2 = scf.for %arg3 = %c0 to  %1 step %c1 iter_args(%o = %arg2)-> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      %66 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %77 = linalg.elemwise_unary ins(%7 : tensor<?xf32>) outs(%66 : tensor<?xf32>) -> tensor<?xf32>
      scf.yield %77 : tensor<?xf32>
    }
    func.return %2 : tensor<?xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // linalg.fill is tileable. The op is tiled and fused.
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module attributes { transform.with_named_sequence } {
  // CHECK: func.func @test_define_op_outside(%[[ARG0:.+]]: index, %[[ARG1:.+]]: tensor<?xf32>, %[[ARG2:.+]]: tensor<?xf32>,
  // CHECK: %[[FOR:.+]] = scf.for
  // CHECK:   %[[EXT:.+]] = tensor.extract_slice %[[ARG1]][%[[OFFSET:.+]]] [%[[SIZE:.+]]] [1]
  // CHECK:   linalg.fill ins(%[[VAL:.+]] : f32) outs(%[[EXT]] : tensor<?xf32>)
  // CHECK-NOT:   linalg.fill ins
  func.func @test_define_op_outside(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]

    %2 = scf.for %arg4 = %c0 to  %1 step %c1 iter_args(%o = %arg2)-> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg4)[%arg0]
      %4 = affine.min #map2(%arg4)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      %66 = tensor.extract_slice %0[%3] [%d0] [1] : tensor<?xf32> to tensor<?xf32>
      %77 = linalg.elemwise_unary ins(%arg3 : tensor<?xf32>) outs(%66 : tensor<?xf32>) -> tensor<?xf32>
      scf.yield %77 : tensor<?xf32>
    }
    func.return %2 : tensor<?xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // linalg.fill is tileable. The op is tiled and fused.
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: %[[FILL:.*]] = linalg.fill ins
  // CHECK: scf.for
  // CHECK: linalg.fill ins
  // CHECK: linalg.elemwise_unary ins(%[[FILL]]
  func.func @test_duplicate_producer(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%0 : f32) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.elemwise_unary {__a__} ins(%1 : tensor<?xf32>) outs(%arg0 : tensor<?xf32>) -> tensor<?xf32>
    %3 = linalg.elemwise_unary ins(%1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    return %2, %3 : tensor<?xf32>, tensor<?xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %for_op = transform.structured.tile_using_for %1 tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // duplicate producer value
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %for_op {duplicate_producer = true} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK-NOT: linalg.fill ins
  // CHECK: scf.for
  // CHECK: linalg.fill ins
  // CHECK: scf.for
  // CHECK: linalg.fill ins
  func.func @test_fuse_into_multiple(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%0 : f32) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.elemwise_unary ins(%1 : tensor<?xf32>) outs(%arg0 : tensor<?xf32>) -> tensor<?xf32>
    %3 = linalg.elemwise_unary ins(%1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    return %2, %3 : tensor<?xf32>, tensor<?xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %for_op = transform.structured.tile_using_for %1 tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2, %3 = transform.split_handle %for_op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // duplicate producer value
    %fused_op:2, %new_containing_op:2 =
      transform.structured.extended_fuse_into_containing_op %0 into %2, %3 {duplicate_producer = true}
        : (!transform.any_op, !transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %4 {
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @test_extract_producer_with_multiple_users(
// CHECK: scf.for
// CHECK: %[[fused_mul_0:.*]] = linalg.elemwise_binary {__b__, 
// CHECK: %[[fused_extract_0:.*]] = tensor.extract %[[fused_mul_0]]{{\[}}] {__a__} 
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[fused_extract_0]] : tensor<16xf32>, f32)
// CHECK: %[[fused_mul_1:.*]] = linalg.elemwise_binary {__b__, 
// CHECK: %[[fused_extract_1:.*]] = tensor.extract %[[fused_mul_1]]{{\[}}] {__a__} 
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[fused_extract_1]] : tensor<16xf32>, f32)
module attributes {transform.with_named_sequence} {
  func.func @test_extract_producer_with_multiple_users(%arg0: tensor<f32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>) -> tensor<32xf32> {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<32xf32>
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.elemwise_binary {__b__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<f32>, tensor<f32>) outs(%1 : tensor<f32>) -> tensor<f32>
    %extracted = tensor.extract %2[] {__a__} : tensor<f32>
    %3:2 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %0, %arg5 = %0) -> (tensor<32xf32>, tensor<32xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[%arg3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted : tensor<16xf32>, f32) outs(%extracted_slice_0 : tensor<16xf32>) -> tensor<16xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3] [16] [1] : tensor<16xf32> into tensor<32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %extracted_slice_2 = tensor.extract_slice %arg5[%arg3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice_1, %extracted : tensor<16xf32>, f32) outs(%extracted_slice_2 : tensor<16xf32>) -> tensor<16xf32>
      %inserted_slice_3 = tensor.insert_slice %5 into %arg5[%arg3] [16] [1] : tensor<16xf32> into tensor<32xf32>
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<32xf32>, tensor<32xf32>
    } {__c__}
    return %3#1 : tensor<32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op_0, %new_containing_op_1 = transform.structured.extended_fuse_into_containing_op %2 into %3 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_rank0_producer_with_multiple_users(
// CHECK: scf.for
// CHECK: %[[fused_mul_0:.*]] = linalg.elemwise_binary {__b__, 
// CHECK: %[[fused_mul_1:.*]] = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%[[fused_mul_0]]
// CHECK: linalg.elemwise_binary {{.*}} ins({{.*}}, %[[fused_mul_1]] : tensor<f32>, tensor<f32>)
// CHECK: %[[fused_mul_2:.*]] = linalg.elemwise_binary {__b__, 
// CHECK: %[[fused_mul_3:.*]] = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%[[fused_mul_2]]
// CHECK: linalg.elemwise_binary {{.*}} ins({{.*}}, %[[fused_mul_3]] : tensor<f32>, tensor<f32>)
module attributes {transform.with_named_sequence} {
  func.func @test_rank0_producer_with_multiple_users(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.elemwise_binary {__b__, fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<f32>, f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %3 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%2, %cst : tensor<f32>, f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %4:2 = scf.for %arg4 = %c0 to %c32 step %c16 iter_args(%arg5 = %0, %arg6 = %0) -> (tensor<f32>, tensor<f32>) {
      %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<f32>, tensor<f32>) outs(%1 : tensor<f32>) -> tensor<f32>
      %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg3, %3 : tensor<f32>, tensor<f32>) outs(%1 : tensor<f32>) -> tensor<f32>
      scf.yield %5, %6 : tensor<f32>, tensor<f32>
    } {__c__}
    return %4#0, %4#1 : tensor<f32>, tensor<f32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op_0, %new_containing_op_1 = transform.structured.extended_fuse_into_containing_op %2 into %3 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_union_producer_users_with_zero_size_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<16x32xf32>, %[[arg1:.*]]: tensor<16x32xf32>
// CHECK-NOT: tensor<10x32xf32>
// CHECK: %[[slice0:.*]] = tensor.extract_slice %[[arg0]]{{\[}}0, 0] {{\[}}2, 32] {{\[}}1, 1] : tensor<16x32xf32> to tensor<2x32xf32>
// CHECK: %[[slice1:.*]] = tensor.extract_slice %[[arg1]]{{\[}}0, 0] {{\[}}2, 32] {{\[}}1, 1] : tensor<16x32xf32> to tensor<2x32xf32>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {{.*}} ins(%[[slice0]], %[[slice1]] : tensor<2x32xf32>, tensor<2x32xf32>)
// CHECK: %[[slice2:.*]] = tensor.extract_slice %[[mul]]{{\[}}0, 0] {{\[}}0, 32] {{\[}}1, 1] : tensor<2x32xf32> to tensor<0x32xf32>
// CHECK: scf.yield %[[mul]], %[[slice2]] : tensor<2x32xf32>, tensor<0x32xf32>
module attributes {transform.with_named_sequence} {
  func.func @test_union_producer_users_with_zero_size_0(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> (tensor<2x32xf32>, tensor<0x32xf32>) {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<2x32xf32>
    %1 = tensor.empty() : tensor<0x32xf32>
    %2 = tensor.empty() : tensor<16x32xf32>
    %3 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %4:2 = scf.for %arg2 = %c0 to %c32 step %c16 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<2x32xf32>, tensor<0x32xf32>) {
      %extracted_slice = tensor.extract_slice %3[0, 0] [2, 32] [1, 1] : tensor<16x32xf32> to tensor<2x32xf32>
      %extracted_slice_0 = tensor.extract_slice %3[10, 0] [0, 32] [1, 1] : tensor<16x32xf32> to tensor<0x32xf32>
      scf.yield %extracted_slice, %extracted_slice_0 : tensor<2x32xf32>, tensor<0x32xf32>
    } {__b__}
    return %4#0, %4#1 : tensor<2x32xf32>, tensor<0x32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_union_producer_users_with_zero_size_1(
// CHECK-NOT: tensor<10x32xf32>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {{.*}} ins({{.*}} : tensor<2x32xf32>, tensor<2x32xf32>)
// CHECK: tensor.extract_slice %[[mul]]{{\[}}0, 0] {{\[}}0, 32] {{\[}}1, 1] : tensor<2x32xf32> to tensor<0x32xf32>
module attributes {transform.with_named_sequence} {
  func.func @test_union_producer_users_with_zero_size_1(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> (tensor<0x32xf32>, tensor<2x32xf32>) {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<0x32xf32>
    %1 = tensor.empty() : tensor<2x32xf32>
    %2 = tensor.empty() : tensor<16x32xf32>
    %3 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %4:2 = scf.for %arg2 = %c0 to %c32 step %c16 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<0x32xf32>, tensor<2x32xf32>) {
      %extracted_slice = tensor.extract_slice %3[10, 0] [0, 32] [1, 1] : tensor<16x32xf32> to tensor<0x32xf32>
      %extracted_slice_0 = tensor.extract_slice %3[0, 0] [2, 32] [1, 1] : tensor<16x32xf32> to tensor<2x32xf32>
      scf.yield %extracted_slice, %extracted_slice_0 : tensor<0x32xf32>, tensor<2x32xf32>
    } {__b__}
    return %4#0, %4#1 : tensor<0x32xf32>, tensor<2x32xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %0 into %1 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_producer_has_other_users_dominated_by_containing_op(
// CHECK:      %1:2 = scf.for %arg4 = %c0 to %c256 step %c64 iter_args(%arg5 = %0, %arg6 = %arg1) -> (tensor<256xf32>, tensor<256xf32>) {
// CHECK-NEXT:   %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [64, 128] [1, 1] : tensor<256x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:   %extracted_slice_0 = tensor.extract_slice %arg6[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %3 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %extracted_slice_0) -> (tensor<64xf32>) {
// CHECK-NEXT:     %5 = scf.for %arg9 = %c0 to %c128 step %c64 iter_args(%arg10 = %arg8) -> (tensor<64xf32>) {
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg7, %arg9] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:  %extracted_slice_5 = tensor.extract_slice %arg10[%arg7] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK-NEXT:       %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_4 : tensor<1x64xf32>) outs(%extracted_slice_5 : tensor<1xf32>) attrs =  {__a__} {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %7 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %7 : f32
// CHECK-NEXT:       } -> tensor<1xf32>
// CHECK-NEXT:       %inserted_slice_6 = tensor.insert_slice %6 into %arg10[%arg7] [1] [1] : tensor<1xf32> into tensor<64xf32>
// CHECK-NEXT:       scf.yield %inserted_slice_6 : tensor<64xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %5 : tensor<64xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %extracted_slice_1 = tensor.extract_slice %arg2[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %extracted_slice_2 = tensor.extract_slice %arg5[%arg4] [64] [1] : tensor<256xf32> to tensor<64xf32>
// CHECK-NEXT:   %4 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%3, %extracted_slice_1 : tensor<64xf32>, tensor<64xf32>) outs(%extracted_slice_2 : tensor<64xf32>) attrs =  {__b__} {
// CHECK-NEXT:   ^bb0(%in: f32, %in_4: f32, %out: f32):
// CHECK-NEXT:     %5 = arith.mulf %in, %in_4 : f32
// CHECK-NEXT:     linalg.yield %5 : f32
// CHECK-NEXT:   } -> tensor<64xf32>
// CHECK-NEXT:   %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4] [64] [1] : tensor<64xf32> into tensor<256xf32>
// CHECK-NEXT:   %inserted_slice_3 = tensor.insert_slice %3 into %arg6[%arg4] [64] [1] : tensor<64xf32> into tensor<256xf32>
// CHECK-NEXT:   scf.yield %inserted_slice, %inserted_slice_3 : tensor<256xf32>, tensor<256xf32>
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
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %1 into %loop {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1:2 = transform.structured.tile_using_for %fused_op tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loop_2 = transform.structured.tile_using_for %3 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield 
  }
}

// -----

// CHECK-LABEL: func.func @test_producer_has_other_users_and_containing_op_has_nested_for(
// CHECK:    %1:2 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %arg1, %arg4 = %0) -> (tensor<64xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:      %4:2 = scf.for %arg5 = %c0 to %c128 step %c64 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<64xf32>, tensor<64x128xf32>) {
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg5] [1, 64] [1, 1] : tensor<64x128xf16> to tensor<1x64xf16>
// CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg7[%arg2, %arg5] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK-NEXT:        %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x64xf16>) outs(%extracted_slice_0 : tensor<1x64xf32>) attrs =  {__a__} {
// CHECK-NEXT:        ^bb0(%in: f16, %out: f32):
// CHECK-NEXT:          %7 = arith.extf %in : f16 to f32
// CHECK-NEXT:          linalg.yield %7 : f32
// CHECK-NEXT:        } -> tensor<1x64xf32>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg6[%arg2] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK-NEXT:        %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<1x64xf32>) outs(%extracted_slice_1 : tensor<1xf32>) attrs =  {__b__} {
// CHECK-NEXT:        ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:          %7 = arith.mulf %in, %in : f32
// CHECK-NEXT:          %8 = arith.addf %7, %out : f32
// CHECK-NEXT:          linalg.yield %8 : f32
// CHECK-NEXT:        } -> tensor<1xf32>
// CHECK-NEXT:        %inserted_slice = tensor.insert_slice %6 into %arg6[%arg2] [1] [1] : tensor<1xf32> into tensor<64xf32>
// CHECK-NEXT:        %inserted_slice_2 = tensor.insert_slice %5 into %arg7[%arg2, %arg5] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
// CHECK-NEXT:        scf.yield %inserted_slice, %inserted_slice_2 : tensor<64xf32>, tensor<64x128xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %4#0, %4#1 : tensor<64xf32>, tensor<64x128xf32>
// CHECK-NEXT:    }
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
    %fused_op, %new_containing_op = transform.structured.extended_fuse_into_containing_op %1 into %loops#0 {duplicate_producer = false} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %3 = transform.structured.match attributes {__c__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_1, %loops_1:2 = transform.structured.tile_using_for %3 tile_sizes [1, 64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield 
  }
}

// -----
 
func.func @by_type() {
  %0 = arith.constant 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1.0 : f32
  return
}
 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_name = transform.structured.match
      ops{["arith.constant"]} filter_result_type = f32 in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name : !transform.any_op
    transform.yield
  }
}
 
// -----
 
func.func @by_operand_type() {
  %c2 = arith.constant 2.0: f32
  %v = arith.constant 8: i32
  %r1 = math.fpowi %c2, %v : f32, i32
  // expected-remark @below {{matched op name}}
  %r2 = arith.addf %c2, %c2 : f32
  // expected-remark @below {{matched op name}}
  %r3 = arith.fptoui %r2 : f32 to i32
  return
}
 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_name1 = transform.structured.match
      ops{["arith.fptoui"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name1, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name1 : !transform.any_op
 
    %match_name2 = transform.structured.match
      ops{["arith.addf"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name2, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name2 : !transform.any_op
 
    %no_match_name1 = transform.structured.match
      ops{["arith.fptoui"]} filter_operand_types = [i32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %no_match_name1, "should not match" : !transform.any_op
    transform.test_consume_operand %no_match_name1 : !transform.any_op
 
    %no_match_name2 = transform.structured.match
      ops{["math.fpowi"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %no_match_name2, "should not match" : !transform.any_op
    transform.test_consume_operand %no_match_name2 : !transform.any_op
    transform.yield
  }
}
 
// -----
 
func.func @foo(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) {
  %c0 = arith.constant 0.0 : f32
  // expected-remark @below {{tileable}}
  %r = linalg.fill ins(%c0 : f32) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  // expected-remark @below {{tileable}}
  linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%r : tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}
 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{TilingInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %matched, "tileable" : !transform.any_op
    transform.yield
  }
}
 
// -----
 
func.func private @callee()
 
func.func @foo(%lb: index, %ub: index, %step: index) {
  // expected-remark @below {{loop-like}}
  scf.for %i = %lb to %ub step %step {
    func.call @callee() : () -> ()
    scf.yield
  }
  // expected-remark @below {{loop-like}}
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    func.call @callee() : () -> ()
    scf.reduce
  }
  // expected-remark @below {{loop-like}}
  scf.forall (%i) in (%ub) {
    func.call @callee() : () -> ()
  }
  return
}
 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %matched, "loop-like" : !transform.any_op
    transform.yield
  }
}
