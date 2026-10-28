// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-vector-loop=0" -split-input-file -verify-diagnostics

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_vector_trip_count_zero(%lb: index, %ub: index, %gm_offset: index, %gm_src: tensor<4x8xf32>, %gm_dst: memref<4x8xf32>) -> (tensor<8xf32>) {
    %c1 = arith.constant 1 : index
    %empty1 = tensor.empty() : tensor<8xf32>
    // expected-error@+1 {{'scf.for' op invalid sub-tiling trip count 0 from 'tile-mix-vector-loop'; expected a positive value}}
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %empty1) -> (tensor<8xf32>) {
      %empty2 = tensor.empty() : tensor<8xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0] [1, 8] [1, 1] : tensor<4x8xf32> to tensor<8xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<8xf32>) outs(%empty2 : tensor<8xf32>) init_out_buffer = false -> tensor<8xf32>
      %subview = memref.subview %gm_dst[%gm_offset, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
      %expanded = tensor.expand_shape %load [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
      hivm.hir.store ins(%expanded : tensor<1x8xf32>) outs(%subview : memref<1x8xf32, strided<[8, 1], offset: ?>>)
      scf.yield %load : tensor<8xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<8xf32>
  }
}

// -----

// Case 2: the invalid compile option is not consulted because the loop is
// skipped for complex control flow.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_vector_trip_count_zero_complex_cf(%lb: index, %ub: index, %gm_offset: index, %gm_src: tensor<4x8xf32>, %gm_dst: memref<4x8xf32>) {
    %c1 = arith.constant 1 : index
    %cond = arith.constant true
    scf.for %arg0 = %lb to %ub step %c1 {
      scf.if %cond {
        %empty2 = tensor.empty() : tensor<8xf32>
        %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0] [1, 8] [1, 1] : tensor<4x8xf32> to tensor<8xf32>
        %load = hivm.hir.load ins(%extracted_slice : tensor<8xf32>) outs(%empty2 : tensor<8xf32>) init_out_buffer = false -> tensor<8xf32>
        %subview = memref.subview %gm_dst[%gm_offset, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
        %expanded = tensor.expand_shape %load [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
        hivm.hir.store ins(%expanded : tensor<1x8xf32>) outs(%subview : memref<1x8xf32, strided<[8, 1], offset: ?>>)
      }
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
