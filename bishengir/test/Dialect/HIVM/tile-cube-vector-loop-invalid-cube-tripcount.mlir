// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-cube-loop=0" -split-input-file -verify-diagnostics

// -----

// Case 1: invalid compile option.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_cube_option_trip_count_zero(
                                          %global_A: memref<256x128xbf16>,
                                          %global_B: memref<128x256xbf16>,
                                          %out_C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    // expected-error@+1 {{'scf.for' op invalid sub-tiling trip count 0 from 'tile-mix-cube-loop'; expected a positive value}}
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B = memref.alloc() : memref<128x256xbf16>
      hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
      %empty_out = tensor.empty() : tensor<256x256xf32>
      %mmad = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// Case 2: invalid IR attribute (zero), which overrides the compile option.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_cube_attr_trip_count_zero(
                                          %global_A: memref<256x128xbf16>,
                                          %global_B: memref<128x256xbf16>,
                                          %out_C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    // expected-error@+1 {{'scf.for' op invalid sub-tiling trip count 0 from 'hivm.tile_mix_cube_num'; expected a positive value}}
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B = memref.alloc() : memref<128x256xbf16>
      hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
      %empty_out = tensor.empty() : tensor<256x256xf32>
      // expected-warning@+1 {{cube loop trip count will depend on attr instead of compile option}}
      %mmad = hivm.hir.mmadL1 {hivm.tile_mix_cube_num = 0 : i32} ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// Case 3: negative IR attribute is rejected.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_cube_attr_trip_count_negative(
                                          %global_A: memref<256x128xbf16>,
                                          %global_B: memref<128x256xbf16>,
                                          %out_C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    // expected-error@+1 {{'scf.for' op invalid sub-tiling trip count -1 from 'hivm.tile_mix_cube_num'; expected a positive value}}
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B = memref.alloc() : memref<128x256xbf16>
      hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
      %empty_out = tensor.empty() : tensor<256x256xf32>
      // expected-warning@+1 {{cube loop trip count will depend on attr instead of compile option}}
      %mmad = hivm.hir.mmadL1 {hivm.tile_mix_cube_num = -1 : i32} ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// Case 4: the invalid compile option is not consulted because the loop is
// skipped for complex control flow.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_cube_trip_count_zero_complex_cf(
                                          %global_A: memref<256x128xbf16>,
                                          %global_B: memref<128x256xbf16>,
                                          %out_C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      scf.if %true {
        %alloc_A = memref.alloc() : memref<256x128xbf16>
        hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
        %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
        %alloc_B = memref.alloc() : memref<128x256xbf16>
        hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
        %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
        %empty_out = tensor.empty() : tensor<256x256xf32>
        %mmad = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
      }
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// Case 5: a legal IR attribute overrides the invalid compile option, so only
// the override warning is emitted.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_cube_attr_overrides_invalid_option(
                                          %global_A: memref<256x128xbf16>,
                                          %global_B: memref<128x256xbf16>,
                                          %out_C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B = memref.alloc() : memref<128x256xbf16>
      hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
      %empty_out = tensor.empty() : tensor<256x256xf32>
      // expected-warning@+1 {{cube loop trip count will depend on attr instead of compile option}}
      %mmad = hivm.hir.mmadL1 {hivm.tile_mix_cube_num = 1 : i32} ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// Case 6: the cube option is not consulted for vector loops, so an invalid
// cube option leaves a vector-only module untouched.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_vector_only_with_invalid_cube_option(%lb: index, %ub: index, %gm_offset: index, %gm_src: tensor<4x8xf32>, %gm_dst: memref<4x8xf32>) -> (tensor<8xf32>) {
    %c1 = arith.constant 1 : index
    %empty1 = tensor.empty() : tensor<8xf32>
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
