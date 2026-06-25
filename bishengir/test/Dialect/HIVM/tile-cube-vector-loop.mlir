// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-cube-loop=4" -split-input-file | FileCheck %s --check-prefix=CHECK-CUBE
// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-vector-loop=4" -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-VECTOR

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_a: memref<?xf16>, %gm_c: memref<128x498xf32, strided<[498, 1], offset: ?>>) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // Define A + Load A
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %a = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
    %tensor_a = bufferization.to_tensor %a restrict writable : memref<128x64xf16>
    // Define B
    %gm_b_tile = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
    %gm_b_tile_cast = memref.cast %gm_b_tile : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
    // CHECK-CUBE: scf.for
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %gm_b_tile_cast) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>) {
      // CHECK-CUBE-NOT: memref.alloc
      // CHECK-CUBE-NOT: hivm.hir.load
      // CHECK-CUBE: scf.for
      // CHECK-CUBE: hivm.hir.load

      // Load B
      %b = memref.alloc() : memref<498x64xf16>
      hivm.hir.load ins(%arg1 : memref<498x64xf16, strided<[?, ?], offset: ?>>) outs(%b : memref<498x64xf16>) init_out_buffer = false
      %tensor_b = bufferization.to_tensor %b restrict writable : memref<498x64xf16>
      // Mmad + fixpipe
      %c = tensor.empty() : tensor<128x498xf32>
      %mmad_out = hivm.hir.mmadL1 {b_transpose} ins(%tensor_a, %tensor_b, %true, %c128, %c64, %c498 : tensor<128x64xf16>, tensor<498x64xf16>, i1, index, index, index) outs(%c : tensor<128x498xf32>) -> tensor<128x498xf32>
      hivm.hir.fixpipe {enable_nz2nd} ins(%mmad_out : tensor<128x498xf32>) outs(%gm_c : memref<128x498xf32, strided<[498, 1], offset: ?>>)
      // Advacne B
      %advance_b = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
      %advance_b_cast = memref.cast %advance_b : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>      scf.yield %advance_b_cast : memref<498x64xf16, strided<[?, ?], offset: ?>>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return %res : memref<498x64xf16, strided<[?, ?], offset: ?>>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_a: memref<?xf16>, %gm_c: memref<128x498xf32, strided<[498, 1], offset: ?>>) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // Define A
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %gm_a_tile_cast = memref.cast %gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16, strided<[?, ?], offset: ?>>
    // Define B
    %gm_b_tile = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
    %gm_b_tile_cast = memref.cast %gm_b_tile : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
    // CHECK-CUBE: scf.for
    %res:2 = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %gm_b_tile_cast, %arg2 = %gm_a_tile_cast) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>) {
      // CHECK-CUBE: hivm.hir.load
      // CHECK-CUBE: scf.for
      // CHECK-CUBE: hivm.hir.load

      // Load A
      %a = memref.alloc() : memref<128x64xf16>
      hivm.hir.load ins(%arg2 : memref<128x64xf16, strided<[?, ?], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
      %tensor_a = bufferization.to_tensor %a restrict writable : memref<128x64xf16>
      // Load B
      %b = memref.alloc() : memref<498x64xf16>
      hivm.hir.load ins(%arg1 : memref<498x64xf16, strided<[?, ?], offset: ?>>) outs(%b : memref<498x64xf16>) init_out_buffer = false
      %tensor_b = bufferization.to_tensor %b restrict writable : memref<498x64xf16>
      // Mmad + fixpipe
      %c = tensor.empty() : tensor<128x498xf32>
      %mmad_out = hivm.hir.mmadL1 {b_transpose} ins(%tensor_a, %tensor_b, %true, %c128, %c64, %c498 : tensor<128x64xf16>, tensor<498x64xf16>, i1, index, index, index) outs(%c : tensor<128x498xf32>) -> tensor<128x498xf32>
      hivm.hir.fixpipe {enable_nz2nd} ins(%mmad_out : tensor<128x498xf32>) outs(%gm_c : memref<128x498xf32, strided<[498, 1], offset: ?>>)
      // Advacne A
      %advance_a = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %advance_a_cast = memref.cast %advance_a : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16, strided<[?, ?], offset: ?>>
      // Advacne B
      %advance_b = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
      %advance_b_cast = memref.cast %advance_b : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %advance_b_cast, %advance_a_cast: memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return %res#0, %res#1 : memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_src: tensor<4x128x498xf32>, %gm_dst: memref<4x128x498xf16>) -> (tensor<128xf32>) {
    %c1 = arith.constant 1 : index
    %accum = tensor.empty() : tensor<128xf32>
    // CHECK-VECTOR: scf.for
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %accum) -> (tensor<128xf32>) {
      // CHECK-VECTOR: scf.for
      // CHECK-VECTOR: hivm.hir.load
      %empty = tensor.empty() : tensor<128x498xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : tensor<4x128x498xf32> to tensor<128x498xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<128x498xf32>) outs(%empty : tensor<128x498xf32>) init_out_buffer = false -> tensor<128x498xf32>
      %empty1 = tensor.empty() : tensor<128x1xf32>
      %reduced = hivm.hir.vreduce <max> ins(%load : tensor<128x498xf32>) outs(%empty1 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
      %collapsed = tensor.collapse_shape %reduced [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %empty2 = tensor.empty() : tensor<128x498xf16>
      // tensor yields
      %cast = hivm.hir.vcast ins(%load : tensor<128x498xf32>) outs(%empty2 : tensor<128x498xf16>) -> tensor<128x498xf16>
      %subview = memref.subview %gm_dst[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : memref<4x128x498xf16> to memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>> into memref<128x498xf16, strided<[498, 1], offset: ?>>
      // memref store
      hivm.hir.store ins(%cast : tensor<128x498xf16>) outs(%collapse_shape : memref<128x498xf16, strided<[498, 1], offset: ?>>)
      scf.yield %collapsed : tensor<128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<128xf32>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  // CHECK-CUBE-LABEL: func @nop
  // CHECK-VECTOR-LABEL: func @nop
  func.func @nop(%gm_offset: index, %gm_a: memref<?xf16>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %a = memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE-NOT: scf.for
    // CHECK-CUBE: hivm.hir.load
    // CHECK-VECTOR-NOT: scf.for
    // CHECK-VECTOR: hivm.hir.load
    hivm.hir.load ins(%gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  // CHECK-CUBE-LABEL: func @test_scope
  // CHECK-VECTOR-LABEL: func @test_scope
  func.func @test_scope(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_src: tensor<4x128x498xf32>, %gm_dst: memref<4x128x498xf16>) -> (tensor<128xf32>) {
    %c1 = arith.constant 1 : index
    %accum = tensor.empty() : tensor<128xf32>
    %res:1 = scope.scope : () -> (tensor<128xf32>) {
      // CHECK-VECTOR: scf.for
      // CHECK-VECTOR: hivm.hir.load
      %empty = tensor.empty() : tensor<128x498xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : tensor<4x128x498xf32> to tensor<128x498xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<128x498xf32>) outs(%empty : tensor<128x498xf32>) init_out_buffer = false -> tensor<128x498xf32>
      %empty1 = tensor.empty() : tensor<128x1xf32>
      %reduced = hivm.hir.vreduce <max> ins(%load : tensor<128x498xf32>) outs(%empty1 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
      %collapsed = tensor.collapse_shape %reduced [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %empty2 = tensor.empty() : tensor<128x498xf16>
      // tensor yields
      %cast = hivm.hir.vcast ins(%load : tensor<128x498xf32>) outs(%empty2 : tensor<128x498xf16>) -> tensor<128x498xf16>
      %subview = memref.subview %gm_dst[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : memref<4x128x498xf16> to memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>> into memref<128x498xf16, strided<[498, 1], offset: ?>>
      // memref store
      hivm.hir.store ins(%cast : tensor<128x498xf16>) outs(%collapse_shape : memref<128x498xf16, strided<[498, 1], offset: ?>>)
      scope.return %collapsed : tensor<128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<128xf32>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  // CHECK-VECTOR-LABEL: func @test_vcmp_tiling_i1_fail
  func.func @test_vcmp_tiling_i1_fail(%lb: index, %ub: index, %gm_offset: index, %gm_src: tensor<4x8xf32>, %gm_dst: memref<4x8xi1>) -> (tensor<8xi1>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0.0 : f32
    %empty1 = tensor.empty() : tensor<8xi1>
    %empty2 = tensor.empty() : tensor<8xf32>
    %brc = hivm.hir.vbrc ins(%c0 :f32) outs(%empty2 : tensor<8xf32>) -> tensor<8xf32>
    // CHECK-VECTOR: scf.for
    // expected-error@+1 {{'scf.for' op Failed to collect vector loop tiling info}}
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %empty1) -> (tensor<8xi1>) {
      // CHECK-VECTOR: hivm.hir.load
      // CHECK-VECTOR: hivm.hir.vcmp ins({{.*}} : tensor<8xf32>, tensor<8xf32>) outs({{.*}} : tensor<8xi1>)
      %empty3 = tensor.empty() : tensor<8xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0] [1, 8] [1, 1] : tensor<4x8xf32> to tensor<8xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<8xf32>) outs(%empty3 : tensor<8xf32>) init_out_buffer = false -> tensor<8xf32>
      %empty4 = tensor.empty() : tensor<8xi1>
      %cmp = hivm.hir.vcmp ins(%load, %brc : tensor<8xf32>, tensor<8xf32>)
                     outs(%empty4 : tensor<8xi1>)
                     compare_mode = #hivm.compare_mode<gt> -> tensor<8xi1>
      %subview = memref.subview %gm_dst[%gm_offset, 0] [1, 8] [1, 1] : memref<4x8xi1> to memref<1x8xi1, strided<[8, 1], offset: ?>>
      %expanded = tensor.expand_shape %cmp [[0, 1]] output_shape [1, 8] : tensor<8xi1> into tensor<1x8xi1>
      hivm.hir.store ins(%expanded : tensor<1x8xi1>) outs(%subview : memref<1x8xi1, strided<[8, 1], offset: ?>>)
      scf.yield %cmp : tensor<8xi1>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<8xi1>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  // CHECK-VECTOR-LABEL: func @test_vcmp_tiling_i1_success
  func.func @test_vcmp_tiling_i1_success(%lb: index, %ub: index, %gm_offset: index, %gm_src: tensor<4x32xf32>, %gm_dst: memref<4x32xi1>) -> (tensor<32xi1>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0.0 : f32
    %empty1 = tensor.empty() : tensor<32xi1>
    %empty2 = tensor.empty() : tensor<32xf32>
    %brc = hivm.hir.vbrc ins(%c0 :f32) outs(%empty2 : tensor<32xf32>) -> tensor<32xf32>
    // CHECK-VECTOR: scf.for
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %empty1) -> (tensor<32xi1>) {
    // CHECK-VECTOR: scf.for
      // CHECK-VECTOR: hivm.hir.load
      // CHECK-VECTOR: hivm.hir.vcmp {vector_producer_to_fuse_0} ins({{.*}} : tensor<8xf32>, tensor<8xf32>) outs({{.*}} : tensor<8xi1>)
      %empty3 = tensor.empty() : tensor<32xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<32xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<32xf32>) outs(%empty3 : tensor<32xf32>) init_out_buffer = false -> tensor<32xf32>
      %empty4 = tensor.empty() : tensor<32xi1>
      %cmp = hivm.hir.vcmp ins(%load, %brc : tensor<32xf32>, tensor<32xf32>)
                     outs(%empty4 : tensor<32xi1>)
                     compare_mode = #hivm.compare_mode<gt> -> tensor<32xi1>
      %subview = memref.subview %gm_dst[%gm_offset, 0] [1, 32] [1, 1] : memref<4x32xi1> to memref<1x32xi1, strided<[32, 1], offset: ?>>
      %expanded = tensor.expand_shape %cmp [[0, 1]] output_shape [1, 32] : tensor<32xi1> into tensor<1x32xi1>
      hivm.hir.store ins(%expanded : tensor<1x32xi1>) outs(%subview : memref<1x32xi1, strided<[32, 1], offset: ?>>)
      scf.yield %cmp : tensor<32xi1>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<32xi1>
  }
}

// -----
 	 
// CHECK-CUBE-DAG: #[[$MAP_M:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_tile_m_axis
  func.func @test_cube_tile_m_axis(
                                  %global_A: memref<256x128xbf16>, 
                                  %global_B: memref<128x256xbf16>, 
                                  %out_C: memref<256x256xf32>) {
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[TRUE:.*]] = arith.constant true
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
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
    // CHECK-CUBE:       scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    // CHECK-CUBE:         %[[EMPTY_B:.*]] = tensor.empty() : tensor<128x256xbf16>
    // CHECK-CUBE:         %[[TENSOR_B:.*]] = bufferization.to_tensor
    // CHECK-CUBE:         %[[LOAD_B:.*]] = hivm.hir.load ins(%[[TENSOR_B]] : tensor<128x256xbf16>) outs(%[[EMPTY_B]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    // CHECK-CUBE-NOT:     cube_producer_to_fuse
    // CHECK-CUBE:         %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<256x256xf32>
    // CHECK-CUBE:         scf.for %[[INNER_IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:           %[[EMPTY_A:.*]] = tensor.empty() {cube_producer_to_fuse_0_group_0} : tensor<256x128xbf16>
    // CHECK-CUBE:           %[[EXTRACT_A:.*]] = tensor.extract_slice %[[EMPTY_A]][%[[INNER_IV]], 0] [64, 128]
    // CHECK-CUBE:           %[[LOAD_A:.*]] = hivm.hir.load ins(%{{.*}}) outs(%[[EXTRACT_A]] : tensor<64x128xbf16>) {cube_producer_to_fuse_0_group_0} -> tensor<64x128xbf16>
    // CHECK-CUBE:           %[[MIN:.*]] = affine.min #[[$MAP_M]](%[[INNER_IV]])
    // CHECK-CUBE:           %[[MMAD:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C256]]{{.*}}
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD]] : tensor<64x256xf32>)
    // CHECK-CUBE:         }
    // CHECK-CUBE:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  }
}

// -----
 	 
// CHECK-CUBE-DAG: #[[$MAP_N:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_tile_n_axis
  func.func @test_cube_tile_n_axis(
                                  %global_A: memref<256x128xbf16>, 
                                  %global_B: memref<128x256xbf16>, 
                                  %out_C: memref<256x256xf32>) {
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[TRUE:.*]] = arith.constant true
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %alloc_A = memref.alloc() : memref<256x128xbf16>
    hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
    %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_B = memref.alloc() : memref<128x256xbf16>
      hivm.hir.load ins(%global_B : memref<128x256xbf16>) outs(%alloc_B : memref<128x256xbf16>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x256xbf16>
      %empty_out = tensor.empty() : tensor<256x256xf32>
      %mmad = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c256 : tensor<256x128xbf16>, tensor<128x256xbf16>, i1, index, index, index) outs(%empty_out : tensor<256x256xf32>) -> tensor<256x256xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad : tensor<256x256xf32>) outs(%out_C : memref<256x256xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
    // CHECK-CUBE:       %[[EMPTY_A:.*]] = tensor.empty() : tensor<256x128xbf16>
    // CHECK-CUBE:       %[[TENSOR_A:.*]] = bufferization.to_tensor
    // CHECK-CUBE:       %[[LOAD_A:.*]] = hivm.hir.load ins(%[[TENSOR_A]] : tensor<256x128xbf16>) outs(%[[EMPTY_A]] : tensor<256x128xbf16>) -> tensor<256x128xbf16>
    // CHECK-CUBE-NOT:   cube_producer_to_fuse
    // CHECK-CUBE:       scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    // CHECK-CUBE:         %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<256x256xf32>
    // CHECK-CUBE:         scf.for %[[INNER_IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:           %[[EMPTY_B:.*]] = tensor.empty() {cube_producer_to_fuse_0_group_0} : tensor<128x256xbf16>
    // CHECK-CUBE:           %[[EXTRACT_B:.*]] = tensor.extract_slice %[[EMPTY_B]][0, %[[INNER_IV]]] [128, 64] [1, 1]
    // CHECK-CUBE:           %[[LOAD_B:.*]] = hivm.hir.load ins(%{{.*}}) outs(%[[EXTRACT_B]] : tensor<128x64xbf16>) {cube_producer_to_fuse_0_group_0} -> tensor<128x64xbf16>
    // CHECK-CUBE:           %[[MIN:.*]] = affine.min #[[$MAP_N]](%[[INNER_IV]])
    // CHECK-CUBE:           %[[MMAD:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B]], %[[TRUE]], %[[C256]], %[[C128]], %[[MIN]]
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD]] : tensor<256x64xf32>)
    // CHECK-CUBE:         }
    // CHECK-CUBE:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  }
}

// -----
	
// CHECK-CUBE-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_loop_fusion_shared_load
  func.func @test_cube_loop_fusion_shared_load(%global_A: memref<256x128xbf16>, 
                                               %global_B1: memref<128x128xbf16>, 
                                               %global_B2: memref<128x128xbf16>, 
                                               %out1: memref<256x128xf32>, 
                                               %out2: memref<256x128xf32>) {
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[TRUE:.*]] = arith.constant true
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B1 = memref.alloc() : memref<128x128xbf16>
      hivm.hir.load ins(%global_B1 : memref<128x128xbf16>) outs(%alloc_B1 : memref<128x128xbf16>)
      %tensor_B1 = bufferization.to_tensor %alloc_B1 restrict writable : memref<128x128xbf16>
      %empty_out1 = tensor.empty() : tensor<256x128xf32>
      %mmad1 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B1, %true, %c256, %c128, %c128 : tensor<256x128xbf16>, tensor<128x128xbf16>, i1, index, index, index) outs(%empty_out1 : tensor<256x128xf32>) -> tensor<256x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad1 : tensor<256x128xf32>) outs(%out1 : memref<256x128xf32>)
      %alloc_B2 = memref.alloc() : memref<128x128xbf16>
      hivm.hir.load ins(%global_B2 : memref<128x128xbf16>) outs(%alloc_B2 : memref<128x128xbf16>)
      %tensor_B2 = bufferization.to_tensor %alloc_B2 restrict writable : memref<128x128xbf16>
      %empty_out2 = tensor.empty() : tensor<256x128xf32>
      %mmad2 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B2, %true, %c256, %c128, %c128 : tensor<256x128xbf16>, tensor<128x128xbf16>, i1, index, index, index) outs(%empty_out2 : tensor<256x128xf32>) -> tensor<256x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad2 : tensor<256x128xf32>) outs(%out2 : memref<256x128xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    // CHECK-CUBE:         scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    // CHECK-CUBE:           %[[EMPTY_B:.*]] = tensor.empty() : tensor<128x128xbf16>
    // CHECK-CUBE:           %[[TENSOR_B1:.*]] = bufferization.to_tensor %arg1{{.*}}
    // CHECK-CUBE:           %[[LOAD_B1:.*]] = hivm.hir.load ins(%[[TENSOR_B1]]{{.*}}) outs(%[[EMPTY_B]]{{.*}}) -> tensor<128x128xbf16>
    // CHECK-CUBE-NOT:       cube_producer_to_fuse
    // CHECK-CUBE:           %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<256x128xf32>
    // CHECK-CUBE:           %[[TENSOR_B2:.*]] = bufferization.to_tensor %arg2{{.*}}
    // CHECK-CUBE:           %[[LOAD_B2:.*]] = hivm.hir.load ins(%[[TENSOR_B2]]{{.*}}) outs(%[[EMPTY_B]]{{.*}}) -> tensor<128x128xbf16>
    // CHECK-CUBE-NOT:       cube_producer_to_fuse
    // CHECK-CUBE:           scf.for %[[INNER_IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:             %[[EMPTY_A:.*]] = tensor.empty() {cube_producer_to_fuse_0_group_0} : tensor<256x128xbf16>
    // CHECK-CUBE:             %[[EXTRACT_A:.*]] = tensor.extract_slice %[[EMPTY_A]][%[[INNER_IV]], 0]{{.*}}
    // CHECK-CUBE:             %[[LOAD_A:.*]] = hivm.hir.load ins(%{{.*}}) outs(%[[EXTRACT_A]]{{.*}}) {cube_producer_to_fuse_0_group_0} -> tensor<64x128xbf16>
    // CHECK-CUBE:             %[[MIN:.*]] = affine.min #[[$MAP]](%[[INNER_IV]])
    // CHECK-CUBE:             %[[MMAD1:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B1]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C128]]{{.*}}
    // CHECK-CUBE:             hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD1]]{{.*}})
    // CHECK-CUBE:             %[[MMAD2:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B2]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C128]]{{.*}}
    // CHECK-CUBE:             hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_1} ins(%[[MMAD2]]{{.*}})
    // CHECK-CUBE:           }
    // CHECK-CUBE:         } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
 	 
// CHECK-CUBE-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_loop_fusion_shared_load_2
  func.func @test_cube_loop_fusion_shared_load_2(
                                              %global_A: memref<256x128xbf16>, 
                                              %global_B1: memref<128x64xbf16>, 
                                              %global_B2: memref<128x64xbf16>, 
                                              %global_B3: memref<128x64xbf16>, 
                                              %out1: memref<256x64xf32>, 
                                              %out2: memref<256x64xf32>,
                                              %out3: memref<256x64xf32>) {
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[TRUE:.*]] = arith.constant true
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A : memref<256x128xbf16>) outs(%alloc_A : memref<256x128xbf16>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xbf16>
      %alloc_B1 = memref.alloc() : memref<128x64xbf16>
      hivm.hir.load ins(%global_B1 : memref<128x64xbf16>) outs(%alloc_B1 : memref<128x64xbf16>)
      %tensor_B1 = bufferization.to_tensor %alloc_B1 restrict writable : memref<128x64xbf16>
      %empty_out1 = tensor.empty() : tensor<256x64xf32>
      %mmad1 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B1, %true, %c256, %c128, %c64 : tensor<256x128xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%empty_out1 : tensor<256x64xf32>) -> tensor<256x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad1 : tensor<256x64xf32>) outs(%out1 : memref<256x64xf32>)
      %alloc_B2 = memref.alloc() : memref<128x64xbf16>
      hivm.hir.load ins(%global_B2 : memref<128x64xbf16>) outs(%alloc_B2 : memref<128x64xbf16>)
      %tensor_B2 = bufferization.to_tensor %alloc_B2 restrict writable : memref<128x64xbf16>
      %empty_out2 = tensor.empty() : tensor<256x64xf32>
      %mmad2 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B2, %true, %c256, %c128, %c64 : tensor<256x128xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%empty_out2 : tensor<256x64xf32>) -> tensor<256x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad2 : tensor<256x64xf32>) outs(%out2 : memref<256x64xf32>)
      %alloc_B3 = memref.alloc() : memref<128x64xbf16>
      hivm.hir.load ins(%global_B3 : memref<128x64xbf16>) outs(%alloc_B3 : memref<128x64xbf16>)
      %tensor_B3 = bufferization.to_tensor %alloc_B3 restrict writable : memref<128x64xbf16>
      %empty_out3 = tensor.empty() : tensor<256x64xf32>
      %mmad3 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B3, %true, %c256, %c128, %c64 : tensor<256x128xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%empty_out3 : tensor<256x64xf32>) -> tensor<256x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad3 : tensor<256x64xf32>) outs(%out3 : memref<256x64xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
    // CHECK-CUBE:       scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    // CHECK-CUBE:         %[[EMPTY_B:.*]] = tensor.empty() : tensor<128x64xbf16>
    // CHECK-CUBE:         %[[LOAD_B1:.*]] = hivm.hir.load ins(%{{.*}} : tensor<128x64xbf16>) outs(%[[EMPTY_B]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
    // CHECK-CUBE-NOT:     cube_producer_to_fuse
    // CHECK-CUBE:         %[[LOAD_B2:.*]] = hivm.hir.load ins(%{{.*}} : tensor<128x64xbf16>) outs(%[[EMPTY_B]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
    // CHECK-CUBE-NOT:     cube_producer_to_fuse
    // CHECK-CUBE:         %[[LOAD_B3:.*]] = hivm.hir.load ins(%{{.*}} : tensor<128x64xbf16>) outs(%[[EMPTY_B]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
    // CHECK-CUBE-NOT:     cube_producer_to_fuse
    // CHECK-CUBE:         scf.for %[[INNER_IV:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:           %[[EMPTY_A:.*]] = tensor.empty() {cube_producer_to_fuse_0_group_0} : tensor<256x128xbf16>
    // CHECK-CUBE:           %[[EXTRACT_A:.*]] = tensor.extract_slice %[[EMPTY_A]][%[[INNER_IV]], 0]
    // CHECK-CUBE:           %[[LOAD_A:.*]] = hivm.hir.load ins(%{{.*}}) outs(%[[EXTRACT_A]] : tensor<64x128xbf16>) {cube_producer_to_fuse_0_group_0} -> tensor<64x128xbf16>
    // CHECK-CUBE:           %[[MIN:.*]] = affine.min #[[$MAP]](%[[INNER_IV]])
    // CHECK-CUBE:           %[[MMAD1:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B1]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C64]]
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD1]] : tensor<64x64xf32>)
    // CHECK-CUBE:           %[[MMAD2:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B2]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C64]]
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_1} ins(%[[MMAD2]] : tensor<64x64xf32>)
    // CHECK-CUBE:           %[[MMAD3:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A]], %[[LOAD_B3]], %[[TRUE]], %[[MIN]], %[[C128]], %[[C64]]
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_2} ins(%[[MMAD3]] : tensor<64x64xf32>)
    // CHECK-CUBE:         }
    // CHECK-CUBE:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  }
}

// -----

// CHECK-CUBE-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_loop_fusion_independence
  func.func @test_cube_loop_fusion_independence(%global_A1: memref<256x128xbf16>, 
                                                %global_A2: memref<256x128xbf16>, 
                                                %global_B1: memref<128x128xbf16>, 
                                                %global_B2: memref<128x128xbf16>, 
                                                %out1: memref<256x128xf32>, 
                                                %out2: memref<256x128xf32>) {
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[TRUE:.*]] = arith.constant true
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %alloc_B1 = memref.alloc() : memref<128x128xbf16>
    hivm.hir.load ins(%global_B1 : memref<128x128xbf16>) outs(%alloc_B1 : memref<128x128xbf16>)
    %tensor_B1 = bufferization.to_tensor %alloc_B1 restrict writable : memref<128x128xbf16>
    %alloc_B2 = memref.alloc() : memref<128x128xbf16>
    hivm.hir.load ins(%global_B2 : memref<128x128xbf16>) outs(%alloc_B2 : memref<128x128xbf16>)
    %tensor_B2 = bufferization.to_tensor %alloc_B2 restrict writable : memref<128x128xbf16>
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A1 = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A1 : memref<256x128xbf16>) outs(%alloc_A1 : memref<256x128xbf16>)
      %tensor_A1 = bufferization.to_tensor %alloc_A1 restrict writable : memref<256x128xbf16>
      %empty_out1 = tensor.empty() : tensor<256x128xf32>
      %mmad1 = hivm.hir.mmadL1 ins(%tensor_A1, %tensor_B1, %true, %c256, %c128, %c128 : tensor<256x128xbf16>, tensor<128x128xbf16>, i1, index, index, index) outs(%empty_out1 : tensor<256x128xf32>) -> tensor<256x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad1 : tensor<256x128xf32>) outs(%out1 : memref<256x128xf32>)
      %alloc_A2 = memref.alloc() : memref<256x128xbf16>
      hivm.hir.load ins(%global_A2 : memref<256x128xbf16>) outs(%alloc_A2 : memref<256x128xbf16>)
      %tensor_A2 = bufferization.to_tensor %alloc_A2 restrict writable : memref<256x128xbf16>
      %empty_out2 = tensor.empty() : tensor<256x128xf32>
      %mmad2 = hivm.hir.mmadL1 ins(%tensor_A2, %tensor_B2, %true, %c256, %c128, %c128 : tensor<256x128xbf16>, tensor<128x128xbf16>, i1, index, index, index) outs(%empty_out2 : tensor<256x128xf32>) -> tensor<256x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mmad2 : tensor<256x128xf32>) outs(%out2 : memref<256x128xf32>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    // CHECK-CUBE:       %[[EMPTY_B:.*]] = tensor.empty() : tensor<128x128xbf16>
    // CHECK-CUBE:       %[[TENSOR_B1:.*]] = bufferization.to_tensor %arg2{{.*}}
    // CHECK-CUBE:       %[[LOAD_B1:.*]] = hivm.hir.load ins(%[[TENSOR_B1]]{{.*}}) outs(%[[EMPTY_B]]{{.*}}) -> tensor<128x128xbf16>
    // CHECK-CUBE:       %[[TENSOR_B2:.*]] = bufferization.to_tensor %arg3{{.*}}
    // CHECK-CUBE:       %[[LOAD_B2:.*]] = hivm.hir.load ins(%[[TENSOR_B2]]{{.*}}) outs(%[[EMPTY_B]]{{.*}}) -> tensor<128x128xbf16>
    // CHECK-CUBE:       scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    // CHECK-CUBE:         scf.for %[[INNER_IV1:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:           %[[LOAD_A1:.*]] = hivm.hir.load ins(%{{.*}}) outs(%{{.*}}) {cube_producer_to_fuse_0_group_0} -> tensor<64x128xbf16>
    // CHECK-CUBE:           %[[MIN1:.*]] = affine.min #[[$MAP]](%[[INNER_IV1]])
    // CHECK-CUBE:           %[[MMAD1:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0} ins(%[[LOAD_A1]], %[[LOAD_B1]], %[[TRUE]], %[[MIN1]], %[[C128]], %[[C128]]{{.*}}
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD1]]{{.*}})
    // CHECK-CUBE:         }
    // CHECK-CUBE:         scf.for %[[INNER_IV2:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
    // CHECK-CUBE:           %[[LOAD_A2:.*]] = hivm.hir.load ins(%{{.*}}) outs(%{{.*}}) {cube_producer_to_fuse_0_group_1} -> tensor<64x128xbf16>
    // CHECK-CUBE:           %[[MIN2:.*]] = affine.min #[[$MAP]](%[[INNER_IV2]])
    // CHECK-CUBE:           %[[MMAD2:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_1} ins(%[[LOAD_A2]], %[[LOAD_B2]], %[[TRUE]], %[[MIN2]], %[[C128]], %[[C128]]{{.*}}
    // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_1} ins(%[[MMAD2]]{{.*}})
    // CHECK-CUBE:         }
    // CHECK-CUBE:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
 	 
// CHECK-CUBE-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (-d0 + 256, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-CUBE-LABEL: func.func @test_cube_loop_fusion_memory_dependency
  func.func @test_cube_loop_fusion_memory_dependency(%global_A: memref<256x128xf32>, 
                                                    %global_B: memref<128x64xf32>, 
                                                    %global_D: memref<64x128xf32>, 
                                                    %out_C: memref<256x64xf32>, 
                                                    %out_E: memref<256x128xf32>) {
    // CHECK-CUBE-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-CUBE-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-CUBE-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-CUBE-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-CUBE-DAG: %[[C128:.*]] = arith.constant 128 : index
    // CHECK-CUBE-DAG: %[[C64:.*]] = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    // CHECK-CUBE: scf.for %[[OUTER_IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
    scf.for %iv = %c0 to %c4 step %c1 {
      %alloc_A = memref.alloc() : memref<256x128xf32>
      hivm.hir.load ins(%global_A : memref<256x128xf32>) outs(%alloc_A : memref<256x128xf32>)
      %tensor_A = bufferization.to_tensor %alloc_A restrict writable : memref<256x128xf32>
      %alloc_B = memref.alloc() : memref<128x64xf32>
      hivm.hir.load ins(%global_B : memref<128x64xf32>) outs(%alloc_B : memref<128x64xf32>)
      %tensor_B = bufferization.to_tensor %alloc_B restrict writable : memref<128x64xf32>
      %empty_C = tensor.empty() : tensor<256x64xf32>
      %C_f32 = hivm.hir.mmadL1 ins(%tensor_A, %tensor_B, %true, %c256, %c128, %c64 : tensor<256x128xf32>, tensor<128x64xf32>, i1, index, index, index) 
                  outs(%empty_C : tensor<256x64xf32>) -> tensor<256x64xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} 
                      ins(%C_f32 : tensor<256x64xf32>) 
                      outs(%out_C : memref<256x64xf32>)
      // CHECK-CUBE:     %[[EMPTY_1:.*]] = tensor.empty() : tensor<256x64xf32>
      // CHECK-CUBE:     scf.for %[[INNER_IV1:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
      // CHECK-CUBE:       %[[MMAD1:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_0}
      // CHECK-CUBE:       hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD1]] : tensor<64x64xf32>)
      // CHECK-CUBE:     }
      %alloc_C_reload = memref.alloc() : memref<256x64xf32>
      hivm.hir.load ins(%out_C : memref<256x64xf32>) outs(%alloc_C_reload : memref<256x64xf32>)
      %tensor_C_reloaded = bufferization.to_tensor %alloc_C_reload restrict writable : memref<256x64xf32>
      %alloc_D = memref.alloc() : memref<64x128xf32>
      hivm.hir.load ins(%global_D : memref<64x128xf32>) outs(%alloc_D : memref<64x128xf32>)
      %tensor_D = bufferization.to_tensor %alloc_D restrict writable : memref<64x128xf32>
      %empty_E = tensor.empty() : tensor<256x128xf32>
      %E_f32 = hivm.hir.mmadL1 ins(%tensor_C_reloaded, %tensor_D, %true, %c256, %c64, %c128 : tensor<256x64xf32>, tensor<64x128xf32>, i1, index, index, index) 
                  outs(%empty_E : tensor<256x128xf32>) -> tensor<256x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} 
                      ins(%E_f32 : tensor<256x128xf32>) 
                      outs(%out_E : memref<256x128xf32>)
      // CHECK-CUBE:     %[[EMPTY_2:.*]] = tensor.empty() : tensor<256x128xf32>
      // CHECK-CUBE:     scf.for %[[INNER_IV2:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
      // CHECK-CUBE:       %[[C_RELOADED:.*]] = hivm.hir.load {{.*}} {cube_producer_to_fuse_0_group_1}
      // CHECK-CUBE:       %[[E_F32:.*]] = hivm.hir.mmadL1 {cube_producer_to_fuse_0_group_1} ins(%[[C_RELOADED]], {{.*}}
      // CHECK-CUBE:       hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_1} ins(%[[E_F32]] : tensor<64x128xf32>)
      // CHECK-CUBE:     }
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    // CHECK-CUBE:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----
 	 
 	 // CHECK-CUBE-DAG: #[[$MAP_REMAIN:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
 	 // CHECK-CUBE-DAG: #[[$MAP_BOUNDED:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
 	 // CHECK-CUBE-DAG: #[[$MAP_MMAD:.*]] = affine_map<()[s0] -> (256, s0)>
 	 module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
 	   
 	   // CHECK-CUBE-LABEL: func.func @test_extracted_cube_loop_pad_load
 	   // CHECK-CUBE-SAME: (%[[ARG_SRC:.*]]: memref<256x128xbf16{{.*}}>, %[[ARG_B:.*]]: tensor<256x128xbf16>, %[[ARG_OUT:.*]]: memref<256x256xf32{{.*}}>, %[[REMAIN_SIZE:.*]]: index, %[[OTHER_SIZE:.*]]: index, %[[IS_PARTIAL:.*]]: i1)
 	   func.func @test_extracted_cube_loop_pad_load(
 	     %arg_src: memref<256x128xbf16, strided<[640, 1], offset: ?>>, 
 	     %arg_tensor_B: tensor<256x128xbf16>,                          
 	     %arg_out: memref<256x256xf32, strided<[256, 1], offset: ?>>,  
 	     %remain_size: index,                                          
 	     %other_size: index,                                           
 	     %is_partial: i1                                               
 	   ) {
 	     // CHECK-CUBE-DAG:   %[[C64:.*]] = arith.constant 64 : index
 	     // CHECK-CUBE-DAG:   %[[C256:.*]] = arith.constant 256 : index
 	     %c0 = arith.constant 0 : index
 	     %c1 = arith.constant 1 : index
 	     %c128 = arith.constant 128 : index
 	     %cst = arith.constant 0.000000e+00 : bf16
 	     %true = arith.constant true
 	     %false = arith.constant false
 	     // CHECK-CUBE:       scf.for %[[IV_OUTER:.*]] = 
 	     scf.for %arg32 = %c0 to %c1 step %c1 {
 	       // CHECK-CUBE:         %[[SUBVIEW_OUTER:.*]] = memref.subview %[[ARG_SRC]][0, 0] [%[[REMAIN_SIZE]], 128] [1, 1]
 	       %subview_22 = memref.subview %arg_src[0, 0] [%remain_size, 128] [1, 1] : memref<256x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
 	       %tensor_in = bufferization.to_tensor %subview_22 restrict : memref<?x128xbf16, strided<[640, 1], offset: ?>>
 	       %empty_tensor_out = tensor.empty() : tensor<256x128xbf16>
 	       // CHECK-CUBE:         scf.for %[[IV_INNER:.*]] = %{{.*}} to %[[C256]] step %[[C64]] {
 	       // CHECK-CUBE:           %[[REMAIN:.*]] = affine.apply #[[$MAP_REMAIN]](%[[IV_INNER]])[%[[REMAIN_SIZE]]]
 	       // CHECK-CUBE:           %[[BOUNDED:.*]] = affine.min #[[$MAP_BOUNDED]](%[[IV_INNER]])[%[[REMAIN_SIZE]]]
 	       // CHECK-CUBE:           %[[COND:.*]] = arith.cmpi slt, %[[REMAIN]], %[[C64]] : index
 	       // CHECK-CUBE:           %[[SRC_SUBVIEW:.*]] = memref.subview %[[SUBVIEW_OUTER]][%[[IV_INNER]], 0] [%[[BOUNDED]], 128] [1, 1]
 	       // CHECK-CUBE:           %[[SRC_TENSOR:.*]] = bufferization.to_tensor %[[SRC_SUBVIEW]] restrict
 	       // CHECK-CUBE:           %[[SLICE_A:.*]] = tensor.extract_slice %{{.*}}[%[[IV_INNER]], 0] [64, 128] [1, 1]
 	       // CHECK-CUBE:           %[[LOAD_RES:.*]] = hivm.hir.load ins(%[[SRC_TENSOR]] : tensor<?x128xbf16>) outs(%[[SLICE_A]] : tensor<64x128xbf16>) {cube_producer_to_fuse_0_group_0} pad_mode = <PadValue> pad_value = %{{.*}} : bf16 left_padding_num = %{{.*}} : index init_out_buffer = true init_condition = %[[COND]] : i1 may_implicit_transpose_with_last_axis = false -> tensor<64x128xbf16>
 	       %loaded_tensor = hivm.hir.load ins(%tensor_in : tensor<?x128xbf16>) outs(%empty_tensor_out : tensor<256x128xbf16>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %is_partial : i1 may_implicit_transpose_with_last_axis = false -> tensor<256x128xbf16>
 	       %84 = tensor.empty() : tensor<256x256xf32>
 	       // CHECK-CUBE:           %[[BOUNDED_MMAD:.*]] = affine.min #[[$MAP_MMAD]]()[%[[OTHER_SIZE]]]
 	       // CHECK-CUBE:           %[[MMAD_RES:.*]] = hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0_group_0, fixpipe_already_inserted = true} ins(%[[LOAD_RES]], %[[ARG_B]], %{{.*}}, %[[BOUNDED]], %{{.*}}, %[[BOUNDED_MMAD]] : tensor<64x128xbf16>, tensor<256x128xbf16>, i1, index, index, index)
 	       %85 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%loaded_tensor, %arg_tensor_B, %true, %remain_size, %c128, %other_size : tensor<256x128xbf16>, tensor<256x128xbf16>, i1, index, index, index) outs(%84 : tensor<256x256xf32>) -> tensor<256x256xf32>
 	       // CHECK-CUBE:           %[[OUT_SUBVIEW:.*]] = memref.subview %[[ARG_OUT]][%[[IV_INNER]], 0] [64, 256] [1, 1]
 	       // CHECK-CUBE:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_branch_0} ins(%[[MMAD_RES]] : tensor<64x256xf32>) outs(%[[OUT_SUBVIEW]] : memref<64x256xf32, strided<[256, 1], offset: ?>>)
 	       hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%85 : tensor<256x256xf32>) outs(%arg_out : memref<256x256xf32, strided<[256, 1], offset: ?>>)
 	     } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
 	     // CHECK-CUBE:         }
 	     // CHECK-CUBE:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
 	     return
 	   }
 	 }

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_shrink_alloc_positive() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // CHECK-CUBE: memref.alloc() : memref<64x64xf16>
    // CHECK-CUBE-NOT: memref.subview
    %alloc = memref.alloc() : memref<128x64xf16>
    %subview = memref.subview %alloc[0, 0] [64, 64] [1, 1] : memref<128x64xf16> to memref<64x64xf16>
    hivm.hir.vadd ins(%subview, %subview : memref<64x64xf16>, memref<64x64xf16>) outs(%subview : memref<64x64xf16>)
    %res = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg = %c0) -> (index) {
      scf.yield %arg : index
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_shrink_alloc_nonzero_offset() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // CHECK-CUBE: memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE: memref.subview
    %alloc = memref.alloc() : memref<128x64xf16>
    %subview = memref.subview %alloc[64, 0] [64, 64] [1, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1], offset: 4096>>
    hivm.hir.vadd ins(%subview, %subview : memref<64x64xf16, strided<[64, 1], offset: 4096>>, memref<64x64xf16, strided<[64, 1], offset: 4096>>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: 4096>>)
    %res = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg = %c0) -> (index) {
      scf.yield %arg : index
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_shrink_alloc_dynamic_offset(%off: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // CHECK-CUBE: memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE: memref.subview
    %alloc = memref.alloc() : memref<128x64xf16>
    %subview = memref.subview %alloc[%off, 0] [64, 64] [1, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    hivm.hir.vadd ins(%subview, %subview : memref<64x64xf16, strided<[64, 1], offset: ?>>, memref<64x64xf16, strided<[64, 1], offset: ?>>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>>)
    %res = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg = %c0) -> (index) {
      scf.yield %arg : index
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_shrink_alloc_nonunit_stride() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // CHECK-CUBE: memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE: memref.subview
    %alloc = memref.alloc() : memref<128x64xf16>
    %subview = memref.subview %alloc[0, 0] [64, 64] [2, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[128, 1]>>
    hivm.hir.vadd ins(%subview, %subview : memref<64x64xf16, strided<[128, 1]>>, memref<64x64xf16, strided<[128, 1]>>) outs(%subview : memref<64x64xf16, strided<[128, 1]>>)
    %res = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg = %c0) -> (index) {
      scf.yield %arg : index
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_shrink_alloc_dynamic_stride(%st: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // CHECK-CUBE: memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE: memref.subview
    %alloc = memref.alloc() : memref<128x64xf16>
    %subview = memref.subview %alloc[0, 0] [64, 64] [%st, 1] : memref<128x64xf16> to memref<64x64xf16, strided<[?, 1]>>
    hivm.hir.vadd ins(%subview, %subview : memref<64x64xf16, strided<[?, 1]>>, memref<64x64xf16, strided<[?, 1]>>) outs(%subview : memref<64x64xf16, strided<[?, 1]>>)
    %res = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg = %c0) -> (index) {
      scf.yield %arg : index
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return
  }
}
