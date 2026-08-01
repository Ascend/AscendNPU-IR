// RUN: bishengir-opt %s --convert-hivm-to-tritongpu -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: tt.func @subview_dynamic_offset_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<i64>,
// CHECK:       %[[ROWRANGE:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK:       %[[ROWCOL:.*]] = tt.reshape %[[ROWRANGE]] : tensor<32xi32> -> tensor<32x1xi32>
// CHECK:       %[[ROWSTRIDE:.*]] = arith.constant dense<16> : tensor<32x1xi32>
// CHECK:       %{{.*}} = arith.muli %[[ROWCOL]], %[[ROWSTRIDE]] : tensor<32x1xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : tensor<32x1xi32>
// CHECK:       %[[BASEPTRS:.*]] = tt.splat %[[ARG0]] : !tt.ptr<i64> -> tensor<32x1x!tt.ptr<i64>>
// CHECK:       %[[ROWPTRS:.*]] = tt.addptr %[[BASEPTRS]], %{{.*}}
// CHECK:       %{{.*}} = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:       %{{.*}} = tt.reshape %{{.*}} : tensor<16xi32> -> tensor<1x16xi32>
// CHECK:       %[[BCASTPTRS:.*]] = tt.broadcast %[[ROWPTRS]] : tensor<32x1x!tt.ptr<i64>> -> tensor<32x16x!tt.ptr<i64>>
// CHECK:       %[[BCASTCOL:.*]] = tt.broadcast %{{.*}} : tensor<1x16xi32> -> tensor<32x16xi32>
// CHECK:       %[[TILEPTRS:.*]] = tt.addptr %[[BCASTPTRS]], %[[BCASTCOL]] : tensor<32x16x!tt.ptr<i64>>, tensor<32x16xi32>
// CHECK:       tt.load %[[TILEPTRS]] evictionPolicy = evict_first : tensor<32x16x!tt.ptr<i64>>
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_dynamic_offset_load(%arg0: memref<?xi64>, %arg1: index, %arg2: memref<32x16xi64>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<?xi64> to memref<64x16xi64, strided<[16, 1]>>
    %0 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg1]
    %subview = memref.subview %reinterpret_cast[%0, 0] [32, 16] [1, 1] : memref<64x16xi64, strided<[16, 1]>> to memref<32x16xi64, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview : memref<32x16xi64, strided<[16, 1], offset: ?>>) outs(%arg2 : memref<32x16xi64>) eviction_policy = <EvictFirst>
    %1 = bufferization.to_tensor %arg2 restrict writable : memref<32x16xi64>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_static_offset_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32>,
// CHECK:       %{{.*}} = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:       %{{.*}} = tt.reshape %{{.*}} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.constant dense<16> : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.constant dense<256> : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : tensor<16x1xi32>
// CHECK:       %{{.*}} = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK:       tt.load {{.*}} evictionPolicy = evict_first

module {
  func.func @subview_static_offset_load(%arg0: memref<?xf32>, %arg1: memref<16x16xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<?xf32> to memref<64x16xf32, strided<[16, 1]>>
    %subview = memref.subview %reinterpret_cast[16, 0] [16, 16] [1, 1] : memref<64x16xf32, strided<[16, 1]>> to memref<16x16xf32, strided<[16, 1], offset: 256>>
    hivm.hir.load ins(%subview : memref<16x16xf32, strided<[16, 1], offset: 256>>) outs(%arg1 : memref<16x16xf32>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf32>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_hivm_address_space_local_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32, 6>,
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.load {{.*}} : tensor<16x16x!tt.ptr<f32, 6>>

module {
  func.func @subview_hivm_address_space_local_load(%arg0: memref<4x16x16xf32, #hivm.address_space<ub>>, %arg1: index) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = hivm.hir.local_load ins(%subview : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) -> tensor<16x16xf32>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_hivm_address_space_local_store
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32, 6>,
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.store {{.*}} : tensor<16x16x!tt.ptr<f32, 6>>

module {
  func.func @subview_hivm_address_space_local_store(%arg0: memref<4x16x16xf32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<16x16xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg2 restrict writable : memref<16x16xf32>
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%subview : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %0 : tensor<16x16xf32>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_to_tensor_dynamic_offset
// CHECK-SAME:  %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i64, 6>,
// CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %[[OFFSET:.*]] = arith.index_cast {{.*}} : index to i32
// CHECK-NEXT:  %[[OFFSET_TENSOR:.*]] = tt.splat %[[OFFSET]] : i32 -> tensor<2xi32>
// CHECK-NEXT:  %[[OFFSETS:.*]] = arith.addi %[[RANGE]], %[[OFFSET_TENSOR]] : tensor<2xi32>
// CHECK-NEXT:  %[[BASE_TENSOR:.*]] = tt.splat %[[BASE]] : !tt.ptr<i64, 6> -> tensor<2x!tt.ptr<i64, 6>>
// CHECK-NEXT:  %[[PTRS:.*]] = tt.addptr %[[BASE_TENSOR]], %[[OFFSETS]] : tensor<2x!tt.ptr<i64, 6>>, tensor<2xi32>
// CHECK-NEXT:  tt.load %[[PTRS]] : tensor<2x!tt.ptr<i64, 6>>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_to_tensor_dynamic_offset(%arg0: memref<8xi64, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<2xi64, #hivm.address_space<ub>>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %offset = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%arg1]
    %subview = memref.subview %arg0[%offset] [2] [1] : memref<8xi64, #hivm.address_space<ub>> to memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<2xi64, #hivm.address_space<ub>>, %0 : tensor<2xi64>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_to_tensor_static_stride
// CHECK-SAME:  %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i64, 6>,
// CHECK-NEXT:  %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %[[STRIDE:.*]] = arith.constant dense<2> : tensor<2xi32>
// CHECK-NEXT:  %[[OFFSETS:.*]] = arith.muli %[[RANGE]], %[[STRIDE]] : tensor<2xi32>
// CHECK-NEXT:  %[[BASE_TENSOR:.*]] = tt.splat %[[BASE]] : !tt.ptr<i64, 6> -> tensor<2x!tt.ptr<i64, 6>>
// CHECK-NEXT:  %[[PTRS:.*]] = tt.addptr %[[BASE_TENSOR]], %[[OFFSETS]] : tensor<2x!tt.ptr<i64, 6>>, tensor<2xi32>
// CHECK-NEXT:  tt.load %[[PTRS]] : tensor<2x!tt.ptr<i64, 6>>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_to_tensor_static_stride(%arg0: memref<8xi64, #hivm.address_space<ub>>, %arg1: memref<2xi64, #hivm.address_space<ub>>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %subview = memref.subview %arg0[0] [2] [2] : memref<8xi64, #hivm.address_space<ub>> to memref<2xi64, strided<[2]>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2xi64, strided<[2]>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg1 : memref<2xi64, #hivm.address_space<ub>>, %0 : tensor<2xi64>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_to_tensor_dynamic_offset_2d
// CHECK-SAME:  %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i64, 6>,
// CHECK:       %{{.*}} = arith.muli {{.*}} : index
// CHECK-NEXT:  %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %{{.*}} = tt.reshape {{.*}} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-NEXT:  %{{.*}} = arith.constant dense<8> : tensor<2x1xi32>
// CHECK-NEXT:  %{{.*}} = arith.muli {{.*}} : tensor<2x1xi32>
// CHECK-NEXT:  %{{.*}} = arith.index_cast {{.*}} : index to i32
// CHECK:       %{{.*}} = arith.addi {{.*}} : tensor<2x1xi32>
// CHECK-NEXT:  %{{.*}} = tt.splat %[[BASE]] : !tt.ptr<i64, 6> -> tensor<2x1x!tt.ptr<i64, 6>>
// CHECK-NEXT:  %[[PTRS:.*]] = tt.addptr {{.*}} : tensor<2x1x!tt.ptr<i64, 6>>, tensor<2x1xi32>
// CHECK:       tt.load {{.*}} : tensor<2x8x!tt.ptr<i64, 6>>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_to_tensor_dynamic_offset_2d(%arg0: memref<4x8xi64, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<2x8xi64, #hivm.address_space<ub>>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %subview = memref.subview %arg0[%arg1, 0] [2, 8] [1, 1] : memref<4x8xi64, #hivm.address_space<ub>> to memref<2x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<2x8xi64, #hivm.address_space<ub>>, %0 : tensor<2x8xi64>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_to_tensor_dynamic_offset_rank_reduced
// CHECK:       %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:  %{{.*}} = arith.muli {{.*}} : index
// CHECK-NEXT:  %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %{{.*}} = tt.reshape {{.*}} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-NEXT:  %{{.*}} = arith.constant dense<8> : tensor<2x1xi32>
// CHECK:       %{{.*}} = tt.load {{.*}} : tensor<2x8x!tt.ptr<i64, 6>>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_to_tensor_dynamic_offset_rank_reduced(%arg0: memref<4x2x8xi64, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<2x8xi64, #hivm.address_space<ub>>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 2, 8] [1, 1, 1] : memref<4x2x8xi64, #hivm.address_space<ub>> to memref<2x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<2x8xi64, #hivm.address_space<ub>>, %0 : tensor<2x8xi64>)
    return
  }
}

// -----

// Test that memref.load consuming a memref.subview result is correctly
// converted without leaving irreconcilable unrealized_conversion_cast ops.
// The subview offset (0) and stride (1) should be composed into the pointer
// arithmetic, and the base pointer should come directly from the function
// argument (after Stage 2 expansion).

// CHECK-LABEL: tt.func @subview_memref_load_scalar
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16, 6>,
// CHECK:       %[[C0:[^ ]+]] = arith.constant 0 : i64
// CHECK:       %[[C1:[^ ]+]] = arith.constant 1 : i64
// CHECK:       %[[OFF0:[^ ]+]] = arith.muli %[[C0]], %[[C1]] : i64
// CHECK:       %[[C0_2:[^ ]+]] = arith.constant 0 : i64
// CHECK:       %[[C1_2:[^ ]+]] = arith.constant 1 : i64
// CHECK:       %[[OFF1:[^ ]+]] = arith.muli %[[C0_2]], %[[C1_2]] : i64
// CHECK:       %[[TOTAL_OFF:[^ ]+]] = arith.addi %[[OFF0]], %[[OFF1]] : i64
// CHECK:       %[[ADDR:[^ ]+]] = tt.addptr %[[ARG0]], %[[TOTAL_OFF]] : !tt.ptr<f16, 6>, i64
// CHECK:       %[[VAL:[^ ]+]] = tt.load %[[ADDR]] : !tt.ptr<f16, 6>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hfusion.disableHfusionVectorize, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @subview_memref_load_scalar(%arg0: memref<4xf16, #hivm.address_space<ub>>, %arg1: memref<128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, memref_attr = array<i32: 0, 1>, no_inline, outline} {
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[%c0] [1] [1] : memref<4xf16, #hivm.address_space<ub>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %val = memref.load %subview[%c0] : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %empty = tensor.empty() : tensor<128xf16>
    %bcast = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%val : f16) outs(%empty : tensor<128xf16>) -> tensor<128xf16>
    hivm.hir.local_store ins(%arg1 : memref<128xf16, #hivm.address_space<ub>>, %bcast : tensor<128xf16>)
    return
  }
}

// -----

// Test that memref.load consuming a memref.subview with dynamic offset is
// correctly converted. The dynamic subview offset should be multiplied by the
// source stride and added to the pointer arithmetic.

// CHECK-LABEL: tt.func @subview_memref_load_dynamic_offset
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16, 6>,
// CHECK:       %[[IDX:[^ ]+]] = arith.index_cast {{.*}} : index to i64
// CHECK:       %[[C1:[^ ]+]] = arith.constant 1 : i64
// CHECK:       %[[OFF0:[^ ]+]] = arith.muli %[[IDX]], %[[C1]] : i64
// CHECK:       %[[C0:[^ ]+]] = arith.constant 0 : i64
// CHECK:       %[[C1_2:[^ ]+]] = arith.constant 1 : i64
// CHECK:       %[[OFF1:[^ ]+]] = arith.muli %[[C0]], %[[C1_2]] : i64
// CHECK:       %[[TOTAL_OFF:[^ ]+]] = arith.addi %[[OFF0]], %[[OFF1]] : i64
// CHECK:       %[[ADDR:[^ ]+]] = tt.addptr %[[ARG0]], %[[TOTAL_OFF]] : !tt.ptr<f16, 6>, i64
// CHECK:       %[[VAL:[^ ]+]] = tt.load %[[ADDR]] : !tt.ptr<f16, 6>
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hfusion.disableHfusionVectorize, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @subview_memref_load_dynamic_offset(%arg0: memref<4xf16, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, memref_attr = array<i32: 0, 1>, no_inline, outline} {
    %subview = memref.subview %arg0[%arg1] [1] [1] : memref<4xf16, #hivm.address_space<ub>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : index
    %val = memref.load %subview[%c0] : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %empty = tensor.empty() : tensor<128xf16>
    %bcast = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%val : f16) outs(%empty : tensor<128xf16>) -> tensor<128xf16>
    hivm.hir.local_store ins(%arg2 : memref<128xf16, #hivm.address_space<ub>>, %bcast : tensor<128xf16>)
    return
  }
}
