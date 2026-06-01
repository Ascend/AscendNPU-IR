// RUN: bishengir-opt --tree-reduce-v2="enable-ar" %s | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  
  func.func @triton_sum_2D_dim0_outlined_vf_0(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<384xf32>) -> tensor<384xf32> attributes {hivm.vector_function, no_inline} {
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %c384 step %c64 iter_args(%arg5 = %arg3) -> (tensor<384xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg4] [64] [1] : tensor<384xf32> to tensor<64xf32>
      %c64_0 = arith.constant 64 : index
      %c0_1 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %1 = vector.transfer_read %extracted_slice[%c0_1], %cst : tensor<64xf32>, vector<64xf32>
      %c0_2 = arith.constant 0 : index
      %cst_3 = arith.constant dense<0.000000e+00> : vector<64xf32>
      %2 = vector.transfer_write %cst_3, %extracted_slice[%c0_2] : vector<64xf32>, tensor<64xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg5[%arg4] [64] [1] : tensor<64xf32> into tensor<384xf32>
      scf.yield %inserted_slice : tensor<384xf32>
    }
    return %0 : tensor<384xf32>
  }

  // CHECK-LABEL: func.func @triton_sum_2D_dim0_outlined_vf_1
  // CHECK-NOT: vector.multi_reduction <add>
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     arith.divui
  // CHECK:     scf.for
  // CHECK:       vector.create_mask
  // CHECK:       scf.for
  // CHECK:         vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x64xf32> }
  // CHECK:         arith.addf {{.*}} : vector<1x64xf32>
  // CHECK:         vector.mask {{.*}} { vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x?xf32> }
  // CHECK:     scf.for
  // CHECK:       vector.create_mask
  // CHECK:       vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x64xf32> }
  // CHECK:       arith.addf {{.*}} : vector<1x64xf32>
  // CHECK:       vector.shape_cast {{.*}} : vector<1x64xf32> to vector<64xf32>
  // CHECK:       vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<?xf32>, vector<64xf32> }
  // CHECK:       arith.addf {{.*}} : vector<64xf32>
  // CHECK:       vector.mask {{.*}} { vector.transfer_write {{.*}} : vector<64xf32>, tensor<?xf32> }
  // CHECK: return

  func.func @triton_sum_2D_dim0_outlined_vf_1(%arg0: tensor<128x384xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: tensor<384xf32>) -> tensor<384xf32> attributes {hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg7 = %c0 to %c128 step %c1 iter_args(%arg8 = %arg6) -> (tensor<384xf32>) {
      %1 = scf.for %arg9 = %c0 to %c384 step %c64 iter_args(%arg10 = %arg8) -> (tensor<384xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg7, %arg9] [1, 64] [1, 1] : tensor<128x384xf32> to tensor<1x64xf32>
        %extracted_slice_0 = tensor.extract_slice %arg10[%arg9] [64] [1] : tensor<384xf32> to tensor<64xf32>
        %c1_1 = arith.constant 1 : index
        %c64_2 = arith.constant 64 : index
        %c0_3 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %2 = vector.transfer_read %extracted_slice[%c0_3, %c0_3], %cst : tensor<1x64xf32>, vector<1x64xf32>
        %cst_4 = arith.constant 0.000000e+00 : f32
        %3 = vector.transfer_read %extracted_slice_0[%c0_3], %cst_4 : tensor<64xf32>, vector<64xf32>
        
        // Target instruction to be replaced
        %4 = vector.multi_reduction <add>, %2, %3 [0] : vector<1x64xf32> to vector<64xf32>
        
        %c0_5 = arith.constant 0 : index
        %5 = vector.transfer_write %4, %extracted_slice_0[%c0_5] : vector<64xf32>, tensor<64xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg10[%arg9] [64] [1] : tensor<64xf32> into tensor<384xf32>
        scf.yield %inserted_slice : tensor<384xf32>
      }
      scf.yield %1 : tensor<384xf32>
    }
    return %0 : tensor<384xf32>
  }

  func.func @triton_sum_2D_dim0_outlined_vf_2(%arg0: tensor<384xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: tensor<384xf16>) -> tensor<384xf16> attributes {hivm.vector_function, no_inline} {
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %c384 step %c64 iter_args(%arg6 = %arg4) -> (tensor<384xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg5] [64] [1] : tensor<384xf32> to tensor<64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg6[%arg5] [64] [1] : tensor<384xf16> to tensor<64xf16>
      %c64_1 = arith.constant 64 : index
      %c0_2 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %1 = vector.transfer_read %extracted_slice[%c0_2], %cst : tensor<64xf32>, vector<64xf32>
      %cst_3 = arith.constant 0.000000e+00 : f16
      %2 = vector.transfer_read %extracted_slice_0[%c0_2], %cst_3 : tensor<64xf16>, vector<64xf16>
      %3 = arith.truncf %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : vector<64xf32> to vector<64xf16>
      %c0_4 = arith.constant 0 : index
      %4 = vector.transfer_write %3, %extracted_slice_0[%c0_4] : vector<64xf16>, tensor<64xf16>
      %inserted_slice = tensor.insert_slice %4 into %arg6[%arg5] [64] [1] : tensor<64xf16> into tensor<384xf16>
      scf.yield %inserted_slice : tensor<384xf16>
    }
    return %0 : tensor<384xf16>
  }

  func.func @triton_sum_2D_dim0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 384], strides: [384, 1] : memref<?xf32> to memref<128x384xf32, strided<[384, 1]>>
    %alloc = memref.alloc() : memref<128x384xf32>
    memref.copy %reinterpret_cast, %alloc : memref<128x384xf32, strided<[384, 1]>> to memref<128x384xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128x384xf32>
    %1 = tensor.empty() : tensor<384xf32>
    %2 = scf.execute_region -> tensor<384xf32> {
      %6 = func.call @triton_sum_2D_dim0_outlined_vf_0(%c0, %c384, %c64, %1) {hivm.vector_function, no_inline} : (index, index, index, tensor<384xf32>) -> tensor<384xf32>
      scf.yield %6 : tensor<384xf32>
    }
    %3 = scf.execute_region -> tensor<384xf32> {
      %6 = func.call @triton_sum_2D_dim0_outlined_vf_1(%0, %c0, %c384, %c64, %c128, %c1, %2) {hivm.vector_function, no_inline} : (tensor<128x384xf32>, index, index, index, index, index, tensor<384xf32>) -> tensor<384xf32>
      scf.yield %6 : tensor<384xf32>
    }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [384], strides: [1] : memref<?xf16> to memref<384xf16, strided<[1]>>
    %4 = tensor.empty() : tensor<384xf16>
    %5 = scf.execute_region -> tensor<384xf16> {
      %6 = func.call @triton_sum_2D_dim0_outlined_vf_2(%3, %c0, %c384, %c64, %4) {hivm.vector_function, no_inline} : (tensor<384xf32>, index, index, index, tensor<384xf16>) -> tensor<384xf16>
      scf.yield %6 : tensor<384xf16>
    }
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_0 : (tensor<384xf16>, memref<384xf16, strided<[1]>>) -> ()
    return
  }
}

#map = affine_map<(d0) -> (-d0 + 16, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  // CHECK-LABEL: func.func @triton_sum_4D_dimm1_outlined_vf_0
  // CHECK: scf.for
  // CHECK-NOT: scf.for {reductionLoop}
  // CHECK: %[[MASK1:.*]] = vector.create_mask {{.*}} : vector<1x8xi1>
  // CHECK: %[[READ1:.*]] = vector.mask %[[MASK1]] { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x8xf32> } : vector<1x8xi1> -> vector<1x8xf32>
  // CHECK: %[[BCAST1:.*]] = vector.broadcast {{.*}} : f32 to vector<1x8xf32>
  // CHECK: %[[SEL1:.*]] = arith.select %[[MASK1]], %[[READ1]], %[[BCAST1]] : vector<1x8xi1>, vector<1x8xf32>
  // CHECK: %[[MASK2:.*]] = vector.create_mask {{.*}} : vector<1x8xi1>
  // CHECK: %[[READ2:.*]] = vector.mask %[[MASK2]] { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x8xf32> } : vector<1x8xi1> -> vector<1x8xf32>
  // CHECK: %[[BCAST2:.*]] = vector.broadcast {{.*}} : f32 to vector<1x8xf32>
  // CHECK: %[[SEL2:.*]] = arith.select %[[MASK2]], %[[READ2]], %[[BCAST2]] : vector<1x8xi1>, vector<1x8xf32>
  // CHECK: %[[ADDF:.*]] = arith.addf %[[SEL1]], %[[SEL2]] : vector<1x8xf32>
  // CHECK: %[[MREDUCE:.*]] = vector.multi_reduction <add>, %[[ADDF]], {{.*}} {withoutInitMergeOp} [1] : vector<1x8xf32> to vector<1xf32>
  // CHECK: vector.transfer_write %[[MREDUCE]], {{.*}} : vector<1xf32>, tensor<1xf32>
  func.func @triton_sum_4D_dimm1_outlined_vf_0(%arg0: tensor<1x64xf32>, %arg1: tensor<1536x16xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: tensor<1536xf32>) -> tensor<1536xf32> attributes {hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c1536 = arith.constant 1536 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg8 = %c0 to %c1536 step %c1 iter_args(%arg9 = %arg7) -> (tensor<1536xf32>) {
      %extracted_slice = tensor.extract_slice %arg9[%arg8] [1] [1] : tensor<1536xf32> to tensor<1xf32>
      %c1_0 = arith.constant 1 : index
      %c64_1 = arith.constant 64 : index
      %c0_2 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %1 = vector.transfer_read %arg0[%c0_2, %c0_2], %cst : tensor<1x64xf32>, vector<1x64xf32>
      %c0_3 = arith.constant 0 : index
      %cst_4 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
      %2 = vector.transfer_write %cst_4, %arg0[%c0_3, %c0_3] : vector<1x64xf32>, tensor<1x64xf32>
      %3 = scf.for %arg10 = %c0 to %c16 step %c64 iter_args(%arg11 = %2) -> (tensor<1x64xf32>) {
        %8 = affine.min #map(%arg10)
        %extracted_slice_11 = tensor.extract_slice %arg1[%arg8, %arg10] [1, %8] [1, 1] : tensor<1536x16xf32> to tensor<1x?xf32>
        %extracted_slice_12 = tensor.extract_slice %arg11[0, 0] [1, %8] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
        %c1_13 = arith.constant 1 : index
        %c1_14 = arith.constant 1 : index
        %dim = tensor.dim %extracted_slice_11, %c1_14 : tensor<1x?xf32>
        %c0_15 = arith.constant 0 : index
        %cst_16 = arith.constant 0.000000e+00 : f32
        %9 = vector.create_mask %c1_13, %dim : vector<1x64xi1>
        %10 = vector.mask %9 { vector.transfer_read %extracted_slice_11[%c0_15, %c0_15], %cst_16 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %cst_17 = arith.constant 0.000000e+00 : f32
        %11 = vector.mask %9 { vector.transfer_read %extracted_slice_12[%c0_15, %c0_15], %cst_17 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %12 = arith.addf %10, %11 {reductionOp} : vector<1x64xf32>
        %c0_18 = arith.constant 0 : index
        %13 = vector.mask %9 { vector.transfer_write %12, %extracted_slice_12[%c0_18, %c0_18] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted_slice_19 = tensor.insert_slice %13 into %arg11[0, 0] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x64xf32>
        scf.yield %inserted_slice_19 : tensor<1x64xf32>
      } {reductionLoop}
      %c1_5 = arith.constant 1 : index
      %c64_6 = arith.constant 64 : index
      %c0_7 = arith.constant 0 : index
      %cst_8 = arith.constant 0.000000e+00 : f32
      %4 = vector.transfer_read %3[%c0_7, %c0_7], %cst_8 : tensor<1x64xf32>, vector<1x64xf32>
      %cst_9 = arith.constant 0.000000e+00 : f32
      %5 = vector.transfer_read %extracted_slice[%c0_7], %cst_9 : tensor<1xf32>, vector<1xf32>
      %6 = vector.multi_reduction <add>, %4, %5 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %c0_10 = arith.constant 0 : index
      %7 = vector.transfer_write %6, %extracted_slice[%c0_10] : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %7 into %arg9[%arg8] [1] [1] : tensor<1xf32> into tensor<1536xf32>
      scf.yield %inserted_slice : tensor<1536xf32>
    }
    return %0 : tensor<1536xf32>
  }
  // CHECK-LABEL: func.func @triton_sum_4D_dimm1
  func.func @triton_sum_4D_dimm1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1536 = arith.constant 1536 : index
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16, 6, 16], strides: [1536, 96, 16, 1] : memref<?xf32> to memref<16x16x6x16xf32, strided<[1536, 96, 16, 1]>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1, 2], [3]] : memref<16x16x6x16xf32, strided<[1536, 96, 16, 1]>> into memref<1536x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<1536x16xf32>
    memref.copy %collapse_shape, %alloc : memref<1536x16xf32, strided<[16, 1]>> to memref<1536x16xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1536x16xf32>
    %1 = tensor.empty() : tensor<1536xf32>
    %2 = tensor.empty() : tensor<1x64xf32>
    %3 = scf.execute_region -> tensor<1536xf32> {
      %4 = func.call @triton_sum_4D_dimm1_outlined_vf_0(%2, %0, %c0, %c16, %c64, %c1536, %c1, %1) {hivm.vector_function, no_inline} : (tensor<1x64xf32>, tensor<1536x16xf32>, index, index, index, index, index, tensor<1536xf32>) -> tensor<1536xf32>
      scf.yield %4 : tensor<1536xf32>
    }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16, 6], strides: [96, 6, 1] : memref<?xf32> to memref<16x16x6xf32, strided<[96, 6, 1]>>
    %collapse_shape_1 = memref.collapse_shape %reinterpret_cast_0 [[0, 1, 2]] : memref<16x16x6xf32, strided<[96, 6, 1]>> into memref<1536xf32, strided<[1]>>
    bufferization.materialize_in_destination %3 in writable %collapse_shape_1 : (tensor<1536xf32>, memref<1536xf32, strided<[1]>>) -> ()
    return
  }
}
