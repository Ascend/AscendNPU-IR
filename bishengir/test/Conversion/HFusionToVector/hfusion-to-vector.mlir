// RUN: bishengir-opt -convert-hfusion-to-vector %s --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @triton_interleave_5_3_340_outlined_vf_0
#map = affine_map<(d0) -> (64, -d0 + 340)>
#map1 = affine_map<()[s0] -> (s0 floordiv 2)>
module{
func.func @triton_interleave_5_3_340_outlined_vf_0(%arg0: tensor<5x3x170xi32>, %arg1: tensor<5x3x170xi32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: tensor<5x3x340xi32>) -> tensor<5x3x340xi32> attributes {hivm.vector_function} {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0_1 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c340 = arith.constant 340 : index
    %c0_2 = arith.constant 0 : index
    %0 = scf.for %arg12 = %c0 to %c5 step %c1 iter_args(%arg13 = %arg11) -> (tensor<5x3x340xi32>) {
      %1 = scf.for %arg14 = %c0_1 to %c3 step %c1_0 iter_args(%arg15 = %arg13) -> (tensor<5x3x340xi32>) {
        %2 = scf.for %arg16 = %c0_2 to %c340 step %c64 iter_args(%arg17 = %arg15) -> (tensor<5x3x340xi32>) {
          %3 = affine.min #map(%arg16)
          %4 = affine.apply #map1()[%3]
          %5 = affine.apply #map1()[%arg16]
          %extracted_slice = tensor.extract_slice %arg0[%arg12, %arg14, %5] [1, 1, %4] [1, 1, 1] : tensor<5x3x170xi32> to tensor<1x1x?xi32>
          %extracted_slice_3 = tensor.extract_slice %arg1[%arg12, %arg14, %5] [1, 1, %4] [1, 1, 1] : tensor<5x3x170xi32> to tensor<1x1x?xi32>
          // CHECK: %[[arg_slice0:.*]] = tensor.extract_slice %{{.*}} : tensor<5x3x170xi32> to tensor<1x1x?xi32>
          // CHECK: %[[arg_slice1:.*]] = tensor.extract_slice %{{.*}} : tensor<5x3x170xi32> to tensor<1x1x?xi32>
          // CHECK: %[[mask0:.*]] = vector.create_mask %[[len0:.*]] : vector<64xi1>
          // CHECK: %[[mask1:.*]] = vector.create_mask %[[len1:.*]] : vector<64xi1>
          // CHECK: %[[v0:.*]] = vector.mask %[[mask0]] { vector.transfer_read %[[arg_slice0]][%{{.*}}], %{{.*}} : tensor<1x1x?xi32>, vector<64xi32> } : vector<64xi1> -> vector<64xi32>
          // CHECK: %[[v1:.*]] = vector.mask %[[mask1]] { vector.transfer_read %[[arg_slice1]][%{{.*}}], %{{.*}} : tensor<1x1x?xi32>, vector<64xi32> } : vector<64xi1> -> vector<64xi32>
          // CHECK: %[[res1:.*]] = vector.interleave %[[v0]], %[[v1]] : vector<64xi32> -> vector<128xi32>
          // CHECK: %[[res2:.*]] = builtin.unrealized_conversion_cast %[[res1]] : vector<128xi32> to vector<64xi32>
          %6 = hfusion.interleave %extracted_slice, %extracted_slice_3 {"hfusion-auto-vectorize-target-1"} : tensor<1x1x?xi32>, tensor<1x1x?xi32> -> tensor<1x1x?xi32>
          // CHECK: %[[mul_len:.*]] = arith.muli %[[len0]], %{{.*}} : index
          // CHECK: %[[mask2:.*]] = vector.create_mask %[[mul_len]] : vector<64xi1>
          // CHECK: %[[write_op:.*]] = vector.mask %[[mask2]] { vector.transfer_write %[[res2]], %{{.*}}[%{{.*}}] : vector<64xi32>, tensor<5x3x340xi32> } : vector<64xi1> -> tensor<5x3x340xi32>
          %inserted_slice = tensor.insert_slice %6 into %arg17[%arg12, %arg14, %arg16] [1, 1, %3] [1, 1, 1] : tensor<1x1x?xi32> into tensor<5x3x340xi32>
          scf.yield %inserted_slice : tensor<5x3x340xi32>
        }
        scf.yield %2 : tensor<5x3x340xi32>
      }
      scf.yield %1 : tensor<5x3x340xi32>
    }
    return %0 : tensor<5x3x340xi32>
  }
}


// -----

// CHECK-LABEL: func.func @triton_split_16_16_2_outlined_vf_with_vf_merge
#map = affine_map<(d0) -> (128, -d0 + 2)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0) -> (128, -d0 + 1)>
#map3 = affine_map<()[s0] -> (s0 * 2)>
module {
  func.func @triton_split_16_16_2_outlined_vf_with_vf_merge(%arg0: tensor<16x16x2xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: tensor<16x16x1xi8>) -> tensor<16x16x1xi8> attributes {hivm.vector_function} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c16_1 = arith.constant 16 : index
    %c0_2 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %0 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %arg10) -> (tensor<16x16x1xi8>) {
      %1 = scf.for %arg13 = %c0_2 to %c16_1 step %c1_0 iter_args(%arg14 = %arg12) -> (tensor<16x16x1xi8>) {
        %2 = scf.for %arg15 = %c0_4 to %c1_3 step %c128 iter_args(%arg16 = %arg14) -> (tensor<16x16x1xi8>) {
          %3 = affine.min #map2(%arg15)
          %4 = affine.apply #map1(%3)
          %5 = affine.apply #map1(%3)
          %6 = affine.apply #map1(%3)
          %7 = affine.apply #map3()[%3]
          %8 = affine.apply #map3()[%arg15]
          %extracted_slice = tensor.extract_slice %arg0[%arg11, %arg13, %8] [1, 1, %7] [1, 1, 1] : tensor<16x16x2xf16> to tensor<1x1x?xf16>
          // CHECK: %[[min_val:.*]] = arith.minsi %{{.*}} : index
          // CHECK: %[[mask_a:.*]] = vector.create_mask %[[min_val]] : vector<128xi1>
          // CHECK: %[[vec_a:.*]] = vector.mask %[[mask_a]] { vector.transfer_read %[[slice_f:.*]][%{{.*}}], %{{.*}} : tensor<1x1x?xf16>, vector<128xf16> } : vector<128xi1> -> vector<128xf16>
          // CHECK: %[[sub:.*]] = arith.subi %{{.*}}, %{{.*}} : index
          // CHECK: %[[tail:.*]] = arith.maxsi %[[sub]], %{{.*}} : index
          // CHECK: %[[mask_b:.*]] = vector.create_mask %[[tail]] : vector<128xi1>
          // CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[tail]], %{{.*}} : index
          // CHECK: %[[sel:.*]] = arith.select %[[cmp]], %{{.*}}, %{{.*}} : index
          // CHECK: %[[vec_b:.*]] = vector.mask %[[mask_b]] { vector.transfer_read %[[slice_f]][%{{.*}}], %{{.*}} : tensor<1x1x?xf16>, vector<128xf16> } : vector<128xi1> -> vector<128xf16>
          // CHECK: %[[vec_ab:.*]] = builtin.unrealized_conversion_cast %[[vec_a]], %[[vec_b]] : vector<128xf16>, vector<128xf16> to vector<256xf16>
          // CHECK: %[[vd0:.*]], %[[vd1:.*]] = vector.deinterleave %[[vec_ab]] : vector<256xf16> -> vector<128xf16>
          %9 = hfusion.deinterleave %extracted_slice channel<1> {"hfusion-auto-vectorize-target-5"} : tensor<1x1x?xf16> -> tensor<1x1x?xf16>
          // CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[vd1]] : vector<128xf16> to vector<1x1x128xf16>
          // CHECK: %[[half:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
          // CHECK: %[[elem_mask:.*]] = vector.create_mask %{{.*}}, %{{.*}}, %[[half]] : vector<1x1x128xi1>
          // CHECK: %[[empty:.*]] = tensor.empty(%[[half]]) : tensor<1x1x?xf16>
          // CHECK: %[[write_vr:.*]] = vector.mask %[[elem_mask]] { vector.transfer_write %[[shape_cast]], %[[empty]][%{{.*}}, %{{.*}}, %{{.*}}] {in_bounds = [true, true, false]} : vector<1x1x128xf16>, tensor<1x1x?xf16> } : vector<1x1x128xi1> -> tensor<1x1x?xf16>
          %extracted_slice_5 = tensor.extract_slice %arg16[%arg11, %arg13, %arg15] [1, 1, %3] [1, 1, 1] : tensor<16x16x1xi8> to tensor<1x1x?xi8>
          %c1_6 = arith.constant 1 : index
          %c1_7 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %dim = tensor.dim %9, %c2 : tensor<1x1x?xf16>
          %c0_8 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f16
          %10 = vector.create_mask %c1_6, %c1_7, %dim : vector<1x1x128xi1>
          %11 = vector.mask %10 { vector.transfer_read %9[%c0_8, %c0_8, %c0_8], %cst {in_bounds = [true, true, true]} : tensor<1x1x?xf16>, vector<1x1x128xf16> } : vector<1x1x128xi1> -> vector<1x1x128xf16>
          %c0_i8 = arith.constant 0 : i8
          %12 = vector.mask %10 { vector.transfer_read %extracted_slice_5[%c0_8, %c0_8, %c0_8], %c0_i8 {in_bounds = [true, true, true]} : tensor<1x1x?xi8>, vector<1x1x128xi8> } : vector<1x1x128xi1> -> vector<1x1x128xi8>
          %13 = arith.fptosi %11 {round_mode = #hfusion.round_mode<trunc>} : vector<1x1x128xf16> to vector<1x1x128xi8>
          %c0_9 = arith.constant 0 : index
          %14 = vector.mask %10 { vector.transfer_write %13, %extracted_slice_5[%c0_9, %c0_9, %c0_9] {in_bounds = [true, true, true]} : vector<1x1x128xi8>, tensor<1x1x?xi8> } : vector<1x1x128xi1> -> tensor<1x1x?xi8>
          %15 = affine.apply #map1(%3)
          %16 = affine.apply #map1(%3)
          %inserted_slice = tensor.insert_slice %14 into %arg16[%arg11, %arg13, %arg15] [1, 1, %3] [1, 1, 1] : tensor<1x1x?xi8> into tensor<16x16x1xi8>
          scf.yield %inserted_slice : tensor<16x16x1xi8>
        }
        scf.yield %2 : tensor<16x16x1xi8>
      }
      scf.yield %1 : tensor<16x16x1xi8>
    }
    return %0 : tensor<16x16x1xi8>
  }
}

// -----

// CHECK-LABEL: func.func @triton_flip
module {
  func.func @triton_flip(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [7, 31, 32], strides: [992, 32, 1] : memref<?xi32> to memref<7x31x32xi32, strided<[992, 32, 1]>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1], [2]] : memref<7x31x32xi32, strided<[992, 32, 1]>> into memref<217x32xi32, strided<[32, 1]>>
    %alloc = memref.alloc() : memref<217x32xi32>
    memref.copy %collapse_shape, %alloc : memref<217x32xi32, strided<[32, 1]>> to memref<217x32xi32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<217x32xi32>
    %1 = hfusion.flip %0 : tensor<217x32xi32> flip_axis = 2 -> tensor<217x32xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [7, 31, 32], strides: [992, 32, 1] : memref<?xi32> to memref<7x31x32xi32, strided<[992, 32, 1]>>
    %collapse_shape_1 = memref.collapse_shape %reinterpret_cast_0 [[0, 1], [2]] : memref<7x31x32xi32, strided<[992, 32, 1]>> into memref<217x32xi32, strided<[32, 1]>>
    bufferization.materialize_in_destination %1 in writable %collapse_shape_1 : (tensor<217x32xi32>, memref<217x32xi32, strided<[32, 1]>>) -> ()
    return
  }
}
