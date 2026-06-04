// RUN: bishengir-opt %s -cse -canonicalize -scf-for-loop-canonicalization  -one-shot-bufferize='bufferize-function-boundaries' -sccp -cse  -scf-for-loop-canonicalization -canonicalize -drop-equivalent-buffer-results -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_loop_dominance_1d
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
module {
  func.func @test_loop_dominance_1d(%arg0: index, %arg1: tensor<?x128xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<1.000000e-01> : vector<64xf32>
    %cst_1 = arith.constant dense<7.812500e-03> : vector<64xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %arg0 step %c64 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
      %1 = affine.min #map(%arg4)[%arg0]
      %extracted_slice = tensor.extract_slice %arg5[%arg4] [%1] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg1[%arg4, 0] [%1, 128] [1, 1] : tensor<?x128xf32> to tensor<?x128xf32>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg4] [%1] [1] : tensor<?xf32> to tensor<?xf32>
      %2 = scf.for %arg6 = %c0 to %1 step %c1 iter_args(%arg7 = %extracted_slice_4) -> (tensor<?xf32>) {
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg6] [1] [1] : tensor<?xf32> to tensor<1xf32>
        %10 = vector.transfer_read %extracted_slice_5[%c0], %cst_2 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
        %11 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %10) -> (vector<1xf32>) {
          %extracted_slice_7 = tensor.extract_slice %extracted_slice_3[%arg6, %arg8] [1, 64] [1, 1] : tensor<?x128xf32> to tensor<1x64xf32>
          %13 = vector.transfer_read %extracted_slice_7[%c0, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
          %14 = arith.mulf %13, %13 : vector<1x64xf32>
          %15 = vector.multi_reduction <add>, %14, %arg9 [1] : vector<1x64xf32> to vector<1xf32>
          scf.yield %15 : vector<1xf32>
        }
        %12 = vector.transfer_write %11, %extracted_slice_5[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
        %inserted_slice_6 = tensor.insert_slice %12 into %arg7[%arg6] [1] [1] : tensor<1xf32> into tensor<?xf32>
        scf.yield %inserted_slice_6 : tensor<?xf32>
      }
      %3 = vector.create_mask %1 : vector<64xi1>
      %4 = vector.transfer_read %2[%c0], %cst_2, %3 {in_bounds = [true]} : tensor<?xf32>, vector<64xf32>
      %5 = arith.mulf %4, %cst_1 : vector<64xf32>
      %6 = arith.addf %5, %cst_0 : vector<64xf32>
      %7 = math.sqrt %6 : vector<64xf32>
      %8 = arith.divf %cst, %7 : vector<64xf32>
      %9 = vector.transfer_write %8, %extracted_slice[%c0], %3 {in_bounds = [true]} : vector<64xf32>, tensor<?xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg5[%arg4] [%1] [1] : tensor<?xf32> into tensor<?xf32>
      scf.yield %inserted_slice : tensor<?xf32>
    }
    return %0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_loop_dominance_2d
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
module {
    func.func @test_loop_dominance_2d(%arg0: tensor<128x128xf32>, %arg00: tensor<128x128xf32>) -> tensor<128x128xf32> {
        %cst = arith.constant dense<1.000000e+00> : vector<64x64xf32>
        %c128 = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst_2 = arith.constant 0.000000e+00 : f32
        %cst_3 = arith.constant dense<2.000000e+00> : tensor<64x64xf32>
        %empty = tensor.empty() : tensor<64x64xf32>
        %0 = scf.for %arg1 = %c0 to %c128 step %c64 iter_args(%arg2 = %arg0) -> (tensor<128x128xf32>) {
            %1 = scf.for %arg3 = %c0 to %c128 step %c64 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
                %extracted_slice = tensor.extract_slice %arg00[%arg1, %arg3] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
                %extracted_slice_1 = tensor.extract_slice %arg4[%arg1, %arg3] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
                %2 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %extracted_slice) -> (tensor<64x64xf32>) {
                    %3 = vector.transfer_read %arg6[%arg5, %arg5], %cst_2 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
                    %4 = arith.addf %3, %cst : vector<64x64xf32>
                    %5 = vector.transfer_write %4, %arg6[%arg5, %arg5] {in_bounds = [true, true]} : vector<64x64xf32>, tensor<64x64xf32>
                    scf.yield %5 : tensor<64x64xf32>
                }
                %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %cst_3: tensor<64x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_1 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %inserted_slice = tensor.insert_slice %6 into %arg4[%arg1, %arg3] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
                scf.yield %inserted_slice: tensor<128x128xf32>
            }
            scf.yield %1 : tensor<128x128xf32>
        }

        return %0 : tensor<128x128xf32>
    }
}

// CHECK-LABEL: func.func @loop_dominance_2d_with_constant_dim
func.func @loop_dominance_2d_with_constant_dim(%arg0: tensor<2x8x1024xf16>, %arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf16>) -> tensor<2x8xf16> attributes {hivm.vector_function} {
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %0 = vector.constant_mask [1, 8] : vector<1x64xi1>
  %1 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> (tensor<2x8xf16>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0] [1, 8, 1024] [1, 1, 1] : tensor<2x8x1024xf16> to tensor<1x8x1024xf16>
    %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<1x8xf32>
    // CHECK-NOT: memref.alloc
    // CHECK-NOT: memref.copy
    %2 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %extracted_slice_1) -> (tensor<1x8xf32>) {
      %extracted_slice_3 = tensor.extract_slice %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x8xf32> to tensor<1x1xf32>
      %6 = vector.transfer_read %extracted_slice_3[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x1xf32>
      %7 = scf.for %arg7 = %c0 to %c1024 step %c64 iter_args(%arg8 = %6) -> (vector<1x1xf32>) {
        %extracted_slice_5 = tensor.extract_slice %extracted_slice[0, %arg5, %arg7] [1, 1, 64] [1, 1, 1] : tensor<1x8x1024xf16> to tensor<1x1x64xf16>
        %9 = vector.transfer_read %extracted_slice_5[%c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true]} : tensor<1x1x64xf16>, vector<1x1x64xf16>
        %10 = arith.extf %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xf16> to vector<1x1x64xf32>
        %11 = vector.multi_reduction <add>, %10, %arg8 [2] : vector<1x1x64xf32> to vector<1x1xf32>
        scf.yield %11 : vector<1x1xf32>
      }
      %8 = vector.transfer_write %7, %extracted_slice_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<1x1xf32>
      %inserted_slice_4 = tensor.insert_slice %8 into %arg6[0, %arg5] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x8xf32>
      scf.yield %inserted_slice_4 : tensor<1x8xf32>
    }
    %extracted_slice_2 = tensor.extract_slice %arg4[%arg3, 0] [1, 8] [1, 1] : tensor<2x8xf16> to tensor<1x8xf16>
    %3 = vector.transfer_read %2[%c0, %c0], %cst, %0 {in_bounds = [true, true]} : tensor<1x8xf32>, vector<1x64xf32>
    %4 = arith.truncf %3 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
    %5 = vector.transfer_write %4, %extracted_slice_2[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x8xf16>
    %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0] [1, 8] [1, 1] : tensor<1x8xf16> into tensor<2x8xf16>
    scf.yield %inserted_slice : tensor<2x8xf16>
  }
  return %1 : tensor<2x8xf16>
}