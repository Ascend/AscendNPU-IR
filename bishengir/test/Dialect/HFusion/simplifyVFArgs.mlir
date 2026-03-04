// RUN: bishengir-opt -hfusion-simplify-vf-arg %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (64, -d0 + s0)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<()[s0] -> (1024 ceildiv s0)>
#map3 = affine_map<()[s0, s1] -> (s0 * s1)>
#map4 = affine_map<()[s0, s1] -> (-(s1 * s0) + 1024, s0)>
module {
  // CHECK: func.func @test_tile_first_reduce_mid_outlined_vf_0(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>)
  func.func @test_tile_first_reduce_mid_outlined_vf_0(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: index, %arg4: index, %arg5: tensor<?xf32>) -> tensor<?xf32> attributes {hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg6 = %c0 to %arg0 step %c64 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
      %1 = affine.min #map(%arg6)[%arg0]
      %2 = affine.apply #map1(%1)
      %3 = affine.apply #map1(%1)
      %4 = affine.apply #map1(%1)
      %5 = affine.apply #map1(%1)
      %extracted_slice = tensor.extract_slice %arg1[%arg6] [%1] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg6] [%1] [1] : tensor<?xf32> to tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%arg6] [%1] [1] : tensor<?xf32> to tensor<?xf32>
      %c0_2 = arith.constant 0 : index
      %dim = tensor.dim %extracted_slice, %c0_2 : tensor<?xf32>
      %c0_3 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %6 = vector.create_mask %dim : vector<64xi1>
      %7 = vector.mask %6 { vector.transfer_read %extracted_slice[%c0_3], %cst {in_bounds = [true]} : tensor<?xf32>, vector<64xf32> } : vector<64xi1> -> vector<64xf32>
      %cst_4 = arith.constant 0.000000e+00 : f32
      %8 = vector.mask %6 { vector.transfer_read %extracted_slice_0[%c0_3], %cst_4 {in_bounds = [true]} : tensor<?xf32>, vector<64xf32> } : vector<64xi1> -> vector<64xf32>
      %cst_5 = arith.constant 0.000000e+00 : f32
      %9 = vector.mask %6 { vector.transfer_read %extracted_slice_1[%c0_3], %cst_5 {in_bounds = [true]} : tensor<?xf32>, vector<64xf32> } : vector<64xi1> -> vector<64xf32>
      %10 = arith.addf %7, %8 : vector<64xf32>
      %c0_6 = arith.constant 0 : index
      %11 = vector.mask %6 { vector.transfer_write %10, %extracted_slice_1[%c0_6] {in_bounds = [true]} : vector<64xf32>, tensor<?xf32> } : vector<64xi1> -> tensor<?xf32>
      %12 = affine.apply #map1(%1)
      %13 = affine.apply #map1(%1)
      %inserted_slice = tensor.insert_slice %11 into %arg7[%arg6] [%1] [1] : tensor<?xf32> into tensor<?xf32>
      scf.yield %inserted_slice : tensor<?xf32>
    }
    return %0 : tensor<?xf32>
  }
  func.func @test_tile_first_reduce_mid(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: i64, %arg4: i64, %arg5: i64) -> tensor<1024xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = arith.index_cast %arg4 : i64 to index
    %2 = affine.apply #map2()[%1]
    %3 = scf.for %arg6 = %c0 to %2 step %c1 iter_args(%arg7 = %arg2) -> (tensor<1024xf32>) {
      %4 = affine.apply #map3()[%arg6, %1]
      %5 = affine.min #map4()[%1, %arg6]
      %extracted_slice = tensor.extract_slice %arg0[%4] [%5] [1] : tensor<1024xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %0[%4] [%5] [1] : tensor<1024xf32> to tensor<?xf32>
      %6 = hfusion.load {__intermediate_producer__} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?xf32>) -> tensor<?xf32>
      annotation.mark %6 {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
      annotation.mark %6 {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%4] [%5] [1] : tensor<1024xf32> to tensor<?xf32>
      %7 = hfusion.load {__intermediate_producer__} ins(%extracted_slice_1 : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?xf32>) -> tensor<?xf32>
      annotation.mark %7 {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
      annotation.mark %7 {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
      %8 = tensor.empty(%5) : tensor<?xf32>
      %c0_2 = arith.constant 0 : index
      %dim = tensor.dim %6, %c0_2 : tensor<?xf32>
      %c0_3 = arith.constant 0 : index
      %dim_4 = tensor.dim %7, %c0_3 : tensor<?xf32>
      %c0_5 = arith.constant 0 : index
      %dim_6 = tensor.dim %8, %c0_5 : tensor<?xf32>
      %c0_7 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %9 = scf.execute_region -> tensor<?xf32> {
        %11 = func.call @test_tile_first_reduce_mid_outlined_vf_0(%dim, %6, %7, %c0_7, %c64, %8) {hivm.vector_function} : (index, tensor<?xf32>, tensor<?xf32>, index, index, tensor<?xf32>) -> tensor<?xf32>
        scf.yield %11 : tensor<?xf32>
      }
      annotation.mark %9 {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
      %extracted_slice_8 = tensor.extract_slice %arg7[%4] [%5] [1] : tensor<1024xf32> to tensor<?xf32>
      %10 = hfusion.store ins(%9 : tensor<?xf32>) outs(%extracted_slice_8 : tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %10 into %arg7[%4] [%5] [1] : tensor<?xf32> into tensor<1024xf32>
      scf.yield %inserted_slice : tensor<1024xf32>
    } {__tiled_for___3}
    return %3 : tensor<1024xf32>
  }
}