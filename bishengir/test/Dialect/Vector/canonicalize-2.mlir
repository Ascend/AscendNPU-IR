// RUN: bishengir-opt -allow-unregistered-dialect -cse -canonicalize -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @check_vector_masc_create_and_for_elimination
func.func @check_vector_masc_create_and_for_elimination(%arg0: tensor<38x16xf32>, %arg1: tensor<38x16xf32>, %arg2: tensor<16x38x16xf32>, %arg3: tensor<38x16xf32>) -> tensor<38x16xf32> attributes {hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x1x64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c38 = arith.constant 38 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index

  // CHECK:     scf.for
  // CHECK-NOT: scf.for
  // CHECK-LABEL: -> (tensor<38x16xf32>) {
  %0 = scf.for %arg4 = %c0 to %c38 step %c1 iter_args(%arg5 = %arg3) -> (tensor<38x16xf32>) {
    %1 = scf.for %arg6 = %c0 to %c16 step %c64 iter_args(%arg7 = %arg5) -> (tensor<38x16xf32>) {
      // CHECK-NOT:  affine.min
      %2 = affine.min affine_map<(d0) -> (-d0 + 16, 64)>(%arg6)
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg6] [1, %2] [1, 1] : tensor<38x16xf32> to tensor<1x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, %arg6] [1, %2] [1, 1] : tensor<38x16xf32> to tensor<1x?xf32>
      // CHECK:      %[[V1:[A-Za-z0-9_]+]] = tensor.empty() : tensor<1x1x16xf32>
      // CHECK-NEXT: %[[V2:[A-Za-z0-9_]+]] = vector.constant_mask [1, 1, 16] : vector<1x1x64xi1>
      // CHECK-NEXT: %[[V3:[A-Za-z0-9_]+]] = vector.transfer_write %cst, %[[V1]][%[[C0:[A-Za-z0-9_]+]], %[[C0]], %[[C0]]], %[[V2]] {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x16xf32>

      %3 = scf.for %arg8 = %c0 to %c1 step %c1 iter_args(%arg9 = %extracted_slice_1) -> (tensor<1x?xf32>) {
        %9 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg8, %arg4]
        %10 = scf.for %arg10 = %c0 to %2 step %c64 iter_args(%arg11 = %arg9) -> (tensor<1x?xf32>) {
          // eCHECK-NOT:  affine.min
          %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg10)[%2]
          %extracted_slice_3 = tensor.extract_slice %arg11[%arg8, %arg10] [1, %11] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
          %12 = tensor.empty(%11) : tensor<1x1x?xf32>
          %dim = tensor.dim %12, %c2 : tensor<1x1x?xf32>
          %13 = vector.create_mask %c1, %c1, %dim : vector<1x1x64xi1>
          // CHECK-NOT: vector.create_mask
          %14 = vector.transfer_write %cst, %12[%c0, %c0, %c0], %13 {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x?xf32>
          %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg10, %arg6]

          // CHECK:     scf.for
          // CHECK-NOT: scf.for
          // CHECK-LABEL: -> (tensor<1x1x?xf32>) {
          %16 = scf.for %arg12 = %c0 to %c16 step %c1 iter_args(%arg13 = %14) -> (tensor<1x1x?xf32>) {
            %extracted_slice_6 = tensor.extract_slice %arg2[%arg12, %9, %15] [1, 1, %11] [1, 1, 1] : tensor<16x38x16xf32> to tensor<1x1x?xf32>
            %extracted_slice_7 = tensor.extract_slice %arg13[0, 0, 0] [1, 1, %11] [1, 1, 1] : tensor<1x1x?xf32> to tensor<1x1x?xf32>
            %23 = vector.create_mask %c1, %c1, %11 : vector<1x1x64xi1>
            // CHECK-NOT: vector.create_mask
            // CHECK-NOT: scf.for
            // CHECK:     vector.constant_mask [1, 1, 16] : vector<1x1x64xi1>
            %24 = vector.transfer_read %extracted_slice_6[%c0, %c0, %c0], %cst_0, %23 {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32>
            %25 = vector.transfer_read %extracted_slice_7[%c0, %c0, %c0], %cst_0, %23 {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32>
            %26 = arith.addf %24, %25 {reductionOp} : vector<1x1x64xf32>
            %27 = vector.transfer_write %26, %extracted_slice_7[%c0, %c0, %c0], %23 {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x?xf32>
            %inserted_slice_8 = tensor.insert_slice %27 into %arg13[0, 0, 0] [1, 1, %11] [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x1x?xf32>
            scf.yield %inserted_slice_8 : tensor<1x1x?xf32>
          } {reductionLoop}
          %dim_4 = tensor.dim %16, %c2 : tensor<1x1x?xf32>
          %17 = vector.create_mask %c1, %c1, %dim_4 : vector<1x1x64xi1>
          %18 = vector.transfer_read %16[%c0, %c0, %c0], %cst_0, %17 {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32>
          %19 = vector.create_mask %c1, %dim_4 : vector<1x64xi1>
          %20 = vector.transfer_read %extracted_slice_3[%c0, %c0], %cst_0, %19 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
          %21 = vector.mask %17 { vector.multi_reduction <add>, %18, %20 {withoutInitMergeOp} [0] : vector<1x1x64xf32> to vector<1x64xf32> } : vector<1x1x64xi1> -> vector<1x64xf32>
          %22 = vector.transfer_write %21, %extracted_slice_3[%c0, %c0], %19 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
          %inserted_slice_5 = tensor.insert_slice %22 into %arg11[%arg8, %arg10] [1, %11] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
          scf.yield %inserted_slice_5 : tensor<1x?xf32>
        }
        scf.yield %10 : tensor<1x?xf32>
      }
      %extracted_slice_2 = tensor.extract_slice %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<38x16xf32> to tensor<1x?xf32>
      %4 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %5 = vector.transfer_read %extracted_slice[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
      %6 = vector.transfer_read %3[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
      %7 = arith.addf %5, %6 : vector<1x64xf32>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0, %c0], %4 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<1x?xf32> into tensor<38x16xf32>
      scf.yield %inserted_slice : tensor<38x16xf32>
    }
    scf.yield %1 : tensor<38x16xf32>
  }
  return %0 : tensor<38x16xf32>
}
