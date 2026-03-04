// RUN: bishengir-opt %s -remove-mask-from-unaligned-reduction-loop -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s
 
#map = affine_map<(d0) -> (-d0 + 150, 64)>
module {
  func.func @reduction_has_tail_block(%arg0: tensor<1x64xf32>, %arg1: tensor<300x150xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: tensor<300xf32>) -> tensor<300xf32> attributes {hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c300 = arith.constant 300 : index
    %c64 = arith.constant 64 : index
    %c150 = arith.constant 150 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg8 = %c0 to %c300 step %c1 iter_args(%arg9 = %arg7) -> (tensor<300xf32>) {
      %extracted_slice = tensor.extract_slice %arg9[%arg8] [1] [1] : tensor<300xf32> to tensor<1xf32>
      %1 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %2 = scf.for %arg10 = %c0 to %c150 step %c64 iter_args(%arg11 = %1) -> (tensor<1x64xf32>) {
        %7 = affine.min #map(%arg10)
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg8, %arg10] [1, %7] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
        %extracted_slice_2 = tensor.extract_slice %arg11[0, 0] [1, %7] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
        %8 = vector.create_mask %c1, %7 : vector<1x64xi1>
        %9 = vector.mask %8 { vector.transfer_read %extracted_slice_1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %10 = vector.mask %8 { vector.transfer_read %extracted_slice_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %11 = arith.addf %9, %10 {reductionOp} : vector<1x64xf32>
        %12 = vector.mask %8 { vector.transfer_write %11, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted_slice_3 = tensor.insert_slice %12 into %arg11[0, 0] [1, %7] [1, 1] : tensor<1x?xf32> into tensor<1x64xf32>
        scf.yield %inserted_slice_3 : tensor<1x64xf32>
      } {reductionLoop}
      %3 = vector.transfer_read %2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %4 = vector.transfer_read %extracted_slice[%c0], %cst_0 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %5 = vector.multi_reduction <add>, %3, %4 [1] : vector<1x64xf32> to vector<1xf32>
      %6 = vector.transfer_write %5, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg9[%arg8] [1] [1] : tensor<1xf32> into tensor<300xf32>
      scf.yield %inserted_slice : tensor<300xf32>
    }
    return %0 : tensor<300xf32>
  }
}

// CHECK-LABEL:   func.func @reduction_has_tail_block(
// CHECK: %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
// CHECK-NEXT: %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c300 = arith.constant 300 : index
// CHECK-NEXT: %c64 = arith.constant 64 : index
// CHECK-NEXT: %c150 = arith.constant 150 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %0 = scf.for %arg8 = %c0 to %c300 step %c1 iter_args(%arg9 = %arg7) -> (tensor<300xf32>) {
// CHECK-NEXT:   %extracted_slice = tensor.extract_slice %arg9[%arg8] [1] [1] : tensor<300xf32> to tensor<1xf32>
// CHECK-NEXT:   %1 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
// CHECK-NEXT:   %2 = scf.for %arg10 = %c0 to %c150 step %c64 iter_args(%arg11 = %1) -> (tensor<1x64xf32>) {
// CHECK-NEXT:     %7 = affine.min #map(%arg10)
// CHECK-NEXT:     %extracted_slice_1 = tensor.extract_slice %arg1[%arg8, %arg10] [1, %7] [1, 1] : tensor<300x150xf32> to tensor<1x?xf32>
// CHECK-NEXT:     %extracted_slice_2 = tensor.extract_slice %arg11[0, 0] [1, %7] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
// CHECK-NEXT:     %8 = vector.create_mask %c1, %7 : vector<1x64xi1>
// CHECK-NEXT:     %9 = vector.mask %8 { vector.transfer_read %extracted_slice_1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
// CHECK-NEXT:     %10 = vector.transfer_read %arg11[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK-NEXT:     %11 = arith.select %8, %9, %cst : vector<1x64xi1>, vector<1x64xf32>
// CHECK-NEXT:     %12 = arith.addf %11, %10 {reductionOp} : vector<1x64xf32>
// CHECK-NEXT:     %13 = vector.transfer_write %12, %arg11[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
// CHECK-NEXT:     scf.yield %13 : tensor<1x64xf32>
// CHECK-NEXT:   } {reductionLoop}
// CHECK-NEXT:   %3 = vector.transfer_read %2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK-NEXT:   %4 = vector.transfer_read %extracted_slice[%c0], %cst_0 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
// CHECK-NEXT:   %5 = vector.multi_reduction <add>, %3, %4 [1] : vector<1x64xf32> to vector<1xf32>
// CHECK-NEXT:   %6 = vector.transfer_write %5, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
// CHECK-NEXT:   %inserted_slice = tensor.insert_slice %6 into %arg9[%arg8] [1] [1] : tensor<1xf32> into tensor<300xf32>
// CHECK-NEXT:   scf.yield %inserted_slice : tensor<300xf32>
// CHECK-NEXT: }