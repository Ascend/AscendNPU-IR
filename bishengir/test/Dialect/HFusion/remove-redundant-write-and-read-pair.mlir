// RUN: bishengir-opt %s -remove-redundant-write-and-read-pair -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @remove_redundant_write_and_read_pair(%arg0: tensor<2x80xf16>, %arg1: tensor<2x80xf16>, %arg2: tensor<2x80xf16>, %arg3: tensor<2x80xf16>) -> tensor<2x80xf16> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c80 = arith.constant 80 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> tensor<2x80xf16> {
    %1 = scf.for %arg6 = %c0 to %c80 step %c64 iter_args(%arg7 = %arg5) -> tensor<2x80xf16> {
      %2 = affine.min affine_map<(d0) -> (-d0 + 80, 64)>(%arg6)
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %4 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %5 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %6 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %7 = arith.addf %5, %6 : vector<1x64xf16>
      %8 = vector.transfer_write %7, %extracted_slice_4[%c0, %c0], %4 {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x?xf16>
      %inserted_slice = tensor.insert_slice %8 into %extracted_slice_4[0, 0] [1, %2] [1, 1] : tensor<1x?xf16> into tensor<1x?xf16>
      %12 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %13 = vector.transfer_read %inserted_slice[%c0, %c0], %cst_0, %12 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %20 = arith.subf %13, %5 : vector<1x64xf16>
      %extracted_slice_8 = tensor.extract_slice %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %22 = vector.transfer_write %20, %extracted_slice_8[%c0, %c0], %12 {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x?xf16>
      %inserted_slice_9 = tensor.insert_slice %22 into %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<1x?xf16> into tensor<2x80xf16>
      scf.yield %inserted_slice_9 : tensor<2x80xf16>
    }
    scf.yield %1 : tensor<2x80xf16>
  }
  return %0 : tensor<2x80xf16>
}

// CHECK-LABEL:   func.func @remove_redundant_write_and_read_pair
// CHECK: %[[ADD:.*]] = arith.addf
// CHECK: %[[SUB:.*]] = arith.subf %[[ADD]], %{{.*}}