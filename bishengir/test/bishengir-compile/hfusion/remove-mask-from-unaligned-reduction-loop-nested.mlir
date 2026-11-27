// RUN: bishengir-opt %s -hfusion-remove-mask-from-unaligned-reduction-loop -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// Two reduction loops, both with a tail block (8183 % 64 != 0), where the
// reduction op of the inner loop is nested inside the outer loop. Collecting
// reduction ops of the outer loop must not pick up the reduction op of the
// inner loop, otherwise matching it against the outer loop fails and the pass
// used to dereference a null transfer_write.

module {
  func.func @nested_reduction_loops_with_tail_block(%arg0: tensor<1x64xf32>, %arg1: tensor<1x8183xf32>, %arg2: tensor<1xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: tensor<1x8183xf32>, %arg7: tensor<1xf32>, %arg8: tensor<1xf32>, %arg9: index, %arg10: tensor<1xf32>, %arg11: tensor<1x8183xf32>) -> (tensor<1xf32>, tensor<1x8183xf32>) attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x64xi32>
    %cst_0 = arith.constant dense<1> : vector<1x64xi32>
    %cst_1 = arith.constant dense<-2139095040> : vector<1x64xi32>
    %cst_2 = arith.constant dense<2147483647> : vector<1x64xi32>
    %cst_3 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c8183 = arith.constant 8183 : index
    %c0 = arith.constant 0 : index
    %0:2 = scf.for %arg12 = %c0 to %c1 step %c1 iter_args(%arg13 = %arg10, %arg14 = %arg11) -> (tensor<1xf32>, tensor<1x8183xf32>) {
      %extracted_slice = tensor.extract_slice %arg13[%arg12] [1] [1] : tensor<1xf32> to tensor<1xf32>
      %1 = vector.transfer_write %cst_5, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %2 = scf.for %arg15 = %c0 to %c8183 step %c64 iter_args(%arg16 = %arg6) -> (tensor<1x8183xf32>) {
        %8 = affine.min affine_map<(d0) -> (-d0 + 8183, 64)>(%arg15)
        %extracted_slice_6 = tensor.extract_slice %arg1[0, %arg15] [1, %8] [1, 1] : tensor<1x8183xf32> to tensor<1x?xf32>
        %extracted_slice_7 = tensor.extract_slice %arg16[0, %arg15] [1, %8] [1, 1] : tensor<1x8183xf32> to tensor<1x?xf32>
        %9 = vector.create_mask %c1, %8 : vector<1x64xi1>
        %10 = vector.mask %9 { vector.transfer_read %extracted_slice_6[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %11 = vector.transfer_read %arg2[%c0], %cst_4 {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (0, 0)>} : tensor<1xf32>, vector<1x64xf32>
        %12 = arith.divf %10, %11 : vector<1x64xf32>
        %13 = vector.mask %9 { vector.transfer_write %12, %extracted_slice_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted_slice_8 = tensor.insert_slice %13 into %arg16[0, %arg15] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x8183xf32>
        scf.yield %inserted_slice_8 : tensor<1x8183xf32>
      }
      %3:2 = scf.for %arg15 = %c0 to %c8183 step %c64 iter_args(%arg16 = %1, %arg17 = %arg14) -> (tensor<1x64xf32>, tensor<1x8183xf32>) {
        %8 = affine.min affine_map<(d0) -> (-d0 + 8183, 64)>(%arg15)
        %extracted_slice_6 = tensor.extract_slice %arg17[%arg12, %arg15] [1, %8] [1, 1] : tensor<1x8183xf32> to tensor<1x?xf32>
        %extracted_slice_7 = tensor.extract_slice %2[%arg12, %arg15] [1, %8] [1, 1] : tensor<1x8183xf32> to tensor<1x?xf32>
        %9 = vector.transfer_write %cst_3, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
        %10 = scf.for %arg18 = %c0 to %c8183 step %c64 iter_args(%arg19 = %9) -> (tensor<1x64xf32>) {
          %26 = affine.min affine_map<(d0) -> (-d0 + 8183, 64)>(%arg18)
          %extracted_slice_12 = tensor.extract_slice %2[0, %arg18] [1, %26] [1, 1] : tensor<1x8183xf32> to tensor<1x?xf32>
          %extracted_slice_13 = tensor.extract_slice %arg19[0, 0] [1, %26] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
          %27 = vector.create_mask %c1, %26 : vector<1x64xi1>
          %28 = vector.mask %27 { vector.transfer_read %extracted_slice_12[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
          %29 = vector.mask %27 { vector.transfer_read %extracted_slice_13[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
          %30 = arith.bitcast %28 : vector<1x64xf32> to vector<1x64xi32>
          %31 = arith.andi %30, %cst_2 : vector<1x64xi32>
          %32 = arith.addi %31, %cst_1 : vector<1x64xi32>
          %33 = arith.minsi %32, %cst_0 : vector<1x64xi32>
          %34 = arith.maxsi %33, %cst : vector<1x64xi32>
          %35 = arith.cmpi ne, %34, %cst : vector<1x64xi32>
          %36 = arith.select %35, %cst_3, %28 : vector<1x64xi1>, vector<1x64xf32>
          %37 = arith.maximumf %36, %29 {reductionOp} : vector<1x64xf32>
          %38 = vector.mask %27 { vector.transfer_write %37, %extracted_slice_13[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
          %inserted_slice_14 = tensor.insert_slice %38 into %arg19[0, 0] [1, %26] [1, 1] : tensor<1x?xf32> into tensor<1x64xf32>
          scf.yield %inserted_slice_14 : tensor<1x64xf32>
        } {reductionLoop}
        %11 = vector.transfer_read %10[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
        %12 = vector.transfer_read %arg7[%c0], %cst_4 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
        %13 = vector.multi_reduction <maximumf>, %11, %12 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
        %14 = vector.transfer_write %13, %arg8[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
        %15 = vector.create_mask %c1, %8 : vector<1x64xi1>
        %16 = vector.mask %15 { vector.transfer_read %extracted_slice_7[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %17 = vector.transfer_read %14[%c0], %cst_4 {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (0, 0)>} : tensor<1xf32>, vector<1x64xf32>
        %18 = arith.subf %16, %17 : vector<1x64xf32>
        %19 = math.exp %18 : vector<1x64xf32>
        %20 = vector.mask %15 { vector.transfer_write %19, %extracted_slice_6[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted_slice_8 = tensor.insert_slice %20 into %extracted_slice_6[0, 0] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
        %extracted_slice_9 = tensor.extract_slice %arg16[0, 0] [1, %8] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
        %dim = tensor.dim %inserted_slice_8, %c1 : tensor<1x?xf32>
        %21 = vector.create_mask %c1, %dim : vector<1x64xi1>
        %22 = vector.mask %21 { vector.transfer_read %inserted_slice_8[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %23 = vector.mask %21 { vector.transfer_read %extracted_slice_9[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %24 = arith.addf %22, %23 {reductionOp} : vector<1x64xf32>
        %25 = vector.mask %21 { vector.transfer_write %24, %extracted_slice_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted_slice_10 = tensor.insert_slice %25 into %arg16[0, 0] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x64xf32>
        %inserted_slice_11 = tensor.insert_slice %inserted_slice_8 into %arg17[%arg12, %arg15] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x8183xf32>
        scf.yield %inserted_slice_10, %inserted_slice_11 : tensor<1x64xf32>, tensor<1x8183xf32>
      } {reductionLoop}
      %4 = vector.transfer_read %3#0[%c0, %c0], %cst_4 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.transfer_read %extracted_slice[%c0], %cst_4 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %6 = vector.multi_reduction <add>, %4, %5 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %7 = vector.transfer_write %6, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %7 into %arg13[%arg12] [1] [1] : tensor<1xf32> into tensor<1xf32>
      scf.yield %inserted_slice, %3#1 : tensor<1xf32>, tensor<1x8183xf32>
    }
    return %0#0, %0#1 : tensor<1xf32>, tensor<1x8183xf32>
  }

}

// CHECK-LABEL:   func.func @nested_reduction_loops_with_tail_block(
