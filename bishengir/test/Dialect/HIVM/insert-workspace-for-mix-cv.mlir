// RUN: bishengir-opt -insert-workspace-for-mix-cv %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @insert_workspace_for_cc(
func.func @insert_workspace_for_cc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3 : tensor<16x16xf16>, %arg4 : tensor<16x16xf16>, %arg5 : tensor<16x16xf32>, %arg6 : memref<16x16xf32, strided<[?, 1], offset: ?>>, %arg7 : memref<16x16xf16, strided<[?, ?], offset: ?>>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %0 = hivm.hir.mmadL1 ins(%arg3, %arg4, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%arg7 : memref<16x16xf16, strided<[?, ?], offset: ?>>) outs(%alloc : memref<16x16xf16>)
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf16>
    // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace() : memref<16x16xf16>
    // CHECK: %[[WORKSPACE_TENSOR:.*]] = bufferization.to_tensor %[[WORKSPACE]] restrict writable : memref<16x16xf16>
    %3 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>} ins(%0 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %4 = tensor.empty() : tensor<16x16xf16>
    %5 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%4 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = hivm.hir.mmadL1 ins(%5, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%8 : tensor<16x16xf32>) outs(%arg6 : memref<16x16xf32, strided<[?, 1], offset: ?>>)
    return
}


// -----
// CHECK-LABEL: @insert_workspace_for_cv(
func.func @insert_workspace_for_cv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2 : tensor<16x16xf16>, %arg3 : tensor<16x16xf16>, %arg4 : memref<16x16xf32, strided<[?, 1], offset: ?>>, %arg5 : tensor<16x16xf32>) attributes {func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<26xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %empty = tensor.empty() : tensor<16x16xf32>
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = hivm.hir.mmadL1 ins(%arg2, %arg3, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace() : memref<16x16xf32>
    // CHECK: %[[OUTPUT_TENSOR:.*]] = bufferization.to_tensor %[[WORKSPACE]] restrict writable : memref<16x16xf32>
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = hivm.hir.fixpipe {channel_split = false, enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<NO_QUANT>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>} ins(%0 : tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %3 = tensor.empty() : tensor<16x16xf32>
    %4 = hivm.hir.load ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.vadd ins(%4, %arg5 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%4 : tensor<16x16xf32>) outs(%arg4 : memref<16x16xf32, strided<[?, 1], offset: ?>>)
    return
}

// -----
// CHECK-LABEL: @insert_workspace_for_vc(
func.func @insert_workspace_for_vc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2 : tensor<16x16xf32>, %arg3 : tensor<16x16xf16>, %arg4 : memref<16x16xf32, strided<[?, 1], offset: ?>>) attributes {func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<26xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %empty = tensor.empty() : tensor<16x16xf16>
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %1 = hivm.hir.vcast ins(%arg2 : tensor<16x16xf32>) outs(%empty : tensor<16x16xf16>) round_mode = <rint> -> tensor<16x16xf16>
    // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace() : memref<16x16xf16>
    // CHECK: %[[OUTPUT_TENSOR:.*]] = bufferization.to_tensor %[[WORKSPACE]] restrict writable : memref<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf16>
    %3 = hivm.hir.store ins(%1 : tensor<16x16xf16>) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %4 = tensor.empty() : tensor<16x16xf16>
    %5 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%4 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %6 = tensor.empty() : tensor<16x16xf32>
    %7 = hivm.hir.mmadL1 ins(%5, %arg3, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%7 : tensor<16x16xf32>) outs(%arg4 : memref<16x16xf32, strided<[?, 1], offset: ?>>)
    return
}

// -----
module {
  func.func @insert_workspace_for_fixpipe(%arg1: memref<?xi8>, %arg2: memref<?xf32> , %arg3: memref<?xf16> , %arg4: memref<?xf16> , %arg5: memref<?xf32>){
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 32, 64], strides: [2048, 64, 1] : memref<?xf16> to memref<2x32x64xf16, strided<[2048, 64, 1]>>
    %alloc = memref.alloc() : memref<2x32x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<2x32x64xf16, strided<[2048, 64, 1]>>) outs(%alloc : memref<2x32x64xf16>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2x32x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 64, 16], strides: [1024, 16, 1] : memref<?xf16> to memref<2x64x16xf16, strided<[1024, 16, 1]>>
    %alloc_1 = memref.alloc() : memref<2x64x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<2x64x16xf16, strided<[1024, 16, 1]>>) outs(%alloc_1 : memref<2x64x16xf16>)
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x64x16xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [2, 32, 16], strides: [512, 16, 1] : memref<?xf32> to memref<2x32x16xf32, strided<[512, 16, 1]>>
    %alloc_3 = memref.alloc() : memref<2x32x16xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<2x32x16xf32, strided<[512, 16, 1]>>) outs(%alloc_3 : memref<2x32x16xf32>)
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<2x32x16xf32>
    // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace() : memref<2x32x16xf32>
    // CHECK: %[[OUTPUT_TENSOR:.*]] = bufferization.to_tensor %[[WORKSPACE]] restrict writable : memref<2x32x16xf32>
    %3 = tensor.empty() : tensor<2x32x16xf32>
    %4 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %3) -> (tensor<2x32x16xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg9, 0, 0] [1, 32, 64] [1, 1, 1] : tensor<2x32x64xf16> to tensor<32x64xf16>
      %extracted_slice_5 = tensor.extract_slice %1[%arg9, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<2x64x16xf16> to tensor<64x16xf16>
      %9 = tensor.empty() : tensor<32x16xf32>
      %10 = hivm.hir.mmadL1 ins(%extracted_slice, %extracted_slice_5, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%9 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %extracted_slice_6 = tensor.extract_slice %3[%arg9, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<2x32x16xf32> to tensor<32x16xf32>
      %11 = hivm.hir.fixpipe {enable_nz2nd} ins(%10 : tensor<32x16xf32>) outs(%extracted_slice_6 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %inserted_slice = tensor.insert_slice %11 into %3[%arg9, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<32x16xf32> into tensor<2x32x16xf32>
      scf.yield %inserted_slice : tensor<2x32x16xf32>
    }
    %5 = tensor.empty() : tensor<2x32x16xf32>
    %6 = hivm.hir.load ins(%4 : tensor<2x32x16xf32>) outs(%5 : tensor<2x32x16xf32>) -> tensor<2x32x16xf32>
    %7 = tensor.empty() : tensor<2x32x16xf32>
    %8 = hivm.hir.vadd ins(%6, %2 : tensor<2x32x16xf32>, tensor<2x32x16xf32>) outs(%7 : tensor<2x32x16xf32>) -> tensor<2x32x16xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 32, 16], strides: [512, 16, 1] : memref<?xf32> to memref<2x32x16xf32, strided<[512, 16, 1]>>
    hivm.hir.store ins(%8 : tensor<2x32x16xf32>) outs(%reinterpret_cast_4 : memref<2x32x16xf32, strided<[512, 16, 1]>>)
    return
  }
}

// -----
module {
  func.func @insert_workspace_for_fixpipe_with_debug(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: tensor<16x16xf32>) {
    // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace() : memref<16x16xf32>
    // CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[WORKSPACE]] restrict writable : memref<16x16xf32>
    // CHECK: {{.*}} = hivm.hir.fixpipe {enable_nz2nd} ins({{.*}}: tensor<16x16xf32>) outs(%[[TO_TENSOR]] : tensor<16x16xf32>) -> tensor<16x16xf32>
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = hivm.hir.fixpipe {enable_nz2nd} ins(%arg1 : tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " fixpipe_out ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<16x16xf32>
    return
  }
}