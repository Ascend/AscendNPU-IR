// RUN: bishengir-opt -hivm-insert-cv-tight-coupled-buffer="only-insert-tightly-coupled-buffer=true" -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @test_collapse_shape_with_annotation(
// CHECK-SAME: %[[VAL_0:.*]]: memref<4x4x16xf16>, %[[VAL_1:.*]]: tensor<16x64xf16>, %[[VAL_2:.*]]: tensor<16x16xf16>) -> tensor<16x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK: %[[VAL_64:.*]] = arith.constant 64 : index
// CHECK: %[[VAL_16:.*]] = arith.constant 16 : index
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%[[VAL_1]] : tensor<16x64xf16>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x64xf32>
// CHECK: %[[TRANSPOSED:.*]] = hivm.hir.vtranspose
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[TRANSPOSED]]
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 512 : i64} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMCAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMCAST]] restrict writable : memref<1x1x16x16xf16>
// CHECK: hivm.hir.copy ins(%[[EXPANDED]] : tensor<1x1x16x16xf16>) outs(%[[MEMCAST]] : memref<1x1x16x16xf16>) {"hivm.inserted-copy"}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%[[TENSOR]], %[[LOAD]], %[[TRUE]], %[[VAL_16]], %[[VAL_16]], %[[VAL_64]] : tensor<1x1x16x16xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%[[EMPTY]] : tensor<16x64xf32>) -> tensor<16x64xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_collapse_shape_with_annotation(%arg0: memref<4x4x16xf16>, %arg1: tensor<16x64xf16>, %arg2: tensor<16x16xf16>) -> tensor<16x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x64xf16>
    %1 = hivm.hir.load ins(%arg1 : tensor<16x64xf16>) outs(%0 : tensor<16x64xf16>) {"hivm.inserted-load"} core_type = <CUBE> -> tensor<16x64xf16>
    %2 = tensor.empty() : tensor<16x64xf32>
    %expanded = tensor.expand_shape %arg2 [[0], [1, 2]] output_shape [16, 1, 16] : tensor<16x16xf16> into tensor<16x1x16xf16>
    %3 = tensor.empty() : tensor<1x16x16xf16>
    %4 = hivm.hir.vtranspose ins(%expanded : tensor<16x1x16xf16>) outs(%3 : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
    %expanded_0 = tensor.expand_shape %4 [[0], [1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
    %5 = tensor.empty() {hivm.address_space = #hivm.address_space<cbuf>, "hivm.inserted-tensor"} : tensor<1x1x16x16xf16>
    %6 = hivm.hir.copy ins(%expanded_0 : tensor<1x1x16x16xf16>) outs(%5 : tensor<1x1x16x16xf16>) {"hivm.inserted-copy"} -> tensor<1x1x16x16xf16>
    %7 = hivm.hir.mmadL1 ins(%6, %1, %true, %c16, %c16, %c64 : tensor<1x1x16x16xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%2 : tensor<16x64xf32>) -> tensor<16x64xf32>
    return %7 : tensor<16x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @func_mmad_l0c_ub(
// CHECK-SAME: %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: tensor<64x32xbf16>, %[[VAL_2:.*]]: tensor<32x64xbf16>, %[[VAL_3:.*]]: tensor<64x64xbf16>, %[[VAL_4:.*]]: tensor<64x64xbf16>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK: %[[MMAD_1:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C}
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]] {buffer_size_in_byte = 16384 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMCAST_1:.*]] = memref.memory_space_cast %[[ALLOC_1]] : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMCAST_1]] restrict writable : memref<64x64xf32>
// CHECK: hivm.hir.fixpipe {"hivm.inserted-fixpipe"} ins(%[[MMAD_1]] : tensor<64x64xf32>) outs(%[[MEMCAST_1]] : memref<64x64xf32>)
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]] {buffer_size_in_byte = 16384 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMCAST_2:.*]] = memref.memory_space_cast %[[ALLOC_2]] : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
// CHECK: %[[TENSOR_2:.*]] = bufferization.to_tensor %[[MEMCAST_2]] restrict writable : memref<64x64xf32>
// CHECK: hivm.hir.fixpipe {"hivm.inserted-fixpipe"} ins(%[[MMAD_1]] : tensor<64x64xf32>) outs(%[[MEMCAST_2]] : memref<64x64xf32>)
// CHECK: %[[VADD:.*]] = hivm.hir.vadd ins(%[[TENSOR_2]], %[[TENSOR]] : tensor<64x64xf32>, tensor<64x64xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @func_mmad_l0c_ub(%arg0: memref<?xi8>, %arg1: tensor<64x32xbf16>, %arg2: tensor<32x64xbf16>, %arg3: tensor<64x64xbf16>, %arg4: tensor<64x64xbf16>, %arg5: i32, %arg6: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<64x32xbf16>
    %1 = hivm.hir.load ins(%arg1 : tensor<64x32xbf16>) outs(%0 : tensor<64x32xbf16>) {"hivm.inserted-load"} core_type = <CUBE> -> tensor<64x32xbf16>
    %2 = tensor.empty() : tensor<32x64xbf16>
    %3 = hivm.hir.load ins(%arg2 : tensor<32x64xbf16>) outs(%2 : tensor<32x64xbf16>) {"hivm.inserted-load"} core_type = <CUBE> -> tensor<32x64xbf16>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = tensor.empty() : tensor<64x64xf32>
    %6 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%1, %3, %true, %c64, %c32, %c64 : tensor<64x32xbf16>, tensor<32x64xbf16>, i1, index, index, index) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %7 = tensor.empty() {hivm.address_space = #hivm.address_space<ub>, "hivm.inserted-tensor"} : tensor<64x64xf32>
    %8 = hivm.hir.fixpipe {"hivm.inserted-fixpipe"} ins(%6 : tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %9 = tensor.empty() {hivm.address_space = #hivm.address_space<ub>, "hivm.inserted-tensor"} : tensor<64x64xf32>
    %10 = hivm.hir.fixpipe {"hivm.inserted-fixpipe"} ins(%6 : tensor<64x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %11 = hivm.hir.vadd ins(%10, %8 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %11 : tensor<64x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @multiple_uses(
// CHECK-SAME: %[[ARG_0:.*]]: tensor<8x8x16x16xf16>, %[[ARG_1:.*]]: tensor<8x8x16x16xf16>) -> (tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]] {buffer_size_in_byte = 32768 : i64} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMCAST_1:.*]] = memref.memory_space_cast %[[ALLOC_1]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xf16>
// CHECK: %[[TENSOR_1:.*]] = bufferization.to_tensor %[[MEMCAST_1]] restrict writable : memref<8x8x16x16xf16>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_2]] {buffer_size_in_byte = 32768 : i64} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMCAST_2:.*]] = memref.memory_space_cast %[[ALLOC_2]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<8x8x16x16xf16>
// CHECK: %[[TENSOR_2:.*]] = bufferization.to_tensor %[[MEMCAST_2]] restrict writable : memref<8x8x16x16xf16>
// CHECK: hivm.hir.copy ins(%[[ARG_0]] : tensor<8x8x16x16xf16>) outs(%[[MEMCAST_1]] : memref<8x8x16x16xf16>) {"hivm.inserted-copy"}
// CHECK: hivm.hir.copy ins(%[[ARG_1]] : tensor<8x8x16x16xf16>) outs(%[[MEMCAST_2]] : memref<8x8x16x16xf16>) {"hivm.inserted-copy"}
// CHECK: return %[[TENSOR_1]], %[[TENSOR_2]] : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @multiple_uses(%arg0: tensor<8x8x16x16xf16>, %arg1: tensor<8x8x16x16xf16>) -> (tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %48 = tensor.empty() {hivm.address_space = #hivm.address_space<cbuf>, "hivm.inserted-tensor"} : tensor<8x8x16x16xf16>
    %49 = hivm.hir.copy ins(%arg0 : tensor<8x8x16x16xf16>) outs(%48 : tensor<8x8x16x16xf16>) {"hivm.inserted-copy"} -> tensor<8x8x16x16xf16>
    %61 = hivm.hir.copy ins(%arg1 : tensor<8x8x16x16xf16>) outs(%48 : tensor<8x8x16x16xf16>) {"hivm.inserted-copy"} -> tensor<8x8x16x16xf16>
    return %49, %61 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>
  }
}