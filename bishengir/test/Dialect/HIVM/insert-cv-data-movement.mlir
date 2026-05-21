// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv="enable-layout-optimization=true enable-dot-scaled-compile=false" -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @insert_ub_to_l1_data_movement
// CHECK: %[[INPUT:.*]] = hivm.hir.vcast
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<4x16x16x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACE:.*]] = memref.memory_space_cast %[[BUFFER]] : memref<4x16x16x32xf8E4M3FN, #hivm.address_space<cbuf>> to memref<4x16x16x32xf8E4M3FN>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMSPACE]] restrict writable : memref<4x16x16x32xf8E4M3FN>
// CHECK: %[[COPY:.*]] = hivm.hir.copy ins(%[[INPUT]] : tensor<4x16x16x32xf8E4M3FN>) outs(%[[TENSOR]] : tensor<4x16x16x32xf8E4M3FN>) {"inserted-copy"} -> tensor<4x16x16x32xf8E4M3FN>
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%[[COPY]]
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @insert_ub_to_l1_data_movement(%arg0: tensor<4x16x16x32xf32>, %arg1: f32) -> tensor<4x16x16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %0 = tensor.empty() : tensor<4x16x16x32xf32>
    %1 = tensor.empty() : tensor<4x16x16x32xf8E4M3FN>
    %2 = hivm.hir.vmul ins(%arg0, %arg1 : tensor<4x16x16x32xf32>, f32) outs(%0 : tensor<4x16x16x32xf32>) -> tensor<4x16x16x32xf32>
    %3 = hivm.hir.vcast ins(%2 : tensor<4x16x16x32xf32>) outs(%1 : tensor<4x16x16x32xf8E4M3FN>) -> tensor<4x16x16x32xf8E4M3FN>
    %alloc = memref.alloc() : memref<2x4x32x32xf8E4M3FN>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<2x4x32x32xf8E4M3FN>
    %5 = tensor.empty() : tensor<4x16x16x16xf32>
    %6 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%3, %4, %true, %c256, %c128, %c64 : tensor<4x16x16x32xf8E4M3FN>, tensor<2x4x32x32xf8E4M3FN>, i1, index, index, index) outs(%5 : tensor<4x16x16x16xf32>) -> tensor<4x16x16x16xf32>
    return %6 : tensor<4x16x16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @insert_l0c_to_ub_data_movement
// CHECK: memref.alloc
// CHECK: memref.memory_space_cast
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.fixpipe
// CHECK: call
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf_mm_func(%arg0: tensor<16x16xf16>) -> tensor<16x16xf16> attributes {hivm.vector_function} {
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.vmul ins(%arg0, %arg0 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %1 : tensor<16x16xf16>
  }
  func.func @insert_l0c_to_ub_data_movement(%arg0: tensor<16x16xf32>) -> tensor<16x16xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %2 = call @vf_mm_func(%1) {hivm.vector_function} : (tensor<16x16xf16>) -> tensor<16x16xf16>
    return %2 : tensor<16x16xf16>
  }
}
