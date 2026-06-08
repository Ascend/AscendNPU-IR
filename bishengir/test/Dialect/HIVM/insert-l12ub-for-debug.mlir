// RUN: bishengir-opt -hivm-insert-l12ub-for-debug %s | FileCheck %s

// CHECK: annotation.mark
// CHECK: hivm.hir.l12ub
// CHECK: %[[VAL:.*]] = bufferization.to_tensor
// CHECK: hivm.hir.debug{{.*}} %[[VAL]]
func.func @triton_device_print(
  %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
  %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
  %arg2: memref<?xf16> {tt.divisibility = 16 : i32},
  %arg3: memref<?xf16> {tt.divisibility = 16 : i32},
  %arg4: memref<?xf32> {tt.divisibility = 16 : i32},
  %arg5: i32, %arg6: i32, %arg7: i32
) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c511 = arith.constant 511 : index
  %c1 = arith.constant 1 : index
  %c63 = arith.constant 63 : index
  %true = arith.constant true
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [511, 1], strides: [1, 1] : memref<?xf16> to memref<511x1xf16, strided<[1, 1]>>
  %alloc = memref.alloc() : memref<511x1xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<511x1xf16, strided<[1, 1]>>) outs(%alloc : memref<511x1xf16>) eviction_policy = <EvictFirst> core_type = <CUBE>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<511x1xf16>
  hivm.hir.debug {debugtype = "print", hex = true, isL1Print = true, prefix = " x0 (hex):\0A: ", tcoretype = #hivm.tcore_type<CUBE>} %2 : tensor<511x1xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 63], strides: [63, 1] : memref<?xf16> to memref<1x63xf16, strided<[63, 1]>>
  %alloc_1 = memref.alloc() : memref<1x63xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<1x63xf16, strided<[63, 1]>>) outs(%alloc_1 : memref<1x63xf16>) eviction_policy = <EvictFirst> core_type = <CUBE>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x63xf16>
  hivm.hir.debug {debugtype = "print", hex = false, memscope = #hivm.address_space<cbuf>, prefix = " ret :\0A: ", tcoretype = #hivm.tcore_type<CUBE>} %3 : tensor<1x63xf16>
  %4 = tensor.empty() : tensor<511x63xf32>
  %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C} ins(%2, %3, %true, %c511, %c1, %c63 : tensor<511x1xf16>, tensor<1x63xf16>, i1, index, index, index) outs(%4 : tensor<511x63xf32>) -> tensor<511x63xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [511, 63], strides: [63, 1] : memref<?xf32> to memref<511x63xf32, strided<[63, 1]>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<511x63xf32>) outs(%reinterpret_cast_2 : memref<511x63xf32, strided<[63, 1]>>)
  hivm.hir.set_ctrl true at ctrl[60]
  return
}