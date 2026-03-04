// RUN: bishengir-opt %s -hivm-inject-sync | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
  %c0_i64 = arith.constant 0 : i64
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
  %2 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  %cast = memref.cast %reinterpret_cast : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>> to memref<1024xf32, #hivm.address_space<gm>>
  %cast_1 = memref.cast %reinterpret_cast_0 : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>> to memref<1024xf32, #hivm.address_space<gm>>
  call @simt_func(%cast, %cast_1, %2) : (memref<1024xf32, #hivm.address_space<gm>>, memref<1024xf32, #hivm.address_space<gm>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
  // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.store ins(%2 : memref<1024xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1024xf32, strided<[1]>, #hivm.address_space<gm>>)
  return
}
 
func.func @simt_func(%arg0: memref<1024xf32, #hivm.address_space<gm>> {memory.effect = #hivm.memory.effect<read>}, %arg1: memref<1024xf32, #hivm.address_space<gm>> {memory.effect = #hivm.memory.effect<read>}, %arg2: memref<1024xf32, #hivm.address_space<ub>> {memory.effect = #hivm.memory.effect<write>}) attributes { hivm.vector_function, simt_wrapper}{
  return
}