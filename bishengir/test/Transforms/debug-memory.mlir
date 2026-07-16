// RUN: bishengir-opt --debug-memory %s | FileCheck %s
// CHECK: func.func @mul_add_kernel_outlined_vf_3

#map = affine_map<()[s0] -> (s0 + 1024)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @mul_add_kernel_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    scf.for %arg1 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg1] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview[%c0], %2, %1 {hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @mul_add_kernel_outlined_vf_1(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    scf.for %arg1 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg1] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview[%c0], %2, %1 {hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @mul_add_kernel_outlined_vf_2(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    scf.for %arg1 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg1] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview[%c0], %2, %1 {hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @mul_add_kernel_outlined_vf_3(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<1024xf32, #hivm.address_space<ub>>, %arg2: memref<1024xf32, #hivm.address_space<ub>>, %arg3: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg4] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg4] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg3[%arg4] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %res_3 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %res_4 = ave.hir.vload <NORM> %subview_1[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> : vector<64xi1>
      %1 = ave.hir.vmul %res, %res_3, %0 : vector<64xf32>, vector<64xi1>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      %3 = ave.hir.vadd %1, %res_4, %2 : vector<64xf32>, vector<64xi1>
      %4 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_2[%c0], %4, %3 {hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @mul_add_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c8192_i64 = arith.constant 8192 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg10, %arg9 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg8 : i32
    %5 = arith.muli %4, %c1024_i32 : i32
    %6 = arith.muli %4, %c2048_i32 : i32
    %7 = arith.index_cast %5 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
    %9 = affine.apply #map()[%7]
    %10 = arith.index_cast %arg7 : i32 to index
    %11 = arith.maxsi %7, %10 : index
    %12 = arith.minsi %9, %11 : index
    %13 = affine.apply #map1()[%12, %7]
    %14 = arith.cmpi slt, %13, %c1024 : index
    scf.if %14 {
      func.call @mul_add_kernel_outlined_vf_0(%8) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    }
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %8[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index eviction_policy = <EvictFirst>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%7], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf32, #hivm.address_space<ub>>
    scf.if %14 {
      func.call @mul_add_kernel_outlined_vf_1(%15) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    }
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    %subview_2 = memref.subview %reinterpret_cast_1[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %15[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview_2 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index eviction_policy = <EvictFirst>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%7], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
    scf.if %14 {
      func.call @mul_add_kernel_outlined_vf_2(%16) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    }
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %subview_5 = memref.subview %reinterpret_cast_4[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %16[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index eviction_policy = <EvictFirst>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @mul_add_kernel_outlined_vf_3(%8, %15, %16, %17) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%7], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %17[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_9 = memref.subview %reinterpret_cast_7[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %18 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%18], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %reinterpret_cast_10[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
