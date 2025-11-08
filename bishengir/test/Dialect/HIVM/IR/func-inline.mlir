// RUN: bishengir-opt --inline %s

//===----------------------------------------------------------------------===//
// Test Function Inlining
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 + 1024)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @load_from_gm_to_ub(
    %source: memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>,
    %dest: memref<?xf32, strided<[1]>, #hivm.address_space<ub>>,
    %padding: index
  ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.load ins(%source : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
               outs(%dest : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
               left_padding_num = %padding : index
               init_out_buffer = false
    return
  }
  func.func @copy_ub_to_gm(
    %result: memref<1024xf32, #hivm.address_space<ub>>,
    %output: memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>,
    %size: index
  ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %subview_result = memref.subview %result[0] [%size] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_output = memref.subview %output[0] [%size] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_result : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_output : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    return
  }
  func.func @add_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg8, %arg7 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg6 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %7 = arith.muli %6, %c1024_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<1024xf32, #hivm.address_space<ub>>
    %9 = affine.apply #map()[%8]
    %10 = arith.index_cast %arg5 : i32 to index
    %11 = arith.maxsi %8, %10 : index
    %12 = arith.minsi %9, %11 : index
    %13 = affine.apply #map1()[%12, %8]
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %alloc[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    call @load_from_gm_to_ub(%subview, %subview_0, %c0) : (memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[1]>, #hivm.address_space<ub>>, index) -> ()
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc_2 = memref.alloc() : memref<1024xf32, #hivm.address_space<ub>>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %alloc_2[0] [%13] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    call @load_from_gm_to_ub(%subview_3, %subview_4, %c0) : (memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[1]>, #hivm.address_space<ub>>, index) -> ()
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_2 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%alloc_5 : memref<1024xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    call @copy_ub_to_gm(%alloc_5, %reinterpret_cast_6, %13) : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, index) -> ()
    return
  }
}