// RUN: bishengir-opt -analyze-vector-layout \
// RUN: -remove-vector-layout-attr -convert-hivmave-to-ave-intrin %s | FileCheck %s
#map = affine_map<()[s0] -> (s0 + 128)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_2(%arg0: memref<128xi64, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg3], sizes: [64], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg3], sizes: [64], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%arg3], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %1 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast = memref.cast %reinterpret_cast : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%1, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast_10 = memref.cast %reinterpret_cast_4 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%2, %cast_10, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %3 = ave.hir.pge <ALL> : vector<64xi1>
      %4 = builtin.unrealized_conversion_cast %3 : vector<64xi1> to vector<256xi1>
      %5 = func.call @_mlir_ciface_vcmp_ge_int64_t(%1, %2, %4) : (!llvm.ptr, !llvm.ptr, vector<256xi1>) -> vector<256xi1>
      %6 = builtin.unrealized_conversion_cast %5 : vector<256xi1> to vector<64xi1>
      %7 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<256xi1>, !llvm.ptr<6>, i32, i32, i32) -> ()
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_9[%c0], %7, %6 : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
    }
    return
  }
  func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi64, #hivm.address_space<ub>>, %arg3: memref<128xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg4], sizes: [64], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%arg4], sizes: [64], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg3 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [%arg4], sizes: [64], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi1>
      %20 = llvm.mlir.constant(1 : i32) : i32
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast = memref.cast %reinterpret_cast_4 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%2, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %3 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast_15 = memref.cast %reinterpret_cast_9 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%3, %cast_15, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %4 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%4, %5, %2, %3) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
      %6 = ave.hir.pge <ALL> : vector<64xi1>
      %cast_16 = memref.cast %reinterpret_cast_14 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %7 = builtin.unrealized_conversion_cast %6 : vector<64xi1> to vector<256xi1>
      func.call @masked_store_NORM_B64_int64_t_rank1(%cast_16, %c0_i64, %7, %4) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64, vector<256xi1>, !llvm.ptr) -> ()
    }
    return
  }

  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @masked_store_NORM_B64_int64_t_rank1(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64, vector<256xi1>, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @_mlir_ciface_vcmp_ge_int64_t(!llvm.ptr, !llvm.ptr, vector<256xi1>) -> vector<256xi1> attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @_mlir_ciface_vdups_int64_t(!llvm.ptr, i64, vector<256xi1>) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c2592_i64 = arith.constant 2592 : i64
    %c2560_i64 = arith.constant 2560 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c800_i32 = arith.constant 800 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg9, %arg8 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg7 : i32
    %5 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xi64, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi64, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    %8 = arith.muli %4, %c800_i32 : i32
    %9 = arith.addi %8, %c800_i32 : i32
    %10 = arith.minsi %9, %arg6 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = hivm.hir.pointer_cast(%c2560_i64) : memref<128xi1, #hivm.address_space<ub>>
    scf.for %arg10 = %c0_i32 to %c7_i32 step %c1_i32  : i32 {
      %13 = arith.muli %arg10, %c128_i32 : i32
      %14 = arith.addi %8, %13 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi64, #hivm.address_space<ub>>
      %17 = affine.apply #map()[%15]
      %18 = arith.maxsi %15, %11 : index
      %19 = arith.minsi %17, %18 : index
      %20 = affine.apply #map1()[%19, %15]
      %21 = arith.cmpi slt, %20, %c128 : index
      scf.if %21 {
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg3 : memref<?xi64, #hivm.address_space<gm>> -> memref<i64, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%15], sizes: [%20], strides: [1] : memref<i64, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %16 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [%20], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %cast = memref.cast %reinterpret_cast : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_5 = memref.cast %reinterpret_cast_4 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int64_t(%cast, %cast_5, %c2_i32, %c0_i64, %c0) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_2(%16, %7, %12) {hivm.vector_function} : (memref<128xi64, #hivm.address_space<ub>>, memref<128xi64, #hivm.address_space<ub>>, memref<128xi1, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      func.call @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%12, %16, %7, %6) {hivm.vector_function} : (memref<128xi1, #hivm.address_space<ub>>, memref<128xi64, #hivm.address_space<ub>>, memref<128xi64, #hivm.address_space<ub>>, memref<128xi64, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %6 : memref<128xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [%20], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %arg4 : memref<?xi64, #hivm.address_space<gm>> -> memref<i64, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%15], sizes: [%20], strides: [1] : memref<i64, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_16 = memref.cast %reinterpret_cast_10 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %reinterpret_cast_15 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_int64_t(%cast_16, %cast_17, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %5 : memref<128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [0], sizes: [%20], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
      %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [%15], sizes: [%20], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      %cast_28 = memref.cast %reinterpret_cast_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_29 = memref.cast %reinterpret_cast_27 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_28, %cast_29, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    return
  }
}