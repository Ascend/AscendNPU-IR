// RUN: bishengir-opt -convert-arith-to-hivmave -convert-vector-to-hivmave -convert-hivmave-to-ave-intrin %s | FileCheck %s 

  func.func private @_mlir_ciface_vcmp_ne_int64_t(!llvm.ptr, !llvm.ptr, vector<256xi1>) -> vector<256xi1> attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @_mlir_ciface_vdups_int64_t(!llvm.ptr, i64, vector<256xi1>) attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func @triton_unk_fused__npu_dtype_cast__softmax_div_eq_masked_fill_2_outlined_vf_6(%arg0: memref<1x1x39xi64, strided<[40, 40, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x1x39xi1, strided<[256, 256, 1]>, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.vector_function} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c39 = arith.constant 39 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %1 = ave.hir.pge <ALLF> : vector<64xi1>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
    %4 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<256xi1>
    call @_mlir_ciface_vdups_int64_t(%3, %c0_i64, %4) : (!llvm.ptr, i64, vector<256xi1>) -> ()
    %res, %new_true_shape = ave.hir.plt %c39 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<1x1x39xi64, strided<[40, 40, 1]>, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [39], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<39xi64, strided<[1]>, #hivm.address_space<ub>>
    %5 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
    %cast = memref.cast %reinterpret_cast : memref<39xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vload_NORM_int64_t_rank1(%5, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
    %6 = builtin.unrealized_conversion_cast %res : vector<64xi1> to vector<256xi1>
    %7 = call @_mlir_ciface_vcmp_ne_int64_t(%5, %3, %6) : (!llvm.ptr, !llvm.ptr, vector<256xi1>) -> vector<256xi1>
    %8 = builtin.unrealized_conversion_cast %7 : vector<256xi1> to vector<64xi1>
    %9 = ave.hir.preg.xor <b8> %8, %1, %res : vector<64xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.pnot.z"({{.*}}, {{.*}}) : (vector<256xi1>, vector<256xi1>) -> vector<256xi1>
    %10 = ave.hir.preg.not <b8> %9, %res : vector<64xi1>
    %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg1 : memref<1x1x39xi1, strided<[256, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [39], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<39xi1, strided<[1]>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B8> %reinterpret_cast_4[%c0], %res, %10 : memref<39xi1, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
    return
  }