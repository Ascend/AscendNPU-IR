// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -convert-hivmave-to-ave-intrin -split-input-file %s | FileCheck %s

// -----
// DEFINE-LABEL: MODULE_MODULE_128_64x_int64
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi64, #hivm.address_space<ub>>, %arg3: memref<128xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_256_64x_int64
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi64, #hivm.address_space<ub>>, %arg2: memref<256xi64, #hivm.address_space<ub>>, %arg3: memref<256xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c256 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_32x_int64
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi64, #hivm.address_space<ub>>, %arg3: memref<128xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_127_64x_int64
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<127xi1, #hivm.address_space<ub>>, %arg1: memref<127xi64, #hivm.address_space<ub>>, %arg2: memref<127xi64, #hivm.address_space<ub>>, %arg3: memref<127xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c127 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<127xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_31x_int64
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi64, #hivm.address_space<ub>>, %arg3: memref<128xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c31 = arith.constant 31 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c31 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}


// -----
// DEFINE-LABEL: MODULE_MODULE_128_64x_int32
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi32, #hivm.address_space<ub>>, %arg2: memref<128xi32, #hivm.address_space<ub>>, %arg3: memref<128xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_256_64x_int32
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi32, #hivm.address_space<ub>>, %arg2: memref<256xi32, #hivm.address_space<ub>>, %arg3: memref<256xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c256 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_32x_int32
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi32, #hivm.address_space<ub>>, %arg2: memref<128xi32, #hivm.address_space<ub>>, %arg3: memref<128xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_127_64x_int32
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<127xi1, #hivm.address_space<ub>>, %arg1: memref<127xi32, #hivm.address_space<ub>>, %arg2: memref<127xi32, #hivm.address_space<ub>>, %arg3: memref<127xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c127 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<127xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_31x_int32
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi32, #hivm.address_space<ub>>, %arg2: memref<128xi32, #hivm.address_space<ub>>, %arg3: memref<128xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c31 = arith.constant 31 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c31 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_64x_int16
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi16, #hivm.address_space<ub>>, %arg2: memref<128xi16, #hivm.address_space<ub>>, %arg3: memref<128xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_256_64x_int16
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi16, #hivm.address_space<ub>>, %arg2: memref<256xi16, #hivm.address_space<ub>>, %arg3: memref<256xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c256 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_32x_int16
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi16, #hivm.address_space<ub>>, %arg2: memref<128xi16, #hivm.address_space<ub>>, %arg3: memref<128xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_127_64x_int16
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<127xi1, #hivm.address_space<ub>>, %arg1: memref<127xi16, #hivm.address_space<ub>>, %arg2: memref<127xi16, #hivm.address_space<ub>>, %arg3: memref<127xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c127 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<127xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_31x_int16
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi16, #hivm.address_space<ub>>, %arg2: memref<128xi16, #hivm.address_space<ub>>, %arg3: memref<128xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c31 = arith.constant 31 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c31 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_64x_int8
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi8, #hivm.address_space<ub>>, %arg2: memref<128xi8, #hivm.address_space<ub>>, %arg3: memref<128xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_256_64x_int8
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi8, #hivm.address_space<ub>>, %arg2: memref<256xi8, #hivm.address_space<ub>>, %arg3: memref<256xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c256 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_32x_int8
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi8, #hivm.address_space<ub>>, %arg2: memref<128xi8, #hivm.address_space<ub>>, %arg3: memref<128xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_127_64x_int8
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<127xi1, #hivm.address_space<ub>>, %arg1: memref<127xi8, #hivm.address_space<ub>>, %arg2: memref<127xi8, #hivm.address_space<ub>>, %arg3: memref<127xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c127 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<127xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_31x_int8
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi8, #hivm.address_space<ub>>, %arg2: memref<128xi8, #hivm.address_space<ub>>, %arg3: memref<128xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c31 = arith.constant 31 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c31 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_64x_int1
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi1, #hivm.address_space<ub>>, %arg2: memref<128xi1, #hivm.address_space<ub>>, %arg3: memref<128xi1, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_256_64x_int1
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi1, #hivm.address_space<ub>>, %arg2: memref<256xi1, #hivm.address_space<ub>>, %arg3: memref<256xi1, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c256 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_32x_int1
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi1, #hivm.address_space<ub>>, %arg2: memref<128xi1, #hivm.address_space<ub>>, %arg3: memref<128xi1, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_127_64x_int1
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<127xi1, #hivm.address_space<ub>>, %arg1: memref<127xi1, #hivm.address_space<ub>>, %arg2: memref<127xi1, #hivm.address_space<ub>>, %arg3: memref<127xi1, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c127 = arith.constant 127 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c127 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<127xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----
// DEFINE-LABEL: MODULE_MODULE_128_31x_int1
module {
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi1, #hivm.address_space<ub>>, %arg2: memref<128xi1, #hivm.address_space<ub>>, %arg3: memref<128xi1, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c31 = arith.constant 31 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c31 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    } {element_alignment_bit_width = 32 : i32}
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}
}
