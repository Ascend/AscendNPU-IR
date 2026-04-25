// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -append-vector-layout -annotate-dist-op-layout \
// RUN: -eliminate-vector-layout -convert-hivmave-to-ave-intrin %s | FileCheck %s
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