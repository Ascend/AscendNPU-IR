// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: --convert-hivmave-to-ave-intrin %s | FileCheck %s

// CHECK-LABEL: func.func @_hstu_attn_bwd_mix_aiv_outlined_vf_6
// CHECK:      %17 = "hivm_regbaseintrins.intr.hivm.vsel"(%13, %15, %16) : (vector<128xi16>, vector<128xi16>, vector<256xi1>) -> vector<128xi16>
// CHECK:      %18 = builtin.unrealized_conversion_cast %17 : vector<128xi16> to vector<64xi16>
// CHECK:          %36 = llvm.bitcast %18 : vector<64xi16> to vector<64xi32>
// CHECK:          %37 = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128f16"(%35, %36, %25) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xf16>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map8 = affine_map<()[s0, s1] -> (s0 * 16 + s1 * 8)>
#map10 = affine_map<()[s0, s1] -> (s0 * 128 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_hstu_attn_bwd_mix_aiv_outlined_vf_6(%arg0: memref<2x8x1x16xf16, #hivm.address_space<ub>>, %arg1: memref<1x16x2x8xf16, #hivm.address_space<ub>>, %arg2: memref<1x16x2x8xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(16 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(16 : i16) : i16
    %6 = llvm.mlir.constant(0 : i16) : i16
    %7 = llvm.mlir.constant(0 : i16) : i16
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = "hivm_regbaseintrins.intr.hivm.vci"(%7, %8) : (i16, i32) -> vector<128xi16>
    %10 = builtin.unrealized_conversion_cast %9 : vector<128xi16> to vector<64xi16>
    %11 = llvm.mlir.constant(8 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%11, %12) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %14 = builtin.unrealized_conversion_cast %13 : vector<256xi1> to vector<64xi1>
    %15 = builtin.unrealized_conversion_cast %14 : vector<64xi1> to vector<256xi1>
    %16 = builtin.unrealized_conversion_cast %10 : vector<64xi16> to vector<128xi16>
    %17 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%16, %5, %15) : (vector<128xi16>, i16, vector<256xi1>) -> vector<128xi16>
    %18 = builtin.unrealized_conversion_cast %17 : vector<128xi16> to vector<64xi16>
    %19 = llvm.mlir.constant(8 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%19, %20) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = builtin.unrealized_conversion_cast %21 : vector<256xi1> to vector<64xi1>
    %23 = builtin.unrealized_conversion_cast %22 : vector<64xi1> to vector<256xi1>
    %24 = llvm.mlir.constant(1 : i32) : i32
    %25 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%6, %23, %24) : (i16, vector<256xi1>, i32) -> vector<128xi16>
    %26 = builtin.unrealized_conversion_cast %25 : vector<128xi16> to vector<64xi16>
    %27 = llvm.mlir.constant(5 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%27, %28) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %30 = builtin.unrealized_conversion_cast %29 : vector<256xi1> to vector<64xi1>
    %31 = builtin.unrealized_conversion_cast %30 : vector<64xi1> to vector<256xi1>
    %32 = builtin.unrealized_conversion_cast %18 : vector<64xi16> to vector<128xi16>
    %33 = builtin.unrealized_conversion_cast %26 : vector<64xi16> to vector<128xi16>
    %34 = "hivm_regbaseintrins.intr.hivm.vsel"(%32, %33, %31) : (vector<128xi16>, vector<128xi16>, vector<256xi1>) -> vector<128xi16>
    %35 = builtin.unrealized_conversion_cast %34 : vector<128xi16> to vector<64xi16>
    scf.for %arg3 = %3 to %2 step %1  : i32 {
      %36 = llvm.sext %arg3 : i32 to i64
      %37 = builtin.unrealized_conversion_cast %36 : i64 to index
      scf.for %arg4 = %3 to %0 step %1  : i32 {
        %38 = llvm.sext %arg4 : i32 to i64
        %39 = builtin.unrealized_conversion_cast %38 : i64 to index
        %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %arg0 : memref<2x8x1x16xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
        %40 = affine.apply #map10()[%39, %37]
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%40], sizes: [1, 8, 1, 1], strides: [128, 16, 16, 1] : memref<f16, #hivm.address_space<ub>> to memref<1x8x1x1xf16, strided<[128, 16, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %41 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<1x8x1x1xf16, strided<[128, 16, 16, 1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<4 x i64>, array<4 x i64>)>
        %42 = llvm.mlir.constant(5 : i32) : i32
        %43 = llvm.mlir.constant(0 : i32) : i32
        %44 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%42, %43) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
        %45 = builtin.unrealized_conversion_cast %44 : vector<256xi1> to vector<64xi1>
        %46 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<4 x i64>, array<4 x i64>)> 
        %47 = llvm.extractvalue %41[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<4 x i64>, array<4 x i64>)> 
        %48 = llvm.getelementptr %46[%47] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
        %49 = llvm.mlir.constant(128 : index) : i64
        %50 = llvm.mul %4, %49 : i64
        %51 = llvm.mlir.constant(16 : index) : i64
        %52 = llvm.mul %4, %51 : i64
        %53 = llvm.add %50, %52 : i64
        %54 = llvm.mlir.constant(16 : index) : i64
        %55 = llvm.mul %4, %54 : i64
        %56 = llvm.add %53, %55 : i64
        %57 = llvm.add %56, %4 : i64
        %58 = llvm.getelementptr %48[%57] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
        %59 = builtin.unrealized_conversion_cast %45 : vector<64xi1> to vector<256xi1>
        %60 = llvm.bitcast %35 : vector<64xi16> to vector<64xi32>
        %61 = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128f16"(%58, %60, %59) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xf16>
        %62 = builtin.unrealized_conversion_cast %61 : vector<128xf16> to vector<64xf16>
        %63 = builtin.unrealized_conversion_cast %62 : vector<64xf16> to vector<128xf16>
        %64 = llvm.mlir.constant(0 : i32) : i32
        %65 = llvm.mlir.constant(0 : i32) : i32
        %66 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%63, %44, %65) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
        %67 = builtin.unrealized_conversion_cast %66 : vector<64xf32> to vector<64xf32>
        %base_buffer_0, %offset_1, %sizes_2:4, %strides_3:4 = memref.extract_strided_metadata %arg2 : memref<1x16x2x8xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
        %68 = affine.apply #map8()[%37, %39]
        %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%68], sizes: [8], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<8xf32, #map1, #hivm.address_space<ub>>
        %69 = builtin.unrealized_conversion_cast %reinterpret_cast_4 : memref<8xf32, #map1, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
        %70 = llvm.extractvalue %69[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
        %71 = llvm.extractvalue %69[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
        %72 = llvm.getelementptr %70[%71] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
        %73 = llvm.getelementptr %72[%4] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
        %74 = llvm.mlir.constant(0 : i32) : i32
        %75 = llvm.mlir.constant(0 : i32) : i32
        %76 = llvm.mlir.constant(2 : i32) : i32
        "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%67, %73, %74, %76, %75, %44) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
      } {element_alignment_bit_width = 32 : i32}
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}