// RUN: bishengir-opt -convert-hivmave-to-ave-intrin %s | FileCheck %s

//----- Case From #2768

func.func @triton_mod_outlined_vf_0(%arg0: memref<1x1x23xf32, #hivm.address_space<ub>>, %arg1: memref<1x1x23xf32, #hivm.address_space<ub>>, %arg2: memref<1x1x23xi16, #hivm.address_space<ub>>) attributes {hivm.vector_function, element_alignment_bit_width = 32 : i32} {
%c0 = arith.constant 0 : index
%c46 = arith.constant 46 : index
%res, %new_true_shape = ave.hir.plt %c46 : vector<128xi1>, index
%0 = builtin.unrealized_conversion_cast %res : vector<128xi1> to vector<1x1x64xi1>
%base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<1x1x23xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
%reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [23], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<23xf32, strided<[1]>, #hivm.address_space<ub>>
%1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] : memref<23xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
%base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg1 : memref<1x1x23xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
%reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [23], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<23xf32, strided<[1]>, #hivm.address_space<ub>>
%2 = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] : memref<23xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
%3 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
%4 = ave.hir.vdiv %1, %2, %3 : vector<64xf32>, vector<64xi1>
%5 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
%6 = ave.hir.vtrc %4, <floor>, %5 : vector<64xf32>, vector<64xi1>, vector<64xf32>
%7 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
%8 = ave.hir.vmul %6, %2, %7 : vector<64xf32>, vector<64xi1>
%9 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
%10 = ave.hir.vsub %1, %8, %9 : vector<64xf32>, vector<64xi1>
%11 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
%12 = ave.hir.vfptosi %10, <truncwithoverflow>, %11 {part = #ave.vcvt_part_type<part_even>, sat = false} : vector<64xf32>, vector<64xi1>, vector<64xi16>
%13 = builtin.unrealized_conversion_cast %0 : vector<1x1x64xi1> to vector<64xi1>
ave.hir.masked_store <PK_B32> %arg2[%c0, %c0, %c0], %13, %12 : memref<1x1x23xi16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
return
}

// CHECK: %[[CONST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: %[[CONST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[CONST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s16.x"(%{{.*}}, %{{.*}}, %[[CONST_4]], %
