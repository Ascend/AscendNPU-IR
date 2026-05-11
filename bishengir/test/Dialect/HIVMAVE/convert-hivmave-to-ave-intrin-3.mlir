// RUN: bishengir-opt -append-vector-layout -annotate-dist-op-layout \
// RUN: -eliminate-vector-layout -ave-normalize-ops -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s

// CHECK-LABEL:   func.func @triton_load_mask_outlined_vf_0
func.func @triton_load_mask_outlined_vf_0(%arg0: memref<32xi8, #hivm.address_space<ub>>, %arg1: memref<32xf8E5M2, #hivm.address_space<ub>>, %arg2: memref<32xf8E5M2, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  // CHECK:           %[[VAL_16:.*]] = builtin.unrealized_conversion_cast %[[VAL_15:.*]] : vector<256xi8> to vector<64xi8>
  // CHECK:           %[[VAL_19:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%{{.*}}, %{{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK:           %[[VAL_23:.*]] = "hivm_regbaseintrins.intr.hivm.vdups.z"(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK:           %[[VAL_26:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%{{.*}}, %{{.*}}) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK:           %[[VAL_28:.*]] = llvm.getelementptr %{{.*}}{{\[}}%{{.*}}] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, i8
   // CHECK:           %[[VAL_32:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[VAL_28:.*]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
   // CHECK:           %[[VAL_35:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[VAL_32:.*]], %[[VAL_32:.*]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
   // CHECK:           %[[VAL_37:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK:           %[[VAL_45:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256f8e5m2"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xf8E5M2>
  // CHECK:           %[[VAL_50:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%{{.*}}, %[[VAL_26:.*]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK:           %[[VAL_52:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"(%{{.*}}, %[[VAL_23:.*]], %[[VAL_26:.*]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
  // CHECK:           %[[VAL_55:.*]] = llvm.bitcast %{{.*}} : vector<256xf8E5M2> to vector<256xi8>
  // CHECK:           %[[VAL_60:.*]] = "hivm_regbaseintrins.intr.hivm.vsel"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>
  // CHECK:           %[[VAL_63:.*]] = llvm.bitcast %{{.*}} : vector<256xi8> to vector<256xf8E5M2>
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i16 = arith.constant 0 : i16
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c0_i16, %0 : i16, vector<64xi1> -> vector<64xi8>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %c0_i32, %2 : i32, vector<64xi1> -> vector<64xi32>
  %4 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<32xi8, #hivm.address_space<ub>> into vector<64xi8>
  %res_0 = ave.hir.vload <NORM> %arg1[%c0] : memref<32xf8E5M2, #hivm.address_space<ub>> into vector<64xf8E5M2>
  %5 = ave.hir.vextsi %res, %4 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  %6 = ave.hir.vcmp <NE> %5, %3, %4 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %7 = vector.bitcast %res_0 : vector<64xf8E5M2> to vector<64xi8>
  %8 = ave.hir.vsel %6, %7, %1 : vector<64xi1>, vector<64xi8>
  %9 = vector.bitcast %8 : vector<64xi8> to vector<64xf8E5M2>
  ave.hir.masked_store <NORM_B8> %arg2[%c0], %4, %9 : memref<32xf8E5M2, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf8E5M2>
  return
}