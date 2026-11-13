// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @vaddcs
// CHECK: %[[ADD_MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: %[[ADD_CARRY_IN:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: %[[ADD_AGG:.*]] = "hivm_regbaseintrins.intr.hivm.vaddcs.v64s32"({{.*}}, {{.*}}, %[[ADD_CARRY_IN]], %[[ADD_MASK]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[ADD_RES:.*]] = llvm.extractvalue %[[ADD_AGG]][0] : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[ADD_CARRY:.*]] = llvm.extractvalue %[[ADD_AGG]][1] : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[ADD_CARRY_OUT:.*]] = builtin.unrealized_conversion_cast %[[ADD_CARRY]] : vector<256xi1> to vector<64xi1>
// CHECK: return %[[ADD_RES]], %[[ADD_CARRY_OUT]] : vector<64xi32>, vector<64xi1>
func.func @vaddcs(%src0: vector<64xi32>, %src1: vector<64xi32>,
                  %carry_in: vector<64xi1>, %mask: vector<64xi1>)
    -> (vector<64xi32>, vector<64xi1>) {
  %res, %carry = ave.hir.vaddcs %src0, %src1, %carry_in, %mask
      {cast = #hivm.cast<cast_signed>}
      : vector<64xi32>, vector<64xi32>, vector<64xi1>, vector<64xi1>
        -> vector<64xi32>, vector<64xi1>
  return %res, %carry : vector<64xi32>, vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @vsubcs
// CHECK: %[[SUB_MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: %[[SUB_CARRY_IN:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: %[[SUB_AGG:.*]] = "hivm_regbaseintrins.intr.hivm.vsubcs.v64s32"({{.*}}, {{.*}}, %[[SUB_CARRY_IN]], %[[SUB_MASK]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[SUB_RES:.*]] = llvm.extractvalue %[[SUB_AGG]][0] : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[SUB_CARRY:.*]] = llvm.extractvalue %[[SUB_AGG]][1] : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
// CHECK: %[[SUB_CARRY_OUT:.*]] = builtin.unrealized_conversion_cast %[[SUB_CARRY]] : vector<256xi1> to vector<64xi1>
// CHECK: return %[[SUB_RES]], %[[SUB_CARRY_OUT]] : vector<64xi32>, vector<64xi1>
func.func @vsubcs(%src0: vector<64xi32>, %src1: vector<64xi32>,
                  %carry_in: vector<64xi1>, %mask: vector<64xi1>)
    -> (vector<64xi32>, vector<64xi1>) {
  %res, %carry = ave.hir.vsubcs %src0, %src1, %carry_in, %mask
      {cast = #hivm.cast<cast_signed>}
      : vector<64xi32>, vector<64xi32>, vector<64xi1>, vector<64xi1>
        -> vector<64xi32>, vector<64xi1>
  return %res, %carry : vector<64xi32>, vector<64xi1>
}
