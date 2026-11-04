// RUN: bishengir-opt %s --analyze-vector-layout -split-input-file | FileCheck %s

// Test for 16-bit index i8 gather whose result must be B8_4VL (8-bits in 32 bit slots).
//
// CHECK-LABEL: func.func @vgather_b8_4vl_shared_b32_mask
// CHECK: ave.hir.pge <VL2> {functionType = #ave.func_dist_type<pb32>}
// CHECK: ave.hir.vci %c0_i16, <INCREASE> {functionType = #ave.func_dist_type<intlv2>}
// CHECK: ave.hir.vgather {{.*}} memref<2x1xi8{{.*}} vector<64xi16>, vector<64xi1> into vector<64xi8>
// CHECK: ave.hir.masked_store <NORM_B8> {{.*}} {functionType = #ave.func_dist_type<pk4>}
// CHECK: ave.hir.vgather {{.*}} memref<2x1xi32{{.*}} vector<64xi32>, vector<64xi1> into vector<64xi32>
// CHECK-NOT: No Solve
func.func @vgather_b8_4vl_shared_b32_mask(%src8: memref<2x1xi8, #hivm.address_space<ub>>, %dst8: memref<2xi8, #hivm.address_space<ub>>, %src32: memref<2x1xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c0_i16 = arith.constant 0 : i16
  %c0_i32 = arith.constant 0 : i32
  %mask = ave.hir.pge <VL2> : vector<64xi1>
  %idx8 = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<64xi16>
  %data = ave.hir.vgather %src8[%c0, %c0] [%idx8], %mask : memref<2x1xi8, #hivm.address_space<ub>>, vector<64xi16>, vector<64xi1> into vector<64xi8>
  ave.hir.masked_store <NORM_B8> %dst8[%c0], %mask, %data : memref<2xi8, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  %idx32 = ave.hir.vci %c0_i32, <INCREASE> : i32, vector<64xi32>
  %addr = ave.hir.vgather %src32[%c0, %c0] [%idx32], %mask : memref<2x1xi32, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xi32>
  return
}
