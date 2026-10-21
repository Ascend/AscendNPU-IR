// RUN: bishengir-opt %s --analyze-vector-layout -split-input-file | FileCheck %s

// A B8 dual gather: each native gather returns its bytes in 16-bit slots
// (vector<128xi16>), each half is packed dense with vpack, and the two halves
// are interleaved. The mask is shared between the gathers and the store, so
// the B8 gathers must accept a b8 mask for the store to stay a plain dense
// NORM store of all 256 lanes (a b16 mask would force a packing store).
// CHECK-LABEL: func.func @dual_gather_vpack_shared_mask
// CHECK: ave.hir.plt {{.*}} {functionType = #ave.func_dist_type<pb8>}
// CHECK-COUNT-2: ave.hir.vpack
// CHECK: ave.hir.vintlv
// CHECK-NOT: intlv2
// CHECK: ave.hir.masked_store <NORM_B8> {{.*}} {functionType = #ave.func_dist_type<norm>}
func.func @dual_gather_vpack_shared_mask(%src: memref<256xi8, #hivm.address_space<ub>>, %dst: memref<200xi8, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c200 = arith.constant 200 : index
  %c0_i16 = arith.constant 0 : i16
  %c1_i16 = arith.constant 1 : i16
  %mask, %n = ave.hir.plt %c200 : vector<256xi1>, index
  %idx0 = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
  %idx1 = ave.hir.vci %c1_i16, <INCREASE> : i16, vector<128xi16>
  %g0 = ave.hir.vgather %src[%c0] [%idx0], %mask : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<128xi16>
  %g1 = ave.hir.vgather %src[%c0] [%idx1], %mask : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<128xi16>
  %p0 = ave.hir.vpack %g0, 0 : vector<128xi16> -> vector<256xi8>
  %p1 = ave.hir.vpack %g1, 0 : vector<128xi16> -> vector<256xi8>
  %r1, %r2 = ave.hir.vintlv %p0, %p1 : vector<256xi8>, vector<256xi8>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %r1 : memref<200xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// -----

// vpack b32 -> b16: a dense b16 result comes from a dense b32 source.
// CHECK-LABEL: func.func @vpack_b32_to_b16
// CHECK: ave.hir.vload <NORM> {{.*}} {functionType = #ave.func_dist_type<norm>}
// CHECK: ave.hir.vpack
// CHECK: ave.hir.masked_store <NORM_B16> {{.*}} {functionType = #ave.func_dist_type<norm>}
func.func @vpack_b32_to_b16(%src: memref<64xi32, #hivm.address_space<ub>>, %dst: memref<128xi16, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %v = ave.hir.vload <NORM> %src[%c0] : memref<64xi32, #hivm.address_space<ub>> into vector<64xi32>
  %p = ave.hir.vpack %v, 0 : vector<64xi32> -> vector<128xi16>
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  ave.hir.masked_store <NORM_B16> %dst[%c0], %mask, %p : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}
