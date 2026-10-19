// RUN: bishengir-opt %s -ave-normalize-ops -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -ave-normalize-ops -ave-normalize-ops -split-input-file | FileCheck %s

// PK inserts one interleave and removes functionType. Repeating normalization
// must not insert another interleave. Both uses must consume the sparse result.
// CHECK-LABEL: func.func @e2b_pk(
// CHECK: %[[LOAD:.*]] = ave.hir.vload <E2B_B16> %{{.*}}[%{{.*}}] : memref<256xf16, #hivm.address_space<ub>> into vector<64xf16>
// CHECK-NEXT: %[[SPARSE:.*]], %{{.*}} = ave.hir.vintlv %[[LOAD]], %[[LOAD]] : vector<64xf16>, vector<64xf16>
// CHECK-NEXT: return %[[SPARSE]], %[[SPARSE]] : vector<64xf16>, vector<64xf16>
func.func @e2b_pk(%src: memref<256xf16, #hivm.address_space<ub>>) -> (vector<64xf16>, vector<64xf16>) {
  %c0 = arith.constant 0 : index
  %load = ave.hir.vload <E2B_B16> %src[%c0] {functionType = #ave.func_dist_type<pk>} : memref<256xf16, #hivm.address_space<ub>> into vector<64xf16>
  return %load, %load : vector<64xf16>, vector<64xf16>
}

// -----

// PK4 inserts two interleaves; users must consume the second result.
// CHECK-LABEL: func.func @blk_pk4(
// CHECK: %[[LOAD:.*]] = ave.hir.vload <BLK> %{{.*}}[%{{.*}}] : memref<256xi8, #hivm.address_space<ub>> into vector<64xi8>
// CHECK-NEXT: %[[SPARSE:.*]], %{{.*}} = ave.hir.vintlv %[[LOAD]], %[[LOAD]] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: %[[SPARSE4:.*]], %{{.*}} = ave.hir.vintlv %[[SPARSE]], %[[SPARSE]] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: return %[[SPARSE4]], %[[SPARSE4]] : vector<64xi8>, vector<64xi8>
func.func @blk_pk4(%src: memref<256xi8, #hivm.address_space<ub>>) -> (vector<64xi8>, vector<64xi8>) {
  %c0 = arith.constant 0 : index
  %load = ave.hir.vload <BLK> %src[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<256xi8, #hivm.address_space<ub>> into vector<64xi8>
  return %load, %load : vector<64xi8>, vector<64xi8>
}
