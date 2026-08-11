// RUN: bishengir-opt %s -ave-normalize-ops -split-input-file | FileCheck %s

// pge ALL on a vector narrower than the register selects its first N lanes.
// With no VLn pattern for N, it becomes plt(N) at the same granularity.
// CHECK-LABEL: func.func @pge_all_200xi8
// CHECK: %[[N:.*]] = arith.constant 200 : index
// CHECK: %[[MASK:.*]], %{{.*}} = ave.hir.plt %[[N]] {functionType = #ave.func_dist_type<pb8>} : vector<200xi1>, index
// CHECK-NOT: ave.hir.pge
// CHECK: ave.hir.masked_store <NORM_B8> %{{.*}}[%{{.*}}], %[[MASK]], %{{.*}}
func.func @pge_all_200xi8(%dst: memref<200xi8, #hivm.address_space<ub>>, %val: vector<200xi8>) {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<200xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %val {functionType = #ave.func_dist_type<norm>} : memref<200xi8, #hivm.address_space<ub>>, vector<200xi1>, vector<200xi8>
  return
}

// -----

// CHECK-LABEL: func.func @pge_all_100xf16
// CHECK: %[[N:.*]] = arith.constant 100 : index
// CHECK: %[[MASK:.*]], %{{.*}} = ave.hir.plt %[[N]] {functionType = #ave.func_dist_type<pb16>} : vector<100xi1>, index
// CHECK-NOT: ave.hir.pge
// CHECK: ave.hir.masked_store <NORM_B16> %{{.*}}[%{{.*}}], %[[MASK]], %{{.*}}
func.func @pge_all_100xf16(%dst: memref<100xf16, #hivm.address_space<ub>>, %val: vector<100xf16>) {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<100xi1>
  ave.hir.masked_store <NORM_B16> %dst[%c0], %mask, %val {functionType = #ave.func_dist_type<norm>} : memref<100xf16, #hivm.address_space<ub>>, vector<100xi1>, vector<100xf16>
  return
}

// -----

// A lane count with a VLn pattern keeps using pge.
// CHECK-LABEL: func.func @pge_all_64xi8
// CHECK: ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1>
// CHECK-NOT: ave.hir.plt
func.func @pge_all_64xi8(%dst: memref<64xi8, #hivm.address_space<ub>>, %val: vector<64xi8>) {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %val {functionType = #ave.func_dist_type<norm>} : memref<64xi8, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  return
}
