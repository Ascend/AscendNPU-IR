// RUN: bishengir-opt %s -normalize-vector -convert-vector-to-hivmave -canonicalize -cse -split-input-file | FileCheck %s

// Split the logical mask into lane-wise 128-lane predicates:
// even[j] = mask[2*j], odd[j] = mask[2*j+1].
// A 199-element prefix must enable 100 even and 99 odd reads.
// Use an allocated 256-byte source so mask correctness is independent of
// safe-index padding.
// CHECK-LABEL: func.func @dual_gather_partial_199
// CHECK-DAG: %[[NE:.*]] = arith.constant 100 : index
// CHECK-DAG: %[[ME:.*]], %{{.*}} = ave.hir.plt %[[NE]] : vector<128xi1>, index
// CHECK-DAG: %[[NO:.*]] = arith.constant 99 : index
// CHECK-DAG: %[[MO:.*]], %{{.*}} = ave.hir.plt %[[NO]] : vector<128xi1>, index
// CHECK: %[[GE:.*]] = ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[ME]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: %[[GO:.*]] = ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[MO]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vpack %[[GE]], 0
// CHECK: ave.hir.vpack %[[GO]], 0
// CHECK: ave.hir.vintlv

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_partial_199(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %even = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [199] : vector<256xi1>
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----

// A 200-element prefix must enable 100 even and 100 odd reads.
// CHECK-LABEL: func.func @dual_gather_partial_200
// CHECK-DAG: %[[NE:.*]] = arith.constant 100 : index
// CHECK-DAG: %[[ME:.*]], %{{.*}} = ave.hir.plt %[[NE]] : vector<128xi1>, index
// CHECK: %[[GE:.*]] = ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[ME]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: %[[GO:.*]] = ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[ME]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vpack %[[GE]], 0
// CHECK: ave.hir.vpack %[[GO]], 0
// CHECK: ave.hir.vintlv

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_partial_200(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %even = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [200] : vector<256xi1>
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}
