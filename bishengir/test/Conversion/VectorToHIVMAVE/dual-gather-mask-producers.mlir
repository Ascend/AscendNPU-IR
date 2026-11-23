// RUN: bishengir-opt %s -normalize-vector -convert-vector-to-hivmave -canonicalize -cse -split-input-file | FileCheck %s

// Direct producers make coverage independent of conversion visitation order.

// CHECK-LABEL: func.func @empty
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALLF> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @empty(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [0] : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @one
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <VL1> : vector<128xi1>
// CHECK-DAG: %[[O:.*]] = ave.hir.pge <ALLF> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[O]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @one(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [1] : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @last_off
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALL> : vector<128xi1>
// CHECK-DAG: %[[NO:.*]] = arith.constant 127 : index
// CHECK-DAG: %[[O:.*]], %{{.*}} = ave.hir.plt %[[NO]] : vector<128xi1>, index
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[O]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @last_off(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [255] : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @plt_prefix
// CHECK-DAG: %[[NE:.*]] = arith.constant 100 : index
// CHECK-DAG: %[[E:.*]], %{{.*}} = ave.hir.plt %[[NE]] : vector<128xi1>, index
// CHECK-DAG: %[[NO:.*]] = arith.constant 99 : index
// CHECK-DAG: %[[O:.*]], %{{.*}} = ave.hir.plt %[[NO]] : vector<128xi1>, index
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[O]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @plt_prefix(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %bound = arith.constant 199 : index
    %mask, %remaining = ave.hir.plt %bound : vector<256xi1>, index
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// A zero PLT bound exercises a different producer from constant_mask and PGE.
// CHECK-LABEL: func.func @plt_zero
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALLF> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @plt_zero(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %bound = arith.constant 0 : index
    %mask, %remaining = ave.hir.plt %bound : vector<256xi1>, index
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @plt_overflow
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALL> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @plt_overflow(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %bound = arith.constant 300 : index
    %mask, %remaining = ave.hir.plt %bound : vector<256xi1>, index
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @pge_prefix
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <VL64> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_prefix(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <VL128> : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// H at B8 enables 128 source lanes, split into 64 lanes per gather.
// CHECK-LABEL: func.func @pge_half_b8
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <VL64> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_half_b8(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// H at B16 enables 64 source lanes, split into 32 lanes per gather.
// CHECK-LABEL: func.func @pge_half_b16
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <VL32> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_half_b16(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb16>} : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// H at B32 enables 32 source lanes, split into 16 lanes per gather.
// CHECK-LABEL: func.func @pge_half_b32
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <VL16> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_half_b32(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb32>} : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @pge_all
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALL> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_all(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <ALL> : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @pge_empty
// CHECK-DAG: %[[E:.*]] = ave.hir.pge <ALLF> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pge_empty(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = ave.hir.pge <ALLF> : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// Flatten a rank-3 prefix mask through rank 2 to the 1-D gather predicate.
// NormalizeVector prepares the multidimensional producer for conversion;
// VectorToHIVMAVE alone conservatively rejects such producers.
// CHECK-LABEL: func.func @shape_cast_chain
// CHECK-DAG: %[[NE:.*]] = arith.constant 100 : index
// CHECK-DAG: %[[E:.*]], %{{.*}} = ave.hir.plt %[[NE]] : vector<128xi1>, index
// CHECK-DAG: %[[NO:.*]] = arith.constant 99 : index
// CHECK-DAG: %[[O:.*]], %{{.*}} = ave.hir.plt %[[NO]] : vector<128xi1>, index
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[O]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shape_cast_chain(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %original = vector.constant_mask [1, 1, 199] : vector<1x1x256xi1>
    %matrix = vector.shape_cast %original : vector<1x1x256xi1> to vector<1x256xi1>
    %mask = vector.shape_cast %matrix : vector<1x256xi1> to vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @dense_false
// CHECK-DAG: %[[E:.*]] = arith.constant dense<false> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dense_false(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = arith.constant dense<false> : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// CHECK-LABEL: func.func @dense_true
// CHECK-DAG: %[[E:.*]] = arith.constant dense<true> : vector<128xi1>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vgather %{{.*}}[%{{.*}}] [%{{.*}}], %[[E]] : memref<256xi8, {{.*}}>, vector<128xi16>, vector<128xi1> into vector<128xi16>
// CHECK: ave.hir.vintlv
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dense_true(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %indices = arith.constant dense<0> : vector<128xi16>
    %mask = arith.constant dense<true> : vector<256xi1>
    %res = vector.gather %src[%c0] [%indices], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}
