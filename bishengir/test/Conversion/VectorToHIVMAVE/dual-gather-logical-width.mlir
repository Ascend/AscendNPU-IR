// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file | FileCheck %s

// A B8 gather with a secondary_index is lowered to two native sub-gathers,
// each returning its bytes in 16-bit slots, packed dense and interleaved. The
// interleave produces the gather's own type directly, so consumers keep operating on the logical width.
// CHECK-LABEL: func.func @dual_gather_200xi8
// CHECK: %[[G0:.*]] = ave.hir.vgather {{.*}} : memref<200x1xi8, {{.*}}>, vector<128xi16>, vector<200xi1> into vector<128xi16>
// CHECK: %[[G1:.*]] = ave.hir.vgather {{.*}} : memref<200x1xi8, {{.*}}>, vector<128xi16>, vector<200xi1> into vector<128xi16>
// CHECK: %[[P0:.*]] = ave.hir.vpack %[[G0]], 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[P1:.*]] = ave.hir.vpack %[[G1]], 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[RES:.*]], %{{.*}} = ave.hir.vintlv %[[P0]], %[[P1]] : vector<256xi8>, vector<200xi8>
// CHECK: arith.addi %[[RES]], %[[RES]] : vector<200xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_200xi8(%src: memref<200x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>) -> vector<200xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<200xi8>
    %idx0 = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [200] : vector<200xi1>
    %0 = vector.gather %src[%c0, %c0] [%idx0], %mask, %pass {secondary_index = dense<16> : vector<128xi16>} : memref<200x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<128xi16>, vector<200xi1>, vector<200xi8> into vector<200xi8>
    %1 = arith.addi %0, %0 : vector<200xi8>
    return %1 : vector<200xi8>
  }
}

// -----

// The full-width case the vectorizer produces for an i8-only chain: the
// interleave result is the whole 256-lane register.
// CHECK-LABEL: func.func @dual_gather_256xi8
// CHECK: %[[G0:.*]] = ave.hir.vgather {{.*}} : memref<256x1xi8, {{.*}}>, vector<128xi16>, vector<256xi1> into vector<128xi16>
// CHECK: %[[G1:.*]] = ave.hir.vgather {{.*}} : memref<256x1xi8, {{.*}}>, vector<128xi16>, vector<256xi1> into vector<128xi16>
// CHECK: %[[P0:.*]] = ave.hir.vpack %[[G0]], 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[P1:.*]] = ave.hir.vpack %[[G1]], 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[RES:.*]], %{{.*}} = ave.hir.vintlv %[[P0]], %[[P1]] : vector<256xi8>, vector<256xi8>
// CHECK: arith.addi %[[RES]], %[[RES]] : vector<256xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_256xi8(%src: memref<256x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %idx0 = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [256] : vector<256xi1>
    %0 = vector.gather %src[%c0, %c0] [%idx0], %mask, %pass {secondary_index = dense<16> : vector<128xi16>} : memref<256x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    %1 = arith.addi %0, %0 : vector<256xi8>
    return %1 : vector<256xi8>
  }
}

// -----

// vpack only produces i8, so for f8 each packed half is bitcast to the f8
// register type before the interleave. Without it the interleave would take
// i8 sources while producing f8, leaving i8 -> f8 casts that cannot be
// reconciled when lowering to LLVM.
// CHECK-LABEL: func.func @dual_gather_200xf8E4M3FN
// CHECK: %[[P0:.*]] = ave.hir.vpack %{{.*}}, 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[F0:.*]] = arith.bitcast %[[P0]] : vector<256xi8> to vector<256xf8E4M3FN>
// CHECK: %[[P1:.*]] = ave.hir.vpack %{{.*}}, 0 : vector<128xi16> -> vector<256xi8>
// CHECK: %[[F1:.*]] = arith.bitcast %[[P1]] : vector<256xi8> to vector<256xf8E4M3FN>
// CHECK: ave.hir.vintlv %[[F0]], %[[F1]] : vector<256xf8E4M3FN>, vector<200xf8E4M3FN>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_200xf8E4M3FN(%src: memref<200x1xf8E4M3FN, strided<[16, 1]>, #hivm.address_space<ub>>) -> vector<200xf8E4M3FN> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0.0> : vector<200xf8E4M3FN>
    %idx0 = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [200] : vector<200xi1>
    %0 = vector.gather %src[%c0, %c0] [%idx0], %mask, %pass {secondary_index = dense<16> : vector<128xi16>} : memref<200x1xf8E4M3FN, strided<[16, 1]>, #hivm.address_space<ub>>, vector<128xi16>, vector<200xi1>, vector<200xf8E4M3FN> into vector<200xf8E4M3FN>
    return %0 : vector<200xf8E4M3FN>
  }
}
