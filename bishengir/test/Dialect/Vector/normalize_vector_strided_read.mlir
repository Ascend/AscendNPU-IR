// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -split-input-file | FileCheck %s --check-prefix=INTRIN

// Physical addresses are 32 + (8 + lane) * 2; the identity map keeps stride 2.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @stride2
  // NORMALIZE-DAG: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, {{.*}}, 124, 126]> : vector<64xi32>
  // NORMALIZE-DAG: %[[START:.*]] = arith.constant 8 : index
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: %[[VALUE:.*]] = vector.gather {{.*}}[%[[START]]] [%[[INDEX]]],
  // NORMALIZE: vector.transfer_write %[[VALUE]]
  // INTRIN-LABEL: func.func @stride2
  // INTRIN-NOT: intr.hivm.vldsx1
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  // INTRIN-NOT: intr.hivm.vldsx1
  func.func @stride2(%src: memref<72xf32, strided<[2], offset: 32>, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c8], %zero {in_bounds = [true]} : memref<72xf32, strided<[2], offset: 32>, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

// Cover 16-bit indices, stride 3 and an identity read with a partial mask.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @masked_f16
  // NORMALIZE: %[[MASK:.*]] = vector.constant_mask [64] : vector<128xi1>
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather {{.*}}, %[[MASK]], {{.*}} : {{.*}}vector<128xi16>
  // INTRIN-LABEL: func.func @masked_f16
  // INTRIN: intr.hivm.vgather2_v300.v128f16
  func.func @masked_f16(%src: memref<64xf16, strided<[3]>, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %mask = vector.constant_mask [64] : vector<128xi1>
    %v = vector.transfer_read %src[%c0], %zero, %mask {in_bounds = [true]} : memref<64xf16, strided<[3]>, #hivm.address_space<ub>>, vector<128xf16>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @contiguous
  // NORMALIZE-NOT: vector.gather
  // NORMALIZE: vector.transfer_read
  // INTRIN-LABEL: func.func @contiguous
  // INTRIN: intr.hivm.vldsx1.v64f32
  func.func @contiguous(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
