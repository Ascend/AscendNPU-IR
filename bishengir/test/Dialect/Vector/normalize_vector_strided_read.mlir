// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -normalize-vector -convert-vector-to-hivmave -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -split-input-file | FileCheck %s --check-prefix=INTRIN

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Each iteration reads x[2*i + 2*lane], not x[2*i + lane].
  // NORMALIZE-LABEL: func.func @stride2_loop
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride2_loop
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, {{.*}}, 124, 126]> : vector<64xi32>
  // CHECK: scf.for
  // CHECK: %[[SLICE:.*]] = memref.subview
  // CHECK-NOT: ave.hir.vload
  // CHECK: ave.hir.vgather %[[SLICE]][{{.*}}] [%[[INDEX]]]
  // CHECK: ave.hir.masked_store
  // INTRIN-LABEL: func.func @stride2_loop
  // INTRIN-NOT: intr.hivm.vldsx1
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  // INTRIN-NOT: intr.hivm.vldsx1
  func.func @stride2_loop(%src: memref<1024xf32, #hivm.address_space<ub>>, %dst: memref<256xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c256 step %c64 {
      %base = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %slice = memref.subview %src[%base] [64] [2] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
      %v = vector.transfer_read %slice[%c0], %zero {in_bounds = [true]} : memref<64xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      vector.transfer_write %v, %dst[%i] {in_bounds = [true]} : vector<64xf32>, memref<256xf32, #hivm.address_space<ub>>
    }
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // The read index is logical: address = 32 + (8 + lane) * 3.
  // NORMALIZE-LABEL: func.func @stride3_nonzero_index
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride3_nonzero_index
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 3, 6, {{.*}}, 186, 189]> : vector<64xi32>
  // CHECK: %[[EIGHT:.*]] = arith.constant 8 : index
  // CHECK: %[[SLICE:.*]] = memref.subview {{.*}}[32] [72] [3]
  // CHECK: ave.hir.vgather %[[SLICE]][%[[EIGHT]]] [%[[INDEX]]]
  // INTRIN-LABEL: func.func @stride3_nonzero_index
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  func.func @stride3_nonzero_index(%src: memref<256xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %slice = memref.subview %src[32] [72] [3] : memref<256xf32, #hivm.address_space<ub>> to memref<72xf32, strided<[3], offset: 32>, #hivm.address_space<ub>>
    %v = vector.transfer_read %slice[%c8], %zero {in_bounds = [true]} : memref<72xf32, strided<[3], offset: 32>, #hivm.address_space<ub>>, vector<64xf32>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @stride2_f16
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride2_f16
  // CHECK: ave.hir.vgather {{.*}} into vector<128xf16>
  // INTRIN-LABEL: func.func @stride2_f16
  // INTRIN: intr.hivm.vgather2_v300.v128f16
  func.func @stride2_f16(%src: memref<128xf16, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<128xf16, strided<[2]>, #hivm.address_space<ub>>, vector<128xf16>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @stride2_bf16
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride2_bf16
  // CHECK: ave.hir.vgather {{.*}} into vector<128xbf16>
  // INTRIN-LABEL: func.func @stride2_bf16
  // INTRIN: intr.hivm.vgather2_v300.v128bf16
  func.func @stride2_bf16(%src: memref<128xbf16, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<128xbf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : bf16
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<128xbf16, strided<[2]>, #hivm.address_space<ub>>, vector<128xbf16>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<128xbf16>, memref<128xbf16, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @stride2_i16
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride2_i16
  // CHECK: ave.hir.vgather {{.*}} into vector<128xi16>
  // INTRIN-LABEL: func.func @stride2_i16
  // INTRIN: intr.hivm.vgather2_v300.v128s16
  func.func @stride2_i16(%src: memref<128xi16, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<128xi16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i16
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<128xi16, strided<[2]>, #hivm.address_space<ub>>, vector<128xi16>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<128xi16>, memref<128xi16, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @stride2_i32
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: vector.gather
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @stride2_i32
  // CHECK: ave.hir.vgather {{.*}} into vector<64xi32>
  // INTRIN-LABEL: func.func @stride2_i32
  // INTRIN: intr.hivm.vgather2_v300.v64s32
  func.func @stride2_i32(%src: memref<64xi32, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<64xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<64xi32, strided<[2]>, #hivm.address_space<ub>>, vector<64xi32>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<64xi32>, memref<64xi32, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // A constant partial mask remains on the existing gather path even though
  // the identity map has no transposed dimensions.
  // NORMALIZE-LABEL: func.func @masked_zero_padding
  // NORMALIZE-NOT: vector.transfer_read
  // NORMALIZE: %[[MASK:.*]] = vector.constant_mask [32]
  // NORMALIZE: vector.gather {{.*}}, %[[MASK]],
  // NORMALIZE-NOT: vector.transfer_read
  // CHECK-LABEL: func.func @masked_zero_padding
  // CHECK: %[[MASK:.*]] = ave.hir.pge <VL32>
  // CHECK: ave.hir.vgather {{.*}}, %[[MASK]]
  // INTRIN-LABEL: func.func @masked_zero_padding
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  func.func @masked_zero_padding(%src: memref<32xf32, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [32] : vector<64xi1>
    %v = vector.transfer_read %src[%c0], %zero, %mask {in_bounds = [true]} : memref<32xf32, strided<[2]>, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @contiguous
  // NORMALIZE-NOT: vector.gather
  // NORMALIZE: vector.transfer_read
  // CHECK-LABEL: func.func @contiguous
  // CHECK-NOT: ave.hir.vgather
  // CHECK: ave.hir.vload <NORM>
  // INTRIN-LABEL: func.func @contiguous
  // INTRIN: intr.hivm.vldsx1.v64f32
  func.func @contiguous(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %out_zero = arith.constant 0 : index
    vector.transfer_write %v, %dst[%out_zero] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
