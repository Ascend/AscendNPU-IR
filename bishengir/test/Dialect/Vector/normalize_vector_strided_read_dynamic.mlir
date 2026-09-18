// RUN: bishengir-opt %s -normalize-vector -mlir-print-elementsattrs-with-hex-if-larger=-1 -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -split-input-file | FileCheck %s --check-prefix=INTRIN

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // A 96-element slice has one full vector and a 32-lane tail. Both iterations
  // must use stride-2 offsets and retain their original runtime mask.
  // CHECK-LABEL: func.func @stride2_tail
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, {{.*}}, 124, 126]> : vector<64xi32>
  // CHECK: scf.for
  // CHECK: %[[SIZE:.*]] = affine.min
  // CHECK: %[[BASE:.*]] = affine.apply
  // CHECK: %[[VIEW:.*]] = memref.subview {{.*}}[%[[BASE]]] [%[[SIZE]]] [2]
  // CHECK: %[[MASK:.*]] = vector.create_mask %[[SIZE]] : vector<64xi1>
  // CHECK-NOT: vector.transfer_read
  // CHECK: %[[READ:.*]] = vector.gather %[[VIEW]][{{.*}}] [%[[INDEX]]], %[[MASK]],
  // CHECK: vector.transfer_write %[[READ]], {{.*}}, %[[MASK]]
  // INTRIN-LABEL: func.func @stride2_tail
  // INTRIN-NOT: intr.hivm.vldsx1
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  // INTRIN-NOT: intr.hivm.vldsx1
  // INTRIN: return
  func.func @stride2_tail(%src: memref<1024xf32, #hivm.address_space<ub>>, %dst: memref<96xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c96 step %c64 {
      %len = affine.min affine_map<(d0) -> (96 - d0, 64)>(%i)
      %base = affine.apply affine_map<(d0) -> (256 + d0 * 2)>(%i)
      %view = memref.subview %src[%base] [%len] [2] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
      %mask = vector.create_mask %len : vector<64xi1>
      %v = vector.transfer_read %view[%c0], %zero, %mask {in_bounds = [true]} : memref<?xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      vector.transfer_write %v, %dst[%i], %mask {in_bounds = [true]} : vector<64xf32>, memref<96xf32, #hivm.address_space<ub>>
    }
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Dynamic source extent does not require a mask when the read is in-bounds.
  // CHECK-LABEL: func.func @dynamic_unmasked
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 2, 4, {{.*}}, 124, 126]> : vector<64xi32>
  // CHECK: %[[MASK:.*]] = vector.constant_mask [64]
  // CHECK: vector.gather {{.*}}[%[[INDEX]]], %[[MASK]],
  // INTRIN-LABEL: func.func @dynamic_unmasked
  // INTRIN: intr.hivm.vgather2_v300.v64f32
  func.func @dynamic_unmasked(%src: memref<?xf32, strided<[2]>, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<?xf32, strided<[2]>, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Preserve the runtime mask independently of whether the source is static.
  // Storing the whole result also keeps masked-off zero padding observable.
  // CHECK-LABEL: func.func @static_source_dynamic_mask
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 3, 6, {{.*}}, 378, 381]> : vector<128xi16>
  // CHECK: %[[MASK:.*]] = vector.create_mask
  // CHECK: vector.gather {{.*}}[%[[INDEX]]], %[[MASK]],
  // INTRIN-LABEL: func.func @static_source_dynamic_mask
  // INTRIN: intr.hivm.vgather2_v300.v128f16
  func.func @static_source_dynamic_mask(%src: memref<128xf16, strided<[3]>, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>, %n: index) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %mask = vector.create_mask %n : vector<128xi1>
    %v = vector.transfer_read %src[%c0], %zero, %mask {in_bounds = [true]} : memref<128xf16, strided<[3]>, #hivm.address_space<ub>>, vector<128xf16>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Only the active indices must fit i16. Inactive lanes must remain zero:
  // computing all 128 offsets would overflow at lane 64 and reject the gather.
  // CHECK-LABEL: func.func @constant_mask_index_range
  // CHECK: %[[INDEX:.*]] = arith.constant dense<[0, 1024, {{.*}}, 31744, 0, {{.*}}]> : vector<128xi16>
  // CHECK: %[[MASK:.*]] = vector.constant_mask [32]
  // CHECK: vector.gather {{.*}}[%[[INDEX]]], %[[MASK]],
  // INTRIN-LABEL: func.func @constant_mask_index_range
  // INTRIN: intr.hivm.vgather2_v300.v128f16
  func.func @constant_mask_index_range(%src: memref<32xf16, strided<[1024]>, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %mask = vector.constant_mask [32] : vector<128xi1>
    %v = vector.transfer_read %src[%c0], %zero, %mask {in_bounds = [true]} : memref<32xf16, strided<[1024]>, #hivm.address_space<ub>>, vector<128xf16>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, #hivm.address_space<ub>>
    return
  }
}
