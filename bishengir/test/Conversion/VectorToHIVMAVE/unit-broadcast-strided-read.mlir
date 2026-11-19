// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize=phase=backend -normalize-vector -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -canonicalize --verify-each | FileCheck %s --implicit-check-not=ave.hir.vload

// CHECK-LABEL: @unit_broadcast_strided_read
// CHECK: ave.hir.vgather
// CHECK-NOT: vector.transfer_read
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960DT_968B7">} {
  func.func @unit_broadcast_strided_read(
      %src: memref<1x16xf32, strided<[128, 8]>, #hivm.address_space<ub>>,
      %dst: memref<16xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %read = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (0, d1)>}
        : memref<1x16xf32, strided<[128, 8]>, #hivm.address_space<ub>>, vector<1x16xf32>
    %flat = vector.shape_cast %read : vector<1x16xf32> to vector<16xf32>
    vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
        : vector<16xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }

  // CHECK-LABEL: @masked_unit_broadcast_strided_read
  // CHECK: arith.constant 8 : i32
  // CHECK: ave.hir.vmuls
  // CHECK: ave.hir.pge <VL16>
  // CHECK: ave.hir.vgather
  // CHECK-NOT: vector.transfer_read
  // CHECK: return
  func.func @masked_unit_broadcast_strided_read(
      %src: memref<1x16xf32, strided<[128, 8]>, #hivm.address_space<ub>>,
      %dst: memref<16xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = vector.constant_mask [16] : vector<64xi1>
    %read = vector.transfer_read %src[%c0, %c0], %zero, %mask
        {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (0, d1)>}
        : memref<1x16xf32, strided<[128, 8]>, #hivm.address_space<ub>>, vector<1x64xf32>
    %flat = vector.shape_cast %read : vector<1x64xf32> to vector<64xf32>
    vector.transfer_write %flat, %dst[%c0], %mask {in_bounds = [true]}
        : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }
}
