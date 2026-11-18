// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize=phase=backend -convert-vector-to-hivmave=enable-triton-compile=true -convert-arith-to-hivmave -analyze-vector-layout -canonicalize -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -canonicalize --verify-each | FileCheck %s --implicit-check-not=vector.extract --implicit-check-not=vector.reduction --implicit-check-not=vector.broadcast --implicit-check-not=ave.hir.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: @unit_parallel_reduction
  // CHECK: hivm_regbaseintrins.intr.hivm.
  // CHECK: return
  func.func @unit_parallel_reduction(
      %src: memref<1x64xf32, #hivm.address_space<ub>>,
      %dst: memref<1xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %read = vector.transfer_read %src[%c0, %c0], %zero
        {in_bounds = [true, true]}
        : memref<1x64xf32, #hivm.address_space<ub>>, vector<1x64xf32>
    %row = vector.extract %read[0] : vector<64xf32> from vector<1x64xf32>
    %init = vector.transfer_read %dst[%c0], %zero {in_bounds = [true]}
        : memref<1xf32, #hivm.address_space<ub>>, vector<1xf32>
    %scalar = vector.extractelement %init[%c0 : index] : vector<1xf32>
    %sum = vector.reduction <add>, %row, %scalar : vector<64xf32> into f32
    %result = vector.broadcast %sum : f32 to vector<1xf32>
    vector.transfer_write %result, %dst[%c0] {in_bounds = [true]}
        : vector<1xf32>, memref<1xf32, #hivm.address_space<ub>>
    return
  }

  // CHECK-LABEL: @rank_zero_reduction_result
  // CHECK: hivm_regbaseintrins.intr.hivm.
  // CHECK: return
  func.func @rank_zero_reduction_result(
      %src: memref<4xi32, #hivm.address_space<ub>>,
      %dst: memref<i32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %init = arith.constant 2147483647 : i32
    %read = vector.transfer_read %src[%c0], %zero {in_bounds = [true]}
        : memref<4xi32, #hivm.address_space<ub>>, vector<4xi32>
    %min = vector.reduction <minsi>, %read, %init : vector<4xi32> into i32
    %result = vector.broadcast %min : i32 to vector<i32>
    vector.transfer_write %result, %dst[]
        : vector<i32>, memref<i32, #hivm.address_space<ub>>
    return
  }
}
