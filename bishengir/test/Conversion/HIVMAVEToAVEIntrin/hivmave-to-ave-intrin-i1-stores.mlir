// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin -cse %s -split-input-file | FileCheck %s

// Only aligned NORM_B8 PB16 stores use PPACK/PSTS.
// PB32, other distributions, and unaligned stores retain PSTU/VSTAS.

// -----

// CHECK-LABEL: @aligned_pb8
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.pstu
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: return
func.func @aligned_pb8(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<256xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #ave.func_dist_type<pb8>} : memref<512xi1, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
  return
}

// -----

// CHECK-LABEL: @aligned_pb16
// CHECK: %[[LOW16:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[PACK16:.*]] = "hivm_regbaseintrins.intr.hivm.ppack.z"({{.*}}, %[[LOW16]])
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.pstu
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[PACK16]],
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: return
func.func @aligned_pb16(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #ave.func_dist_type<pb16>} : memref<512xi1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}

// -----

// CHECK-LABEL: @aligned_pb32
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK: return
func.func @aligned_pb32(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<64xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #ave.func_dist_type<pb32>} : memref<512xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// CHECK-LABEL: @unaligned_pb16
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK: return
func.func @unaligned_pb16(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 128 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb16>} : memref<512xi1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}

// -----

// CHECK-LABEL: @unaligned_pb32
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK: return
func.func @unaligned_pb32(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<64xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 128 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<512xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// ONEPT is outside the ordinary aligned PB16 workaround, even without the
// continuous-store marker that would select the earlier dedicated branch.
// CHECK-LABEL: @onept_pb16
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK: return
func.func @onept_pb16(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  ave.hir.masked_store <ONEPT_B8> %dst[%c0], %mask, %data {functionType = #ave.func_dist_type<pb16>} : memref<512xi1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}

// -----

// An unaligned ONEPT store bypasses the first !isONEPTDist branch. The explicit
// guard must still retain the original sparse-predicate store in this case.
// CHECK-LABEL: @unaligned_onept_pb16
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.psts.b8"
// CHECK: return
func.func @unaligned_onept_pb16(%dst: memref<512xi1, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c128 = arith.constant 128 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  ave.hir.masked_store <ONEPT_B8> %dst[%c128], %mask, %data {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb16>} : memref<512xi1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}
