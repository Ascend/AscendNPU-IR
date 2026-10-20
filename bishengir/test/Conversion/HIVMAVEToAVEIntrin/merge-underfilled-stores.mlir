// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --enable-var-scope
// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --enable-var-scope

// Effective store payload, not the producer or vector spelling, determines
// eligibility. Sparse vector stores can fill a 256-byte register. Predicate
// stores have a separate 256-bit limit. Dense unaligned prefixes share a UST
// and one final flush when no cheaper register packing is available.

// -----

// CHECK-LABEL: func.func @pack_two_i8_halves(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-COUNT-1: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK-NOT: hivm.vdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"({{.*}}, %[[DIST]], {{.*}})
// CHECK-NOT: hivm.vsts
// CHECK: return
func.func @pack_two_i8_halves(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<128xi16>, %a1: vector<128xi16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B16> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<pk>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B16> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<pk>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}

// -----

// CHECK-LABEL: func.func @pack_two_f16_halves(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-COUNT-1: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK-NOT: hivm.vdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"({{.*}}, %[[DIST]], {{.*}})
// CHECK-NOT: hivm.vsts
// CHECK: return
func.func @pack_two_f16_halves(%dst: memref<1024xf16, #hivm.address_space<ub>>, %a0: vector<64xf32>, %a1: vector<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B32> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<pk>} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B32> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<pk>} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// -----

// CHECK-LABEL: func.func @pack_four_i8_quarters(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-COUNT-3: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK-NOT: hivm.vdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"({{.*}}, %[[DIST]], {{.*}})
// CHECK-NOT: hivm.vsts
// CHECK: return
func.func @pack_four_i8_quarters(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<64xi32>, %a1: vector<64xi32>, %a2: vector<64xi32>, %a3: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  %m2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v2 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v2[%c0], %m2, %a2 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  %m3 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v3 = memref.reinterpret_cast %dst to offset: [192], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v3[%c0], %m3, %a3 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// -----

// CHECK-LABEL: func.func @pack_two_i8_quarters_preserve_padding(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(6 : i32)
// CHECK-COUNT-1: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK-NOT: hivm.vdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v128s16"({{.*}}, %[[DIST]], {{.*}})
// CHECK-NOT: hivm.vsts
// CHECK: return
func.func @pack_two_i8_quarters_preserve_padding(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<64xi32>, %a1: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK4_B32> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// -----

// Three PB32 stores cannot form a power-of-two group. Merge the first two
// into a 16-byte PB16 store and preserve the third 8-byte PB32 store.
// CHECK-LABEL: func.func @three_pb32_arguments(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NOT: hivm.pst
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NOT: hivm.pdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{[^,]+}}, {{[^,]+}}, %[[PACK]],
// CHECK-NOT: hivm.psts
// CHECK-NOT: hivm.pdintlv
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK-NOT: hivm.pst
// CHECK-NOT: hivm.pdintlv
// CHECK: return
func.func @three_pb32_arguments(%dst: memref<1024xi1, #hivm.address_space<ub>>, %a0: vector<64xi1>, %a1: vector<64xi1>, %a2: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %m2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v2 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v2[%c0], %m2, %a2 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// CHECK-LABEL: func.func @four_pb32_arguments(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NOT: hivm.pstu
// CHECK-COUNT-2: "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.pdintlv.b8"
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, %[[DIST]],
// CHECK-NOT: hivm.psts
// CHECK-NOT: hivm.pstu
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @four_pb32_arguments(%dst: memref<1024xi1, #hivm.address_space<ub>>, %a0: vector<64xi1>, %a1: vector<64xi1>, %a2: vector<64xi1>, %a3: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %m2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v2 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v2[%c0], %m2, %a2 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %m3 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v3 = memref.reinterpret_cast %dst to offset: [192], sizes: [64], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v3[%c0], %m3, %a3 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// CHECK-LABEL: func.func @pb16_arguments(
// CHECK: %[[DIST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NOT: hivm.pstu
// CHECK: "hivm_regbaseintrins.intr.hivm.pdintlv.b8"
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, %[[DIST]],
// CHECK-NOT: hivm.psts
// CHECK-NOT: hivm.pstu
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @pb16_arguments(%dst: memref<1024xi1, #hivm.address_space<ub>>, %a0: vector<128xi1>, %a1: vector<128xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [128], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<128xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb16>} : memref<128xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [128], strides: [1] : memref<1024xi1, #hivm.address_space<ub>> to memref<128xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb16>} : memref<128xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}

// -----

// Three unequal prefixes share one flush, including a nonzero start offset.
// CHECK-LABEL: func.func @dense_i8_mixed_prefixes(
// CHECK-DAG: %[[N64:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-DAG: %[[N128:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: memref.reinterpret_cast {{.*}} offset: [3]
// CHECK: %[[INIT:.*]] = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
// CHECK: %[[S0:.*]] = "hivm_regbaseintrins.intr.hivm.vstus.post.s8"(%arg1, {{.*}}, %[[N64]], %[[INIT]])
// CHECK: %[[U0:.*]] = llvm.extractvalue %[[S0]][0]
// CHECK: %[[P0:.*]] = llvm.extractvalue %[[S0]][1]
// CHECK-NOT: hivm.vstas
// CHECK: %[[S1:.*]] = "hivm_regbaseintrins.intr.hivm.vstus.post.s8"(%arg2, %[[P0]], %[[N128]], %[[U0]])
// CHECK: %[[U1:.*]] = llvm.extractvalue %[[S1]][0]
// CHECK: %[[P1:.*]] = llvm.extractvalue %[[S1]][1]
// CHECK-NOT: hivm.vstas
// CHECK: %[[S2:.*]] = "hivm_regbaseintrins.intr.hivm.vstus.post.s8"(%arg3, %[[P1]], %[[N64]], %[[U1]])
// CHECK: %[[U2:.*]] = llvm.extractvalue %[[S2]][0]
// CHECK: %[[P2:.*]] = llvm.extractvalue %[[S2]][1]
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"(%[[U2]], %[[P2]],
// CHECK-NOT: hivm.vstas
// CHECK-NOT: hivm.vstus
// CHECK: return
func.func @dense_i8_mixed_prefixes(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>, %a2: vector<256xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [3], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m1 = ave.hir.pge <VL128> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [67], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %v2 = memref.reinterpret_cast %dst to offset: [195], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v2[%c0], %m0, %a2 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// -----

// CHECK-LABEL: func.func @dense_f32_constant_tail(
// CHECK-DAG: %[[BYTES64:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-DAG: %[[BYTES128:.*]] = llvm.mlir.constant(128 : i32)
// CHECK-NOT: hivm.vstas
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"({{[^,]+}}, {{[^,]+}}, %[[BYTES64]],
// CHECK-NOT: hivm.vstas
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"({{[^,]+}}, {{[^,]+}}, %[[BYTES128]],
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: hivm.vstas
// CHECK-NOT: hivm.vstus
// CHECK: return
func.func @dense_f32_constant_tail(%dst: memref<1024xf32, #hivm.address_space<ub>>, %a0: vector<64xf32>, %a1: vector<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %n0 = arith.constant 16 : index
  %m0, %remaining0 = ave.hir.plt %n0 {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>, index
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [16], strides: [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B32> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  %n1 = arith.constant 32 : index
  %m1, %remaining1 = ave.hir.plt %n1 {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>, index
  %v1 = memref.reinterpret_cast %dst to offset: [16], sizes: [32], strides: [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B32> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// -----

// CHECK-LABEL: func.func @keep_full_vectors(
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: hivm.vst
// CHECK: return
func.func @keep_full_vectors(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [256], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [256], sizes: [256], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// -----

// CHECK-LABEL: func.func @keep_dense_aligned(
// CHECK-COUNT-2: "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"
// CHECK: return
func.func @keep_dense_aligned(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m1 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// -----

// With ALL masks this PK_B16 pair could merge. M3 leaves holes in each
// payload, so treating the stores as dense 128-byte prefixes is invalid.
// CHECK-LABEL: func.func @keep_sparse_nonprefix_mask(
// CHECK-NOT: hivm.vdintlv
// CHECK-COUNT-2: "hivm_regbaseintrins.intr.hivm.vstsx1.v128s16"
// CHECK-NOT: hivm.vdintlv
// CHECK-NOT: hivm.vst
// CHECK: return
func.func @keep_sparse_nonprefix_mask(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<128xi16>, %a1: vector<128xi16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <M3> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B16> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<pk>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  %m1 = ave.hir.pge <M3> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <PK_B16> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<pk>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}

// -----

// 3 x 128B cannot form one 256B group; the third store has its own flush.
// CHECK-LABEL: func.func @split_at_register_capacity(
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK-NOT: hivm.vstas
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: hivm.vst
// CHECK: return
func.func @split_at_register_capacity(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>, %a2: vector<256xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <VL128> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m1 = ave.hir.pge <VL128> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m2 = ave.hir.pge <VL128> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v2 = memref.reinterpret_cast %dst to offset: [256], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v2[%c0], %m2, %a2 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// -----

// Logical vector size alone does not establish a short store footprint.
// CHECK-LABEL: func.func @keep_small_vector_full_mask(
// CHECK-NOT: hivm.vdintlv
// CHECK-COUNT-2: "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"
// CHECK: return
func.func @keep_small_vector_full_mask(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<64xi8>, %a1: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<64xi8>
  %m1 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<64xi8>
  return
}

// -----

// Check every delayed store, not only the first store in the group.
// CHECK-LABEL: func.func @stop_at_read_of_second_store(
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK-NOT: hivm.vstas
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: hivm.vst
// CHECK: return
func.func @stop_at_read_of_second_store(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>, %a2: vector<256xi8>) -> vector<64xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %m1 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %read_view = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %read = ave.hir.vload <UNPK4_B8> %read_view[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %m2 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v2 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v2[%c0], %m2, %a2 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return %read : vector<64xi8>
}

// -----

// A marked cast changes aligned_ptr: the two stores cover [3, 67) and
// [131, 259). Combining their flushes would fill the gap and shift the data.
// CHECK-LABEL: func.func @keep_folded_pointer_gap(
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"(%arg1,
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s8"(%arg2,
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: hivm.vstus
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @keep_folded_pointer_gap(%dst: memref<1024xi8, #hivm.address_space<ub>>, %a0: vector<256xi8>, %a1: vector<256xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %m0 = ave.hir.pge <VL64> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [3], sizes: [64], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %m0, %a0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  %off = arith.constant 64 : index
  %shifted = memref.reinterpret_cast %dst to offset: [%off], sizes: [1024], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<1024xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %folded = memref.cast %shifted {fold_offset_into_ptr} : memref<1024xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<1024xi8, #hivm.address_space<ub>>
  %m1 = ave.hir.pge <VL128> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %v1 = memref.reinterpret_cast %folded to offset: [67], sizes: [128], strides: [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %m1, %a1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<128xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}
