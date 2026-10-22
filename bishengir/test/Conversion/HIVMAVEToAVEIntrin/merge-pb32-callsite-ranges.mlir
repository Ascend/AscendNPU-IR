// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN: -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --implicit-check-not=_packed_stores --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts --implicit-check-not=hivm.pstu

// -----

// Absolute offsets are unknown, but the load starts exactly 64 bits after
// the first store: (idx * 32 + 8) * 8 - idx * 256 = 64.
// CHECK-LABEL: func.func @dynamic_touching_ranges(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @dynamic_touching_ranges(%dst: memref<4096xi1, #hivm.address_space<ub>>, %src: memref<512xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>, %idx: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %dst_offset = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%idx]
  %src_offset = affine.apply affine_map<()[s0] -> (s0 * 32 + 8)>()[%idx]
  %next_offset = affine.apply affine_map<()[s0] -> (s0 * 256 + 64)>()[%idx]
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [%dst_offset], sizes: [64], strides: [1] : memref<4096xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [%src_offset], sizes: [64], strides: [1] : memref<512xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [%next_offset], sizes: [64], strides: [1] : memref<4096xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_dynamic_touching_ranges(
// CHECK: call @dynamic_touching_ranges(
// CHECK: call @dynamic_touching_ranges(
// CHECK: return
func.func @caller_dynamic_touching_ranges(%a: vector<64xi8>, %b: vector<64xi8>, %idx: index) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<4096xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<512xi8, #hivm.address_space<ub>>
  func.call @dynamic_touching_ranges(%d, %sbase, %a, %b, %idx) : (memref<4096xi1, #hivm.address_space<ub>>, memref<512xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>, index) -> ()
  %alt = arith.constant 8192 : i64
  %other = hivm.hir.pointer_cast(%alt) : memref<4096xi1, #hivm.address_space<ub>>
  func.call @dynamic_touching_ranges(%other, %sbase, %a, %b, %idx) : (memref<4096xi1, #hivm.address_space<ub>>, memref<512xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>, index) -> ()
  return
}

// -----

// The same dynamic base cancels, but a 56-bit delta overlaps the 64-bit
// first store by one byte. Do not sink that store past the load.
// CHECK-LABEL: func.func @dynamic_one_byte_overlap(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @dynamic_one_byte_overlap(%dst: memref<4096xi1, #hivm.address_space<ub>>, %src: memref<512xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>, %idx: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %dst_offset = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%idx]
  %src_offset = affine.apply affine_map<()[s0] -> (s0 * 32 + 7)>()[%idx]
  %next_offset = affine.apply affine_map<()[s0] -> (s0 * 256 + 64)>()[%idx]
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [%dst_offset], sizes: [64], strides: [1] : memref<4096xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [%src_offset], sizes: [64], strides: [1] : memref<512xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [%next_offset], sizes: [64], strides: [1] : memref<4096xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_dynamic_one_byte_overlap(
// CHECK: call @dynamic_one_byte_overlap(
// CHECK: return
func.func @caller_dynamic_one_byte_overlap(%a: vector<64xi8>, %b: vector<64xi8>, %idx: index) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<4096xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<512xi8, #hivm.address_space<ub>>
  func.call @dynamic_one_byte_overlap(%d, %sbase, %a, %b, %idx) : (memref<4096xi1, #hivm.address_space<ub>>, memref<512xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>, index) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @unknown_offset(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @unknown_offset(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>, %idx: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [%idx], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_unknown_offset(
// CHECK: call @unknown_offset(
// CHECK: return
func.func @caller_unknown_offset(%idx: index, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @unknown_offset(%d, %sbase, %a, %b, %idx) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>, index) -> ()
  return
}

// -----

// PlanMemory keeps these distinct allocation bases disjoint. The intervening
// broadcast has a runtime offset and an unmodeled footprint, neither of which
// is needed for the allocation-level proof. Check all selects and call sites.
// CHECK-LABEL: func.func @distinct_bases_dynamic_broadcast(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @distinct_bases_dynamic_broadcast(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<1024xf32, #hivm.address_space<ub>>, %a: vector<64xf32>, %b: vector<64xf32>, %idx: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xf32>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [%idx], sizes: [1], strides: [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <BRC_B32> %s[%c0] {functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xf32>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_distinct_bases_dynamic_broadcast(
// CHECK: call @distinct_bases_dynamic_broadcast(
// CHECK: call @distinct_bases_dynamic_broadcast(
// CHECK: return
func.func @caller_distinct_bases_dynamic_broadcast(%choose: i1, %idx: index, %a: vector<64xf32>, %b: vector<64xf32>) {
  %dc = arith.constant 0 : i64
  %sc0 = arith.constant 4096 : i64
  %sc1 = arith.constant 8192 : i64
  %sc2 = arith.constant 12288 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %s0 = hivm.hir.pointer_cast(%sc0) : memref<1024xf32, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%sc1) : memref<1024xf32, #hivm.address_space<ub>>
  %s2 = hivm.hir.pointer_cast(%sc2) : memref<1024xf32, #hivm.address_space<ub>>
  %s = arith.select %choose, %s0, %s1 : memref<1024xf32, #hivm.address_space<ub>>
  func.call @distinct_bases_dynamic_broadcast(%d, %s, %a, %b, %idx) : (memref<256xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, vector<64xf32>, vector<64xf32>, index) -> ()
  func.call @distinct_bases_dynamic_broadcast(%d, %s2, %a, %b, %idx) : (memref<256xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, vector<64xf32>, vector<64xf32>, index) -> ()
  return
}

// -----

// The first call uses a distinct allocation. The second selects a base that
// may overlap the load. Check every call and every select alternative.
// CHECK-LABEL: func.func @multiple_calls_one_overlap(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @multiple_calls_one_overlap(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [0], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_multiple_calls_one_overlap(
// CHECK: call @multiple_calls_one_overlap(
// CHECK: call @multiple_calls_one_overlap(
// CHECK: return
func.func @caller_multiple_calls_one_overlap(%choose: i1, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 8192 : i64
  %sc = arith.constant 0 : i64
  %overlap = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @multiple_calls_one_overlap(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  %d2 = hivm.hir.pointer_cast(%overlap) : memref<256xi1, #hivm.address_space<ub>>
  %selected = arith.select %choose, %d, %d2 : memref<256xi1, #hivm.address_space<ub>>
  func.call @multiple_calls_one_overlap(%selected, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multiple_calls_one_unknown(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @multiple_calls_one_unknown(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_multiple_calls_one_unknown(
// CHECK: call @multiple_calls_one_unknown(
// CHECK: call @multiple_calls_one_unknown(
// CHECK: return
func.func @caller_multiple_calls_one_unknown(%unknown: i64, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @multiple_calls_one_unknown(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  %d2 = hivm.hir.pointer_cast(%unknown) : memref<256xi1, #hivm.address_space<ub>>
  func.call @multiple_calls_one_unknown(%d2, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @wrapped_address(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @wrapped_address(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_wrapped_address(
// CHECK: call @wrapped_address(
// CHECK: return
func.func @caller_wrapped_address(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 4294967296 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @wrapped_address(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @offset_unit_overflow(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @offset_unit_overflow(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [2305843009213694016], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_offset_unit_overflow(
// CHECK: call @offset_unit_overflow(
// CHECK: return
func.func @caller_offset_unit_overflow(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @offset_unit_overflow(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// Both arguments originate from allocation base 0; the store offset is 128B.
// The marked cast folds offset 64 into the source pointer. The VF load
// is at [128, 192), overlapping the first store at [128, 136).
// CHECK-LABEL: func.func @marked_cast_source_offset(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @marked_cast_source_offset(%dst: memref<2048xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [1024], sizes: [64], strides: [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [1088], sizes: [64], strides: [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_marked_cast_source_offset(
// CHECK: call @marked_cast_source_offset(
// CHECK: return
func.func @caller_marked_cast_source_offset(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<2048xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %off = arith.constant 64 : index
  %sview = memref.reinterpret_cast %sbase to offset: [%off], sizes: [256], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %folded = memref.cast %sview {fold_offset_into_ptr} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<256xi8, #hivm.address_space<ub>>
  func.call @marked_cast_source_offset(%d, %folded, %a, %b) : (memref<2048xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// Both arguments originate from allocation base 0; the store offset is 128B.
// An ordinary cast preserves aligned_ptr. The VF reinterpret_cast replaces
// offset 64, so the load is at [64, 128), disjoint from the store at 128.
// CHECK-LABEL: func.func @ordinary_cast_source_offset(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @ordinary_cast_source_offset(%dst: memref<2048xi1, #hivm.address_space<ub>>, %src: memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [1024], sizes: [64], strides: [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [1088], sizes: [64], strides: [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_ordinary_cast_source_offset(
// CHECK: call @ordinary_cast_source_offset(
// CHECK: return
func.func @caller_ordinary_cast_source_offset(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<2048xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %off = arith.constant 64 : index
  %sview = memref.reinterpret_cast %sbase to offset: [%off], sizes: [256], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %folded = memref.cast %sview : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  func.call @ordinary_cast_source_offset(%d, %folded, %a, %b) : (memref<2048xi1, #hivm.address_space<ub>>, memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}
