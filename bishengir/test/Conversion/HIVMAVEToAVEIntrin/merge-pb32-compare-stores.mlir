// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin -cse | FileCheck %s \
// RUN: --implicit-check-not=hivm.ppack --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts

// Match normalized PB32 comparison stores. A successful pair becomes one
// PDINTLV.B16 followed by PSTS DIST_PACK=1 (16 bytes), preserving predicate
// order and the original comparison results for other users.

// -----

// Adjacent full stores merge even when their comparisons use a partial mask.
// CHECK-LABEL: @merge_adjacent
// CHECK-NOT: hivm.pstu
// CHECK: %[[CMP0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P0:.*]] = builtin.unrealized_conversion_cast %[[CMP0]] : vector<256xi1> to vector<64xi1>
// CHECK: %[[CMP1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.gt.s.z"
// CHECK: %[[P1:.*]] = builtin.unrealized_conversion_cast %[[CMP1]] : vector<256xi1> to vector<64xi1>
// CHECK: %[[A:.*]] = builtin.unrealized_conversion_cast %[[P0]] : vector<64xi1> to vector<256xi1>
// CHECK: %[[B:.*]] = builtin.unrealized_conversion_cast %[[P1]] : vector<64xi1> to vector<256xi1>
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[A]], %[[B]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: %[[JOINED:.*]] = builtin.unrealized_conversion_cast %[[EVEN]] : vector<256xi1> to vector<128xi1>
// CHECK: %[[DATA:.*]] = builtin.unrealized_conversion_cast %[[JOINED]] : vector<128xi1> to vector<256xi1>
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[DATA]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.pstu
// CHECK: return
func.func @merge_adjacent(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <VL32> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// A gap must not be filled by the merged write.
// CHECK-LABEL: @keep_nonadjacent
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_nonadjacent(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Two contiguous stores starting at byte 8 cannot use aligned PB16.
// CHECK-LABEL: @keep_8_byte_aligned
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_8_byte_aligned(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [128], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// A dynamic-offset function argument does not establish base alignment.
// CHECK-LABEL: @keep_unknown_base_alignment
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_unknown_base_alignment(%dst: memref<256xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Partial store masks do not authorize a full 128-bit store.
// CHECK-LABEL: @keep_tail_store_mask
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_tail_store_mask(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <VL32> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Consecutive stores to different buffers must not be concatenated.
// CHECK-LABEL: @keep_other_buffer
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_other_buffer(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>, %other: memref<256xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %other to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Do not sink the first store past a read of its destination.
// CHECK-LABEL: @keep_aliasing_read
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_aliasing_read(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) -> vector<64xi1> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %read = ave.hir.vload <NORM> %v0[%c0] {functionType = #ave.func_dist_type<pb32>, ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return %read : vector<64xi1>
}

// -----

// Do not sink the first store past an intervening write.
// CHECK-LABEL: @keep_aliasing_write
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_aliasing_write(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %false = arith.constant false
  memref.store %false, %dst[%c0] : memref<256xi1, #hivm.address_space<ub>>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Distinct SSA function arguments may alias.
// CHECK-LABEL: @keep_possible_alias
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_possible_alias(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>, %other: memref<256xi1, #hivm.address_space<ub>>) -> vector<64xi1> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %read = ave.hir.vload <NORM> %other[%c0] {functionType = #ave.func_dist_type<pb32>, ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<256xi1, #hivm.address_space<ub>> into vector<64xi1>
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return %read : vector<64xi1>
}

// -----

// Do not move the first write across a region containing a barrier.
// CHECK-LABEL: @keep_control_flow
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_control_flow(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  scf.if %cond {
    %bar = arith.constant 0 : i32
    ave.hir.membar %bar
  }
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// Unknown call effects stop store merging.
func.func private @touch(memref<256xi1, #hivm.address_space<ub>>)
// CHECK-LABEL: @keep_unknown_call
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @keep_unknown_call(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi32>, %b: vector<64xi32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %cmpmask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  func.call @touch(%dst) : (memref<256xi1, #hivm.address_space<ub>>) -> ()
  %p1 = ave.hir.vcmp <GT> %a, %b, %cmpmask : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  %v1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
