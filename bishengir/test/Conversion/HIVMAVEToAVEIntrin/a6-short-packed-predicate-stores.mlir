// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s --check-prefix=A5 --implicit-check-not=hivm.vstus

#pb16 = #ave.func_dist_type<pb16>
#pb32 = #ave.func_dist_type<pb32>
#unaligned = #ave.unaligned_ub_access
!row16 = memref<16xi1, strided<[1], offset: 16>, #hivm.address_space<ub>>
!row32 = memref<32xi1, strided<[1], offset: 32>, #hivm.address_space<ub>>
!full = memref<256xi1, #hivm.address_space<ub>>

// A 32-bit PB16 row starts at byte four. The flush must consume the alignment
// state and the updated pointer from the four-byte store.
// CHECK-LABEL: @pb16
// CHECK: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[WRITE:.*]] = "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[FOUR]], %{{[^)]+}})
// CHECK-DAG: %[[TAIL:.*]] = llvm.extractvalue %[[WRITE]][0]
// CHECK-DAG: %[[NEXT:.*]] = llvm.extractvalue %[[WRITE]][1]
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"(%[[TAIL]], %[[NEXT]],
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"
// A5-LABEL: @pb16
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// A5: "hivm_regbaseintrins.intr.hivm.psts.b8"
func.func @pb16(%row: !row32, %full: !full, %data: vector<128xi1>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %short, %rem = ave.hir.plt %c32 {functionType = #pb16} : vector<128xi1>, index
  %all = ave.hir.pge <ALL> {functionType = #pb16} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %row[%c0], %short, %data {ave.unaligned_ub_access = #unaligned, functionType = #pb16} : !row32, vector<128xi1>, vector<128xi1>
  ave.hir.masked_store <NORM_B8> %full[%c0], %all, %data {functionType = #pb16} : !full, vector<128xi1>, vector<128xi1>
  return
}

// A 16-bit PB32 row at byte two must leave the following row untouched.
// CHECK-LABEL: @pb32
// CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[TWO]], %{{[^)]+}})
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// A5-LABEL: @pb32
// A5-COUNT-2: "hivm_regbaseintrins.intr.hivm.pstu.b32"
func.func @pb32(%row: !row16, %full: !full, %data: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %short = ave.hir.pge <VL16> {functionType = #pb32} : vector<64xi1>
  %all = ave.hir.pge <ALL> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %row[%c0], %short, %data {ave.unaligned_ub_access = #unaligned, functionType = #pb32} : !row16, vector<64xi1>, vector<64xi1>
  ave.hir.masked_store <NORM_B8> %full[%c0], %all, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  return
}

// Forty PB32 bits use five bytes. An aligned destination can still contain
// live bits immediately after those five bytes.
// CHECK-LABEL: @pb32_five_bytes
// CHECK-DAG: %[[FIVE:.*]] = llvm.mlir.constant(5 : i32)
// CHECK-DAG: "hivm_regbaseintrins.intr.hivm.vpackv2.v128u16.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[FIVE]], %{{[^)]+}})
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// A5-LABEL: @pb32_five_bytes
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b32"
func.func @pb32_five_bytes(%dst: !full, %data: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %mask, %rest = ave.hir.plt %c40 {functionType = #pb32} : vector<64xi1>, index
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  return
}

// Q selects the first quarter of 128 PB16 lanes.
// CHECK-LABEL: @pb16_quarter
// CHECK: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[FOUR]], %{{[^)]+}})
// A5-LABEL: @pb16_quarter
// A5: "hivm_regbaseintrins.intr.hivm.psts.b8"
func.func @pb16_quarter(%dst: !full, %data: vector<128xi1>) {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <Q> {functionType = #pb16} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #pb16} : !full, vector<128xi1>, vector<128xi1>
  return
}

// CHECK-LABEL: @empty
// CHECK-NOT: hivm.pstu
// CHECK-NOT: hivm.vstus
// CHECK: return
// A5-LABEL: @empty
func.func @empty(%dst: !full, %data: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALLF> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  return
}

// Full packets also require an aligned starting byte. The PB16 address is
// marked unaligned. The PB32 memref starts four bytes from its aligned base.
// CHECK-LABEL: @full_unaligned
// CHECK: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[SIXTEEN]], %{{[^)]+}})
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[EIGHT]], %{{[^)]+}})
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// A5-LABEL: @full_unaligned
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b32"
func.func @full_unaligned(%dst16: !full, %dst32: memref<64xi1, strided<[1], offset: 32>, #hivm.address_space<ub>>, %data16: vector<128xi1>, %data32: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %all16 = ave.hir.pge <ALL> {functionType = #pb16} : vector<128xi1>
  %all32 = ave.hir.pge <ALL> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst16[%c0], %all16, %data16 {ave.unaligned_ub_access = #unaligned, functionType = #pb16} : !full, vector<128xi1>, vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst32[%c0], %all32, %data32 {functionType = #pb32} : memref<64xi1, strided<[1], offset: 32>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// A nonzero i1 index counts bits. Index eight consumes one byte of alignment.
// CHECK-LABEL: @indexed_full
// CHECK: llvm.sdiv
// CHECK: llvm.getelementptr {{.*}} i8
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// A5-LABEL: @indexed_full
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b32"
func.func @indexed_full(%dst: !full, %data: vector<64xi1>) {
  %eight = arith.constant 8 : index
  %mask = ave.hir.pge <ALL> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%eight], %mask, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  return
}

// The comparison store uses PB32. A subsequent PB8 read at the same i1 index
// must recover the same address even though the predicate layout changed.
// CHECK-LABEL: @readback
// CHECK: %[[ADDR:.*]] = llvm.getelementptr {{.*}}, i8
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"({{.*}}, %[[ADDR]],
// CHECK: "hivm_regbaseintrins.intr.hivm.mem.bar.vst.vld"
// CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"(%[[ADDR]],
// A5-LABEL: @readback
// A5: %[[ADDR:.*]] = llvm.getelementptr {{.*}}, i1
// A5: "hivm_regbaseintrins.intr.hivm.pstu.b32"({{.*}}, %[[ADDR]],
// A5: "hivm_regbaseintrins.intr.hivm.plds.b8"(%[[ADDR]],
func.func @readback(%work: memref<4096xi1, #hivm.address_space<ub>>, %data: vector<64xi1>) -> vector<64xi1> {
  %at = arith.constant 256 : index
  %bar = arith.constant 1 : i32
  %all = ave.hir.pge <ALL> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %work[%at], %all, %data {functionType = #pb32} : memref<4096xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  ave.hir.membar %bar
  %bits = ave.hir.vload <NORM> %work[%at] {functionType = #ave.func_dist_type<pb8>} : memref<4096xi1, #hivm.address_space<ub>> into vector<64xi1>
  return %bits : vector<64xi1>
}

// The neighboring row starts at byte four. The second store uses the actual
// count, even if a different call site left a zero tag on the VF argument.
// CHECK-LABEL: @adjacent_runtime_row
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[FOUR]],
// CHECK: %[[ACTIVE:.*]] = llvm.intr.umin
// CHECK: %[[WHOLE:.*]] = llvm.lshr %[[ACTIVE]]
// CHECK: %[[ROUNDED:.*]] = llvm.add %[[ACTIVE]]
// CHECK: %[[BYTES:.*]] = llvm.lshr %[[ROUNDED]]
// CHECK: %[[PARTIAL:.*]] = llvm.icmp "ne" %[[WHOLE]], %[[BYTES]]
// CHECK: %[[CAP:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: %[[READ_LANES:.*]] = llvm.select %[[PARTIAL]], %[[CAP]], %[[ZERO]]
// CHECK: %[[PLT:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%[[READ_LANES]])
// CHECK: %[[READ:.*]] = llvm.extractvalue %[[PLT]][0]
// CHECK: %[[LAST:.*]] = llvm.getelementptr {{.*}}[%[[WHOLE]]]
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"(%[[LAST]], {{.*}}, %[[READ]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vxor.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.vand.x"
// CHECK: %[[MERGED:.*]] = "hivm_regbaseintrins.intr.hivm.vxor.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"(%[[MERGED]], {{.*}}, %[[BYTES]],
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// A5-LABEL: @adjacent_runtime_row
// A5-COUNT-2: "hivm_regbaseintrins.intr.hivm.pstu.b32"
func.func @adjacent_runtime_row(%dst: !full, %data: vector<64xi1>, %count: index {hivm.constant_value = 0 : i64}) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %neighbor = ave.hir.pge <VL32> {functionType = #pb32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c32], %neighbor, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  %prefix, %rest = ave.hir.plt %count {functionType = #pb32} : vector<64xi1>, index
  ave.hir.masked_store <NORM_B8> %dst[%c0], %prefix, %data {functionType = #pb32} : !full, vector<64xi1>, vector<64xi1>
  return
}

// A three-bit prefix shares its first byte with five inactive bits.
// CHECK-LABEL: @three_bits
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s16"({{.*}}, %[[ONE]],
// A5-LABEL: @three_bits
// A5: "hivm_regbaseintrins.intr.hivm.psts.b8"
func.func @three_bits(%dst: !full, %data: vector<128xi1>) {
  %zero = arith.constant 0 : index
  %three = ave.hir.pge <VL3> {functionType = #pb16} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst[%zero], %three, %data {functionType = #pb16} : !full, vector<128xi1>, vector<128xi1>
  return
}
