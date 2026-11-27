// RUN: split-file %s %t
// RUN: bishengir-opt %t/both.mlir -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-vector-to-hivmave -canonicalize | FileCheck %s --check-prefixes=COMMON,A6
// RUN: bishengir-opt %t/both.mlir -hacc-append-device-spec=target=Ascend950PR_9579 -convert-vector-to-hivmave -canonicalize | FileCheck %s --check-prefixes=COMMON,A5
// RUN: bishengir-opt %t/mask.mlir -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-vector-to-hivmave -analyze-vector-layout -canonicalize | FileCheck %s --check-prefix=MASK

//--- both.mlir

// The inverse writes four rows of 16 floats into rows pitched by 64 floats.
// COMMON-LABEL: func.func @inverse_rows
// A6-DAG: %[[SHIFT:.+]] = arith.constant 4 : i32
// A6-DAG: %[[GAP:.+]] = arith.constant 48 : i32
// A6: %[[LANE:.+]] = ave.hir.vci
// A6: %[[ROW:.+]] = ave.hir.vshrs %[[LANE]], %[[SHIFT]]
// A6: %[[EXTRA:.+]] = ave.hir.vmuls %[[ROW]], %[[GAP]]
// A6: %[[ADDR:.+]] = ave.hir.vadd %[[LANE]], %[[EXTRA]]
// A6: ave.hir.vscatter %arg0[%arg2, %arg3] [%[[ADDR]]]
// A5: ave.hir.masked_store <NORM_B32>
func.func @inverse_rows(%dst: memref<4x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, %value: vector<4x16xf32>, %row: index, %col: index) {
  vector.transfer_write %value, %dst[%row, %col] {in_bounds = [true, true]} : vector<4x16xf32>, memref<4x16xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  return
}

// The 128x528 f16 transpose has a runtime-masked tail on a stride-128 output.
// COMMON-LABEL: func.func @masked_column
// A6: %[[PITCH:.+]] = arith.constant 128 : i16
// A6: %[[LANE:.+]] = ave.hir.vci
// A6: %[[ADDR:.+]] = ave.hir.vmuls %[[LANE]], %[[PITCH]], %arg2
// A6: ave.hir.vscatter %arg0[%arg3] [%[[ADDR]]], %arg2, %arg1
// A5: ave.hir.masked_store <NORM_B16>
func.func @masked_column(%dst: memref<?xf16, strided<[128], offset: ?>, #hivm.address_space<ub>>, %value: vector<128xf16>, %mask: vector<128xi1>, %base: index) {
  vector.transfer_write %value, %dst[%base], %mask {in_bounds = [true]} : vector<128xf16>, memref<?xf16, strided<[128], offset: ?>, #hivm.address_space<ub>>
  return
}

// V920R1 VSCATTER.B16 uses unsigned 16-bit element indices (SIMD ISA 9.10.3).
// Lane 127 writes 127 * 516 = 65532 elements past the base. Increasing the
// pitch to 517 exceeds UINT16_MAX, as checked by @halfword_overflow.
// COMMON-LABEL: func.func @largest_fitting_pitch
// A6: %[[PITCH:.+]] = arith.constant 516 : i16
// A6: %[[ADDR:.+]] = ave.hir.vmuls {{.*}}, %[[PITCH]]
// A6: ave.hir.vscatter %arg0[%arg3] [%[[ADDR]]], %arg2, %arg1
// A5: ave.hir.masked_store <NORM_B16>
func.func @largest_fitting_pitch(%dst: memref<128xi16, strided<[516], offset: ?>, #hivm.address_space<ub>>, %value: vector<128xi16>, %mask: vector<128xi1>, %base: index) {
  vector.transfer_write %value, %dst[%base], %mask {in_bounds = [true]} : vector<128xi16>, memref<128xi16, strided<[516], offset: ?>, #hivm.address_space<ub>>
  return
}

// COMMON-LABEL: func.func @dense
// COMMON: ave.hir.masked_store <NORM_B32>
func.func @dense(%dst: memref<4x16xf32, #hivm.address_space<ub>>, %value: vector<4x16xf32>, %i: index) {
  vector.transfer_write %value, %dst[%i, %i] {in_bounds = [true, true]} : vector<4x16xf32>, memref<4x16xf32, #hivm.address_space<ub>>
  return
}

// COMMON-LABEL: func.func @one_block_per_row
// COMMON: ave.hir.store_with_stride
func.func @one_block_per_row(%dst: memref<8x8xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, %value: vector<8x8xf32>, %i: index) {
  vector.transfer_write %value, %dst[%i, %i] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  return
}

// The physical pitch would overflow an i16 element index on the second row.
// Only row zero is active, so every address used by the mask fits a flat store.
// COMMON-LABEL: func.func @inactive_row_past_scatter_range
// COMMON: ave.hir.pge <VL32>
// COMMON: ave.hir.masked_store <NORM_B16>
func.func @inactive_row_past_scatter_range(%dst: memref<4x32xi16, strided<[65539, 1]>, #hivm.address_space<ub>>, %value: vector<4x32xi16>, %i: index) {
  %mask = vector.constant_mask [1, 32] : vector<4x32xi1>
  vector.transfer_write %value, %dst[%i, %i], %mask {in_bounds = [true, true]} : vector<4x32xi16>, memref<4x32xi16, strided<[65539, 1]>, #hivm.address_space<ub>>
  return
}

// The first 64 bytes are physically contiguous. The second row begins after
// a gap and has no active lanes.
// COMMON-LABEL: func.func @first_row_only
// COMMON: ave.hir.pge <VL64>
// COMMON: ave.hir.masked_store <NORM_B8>
func.func @first_row_only(%dst: memref<4x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, %value: vector<4x64xi8>, %i: index) {
  %mask = vector.constant_mask [1, 64] : vector<4x64xi1>
  vector.transfer_write %value, %dst[%i, %i], %mask {in_bounds = [true, true]} : vector<4x64xi8>, memref<4x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
  return
}

// COMMON-LABEL: func.func @one_strided_lane
// COMMON: ave.hir.masked_store <ONEPT_B32>
func.func @one_strided_lane(%dst: memref<16xf32, strided<[2]>, #hivm.address_space<ub>>, %value: vector<16xf32>, %i: index) {
  %mask = vector.constant_mask [1] : vector<16xi1>
  vector.transfer_write %value, %dst[%i], %mask : vector<16xf32>, memref<16xf32, strided<[2]>, #hivm.address_space<ub>>
  return
}

// Only lane 0 fits a packed store at this pitch. The unsigned minimum
// proves that at most that lane can write, even when the count is dynamic.
// COMMON-LABEL: func.func @at_most_one_lane
// COMMON: %[[COUNT:.+]] = arith.minui
// COMMON: %[[TAIL:.+]], %{{.*}} = ave.hir.plt %[[COUNT]]
// COMMON: ave.hir.masked_store <NORM_B16> {{.*}}, %[[TAIL]],
func.func @at_most_one_lane(%dst: memref<128xf16, strided<[517]>, #hivm.address_space<ub>>, %value: vector<128xf16>, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %count = arith.minui %n, %one : index
  %mask = vector.create_mask %count : vector<128xi1>
  vector.transfer_write %value, %dst[%zero], %mask {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[517]>, #hivm.address_space<ub>>
  return
}

// COMMON-LABEL: func.func @empty_mask
// A6-NEXT: return
// A5: ave.hir.masked_store <NORM_B16>
func.func @empty_mask(%dst: memref<128xf16, strided<[517]>, #hivm.address_space<ub>>, %value: vector<128xf16>) {
  %zero = arith.constant 0 : index
  %none = arith.constant dense<false> : vector<128xi1>
  vector.transfer_write %value, %dst[%zero], %none {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[517]>, #hivm.address_space<ub>>
  return
}

//--- mask.mlir

// VSSTB takes its pitch in blocks. The last row is beyond the B16 scatter
// index range. The full-width predicate selects the first two rows.
// MASK-LABEL: func.func @masked_native_blocks
// MASK: %[[PRED:.+]] = ave.hir.pge <VL32>
// MASK: ave.hir.store_with_stride {{.*}}%[[PRED]]{{.*}}vector<128xi1>
func.func @masked_native_blocks(%dst: memref<8x16xf16, strided<[9376, 1]>, #hivm.address_space<ub>>, %value: vector<8x16xf16>, %i: index) {
  %prefix = vector.constant_mask [32] : vector<128xi1>
  %mask = vector.shape_cast %prefix : vector<128xi1> to vector<8x16xi1>
  vector.transfer_write %value, %dst[%i, %i], %mask {in_bounds = [true, true]} : vector<8x16xf16>, memref<8x16xf16, strided<[9376, 1]>, #hivm.address_space<ub>>
  return
}

// The two physical gaps add 24 elements every 16 lanes and 15920 every 32.
// The last row crosses the signed B16 index boundary. Any subset of lanes
// can be active, including separate rows of the 3-D mask argument.
// MASK-LABEL: func.func @masked_tile
// MASK-DAG: %[[GAP1:.+]] = arith.constant 24 : i16
// MASK-DAG: %[[GAP2:.+]] = arith.constant 15920 : i16
// MASK: %[[MASK:.+]] = builtin.unrealized_conversion_cast %arg2 : vector<4x2x16xi1> to vector<128xi1>
// MASK: ave.hir.vmuls {{.*}}, %[[GAP1]], %[[MASK]]
// MASK: ave.hir.vmuls {{.*}}, %[[GAP2]], %[[MASK]]
// MASK: ave.hir.vscatter %arg0[{{.*}}] [{{.*}}], %[[MASK]]
func.func @masked_tile(%dst: memref<4x2x16xf16, strided<[16000, 40, 1], offset: ?>, 6>, %value: vector<4x2x16xf16>, %mask: vector<4x2x16xi1>, %i: index) {
  vector.transfer_write %value, %dst[%i, %i, %i], %mask {in_bounds = [true, true, true]} : vector<4x2x16xf16>, memref<4x2x16xf16, strided<[16000, 40, 1], offset: ?>, 6>
  return
}

// Both calls use the same store. A bound of 32 fits row zero, but a bound
// of 128 also writes the other three rows at their physical pitch.
// MASK-LABEL: func.func private @store_rows
// MASK: %[[TAIL:.+]], %{{.*}} = ave.hir.plt %arg2
// MASK: ave.hir.vscatter {{.*}}, %[[TAIL]],
func.func private @store_rows(%dst: memref<4x32xi16, strided<[64, 1]>, #hivm.address_space<ub>>, %value: vector<4x32xi16>, %count: index) {
  %zero = arith.constant 0 : index
  %prefix = vector.create_mask %count : vector<128xi1>
  %mask = vector.shape_cast %prefix : vector<128xi1> to vector<4x32xi1>
  vector.transfer_write %value, %dst[%zero, %zero], %mask {in_bounds = [true, true]} : vector<4x32xi16>, memref<4x32xi16, strided<[64, 1]>, #hivm.address_space<ub>>
  return
}

func.func @call_store_rows(%dst: memref<4x32xi16, strided<[64, 1]>, #hivm.address_space<ub>>, %value: vector<4x32xi16>) {
  %full = arith.constant 128 : index
  %one_row = arith.constant 32 : index
  func.call @store_rows(%dst, %value, %full) : (memref<4x32xi16, strided<[64, 1]>, #hivm.address_space<ub>>, vector<4x32xi16>, index) -> ()
  func.call @store_rows(%dst, %value, %one_row) : (memref<4x32xi16, strided<[64, 1]>, #hivm.address_space<ub>>, vector<4x32xi16>, index) -> ()
  return
}
