// RUN: bishengir-opt -memref-dse -split-input-file %s | FileCheck %s

// CHECK-LABEL: example
// CHECK: %[[cstOne:.*]] = arith.constant 4.200000
// CHECK: %[[res:.*]] = arith.addf %[[cstOne]], %[[cstTwo:.*]]
// CHECK: linalg.elemwise_unary
// CHECK-NEXT: %[[load_after_unary:.*]] = memref.load
// CHECK: return %[[res]], %[[load_after_unary]], %[[cstOne:.*]], %[[cstOne:.*]]
func.func @example() -> (f32, f32, f32, f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<4xf32>
    %alloc_0 = memref.alloc() : memref<4xf32>
    %cst = arith.constant 4.200000e+01 : f32
    %cst_2 = arith.constant 8.400000e+01 : f32
    memref.store %cst, %alloc_0[%c0] : memref<4xf32>
    memref.store %cst_2, %alloc[%c1] : memref<4xf32>
    %0 = memref.load %alloc_0[%c0] : memref<4xf32>
    %result = arith.addf %0, %cst_2 : f32
    memref.store %result, %alloc[%c0] : memref<4xf32>
    %1 = memref.load %alloc[%c0] : memref<4xf32>
    linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%alloc : memref<4xf32>) outs(%alloc : memref<4xf32>)
    %4 = memref.load %alloc[%c0] : memref<4xf32>
    memref.store %cst, %alloc[%c0] : memref<4xf32>
    %5 = memref.load %alloc[%c0] : memref<4xf32>
    %6 = memref.load %alloc[%c0] : memref<4xf32>
    return %1, %4, %5, %6 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: long_dse
func.func @long_dse(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}) attributes {debug_instruction_number = 257 : i32, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, false, false]> : vector<13xi1>, global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant {debug_instruction_number = 0 : i32} 0 : index
    %c1 = arith.constant {debug_instruction_number = 1 : i32} 1 : index
    %c3072 = arith.constant {debug_instruction_number = 2 : i32} 3072 : index
    %c1024 = arith.constant {debug_instruction_number = 3 : i32} 1024 : index
    %c48_i32 = arith.constant {debug_instruction_number = 4 : i32} 48 : i32
    %c1_i32 = arith.constant {debug_instruction_number = 5 : i32} 1 : i32
    %c3_i32 = arith.constant {debug_instruction_number = 6 : i32} 3 : i32
    %c1024_i32 = arith.constant {debug_instruction_number = 7 : i32} 1024 : i32
    %c12_i32 = arith.constant {debug_instruction_number = 8 : i32} 12 : i32
    %c32_i32 = arith.constant {debug_instruction_number = 9 : i32} 32 : i32
    %c24_i32 = arith.constant {debug_instruction_number = 10 : i32} 24 : i32
    %cst = arith.constant {debug_instruction_number = 11 : i32} 1.000000e+00 : f32
    %cst_0 = arith.constant {debug_instruction_number = 12 : i32} 2.71267363E-5 : f32
    %c0_i32 = arith.constant {debug_instruction_number = 13 : i32} 0 : i32
    %c1179648_i32 = arith.constant {debug_instruction_number = 14 : i32} 1179648 : i32
    %c36864_i32 = arith.constant {debug_instruction_number = 15 : i32} 36864 : i32
    %c3072_i32 = arith.constant {debug_instruction_number = 16 : i32} 3072 : i32
    %cst_1 = arith.constant {debug_instruction_number = 17 : i32} 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 18 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc {debug_instruction_number = 19 : i32} ins(%cst_1 : f32) outs(%alloc : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 20 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
    memref.store %cst_1, %alloc_2[%c0, %c0, %c0, %c0] {debug_instruction_number = 21 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 22 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
    memref.store %cst_0, %alloc_3[%c0, %c0, %c0, %c0] {debug_instruction_number = 23 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 24 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc {debug_instruction_number = 25 : i32} ins(%cst : f32) outs(%alloc_4 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
    %0 = arith.divsi %arg11, %c48_i32 {debug_instruction_number = 26 : i32} : i32
    %1 = arith.remsi %arg11, %c48_i32 {debug_instruction_number = 27 : i32} : i32
    %2 = hivm.hir.get_block_idx {debug_instruction_number = 28 : i32} -> i64
    %3 = arith.trunci %2 {debug_instruction_number = 29 : i32} : i64 to i32
    %4 = arith.muli %3, %0 {debug_instruction_number = 30 : i32} : i32
    %5 = arith.addi %4, %1 {debug_instruction_number = 31 : i32} : i32
    %6 = arith.cmpi slt, %3, %1 {debug_instruction_number = 32 : i32} : i32
    %7:2 = scf.if %6 -> (i32, i32) {
      %8 = arith.addi %0, %c1_i32 {debug_instruction_number = 33 : i32} : i32
      %9 = arith.muli %3, %8 {debug_instruction_number = 34 : i32} : i32
      scf.yield {debug_instruction_number = 35 : i32} %8, %9 : i32, i32
    } else {
      scf.yield {debug_instruction_number = 36 : i32} %0, %5 : i32, i32
    } {debug_instruction_number = 37 : i32}
    scf.for %arg13 = %c0_i32 to %7#0 step %c1_i32  : i32 {
      %8 = arith.addi %7#1, %arg13 {debug_instruction_number = 38 : i32} : i32
      %9 = arith.remsi %8, %c3_i32 {debug_instruction_number = 39 : i32} : i32
      %10 = arith.muli %9, %c1024_i32 {debug_instruction_number = 40 : i32} : i32
      %11 = arith.divsi %8, %c3_i32 {debug_instruction_number = 41 : i32} : i32
      %12 = arith.remsi %11, %c12_i32 {debug_instruction_number = 42 : i32} : i32
      %13 = arith.divsi %11, %c12_i32 {debug_instruction_number = 43 : i32} : i32
      %14 = arith.remsi %13, %c32_i32 {debug_instruction_number = 44 : i32} : i32
      %15 = arith.divsi %13, %c32_i32 {debug_instruction_number = 45 : i32} : i32
      %16 = arith.remsi %15, %c24_i32 {debug_instruction_number = 46 : i32} : i32
      %17 = arith.cmpi slt, %12, %c12_i32 {debug_instruction_number = 47 : i32} : i32
      %18 = arith.cmpi slt, %14, %c32_i32 {debug_instruction_number = 48 : i32} : i32
      %19 = arith.cmpi slt, %16, %c24_i32 {debug_instruction_number = 49 : i32} : i32
      %20 = arith.muli %16, %c32_i32 {debug_instruction_number = 50 : i32} : i32
      %21 = arith.addi %14, %20 {debug_instruction_number = 51 : i32} : i32
      %22 = arith.andi %19, %18 {debug_instruction_number = 52 : i32} : i1
      %23 = arith.muli %12, %c3072_i32 {debug_instruction_number = 53 : i32} : i32
      %24 = arith.muli %14, %c36864_i32 {debug_instruction_number = 54 : i32} : i32
      %25 = arith.muli %16, %c1179648_i32 {debug_instruction_number = 55 : i32} : i32
      %26 = arith.andi %17, %19 {debug_instruction_number = 56 : i32} : i1
      %27 = arith.index_cast %10 {debug_instruction_number = 57 : i32} : i32 to index
      %28 = arith.index_cast %23 {debug_instruction_number = 58 : i32} : i32 to index
      %29 = arith.addi %27, %28 {debug_instruction_number = 59 : i32} : index
      %30 = arith.index_cast %24 {debug_instruction_number = 60 : i32} : i32 to index
      %31 = arith.addi %29, %30 {debug_instruction_number = 61 : i32} : index
      %32 = arith.index_cast %25 {debug_instruction_number = 62 : i32} : i32 to index
      %33 = arith.addi %31, %32 {debug_instruction_number = 63 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 64 : i32} : memref<?xbf16, #hivm.address_space<gm>> to memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_5 = memref.alloc() {debug_instruction_number = 65 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>
      %34 = arith.index_cast %17 {debug_instruction_number = 66 : i32} : i1 to index
      %35 = arith.muli %34, %c1024 {debug_instruction_number = 67 : i32} : index
      %36 = arith.addi %27, %c1024 {debug_instruction_number = 68 : i32} : index
      %37 = arith.maxsi %27, %c3072 {debug_instruction_number = 69 : i32} : index
      %38 = arith.minsi %36, %37 {debug_instruction_number = 70 : i32} : index
      %39 = arith.subi %38, %27 {debug_instruction_number = 71 : i32} : index
      %40 = arith.minsi %34, %c1 {debug_instruction_number = 72 : i32} : index
      %41 = arith.minsi %35, %39 {debug_instruction_number = 73 : i32} : index
      %42 = arith.index_cast %18 {debug_instruction_number = 74 : i32} : i1 to index
      %43 = arith.muli %42, %c1024 {debug_instruction_number = 75 : i32} : index
      %44 = arith.minsi %40, %42 {debug_instruction_number = 76 : i32} : index
      %45 = arith.minsi %41, %43 {debug_instruction_number = 77 : i32} : index
      %46 = arith.index_cast %19 {debug_instruction_number = 78 : i32} : i1 to index
      %47 = arith.muli %46, %c1024 {debug_instruction_number = 79 : i32} : index
      %48 = arith.minsi %44, %46 {debug_instruction_number = 80 : i32} : index
      %49 = arith.minsi %45, %47 {debug_instruction_number = 81 : i32} : index
      %subview = memref.subview %reinterpret_cast[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 82 : i32} : memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %alloc_5[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 83 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 84 : i32}
      %alloc_7 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 85 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vcast {debug_instruction_number = 86 : i32} ins(%alloc_5 : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>) outs(%alloc_7 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 87 : i32} : memref<?xbf16, #hivm.address_space<gm>> to memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_9 = memref.alloc() {debug_instruction_number = 88 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>
      %subview_10 = memref.subview %reinterpret_cast_8[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 89 : i32} : memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_11 = memref.subview %alloc_9[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 90 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_10 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 91 : i32}
      %alloc_12 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 92 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vcast {debug_instruction_number = 93 : i32} ins(%alloc_9 : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>) outs(%alloc_12 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %50 = arith.index_cast %21 {debug_instruction_number = 94 : i32} : i32 to index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%50], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] {debug_instruction_number = 95 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_14 = memref.alloc() {debug_instruction_number = 96 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %51 = arith.index_cast %22 {debug_instruction_number = 97 : i32} : i1 to index
      %subview_15 = memref.subview %reinterpret_cast_13[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 98 : i32} : memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_16 = memref.subview %alloc_14[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 99 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_15 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_16 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 100 : i32}
      %52 = arith.muli %14, %c12_i32 {debug_instruction_number = 101 : i32} : i32
      %53 = arith.addi %12, %52 {debug_instruction_number = 102 : i32} : i32
      %54 = arith.andi %17, %18 {debug_instruction_number = 103 : i32} : i1
      %55 = arith.index_cast %53 {debug_instruction_number = 104 : i32} : i32 to index
      %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [%55], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] {debug_instruction_number = 105 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_18 = memref.alloc() {debug_instruction_number = 106 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %56 = arith.index_cast %54 {debug_instruction_number = 107 : i32} : i1 to index
      %subview_19 = memref.subview %reinterpret_cast_17[0, 0, 0, 0] [%56, %56, %56, %56] [1, 1, 1, 1] {debug_instruction_number = 108 : i32} : memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_20 = memref.subview %alloc_18[0, 0, 0, 0] [%56, %56, %56, %56] [1, 1, 1, 1] {debug_instruction_number = 109 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_19 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_20 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 110 : i32}
      %reinterpret_cast_21 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 111 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1024xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_22 = memref.alloc() {debug_instruction_number = 112 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      %subview_23 = memref.subview %reinterpret_cast_21[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 113 : i32} : memref<1x1x1x1024xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_24 = memref.subview %alloc_22[0, 0, 0, 0] [%48, %48, %48, %49] [1, 1, 1, 1] {debug_instruction_number = 114 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_23 : memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_24 : memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 115 : i32}
      %reinterpret_cast_25 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] {debug_instruction_number = 116 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_26 = memref.alloc() {debug_instruction_number = 117 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %subview_27 = memref.subview %reinterpret_cast_25[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 118 : i32} : memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %alloc_26[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 119 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_27 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_28 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 120 : i32}
      %reinterpret_cast_29 = memref.reinterpret_cast %arg7 to offset: [%50], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] {debug_instruction_number = 121 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_30 = memref.alloc() {debug_instruction_number = 122 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %subview_31 = memref.subview %reinterpret_cast_29[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 123 : i32} : memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_32 = memref.subview %alloc_30[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 124 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_31 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_32 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 125 : i32}
      %reinterpret_cast_33 = memref.reinterpret_cast %arg8 to offset: [%50], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] {debug_instruction_number = 126 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_34 = memref.alloc() {debug_instruction_number = 127 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %subview_35 = memref.subview %reinterpret_cast_33[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 128 : i32} : memref<1x1x1x1xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_36 = memref.subview %alloc_34[0, 0, 0, 0] [%51, %51, %51, %51] [1, 1, 1, 1] {debug_instruction_number = 129 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_35 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<?x?x?x?xf32, strided<[1, 1, 1, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 130 : i32}
      %reinterpret_cast_37 = memref.reinterpret_cast %arg9 to offset: [%33], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 131 : i32} : memref<?xbf16, #hivm.address_space<gm>> to memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_38 = memref.alloc() {debug_instruction_number = 132 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>
      %57 = arith.index_cast %26 {debug_instruction_number = 133 : i32} : i1 to index
      %58 = arith.muli %57, %c1024 {debug_instruction_number = 134 : i32} : index
      %59 = arith.minsi %57, %c1 {debug_instruction_number = 135 : i32} : index
      %60 = arith.minsi %58, %39 {debug_instruction_number = 136 : i32} : index
      %61 = arith.minsi %59, %42 {debug_instruction_number = 137 : i32} : index
      %62 = arith.minsi %60, %43 {debug_instruction_number = 138 : i32} : index
      %subview_39 = memref.subview %reinterpret_cast_37[0, 0, 0, 0] [%61, %61, %61, %62] [1, 1, 1, 1] {debug_instruction_number = 139 : i32} : memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_40 = memref.subview %alloc_38[0, 0, 0, 0] [%61, %61, %61, %62] [1, 1, 1, 1] {debug_instruction_number = 140 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_39 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_40 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) {debug_instruction_number = 141 : i32}
      %alloc_41 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 142 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vcast {debug_instruction_number = 143 : i32} ins(%alloc_38 : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>) outs(%alloc_41 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_42 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 144 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vsub {debug_instruction_number = 145 : i32} ins(%alloc, %alloc_12 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_42 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_43 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 146 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vexp {debug_instruction_number = 147 : i32} ins(%alloc_42 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_43 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_44 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 148 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vadd {debug_instruction_number = 149 : i32} ins(%alloc_43, %cst : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, f32) outs(%alloc_44 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_45 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 150 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vdiv {debug_instruction_number = 151 : i32} ins(%alloc_4, %alloc_44 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_45 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_46 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 152 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vsub {debug_instruction_number = 153 : i32} ins(%alloc_4, %alloc_45 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_46 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_47 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 154 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vmul {debug_instruction_number = 155 : i32} ins(%alloc_12, %alloc_46 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_47 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_48 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 156 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vadd {debug_instruction_number = 157 : i32} ins(%alloc_47, %cst : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, f32) outs(%alloc_48 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_49 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 158 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vmul {debug_instruction_number = 159 : i32} ins(%alloc_45, %alloc_48 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_49 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_50 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 160 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vmul {debug_instruction_number = 161 : i32} ins(%alloc_7, %alloc_49 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_50 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_51 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 162 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %63 = memref.load %alloc_14[%c0, %c0, %c0, %c0] {debug_instruction_number = 163 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %64 = memref.load %alloc_18[%c0, %c0, %c0, %c0] {debug_instruction_number = 164 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %65 = arith.mulf %63, %64 {debug_instruction_number = 165 : i32} : f32
      memref.store %65, %alloc_51[%c0, %c0, %c0, %c0] {debug_instruction_number = 166 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %alloc_51 [[0], [1], [2, 3]] {debug_instruction_number = 167 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> into memref<1x1x1xf32, #hivm.address_space<ub>>
      %66 = memref.load %collapse_shape[%c0, %c0, %c0] {debug_instruction_number = 168 : i32} : memref<1x1x1xf32, #hivm.address_space<ub>>
      %alloc_52 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 169 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vmul {debug_instruction_number = 170 : i32} ins(%alloc_50, %66 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, f32) outs(%alloc_52 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_53 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 171 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %67 = memref.load %alloc_26[%c0, %c0, %c0, %c0] {debug_instruction_number = 172 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %68 = memref.load %alloc_30[%c0, %c0, %c0, %c0] {debug_instruction_number = 173 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %69 = arith.mulf %67, %68 {debug_instruction_number = 174 : i32} : f32
      memref.store %69, %alloc_53[%c0, %c0, %c0, %c0] {debug_instruction_number = 175 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_54 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 176 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %70 = memref.load %alloc_53[%c0, %c0, %c0, %c0] {debug_instruction_number = 177 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %71 = memref.load %alloc_34[%c0, %c0, %c0, %c0] {debug_instruction_number = 178 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %72 = arith.subf %70, %71 {debug_instruction_number = 179 : i32} : f32
      memref.store %72, %alloc_54[%c0, %c0, %c0, %c0] {debug_instruction_number = 180 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_55 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 181 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %73 = memref.load %alloc_54[%c0, %c0, %c0, %c0] {debug_instruction_number = 182 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %74 = memref.load %alloc_14[%c0, %c0, %c0, %c0] {debug_instruction_number = 183 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %75 = arith.mulf %73, %74 {debug_instruction_number = 184 : i32} : f32
      memref.store %75, %alloc_55[%c0, %c0, %c0, %c0] {debug_instruction_number = 185 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_56 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 186 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %76 = memref.load %alloc_55[%c0, %c0, %c0, %c0] {debug_instruction_number = 187 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %77 = memref.load %alloc_14[%c0, %c0, %c0, %c0] {debug_instruction_number = 188 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %78 = arith.mulf %76, %77 {debug_instruction_number = 189 : i32} : f32
      memref.store %78, %alloc_56[%c0, %c0, %c0, %c0] {debug_instruction_number = 190 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_57 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 191 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %79 = memref.load %alloc_56[%c0, %c0, %c0, %c0] {debug_instruction_number = 192 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %80 = memref.load %alloc_14[%c0, %c0, %c0, %c0] {debug_instruction_number = 193 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %81 = arith.mulf %79, %80 {debug_instruction_number = 194 : i32} : f32
      memref.store %81, %alloc_57[%c0, %c0, %c0, %c0] {debug_instruction_number = 195 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_58 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 196 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %82 = memref.load %alloc_57[%c0, %c0, %c0, %c0] {debug_instruction_number = 197 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %83 = memref.load %alloc_3[%c0, %c0, %c0, %c0] {debug_instruction_number = 198 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %84 = arith.mulf %82, %83 {debug_instruction_number = 199 : i32} : f32
      memref.store %84, %alloc_58[%c0, %c0, %c0, %c0] {debug_instruction_number = 200 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %alloc_58 [[0], [1], [2, 3]] {debug_instruction_number = 201 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> into memref<1x1x1xf32, #hivm.address_space<ub>>
      %85 = memref.load %collapse_shape_59[%c0, %c0, %c0] {debug_instruction_number = 202 : i32} : memref<1x1x1xf32, #hivm.address_space<ub>>
      %alloc_60 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 203 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vmul {debug_instruction_number = 204 : i32} ins(%alloc_22, %85 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, f32) outs(%alloc_60 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_61 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 205 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vadd {debug_instruction_number = 206 : i32} ins(%alloc_52, %alloc_60 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_61 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_62 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 207 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %86 = memref.load %alloc_2[%c0, %c0, %c0, %c0] {debug_instruction_number = 208 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %87 = memref.load %alloc_58[%c0, %c0, %c0, %c0] {debug_instruction_number = 209 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %88 = arith.subf %86, %87 {debug_instruction_number = 210 : i32} : f32
      memref.store %88, %alloc_62[%c0, %c0, %c0, %c0] {debug_instruction_number = 211 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_63 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 212 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %89 = memref.load %alloc_62[%c0, %c0, %c0, %c0] {debug_instruction_number = 213 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %90 = memref.load %alloc_30[%c0, %c0, %c0, %c0] {debug_instruction_number = 214 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      // CHECK: %[[var1:.*]] = arith.mulf %[[old213:.*]], %[[old214:.*]] {debug_instruction_number = 215 : i32
      %91 = arith.mulf %89, %90 {debug_instruction_number = 215 : i32} : f32
      memref.store %91, %alloc_63[%c0, %c0, %c0, %c0] {debug_instruction_number = 216 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_64 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 217 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %92 = memref.load %alloc_26[%c0, %c0, %c0, %c0] {debug_instruction_number = 218 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %93 = memref.load %alloc_14[%c0, %c0, %c0, %c0] {debug_instruction_number = 219 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %94 = arith.mulf %92, %93 {debug_instruction_number = 220 : i32} : f32
      memref.store %94, %alloc_64[%c0, %c0, %c0, %c0] {debug_instruction_number = 221 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_65 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 222 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %95 = memref.load %alloc_64[%c0, %c0, %c0, %c0] {debug_instruction_number = 223 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %96 = memref.load %alloc_3[%c0, %c0, %c0, %c0] {debug_instruction_number = 224 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      // CHECK: %[[var2:.*]] = arith.mulf %[[old223:.*]], %[[old224:.*]][[dump:.*]]debug_instruction_number = 225 : i32
      %97 = arith.mulf %95, %96 {debug_instruction_number = 225 : i32} : f32
      memref.store %97, %alloc_65[%c0, %c0, %c0, %c0] {debug_instruction_number = 226 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %alloc_66 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 227 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %98 = memref.load %alloc_63[%c0, %c0, %c0, %c0] {debug_instruction_number = 228 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %99 = memref.load %alloc_65[%c0, %c0, %c0, %c0] {debug_instruction_number = 229 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      // CHECK: arith.subf %[[var1]], %[[var2]] {debug_instruction_number = 230
      %100 = arith.subf %98, %99 {debug_instruction_number = 230 : i32} : f32
      memref.store %100, %alloc_66[%c0, %c0, %c0, %c0] {debug_instruction_number = 231 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>>
      %collapse_shape_67 = memref.collapse_shape %alloc_66 [[0], [1], [2, 3]] {debug_instruction_number = 232 : i32} : memref<1x1x1x1xf32, #hivm.address_space<ub>> into memref<1x1x1xf32, #hivm.address_space<ub>>
      %101 = memref.load %collapse_shape_67[%c0, %c0, %c0] {debug_instruction_number = 233 : i32} : memref<1x1x1xf32, #hivm.address_space<ub>>
      %alloc_68 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 234 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vadd {debug_instruction_number = 235 : i32} ins(%alloc_61, %101 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, f32) outs(%alloc_68 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %alloc_69 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 236 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>>
      hivm.hir.vadd {debug_instruction_number = 237 : i32} ins(%alloc_41, %alloc_68 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>, memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_69 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>)
      %102 = arith.muli %21, %c36864_i32 {debug_instruction_number = 238 : i32} : i32
      %103 = arith.index_cast %102 {debug_instruction_number = 239 : i32} : i32 to index
      %104 = arith.addi %29, %103 {debug_instruction_number = 240 : i32} : index
      %reinterpret_cast_70 = memref.reinterpret_cast %arg0 to offset: [%104], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 241 : i32} : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x1x1024xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %105 = arith.muli %51, %c1024 {debug_instruction_number = 242 : i32} : index
      %106 = arith.minsi %40, %51 {debug_instruction_number = 243 : i32} : index
      %107 = arith.minsi %41, %105 {debug_instruction_number = 244 : i32} : index
      %subview_71 = memref.subview %alloc_68[0, 0, 0, 0] [%106, %106, %106, %107] [1, 1, 1, 1] {debug_instruction_number = 245 : i32} : memref<1x1x1x1024xf32, #hivm.address_space<ub>> to memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      %subview_72 = memref.subview %reinterpret_cast_70[0, 0, 0, 0] [%106, %106, %106, %107] [1, 1, 1, 1] {debug_instruction_number = 246 : i32} : memref<1x1x1x1024xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_71 : memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) outs(%subview_72 : memref<?x?x?x?xf32, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) {debug_instruction_number = 247 : i32}
      %reinterpret_cast_73 = memref.reinterpret_cast %arg10 to offset: [%33], sizes: [1, 1, 1, 1024], strides: [1024, 1024, 1024, 1] {debug_instruction_number = 248 : i32} : memref<?xbf16, #hivm.address_space<gm>> to memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_74 = memref.alloc() {alignment = 64 : i64, debug_instruction_number = 249 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>
      hivm.hir.vcast {debug_instruction_number = 250 : i32} ins(%alloc_69 : memref<1x1x1x1024xf32, #hivm.address_space<ub>>) outs(%alloc_74 : memref<1x1x1x1024xbf16, #hivm.address_space<ub>>) round_mode = <rint>
      %subview_75 = memref.subview %alloc_74[0, 0, 0, 0] [%61, %61, %61, %62] [1, 1, 1, 1] {debug_instruction_number = 251 : i32} : memref<1x1x1x1024xbf16, #hivm.address_space<ub>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>
      %subview_76 = memref.subview %reinterpret_cast_73[0, 0, 0, 0] [%61, %61, %61, %62] [1, 1, 1, 1] {debug_instruction_number = 252 : i32} : memref<1x1x1x1024xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_75 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1]>, #hivm.address_space<ub>>) outs(%subview_76 : memref<?x?x?x?xbf16, strided<[1024, 1024, 1024, 1], offset: ?>, #hivm.address_space<gm>>) {debug_instruction_number = 253 : i32}
    } {debug_instruction_number = 255 : i32}
    return {debug_instruction_number = 256 : i32}
}
// -----
// CHECK-LABEL: @shape_cal_0(
// CHECK: return %alloc
func.func @shape_cal_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> memref<2xindex> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<?x?xf32>
  %dim_0 = memref.dim %arg3, %c0 : memref<?x?xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
  memref.store %dim, %alloc[%c0] : memref<2xindex>
  memref.store %dim_0, %alloc[%c1] : memref<2xindex>
  return %alloc : memref<2xindex>
}

// -----

module {
// CHECK-LABEL: @indirect_changes_0(
  func.func @indirect_changes_0(%arg1 : memref<4xf32>) -> (f32, f32) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.100000e+01 : f32
    memref.store %cst, %arg1[%c0] : memref<4xf32>
    %load = memref.load %arg1[%c0] : memref<4xf32>
    %alloc = memref.alloc() : memref<4xf32>
    memref.store %cst, %alloc[%c0] : memref<4xf32>
    %load_1 = memref.load %alloc[%c0] : memref<4xf32>
    return %load, %load_1 : f32, f32
  }
// CHECK-LABEL: @indirect_changes(
  func.func @indirect_changes() -> (f32, f32, f32) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 4.200000e+01 : f32
    %step = arith.constant 1 : i32
    %start = arith.constant 1 : i32
    %end = arith.constant 10 : i32
    %alloc = memref.alloc() : memref<4xf32>
// CHECK: memref.store
    memref.store %cst, %alloc[%c0] : memref<4xf32>
// CHECK: scf.for
    scf.for %i = %start to %end step %step  : i32 {
// CHECK: memref.load
      %load = memref.load %alloc[%c0] : memref<4xf32>
// CHECK: arith.addf
      %1 = arith.addf %load, %load : f32
// CHECK: memref.store
      memref.store %1, %alloc[%c0] : memref<4xf32>
    }
// CHECK: memref.load
    %load_1 = memref.load %alloc[%c0]: memref<4xf32>
// CHECK: scf.for
    scf.for %i = %start to %end step %step  : i32 {
      %cst_1 = arith.constant 2.200000e+01 : f32
      %alloc_1 = memref.alloc() : memref<4xf32>
// CHECK-NOT: memref.store
      memref.store %cst_1, %alloc_1[%c0] : memref<4xf32>
// CHECK-NOT: memref.load
      %load_2 = memref.load %alloc_1[%c0] : memref<4xf32>
// CHECK: scf.for
      scf.for %j = %start to %end step %step  : i32 {
// CHECK: memref.store
        memref.store %cst, %alloc[%c0] : memref<4xf32>
// CHECK-NOT: memref.load
        %load = memref.load %alloc[%c0] : memref<4xf32>
// CHECK: arith.addf
        %1 = arith.addf %load, %load : f32
// CHECK: memref.store
        memref.store %1, %alloc[%c0] : memref<4xf32>
      }
      %1 = arith.addf %load_2, %load_2 : f32
      memref.store %1, %alloc[%c0] : memref<4xf32>
    }
// CHECK: memref.load
    %load_2 = memref.load %alloc[%c0]: memref<4xf32>
// CHECK: memref.store
    memref.store %cst, %alloc[%c0] : memref<4xf32>
// CHECK: call @indirect_changes_0
    %returned:2 = call @indirect_changes_0(%alloc) : (memref<4xf32>) -> (f32, f32)
// CHECK: memref.load
    %load_3 = memref.load %alloc[%c0]: memref<4xf32>
// CHECK: return 
    return %load_1, %load_2, %load_3 : f32, f32, f32
  }
}

// -----
// CHECK-LABEL: func.func @memref_view_subview
// CHECK: memref.subview
// CHECK: memref.store
// CHECK: hivm.hir.vreduce
// CHECK: memref.load
// CHECK: memref.view
// CHECK: memref.store
// CHECK: hivm.hir.vreduce
// CHECK: memref.load
// CHECK: return
func.func @memref_view_subview(%arg1 : memref<4xi8>) -> (i8, i8) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1 : i8

  %alloc = memref.alloc() : memref<4xi8>
  %subview = memref.subview %alloc[0][1][5] : memref<4xi8> to memref<1xi8, strided<[5]>>
  memref.store %cst, %alloc[%c0] : memref<4xi8>

  hivm.hir.vreduce <min> ins(%alloc : memref<4xi8>) outs(%subview : memref<1xi8, strided<[5]>>) reduce_dims = [0]
  %load = memref.load %alloc[%c0] : memref<4xi8>

  %view = memref.view %alloc[%c0][] : memref<4xi8> to memref<1xi8>
  memref.store %cst, %alloc[%c0] : memref<4xi8>

  hivm.hir.vreduce <min> ins(%alloc : memref<4xi8>) outs(%view : memref<1xi8>) reduce_dims = [0]
  %load_1 = memref.load %alloc[%c0] : memref<4xi8>

  return %load, %load_1 : i8, i8
}

// -----

// CHECK-LABEL: @expand_shape_as_alias(
// CHECK: %[[I0:.*]] = arith.constant 0 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.store %[[C1]], %[[ALLOC]] 
// CHECK: %[[ALIAS:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK: memref.store %[[C0]], %[[ALIAS]]
// CHECK: %[[ALLOC_0:.*]] = memref.alloc
// CHECK: memref.store %[[C0]], %[[ALLOC_0]]
func.func @expand_shape_as_alias(%arg0: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<i32>
  memref.store %c1_i32, %alloc[] : memref<i32>
  %expand_shape = memref.expand_shape %alloc [] output_shape [1] : memref<i32> into memref<1xi32>
  memref.store %c0_i32, %expand_shape[%c0] : memref<1xi32>
  %0 = memref.load %alloc[] : memref<i32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xi32>
  memref.store %0, %alloc_0[%c0] : memref<1xi32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  hivm.hir.store ins(%alloc_0 : memref<1xi32>) outs(%reinterpret_cast : memref<1xi32, strided<[1]>>)
  return
}

// -----

// CHECK-LABEL: @view_like_operation
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32, #hivm.address_space<ub>>
// CHECK: memref.store %cst, %[[ALLOC_1]][%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC_1]]
// CHECK: %[[BASE_BUFFER:.*]], %offset[[_:.*]] = memref.extract_strided_metadata %[[SUBVIEW]]
// CHECK: %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BASE_BUFFER]]
// CHECK: hivm.hir.vreduce <sum> [[_:.*]] outs(%[[REINTERPRET_CAST]] : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOC_1]]
// CHECK: return %[[LOAD]] : f32
module {
  func.func @view_like_operation() -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<29x8x1x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0, 0, 0] [29, 1, 1, 1] [1, 1, 1, 1] : memref<29x8x1x1xf32, #hivm.address_space<ub>> to memref<29x1x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %subview [[0], [1, 2]] : memref<29x1x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>> into memref<29x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %collapse_shape[0, 0] [29, 1] [1, 1] : memref<29x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<29xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview_0 : memref<29xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [29, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<29x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32, #hivm.address_space<ub>>
    memref.store %cst, %alloc_1[%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
    %subview_2 = memref.subview %alloc_1[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %subview_2 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<232xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum> ins(%reinterpret_cast : memref<29x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%alloc_8 : memref<232xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %0 = memref.load %alloc_1[%c0, %c0] : memref<1x1xf32, #hivm.address_space<ub>>
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: @prevent_removal_alloc_1
// CHECK: scf.for
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: scf.yield %[[ALLOC]]
module {
  func.func @prevent_removal_alloc_1(%arg0: memref<?xi64> {tt.divisibility = 16 : i32}) -> memref<1xi64> {
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %alloc = memref.alloc() : memref<1xi64>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
    %0 = scf.for %arg1 = %c2_i32 to %c5_i32 step %c2_i32 iter_args(%arg2 = %alloc_0) -> (memref<1xi64>)  : i32 {
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      %1 = memref.load %arg2[%c0] : memref<1xi64>
      %2 = memref.load %alloc[%c0] : memref<1xi64>
      %3 = arith.addi %1, %2 : i64
      memref.store %3, %alloc_1[%c0] : memref<1xi64>
      scf.yield %alloc_1 : memref<1xi64>
    }
    return %0 : memref<1xi64>
  }
}

// -----

// CHECK-LABEL: @ternary_op
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST_1:.*]] = arith.constant 1
// CHECK: %[[CST_2:.*]] = arith.constant 2
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK: memref.store %[[CST_1]], %[[ALLOC]][%[[C0]]] : memref<1xf32>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK: memref.store %[[CST_2]], %[[ALLOC_2]][%[[C0]]] : memref<1xf32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK: hivm.hir.vsel ins(%[[TRUE]], %[[ALLOC_2]], %[[ALLOC]] : i1, memref<1xf32>, memref<1xf32>
// CHECK: arith.mulf %[[CST_1]]
func.func @ternary_op() {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
  memref.store %cst_0, %alloc[%c0] : memref<1xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
  memref.store %cst_1, %alloc_1[%c0] : memref<1xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
  hivm.hir.vsel ins(%true, %alloc_1, %alloc : i1, memref<1xf32>, memref<1xf32>) outs(%alloc_2 : memref<1xf32>)
  %0 = memref.load %alloc[%c0] : memref<1xf32>
  %1 = arith.mulf %0, %cst : f32
  return
}

// -----

// CHECK: @alloc_on_function(%[[ARG0:.*]]: memref<1xf32>)
// CHECK: %[[CST:.*]] = arith.constant 1
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: memref.store %cst, %[[ARG0]][%[[C0]]] : memref<1xf32>
// CHECK: %[[CST]]
func.func @alloc_on_function(%arg0: memref<1xf32>) -> f32 {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  memref.store %cst, %arg0[%c0] : memref<1xf32>
  %0 = memref.load %arg0[%c0] : memref<1xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   func.func @double_free(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<1xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: memref<1xf32>) -> memref<1xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_6]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           hivm.hir.atomic_cas ins(%[[VAL_5]], %[[VAL_6]] : memref<1xf32>, memref<1xf32>) outs(%[[VAL_7]] : memref<1xf32>)
// CHECK:           return %[[VAL_7]] : memref<1xf32>
// CHECK:         }
func.func @double_free(%arg0: memref<1xf32>,%arg1: memref<1xf32>) -> memref<1xf32> {
  %c0 = arith.constant 0 : index
  %0 = memref.load %arg0[%c0] : memref<1xf32>
  %1 = memref.load %arg1[%c0] : memref<1xf32>
  %alloc_0 = memref.alloc(): memref<1xf32>
  memref.store %0, %alloc_0[%c0]: memref<1xf32>
  %alloc_1 = memref.alloc(): memref<1xf32>
  memref.store %1, %alloc_1[%c0]: memref<1xf32>
  %alloc_2 = memref.alloc(): memref<1xf32>
  hivm.hir.atomic_cas ins(%alloc_0, %alloc_1: memref<1xf32>, memref<1xf32>) outs(%alloc_2: memref<1xf32>)
  return %alloc_2 : memref<1xf32>
}

// -----
// CHECK-LABEL:   func.func @atomic_cas(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<1xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: memref<1xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_6]]{{\[}}%[[VAL_2]]] : memref<1xf32>
// CHECK:           hivm.hir.atomic_cas ins(%[[VAL_5]], %[[VAL_6]] : memref<1xf32>, memref<1xf32>) outs(%[[VAL_0]] : memref<1xf32>)
// CHECK:           return
// CHECK:         }
func.func @atomic_cas(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %0 = memref.load %arg0[%c0] : memref<1xf32>
  %1 = memref.load %arg1[%c0] : memref<1xf32>
  %alloc = memref.alloc() : memref<1xf32>
  memref.store %0, %alloc[%c0] : memref<1xf32>
  %alloc_0 = memref.alloc() : memref<1xf32>
  memref.store %1, %alloc_0[%c0] : memref<1xf32>
  hivm.hir.atomic_cas ins(%alloc, %alloc_0 : memref<1xf32>, memref<1xf32>) outs(%arg0 : memref<1xf32>)
  return
}