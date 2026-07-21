// REQUIRES: regbase
// TODO: enable after migrating regbase pipeline dependencies
// RUN: bishengir-opt -convert-linalg-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_atomic_add
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_add(%arg0 : memref<?xf32> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xf32>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xf32, strided<[1]>>
  // CHECK:       hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%[[UB_MEMREF:.*]] : memref<256xf32, strided<[1]>>) outs(%[[GM_MEMREF:.*]] : memref<256xf32, strided<[1], offset: ?>>)
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xf32, strided<[1], offset: ?>>, memref<256xf32, strided<[1]>>) outs(%reinterpret_cast : memref<256xf32, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_atomic_umax
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_umax(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
  // CHECK:       hfusion.store {atomic_kind = #hfusion.atomic_kind<umax>} ins(%[[VAL:.*]] : memref<256xi16, strided<[1]>>) outs(%[[OUT:.*]] : memref<256xi16, strided<[1], offset: ?>>)
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xi16, strided<[1], offset: ?>>, memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "umax", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: i16, %in_0: i16, %out: i16):
      %3 = arith.maxui %in, %in_0 : i16
      linalg.yield %3 : i16
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index
func.func @test_reduce_with_index(%arg0 : tensor<256x64xf32>, %arg1 : tensor<256x64xi32>) -> tensor<256xf32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<256xf32>
  %1 = tensor.empty() : tensor<256xi32>
  //CHECK:  %[[REDUCED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%[[INPUT0:.*]], %[[INPUT1:.*]] : tensor<256x64xf32>, tensor<256x64xi32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
  %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1]  {reduce_mode = "max_with_index", unsigned_src = "false"}
    (%in: f32, %in_1: i32, %init: f32, %init_1: i32) {
      %7 = arith.cmpf ogt, %in, %init : f32
      %8 = arith.cmpf oeq, %in, %init : f32
      %9 = arith.cmpf une, %in, %in : f32
      %10 = arith.cmpf une, %init, %init : f32
      %11 = arith.xori %10, %true : i1
      %12 = arith.andi %9, %11 : i1
      %13 = arith.ori %7, %12 : i1
      %14 = arith.andi %9, %10 : i1
      %15 = arith.ori %8, %14 : i1
      %16 = arith.cmpi slt, %in_1, %init_1 : i32
      %17 = arith.andi %15, %16 : i1
      %18 = arith.ori %13, %17 : i1
      %19 = arith.select %18, %in, %init : f32
      %20 = arith.select %18, %in_1, %init_1 : i32
      linalg.yield %19, %20 : f32, i32
    }
  return %0 : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index
func.func @test_reduce_with_index(%arg0 : tensor<256x64xf32>, %arg1 : tensor<256x64xi32>) -> tensor<256xf32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<256xf32>
  %1 = tensor.empty() : tensor<256xi32>
  //CHECK:  %[[REDUCED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = false, unsigned_src = false} <max> ins(%[[INPUT0:.*]], %[[INPUT1:.*]] : tensor<256x64xf32>, tensor<256x64xi32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
  %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1]  {reduce_mode = "max_with_index", unsigned_src = "false", tie_break_left = "false"}
    (%in: f32, %in_1: i32, %init: f32, %init_1: i32) {
      %7 = arith.cmpf ogt, %in, %init : f32
      %8 = arith.cmpf oeq, %in, %init : f32
      %9 = arith.cmpf une, %in, %in : f32
      %10 = arith.cmpf une, %init, %init : f32
      %11 = arith.xori %10, %true : i1
      %12 = arith.andi %9, %11 : i1
      %13 = arith.ori %7, %12 : i1
      %14 = arith.andi %9, %10 : i1
      %15 = arith.ori %8, %14 : i1
      %16 = arith.cmpi slt, %in_1, %init_1 : i32
      %17 = arith.andi %15, %16 : i1
      %18 = arith.ori %13, %17 : i1
      %19 = arith.select %18, %in, %init : f32
      %20 = arith.select %18, %in_1, %init_1 : i32
      linalg.yield %19, %20 : f32, i32
    }
  return %0 : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @test_rintf
func.func private @__hmf_rint(f32) -> f32 attributes {llvm.readnone}
func.func @test_rintf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK: %[[RET:.*]] = hfusion.cast {
  // CHECK-SAME: cast = #hfusion.type_fn<cast_signed>,
  // CHECK-SAME: enable_overflow = true,
  // CHECK-SAME: enable_saturate = true,
  // CHECK-SAME: round_mode = #hfusion.round_mode<rint>
  %ret = linalg.map { func.call {callee = @__hmf_rint} } ins(%arg0 : tensor<6x6xf32>) outs(%arg0 : tensor<6x6xf32>)
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_gather_normal
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<5x6x1xf16> into tensor<5x6xf16>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<5x3x1xi32> into tensor<5x3xi32>
// CHECK: tensor.empty() : tensor<5x3xf16>
// CHECK: hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins({{.*}}, {{.*}} : tensor<5x6xf16>, tensor<5x3xi32>) outs({{.*}} : tensor<5x3xf16>) axis = 1 -> tensor<5x3xf16>
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}5, 3, 1] : tensor<5x3xf16> into tensor<5x3x1xf16>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @test_gather_normal(%arg0: tensor<5x6x1xf16>, %arg1: tensor<5x3x1xi32>) -> tensor<5x3x1xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<5x6x1xf16> into tensor<5x6xf16>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<5x3x1xi32> into tensor<5x3xi32>
  %0 = tensor.empty() : tensor<5x3xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "gather"]} ins(%collapsed, %collapsed_0 : tensor<5x6xf16>, tensor<5x3xi32>) outs(%0 : tensor<5x3xf16>) {
  ^bb0(%in: f16, %in_1: i32, %out: f16):
    %2 = linalg.index 2 : index
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.cmpi eq, %3, %in_1 : i32
    %5 = arith.select %4, %in, %out : f16
    linalg.yield %5 : f16
  } -> tensor<5x3xf16>
  %expanded = tensor.expand_shape %1 [[0], [1, 2]] output_shape [5, 3, 1] : tensor<5x3xf16> into tensor<5x3x1xf16>
  return %expanded : tensor<5x3x1xf16>
}
