// RUN: bishengir-opt --hfusion-flatten-ops="flatten-mode=tidy register-based=true" --split-input-file %s | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true" -split-input-file --cse --canonicalize| FileCheck %s --check-prefix=CSE

// -----

// CHECK-LABEL: func.func @memref_load_store(
// CHECK: memref.load
// CHECK-SAME: memref<111xi8>
// CHECK: memref.store
// CHECK-SAME: memref<111xi8>
func.func @memref_load_store(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c111 = arith.constant 111 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 111], strides: [111, 1] : memref<?xi8> to memref<1x111xi8, strided<[111, 1]>>
  %alloc = memref.alloc() : memref<1x111xi8>
  memref.copy %reinterpret_cast, %alloc : memref<1x111xi8, strided<[111, 1]>> to memref<1x111xi8>
  %alloc_0 = memref.alloc() : memref<1x111xi8>
  %0 = memref.load %alloc[%c0, %c0] : memref<1x111xi8>
  memref.store %0, %alloc_0[%c0, %c0] : memref<1x111xi8>
  scf.for %arg10 = %c1 to %c111 step %c1 {
    %2 = arith.subi %arg10, %c1 : index
    %3 = memref.load %alloc[%c0, %arg10] : memref<1x111xi8>
    %4 = memref.load %alloc_0[%c0, %2] : memref<1x111xi8>
    %5 = arith.andi %4, %3 : i8
    memref.store %5, %alloc_0[%c0, %arg10] : memref<1x111xi8>
  }
  %1 = bufferization.to_tensor %alloc_0 restrict : memref<1x111xi8>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 111], strides: [111, 1] : memref<?xi8> to memref<1x111xi8, strided<[111, 1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast_1 : (tensor<1x111xi8>, memref<1x111xi8, strided<[111, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: test_arange_0
// CHECK: hfusion.arange offset{{.*}} strides{{.*}} outs({{.*}} : tensor<150xi32>) -> tensor<150xi32>
func.func @test_arange_0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c150 = arith.constant 150 : index
  %c4_i32 = arith.constant 4 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<1x150xi32>
  %1 = tensor.empty() : tensor<4x1x150xi64>
  %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<4x1x150xi64>) -> tensor<4x1x150xi64>
  %3 = arith.muli %arg9, %c4_i32 : i32
  %4 = arith.addi %3, %c4_i32 : i32
  %5 = arith.minsi %4, %arg4 : i32
  %6 = hfusion.arange offset[%c0] strides[%c150, %c1] outs(%0 : tensor<1x150xi32>) -> tensor<1x150xi32>
  %7 = arith.index_cast %3 : i32 to index
  %8 = arith.muli %7, %c150 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [4, 150], strides: [150, 1] : memref<?xi64> to memref<4x150xi64, strided<[150, 1], offset: ?>>
  %alloc = memref.alloc() : memref<4x1x150xi64>
  %collapse_shape = memref.collapse_shape %alloc [[0], [1, 2]] : memref<4x1x150xi64> into memref<4x150xi64>
  %9 = arith.addi %7, %c4 : index
  %10 = arith.index_cast %5 : i32 to index
  %11 = arith.maxsi %7, %10 : index
  %12 = arith.minsi %9, %11 : index
  %13 = arith.subi %12, %7 : index
  %14 = arith.cmpi slt, %13, %c4 : index
  scf.if %14 {
    linalg.fill ins(%c0_i64 : i64) outs(%collapse_shape : memref<4x150xi64>)
  } {hivm.unlikely_condition}
  %subview = memref.subview %reinterpret_cast[0, 0] [%13, 150] [1, 1] : memref<4x150xi64, strided<[150, 1], offset: ?>> to memref<?x150xi64, strided<[150, 1], offset: ?>>
  %subview_0 = memref.subview %collapse_shape[0, 0] [%13, 150] [1, 1] : memref<4x150xi64> to memref<?x150xi64, strided<[150, 1]>>
  memref.copy %subview, %subview_0 : memref<?x150xi64, strided<[150, 1], offset: ?>> to memref<?x150xi64, strided<[150, 1]>>
  %15 = bufferization.to_tensor %alloc restrict writable : memref<4x1x150xi64>
  %16 = tensor.empty() : tensor<4x1x150xi1>
  %17 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%15, %2 : tensor<4x1x150xi64>, tensor<4x1x150xi64>) outs(%16 : tensor<4x1x150xi1>) -> tensor<4x1x150xi1>
  %18 = tensor.empty() : tensor<4x1x150xf32>
  %19 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%17 : tensor<4x1x150xi1>) outs(%18 : tensor<4x1x150xf32>) -> tensor<4x1x150xf32>
  %20 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%19 : tensor<4x1x150xf32>) outs(%1 : tensor<4x1x150xi64>) -> tensor<4x1x150xi64>
  %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %c1_i32 : tensor<1x150xi32>, i32) outs(%0 : tensor<1x150xi32>) -> tensor<1x150xi32>
  %22 = tensor.empty() : tensor<1x150xi64>
  %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%21 : tensor<1x150xi32>) outs(%22 : tensor<1x150xi64>) -> tensor<1x150xi64>
  %broadcasted = linalg.broadcast ins(%23 : tensor<1x150xi64>) outs(%1 : tensor<4x1x150xi64>) dimensions = [0]
  %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %20 : tensor<4x1x150xi64>, tensor<4x1x150xi64>) outs(%1 : tensor<4x1x150xi64>) -> tensor<4x1x150xi64>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [4, 150], strides: [150, 1] : memref<?xi64> to memref<4x150xi64, strided<[150, 1], offset: ?>>
  %extracted_slice = tensor.extract_slice %24[0, 0, 0] [%13, 1, 150] [1, 1, 1] : tensor<4x1x150xi64> to tensor<?x1x150xi64>
  %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%13, 150] [1, 1] : memref<4x150xi64, strided<[150, 1], offset: ?>> to memref<?x150xi64, strided<[150, 1], offset: ?>>
  %expand_shape = memref.expand_shape %subview_2 [[0], [1, 2]] output_shape [%13, 1, 150] : memref<?x150xi64, strided<[150, 1], offset: ?>> into memref<?x1x150xi64, strided<[150, 150, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %expand_shape : (tensor<?x1x150xi64>, memref<?x1x150xi64, strided<[150, 150, 1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_subview_strided_memref_copy_and_elemwise
// CHECK-SAME: %[[VAL_0:.*]]: memref<1x8x2x8x3x1x85xf32>,
// Important things about the new memref is prioritizing unit dimension to be collapsed with contiguous ones:
// CHECK: %[[VAL_5:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0, 1, 2, 3], [4, 5, 6]] : memref<1x8x2x8x3x1x85xf32> into memref<128x255xf32>
// CHECK: %[[VAL_6:.*]] = memref.subview %[[VAL_5]][0, 0] [128, 170] [1, 1] : memref<128x255xf32> to memref<128x170xf32, strided<[255, 1]>>
func.func @test_subview_strided_memref_copy_and_elemwise(%arg0: memref<1x8x2x8x3x1x85xf32>,
                                                          %arg1: tensor<1x8x2x8x2x1x85xf32>,
                                                          %arg2: tensor<1x8x2x8x2x1x85xf32>) -> tensor<1x8x2x8x2x1x85xf32> {
  %subview = memref.subview %arg0[0, 0, 0, 0, 0, 0, 0] [1, 8, 2, 8, 2, 1, 85] [1, 1, 1, 1, 1, 1, 1]
             : memref<1x8x2x8x3x1x85xf32> to memref<1x8x2x8x2x1x85xf32, strided<[32640, 4080, 2040, 255, 85, 85, 1]>>
  %alloc = memref.alloc() : memref<1x8x2x8x2x1x85xf32>
  %empty = tensor.empty() : tensor<1x8x2x8x2x1x85xf32>
  memref.copy %subview, %alloc : memref<1x8x2x8x2x1x85xf32, strided<[32640, 4080, 2040, 255, 85, 85, 1]>> to memref<1x8x2x8x2x1x85xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<1x8x2x8x2x1x85xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
         ins(%tensor, %arg1 : tensor<1x8x2x8x2x1x85xf32>, tensor<1x8x2x8x2x1x85xf32>)
         outs(%empty : tensor<1x8x2x8x2x1x85xf32>) -> tensor<1x8x2x8x2x1x85xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
         ins(%mul, %arg2 : tensor<1x8x2x8x2x1x85xf32>, tensor<1x8x2x8x2x1x85xf32>)
         outs(%empty : tensor<1x8x2x8x2x1x85xf32>) -> tensor<1x8x2x8x2x1x85xf32>
  memref.dealloc %alloc : memref<1x8x2x8x2x1x85xf32>
  return %add : tensor<1x8x2x8x2x1x85xf32>
}

// -----

// CHECK-LABEL: @test_strided_memref_copy_and_elemwise(
// CHECK-SAME: %[[VAL_0:.*]]: memref<1x8x2x8x2x1x1x1x85xf32, strided<[326400, 40800, 20400, 2550, 1275, 255, 85, 85, 1], offset: ?>>,
// Important things about the new memref is prioritizing unit dimension to be collapsed with contiguous ones:
// [0, 1, 2, 3, 4, 5], [6, 7, 8]
// CHECK-DAG: %[[VAL_5:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0, 1, 2, 3, 4, 5], [6, 7, 8]] : memref<1x8x2x8x2x1x1x1x85xf32, strided<[326400, 40800, 20400, 2550, 1275, 255, 85, 85, 1], offset: ?>> into memref<256x85xf32, strided<[1275, 1], offset: ?>>
// CHECK: memref.copy %[[VAL_5]], %{{.*}} : memref<256x85xf32, strided<[1275, 1], offset: ?>> to memref<256x85xf32>
func.func @test_strided_memref_copy_and_elemwise(%arg0: memref<1x8x2x8x2x1x1x1x85xf32, strided<[326400, 40800, 20400, 2550, 1275, 255, 85, 85, 1], offset: ?>>,
                                                  %arg1: tensor<1x8x2x8x2x1x1x1x85xf32>,
                                                  %arg2: tensor<1x8x2x8x2x1x1x1x85xf32>) -> tensor<1x8x2x8x2x1x1x1x85xf32> {
  %alloc = memref.alloc() : memref<1x8x2x8x2x1x1x1x85xf32>
  %empty = tensor.empty() : tensor<1x8x2x8x2x1x1x1x85xf32>
  memref.copy %arg0, %alloc : memref<1x8x2x8x2x1x1x1x85xf32, strided<[326400, 40800, 20400, 2550, 1275, 255, 85, 85, 1], offset: ?>> to memref<1x8x2x8x2x1x1x1x85xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<1x8x2x8x2x1x1x1x85xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
         ins(%tensor, %arg1 : tensor<1x8x2x8x2x1x1x1x85xf32>, tensor<1x8x2x8x2x1x1x1x85xf32>)
         outs(%empty : tensor<1x8x2x8x2x1x1x1x85xf32>) -> tensor<1x8x2x8x2x1x1x1x85xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
         ins(%mul, %arg2 : tensor<1x8x2x8x2x1x1x1x85xf32>, tensor<1x8x2x8x2x1x1x1x85xf32>)
         outs(%empty : tensor<1x8x2x8x2x1x1x1x85xf32>) -> tensor<1x8x2x8x2x1x1x1x85xf32>
  memref.dealloc %alloc : memref<1x8x2x8x2x1x1x1x85xf32>
  return %add : tensor<1x8x2x8x2x1x1x1x85xf32>
}

// -----
// CHECK-LABEL: test_subview_reshape(

// CHECK: %[[VAL_12:.*]] = memref.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [64, 1] : memref<?xi64, strided<[1]>> into memref<?x1xi64>
// CHECK: %[[VAL_13:.*]] = memref.cast %[[VAL_12]] : memref<?x1xi64> to memref<?x1xi64, strided<[1, 1]>>
// CHECK: %[[VAL_14:.*]] = memref.collapse_shape %[[VAL_13]] {{\[\[}}0, 1]] : memref<?x1xi64, strided<[1, 1]>> into memref<?xi64, strided<[1]>>
func.func @test_subview_reshape() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %alloc = memref.alloc() : memref<64x1xi64>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<64x1xi64> to memref<64x1xi64, strided<[1, 1], offset: ?>>
  %alloc_0 = memref.alloc() : memref<64x1xi64>
  %subview = memref.subview %reinterpret_cast[0, 0] [%c32, 1] [1, 1] : memref<64x1xi64, strided<[1, 1], offset: ?>> to memref<?x1xi64, strided<[1, 1], offset: ?>>
  %subview_1 = memref.subview %alloc_0[0, 0] [%c32, 1] [1, 1] : memref<64x1xi64> to memref<?x1xi64, strided<[1, 1]>>
  %collapse_shape = memref.collapse_shape %subview_1 [[0, 1]] : memref<?x1xi64, strided<[1, 1]>> into memref<?xi64, strided<[1]>>
  %dim = memref.dim %collapse_shape, %c0 : memref<?xi64, strided<[1]>>
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%dim, 1] : memref<?xi64, strided<[1]>> into memref<?x1xi64>
  memref.copy %subview, %expand_shape : memref<?x1xi64, strided<[1, 1], offset: ?>> to memref<?x1xi64>
  %0 = bufferization.to_tensor %alloc_0 restrict writable : memref<64x1xi64>
  memref.dealloc %alloc : memref<64x1xi64>
  memref.dealloc %alloc_0 : memref<64x1xi64>
  return
}


// -----

// CHECK-LABEL: func.func @flatten_sort(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<7x4x32xi32>)
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], {{\[}}2]] : tensor<7x4x32xi32> into tensor<28x32xi32>
// CHECK: %[[SORTED:.*]] = hfusion.sort ins(%[[COLLAPSED]] : tensor<28x32xi32>) descending = true sort_axis = 1 -> tensor<28x32xi32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[SORTED]] {{\[\[}}0, 1], {{\[}}2]] output_shape {{\[}}7, 4, 32] : tensor<28x32xi32> into tensor<7x4x32xi32>
// CHECK: return %[[EXPANDED]] : tensor<7x4x32xi32>
func.func @flatten_sort(%arg0: tensor<7x4x32xi32>) -> tensor<7x4x32xi32> {
    %0 = hfusion.sort ins(%arg0 : tensor<7x4x32xi32>) descending = true sort_axis = 2 -> tensor<7x4x32xi32>
    return %0 : tensor<7x4x32xi32>
}

// -----

// CHECK-LABEL:   func.func @flatten_scf_if(
// CHECK-SAME:                                      %[[ARG:.*]]: tensor<1x6x4xf32>) {
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1, 2]] : tensor<1x6x4xf32> into tensor<24xf32>
// CHECK:           %[[BOOL:.*]] = arith.cmpi
// CHECK:           %[[RET:.*]] = scf.if %[[BOOL]] -> (tensor<24xf32>) {
// CHECK:             scf.yield %[[COLLAPSED]] : tensor<24xf32>
// CHECK:           } else {
// CHECK:             %[[EXTRACTED:.*]] = tensor.extract_slice %[[COLLAPSED]][0] [4] [1] : tensor<24xf32> to tensor<4xf32>
// CHECK:             %[[INSERTED:.*]] = tensor.insert_slice %[[EXTRACTED]] into %[[COLLAPSED]][0] [4] [1] : tensor<4xf32> into tensor<24xf32>
// CHECK:             scf.yield %[[INSERTED]] : tensor<24xf32>
// CHECK:           }
func.func @flatten_scf_if(%arg0: tensor<1x6x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.cmpi slt, %c1, %c0 : index
  %1 = scf.if %0 -> (tensor<1x6x4xf32>) {
    scf.yield %arg0 : tensor<1x6x4xf32>
  } else {
    %2 = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 4] [1, 1, 1] : tensor<1x6x4xf32> to tensor<1x1x4xf32>
    %3 = tensor.insert_slice %2 into %arg0[0, 0, 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<1x6x4xf32>
    scf.yield %3 : tensor<1x6x4xf32>
  }
  return
}

// -----

// CHECK-LABEL:   func.func @flatten_scf_for(
// CHECK-SAME:                               %[[ARG:.*]]: tensor<8x32xf32>, %[[END:.*]]: i32)
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1]] : tensor<8x32xf32> into tensor<256xf32>
// CHECK:           %[[BEGIN:.*]] = arith.constant 0 : i32
// CHECK:           %[[STEP:.*]] = arith.constant 1 : i32
// CHECK:           %[[RET:.*]] = scf.for %[[INDEX:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] iter_args(%[[ITER:.*]] = %[[COLLAPSED]]) -> (tensor<256xf32>) : i32 {
// CHECK:                        %[[EMPTY:.*]] = tensor.empty() : tensor<8x32xf32>
// CHECK:                        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1]] : tensor<8x32xf32> into tensor<256xf32>
// CHECK:                        %[[BINARY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ITER]], %[[ITER]] : tensor<256xf32>, tensor<256xf32>) outs(%[[COLLAPSED_0]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:                        scf.yield %[[BINARY]] : tensor<256xf32>
// CHECK:           %[[EXPENDED:.*]] = tensor.expand_shape %[[RET]] {{\[\[}}0, 1]] output_shape [8, 32] : tensor<256xf32> into tensor<8x32xf32>
// CHECK:           return %[[EXPENDED]] : tensor<8x32xf32>
func.func @flatten_scf_for(%arg0: tensor<8x32xf32>, %arg1: i32) -> tensor<8x32xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = scf.for %arg11 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg3 = %arg0) -> (tensor<8x32xf32>)  :i32 {
      %1 = tensor.empty() : tensor<8x32xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg3, %arg3 : tensor<8x32xf32>,tensor<8x32xf32>) outs(%1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.yield %2 : tensor<8x32xf32>
  }
  return %0 : tensor<8x32xf32>
}

// -----

// CSE-LABEL: func.func @flatten_scf_for_02
// CSE: scf.yield %[[VAL2:.*]] : tensor<1x8x32xf32>
// CSE: tensor.collapse_shape %[[VAL1:.*]] {{\[}}[0, 1], [2]] : tensor<1x8x32xf32> into tensor<8x32xf32>
func.func @flatten_scf_for_02(%arg0: tensor<1x32x8xf32>, %arg1: tensor<32x8xf32>, %arg2: i32) -> tensor<8x32xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x8x32xf32>
  %1 = tensor.empty() : tensor<8x32xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<1x32x8xf32>) outs(%0 : tensor<1x8x32xf32>) permutation = [0, 2, 1]
  %transposed_0 = linalg.transpose ins(%arg1 : tensor<32x8xf32>) outs(%1 : tensor<8x32xf32>) permutation = [1, 0]
  %2:2 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %transposed_0, %arg5 = %transposed) -> (tensor<8x32xf32>, tensor<1x8x32xf32>)  : i32 {
    %3 = tensor.empty() : tensor<1x8x32xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg5, %arg5 : tensor<1x8x32xf32>, tensor<1x8x32xf32>) outs(%3 : tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
    %collapsed_1 = tensor.collapse_shape %4 [[0, 1], [2]] : tensor<1x8x32xf32> into tensor<8x32xf32>
    scf.yield %collapsed_1, %4 : tensor<8x32xf32>, tensor<1x8x32xf32>
  }
  %collapsed = tensor.collapse_shape %2#1 [[0, 1], [2]] : tensor<1x8x32xf32> into tensor<8x32xf32>
  return %collapsed : tensor<8x32xf32>
}

// -----
// CHECK-LABEL: func.func @flatten_scf_if_2(
// CHECK-SAME:                            %[[ARG:.*]]: tensor<4x8x32xf32>
// CHECK:         %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1, 2]] : tensor<4x8x32xf32> into tensor<1024xf32>
// CHECK:         %[[RET:.*]] = scf.if %0 -> (tensor<1024xf32>) {
// CHECK:           %2 = tensor.empty() : tensor<4x8x32xf32>
// CHECK:           %[[COLLAPSED_0:.*]] = tensor.collapse_shape %2
// CHECK:           %[[BINARY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME:        ins(%[[COLLAPSED]], %[[COLLAPSED]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME:        outs(%[[COLLAPSED_0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           scf.yield %[[BINARY]] : tensor<1024xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[COLLAPSED]] : tensor<1024xf32>
func.func @flatten_scf_if_2(%arg0: tensor<4x8x32xf32>) {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %0 = arith.cmpi slt, %c1, %c0 : index
 %1 = scf.if %0 -> (tensor<4x8x32xf32>) {
  %2 = tensor.empty() : tensor<4x8x32xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<4x8x32xf32>, tensor<4x8x32xf32>) outs(%2 : tensor<4x8x32xf32>) -> tensor<4x8x32xf32>
  scf.yield %3 : tensor<4x8x32xf32>
 } else {
  scf.yield %arg0 : tensor<4x8x32xf32>
 }
 return
}

// -----

// CHECK-LABEL: func.func @flatten_reduce_right(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2]] : tensor<7x15x32xf32> into tensor<105x32xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7x15xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1]] : tensor<7x15xf32> into tensor<105xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%cst : f32) outs(%[[COLLAPSED_0]] : tensor<105xf32>) -> tensor<105xf32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[COLLAPSED]] : tensor<105x32xf32>) outs(%[[FILL]] : tensor<105xf32>) dimensions = [1]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1]] output_shape {{\[}}7, 15] : tensor<105xf32> into tensor<7x15xf32>
func.func @flatten_reduce_right(%arg0: tensor<7x15x32xf32>) -> (tensor<7x15xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<7x15xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7x15xf32>) -> tensor<7x15xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<7x15x32xf32>) outs(%1 : tensor<7x15xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  return %reduced : tensor<7x15xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_reduce_left(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0], {{\[}}1, 2]] : tensor<7x15x32xf32> into tensor<7x480xf32>

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<15x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1]] : tensor<15x32xf32> into tensor<480xf32>

// CHECK: %[[FILL:.*]] = linalg.fill ins(%cst : f32) outs(%[[COLLAPSED_0]] : tensor<480xf32>) -> tensor<480xf32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[COLLAPSED]] : tensor<7x480xf32>) outs(%[[FILL]] : tensor<480xf32>) dimensions = [0]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1]] output_shape {{\[}}15, 32] : tensor<480xf32> into tensor<15x32xf32>
// CHECK: return %[[EXPANDED]] : tensor<15x32xf32>
func.func @flatten_reduce_left(%arg0: tensor<7x15x32xf32>) -> (tensor<15x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<15x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<15x32xf32>) -> tensor<15x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<7x15x32xf32>) outs(%1 : tensor<15x32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  return %reduced : tensor<15x32xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_reduce_middle(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x100x8x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2], {{\[}}3, 4]] : tensor<7x15x100x8x32xf32> into tensor<105x100x256xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7x15x8x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2, 3]] : tensor<7x15x8x32xf32> into tensor<105x256xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%cst : f32) outs(%[[COLLAPSED_0]] : tensor<105x256xf32>) -> tensor<105x256xf32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[COLLAPSED]] : tensor<105x100x256xf32>) outs(%[[FILL]] : tensor<105x256xf32>) dimensions = [1]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1], {{\[}}2, 3]] output_shape {{\[}}7, 15, 8, 32] : tensor<105x256xf32> into tensor<7x15x8x32xf32>
func.func @flatten_reduce_middle(%arg0: tensor<7x15x100x8x32xf32>) -> (tensor<7x15x8x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<7x15x8x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7x15x8x32xf32>) -> tensor<7x15x8x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<7x15x100x8x32xf32>) outs(%1 : tensor<7x15x8x32xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  return %reduced : tensor<7x15x8x32xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_reduce_2_axis_split(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x100x8x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0], {{\[}}1, 2], {{\[}}3], {{\[}}4]] : tensor<7x15x100x8x32xf32> into tensor<7x1500x8x32xf32>
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<15x100x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2]] : tensor<15x100x32xf32> into tensor<1500x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[COLLAPSED_0]] : tensor<1500x32xf32>) -> tensor<1500x32xf32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[COLLAPSED]] : tensor<7x1500x8x32xf32>) outs(%[[FILL]] : tensor<1500x32xf32>) dimensions = [0, 2]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1], {{\[}}2]] output_shape {{\[}}15, 100, 32] : tensor<1500x32xf32> into tensor<15x100x32xf32>
func.func @flatten_reduce_2_axis_split(%arg0: tensor<7x15x100x8x32xf32>) -> (tensor<15x100x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<15x100x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<15x100x32xf32>) -> tensor<15x100x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<7x15x100x8x32xf32>) outs(%1 : tensor<15x100x32xf32>) dimensions = [0,3]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  return %reduced : tensor<15x100x32xf32>
}


// -----

// CHECK-LABEL: func.func @flatten_reduce_2_axis_join(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x100x8x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2, 3], {{\[}}4]] : tensor<7x15x100x8x32xf32> into tensor<105x800x32xf32>
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7x15x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2]] : tensor<7x15x32xf32> into tensor<105x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[COLLAPSED_0]] : tensor<105x32xf32>) -> tensor<105x32xf32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[COLLAPSED]] : tensor<105x800x32xf32>) outs(%[[FILL]] : tensor<105x32xf32>) dimensions = [1]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] {{\[\[}}0, 1], {{\[}}2]] output_shape {{\[}}7, 15, 32] : tensor<105x32xf32> into tensor<7x15x32xf32>
func.func @flatten_reduce_2_axis_join(%arg0: tensor<7x15x100x8x32xf32>) -> (tensor<7x15x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<7x15x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7x15x32xf32>) -> tensor<7x15x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<7x15x100x8x32xf32>) outs(%1 : tensor<7x15x32xf32>) dimensions = [2,3]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  return %reduced : tensor<7x15x32xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_broadcast_left(
// CHECK-SAME:  %[[ARG:.*]]: tensor<8x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1]] : tensor<8x32xf32> into tensor<256xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0], {{\[}}1, 2]] : tensor<4x8x32xf32> into tensor<4x256xf32>
// CHECK: %[[BROADCASTED:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<256xf32>) outs(%[[COLLAPSED_0]] : tensor<4x256xf32>) dimensions = [0]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCASTED]] {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}4, 8, 32] : tensor<4x256xf32> into tensor<4x8x32xf32>
func.func @flatten_broadcast_left(%arg0: tensor<8x32xf32>) -> tensor<4x8x32xf32> {
%0 = tensor.empty() : tensor<4x8x32xf32>
%1 = linalg.broadcast
    ins(%arg0 : tensor<8x32xf32>)
    outs(%0 : tensor<4x8x32xf32>)
    dimensions = [0]
return %1 : tensor<4x8x32xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_broadcast_2_axis_join(
// CHECK-SAME:  %[[ARG:.*]]: tensor<4x8x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2]] : tensor<4x8x32xf32> into tensor<32x32xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8x32x3x32xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2, 3], {{\[}}4]] : tensor<4x8x32x3x32xf32> into tensor<32x96x32xf32>
// CHECK: %[[BROADCASTED:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<32x32xf32>) outs(%[[COLLAPSED_0]] : tensor<32x96x32xf32>) dimensions = [1]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCASTED]] {{\[\[}}0, 1], {{\[}}2, 3], {{\[}}4]] output_shape {{\[}}4, 8, 32, 3, 32] : tensor<32x96x32xf32> into tensor<4x8x32x3x32xf32>
func.func @flatten_broadcast_2_axis_join(%arg0: tensor<4x8x32xf32>) -> tensor<4x8x32x3x32xf32> {
%0 = tensor.empty() : tensor<4x8x32x3x32xf32>
%1 = linalg.broadcast
    ins(%arg0 : tensor<4x8x32xf32>)
    outs(%0 : tensor<4x8x32x3x32xf32>)
    dimensions = [2, 3]
return %1 : tensor<4x8x32x3x32xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_broadcast_right(
// CHECK-SAME:  %[[ARG:.*]]: tensor<2x5xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1]] : tensor<2x5xf32> into tensor<10xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x5x7xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2]] : tensor<2x5x7xf32> into tensor<10x7xf32>
// CHECK: %[[BROADCASTED:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<10xf32>) outs(%[[COLLAPSED_0]] : tensor<10x7xf32>) dimensions = [1]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCASTED]] {{\[\[}}0, 1], {{\[}}2]] output_shape {{\[}}2, 5, 7] : tensor<10x7xf32> into tensor<2x5x7xf32>
func.func @flatten_broadcast_right(%arg0: tensor<2x5xf32>) -> tensor<2x5x7xf32> {
  %0 = tensor.empty() : tensor<2x5x7xf32>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<2x5xf32>)
    outs(%0 : tensor<2x5x7xf32>)
    dimensions = [2]
  return %1 : tensor<2x5x7xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_broadcast_2_axis_split(
// CHECK-SAME:  %[[ARG:.*]]: tensor<8x4x2x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2, 3]] : tensor<8x4x2x32xf32> into tensor<32x64xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x4x7x2x32x9xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1], {{\[}}2], {{\[}}3, 4], {{\[}}5]] : tensor<8x4x7x2x32x9xf32> into tensor<32x7x64x9xf32>
// CHECK: %[[BROADCASTED:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<32x64xf32>) outs(%[[COLLAPSED_0]] : tensor<32x7x64x9xf32>) dimensions = [1, 3]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCASTED]] {{\[\[}}0, 1], {{\[}}2], {{\[}}3, 4], {{\[}}5]] output_shape {{\[}}8, 4, 7, 2, 32, 9] : tensor<32x7x64x9xf32> into tensor<8x4x7x2x32x9xf32>
func.func @flatten_broadcast_2_axis_split(%arg0: tensor<8x4x2x32xf32>) -> tensor<8x4x7x2x32x9xf32> {
  %0 = tensor.empty() : tensor<8x4x7x2x32x9xf32>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8x4x2x32xf32>)
    outs(%0 : tensor<8x4x7x2x32x9xf32>)
    dimensions = [2, 5]
  return %1 : tensor<8x4x7x2x32x9xf32>
}

// -----

// CHECK-LABEL: func.func @flatten_cast(
// CHECK-SAME:  %[[ARG:.*]]: tensor<7x15x13xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1, 2]] : tensor<7x15x13xf32> into tensor<1365xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7x15x13xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[EMPTY]] {{\[\[}}0, 1, 2]] : tensor<7x15x13xf32> into tensor<1365xf32>
// CHECK: %[[CASTED:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[COLLAPSED]] : tensor<1365xf32>) outs(%[[COLLAPSED_0]] : tensor<1365xf32>) -> tensor<1365xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[CASTED]] {{\[\[}}0, 1, 2]] output_shape {{\[}}7, 15, 13] : tensor<1365xf32> into tensor<7x15x13xf32>
// CHECK: return %[[EXPANDED]] : tensor<7x15x13xf32>
func.func @flatten_cast(%arg0: tensor<7x15x13xf32>) -> (tensor<7x15x13xf32>) {
    %0 = tensor.empty() : tensor<7x15x13xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x15x13xf32>) outs(%0 : tensor<7x15x13xf32>) -> tensor<7x15x13xf32>
    return %1 : tensor<7x15x13xf32>
}

// -----

// CHECK-LABEL: func.func @flip(
// CHECK-SAME:  %[[ARG:.*]]: tensor<4x8x17x1x32xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[}}0, 1], {{\[}}2], {{\[}}3, 4]] : tensor<4x8x17x1x32xf32> into tensor<32x17x32xf32>
// CHECK: %[[FLIPPED:.*]] = hfusion.flip %[[COLLAPSED]] : tensor<32x17x32xf32> flip_axis = 1 -> tensor<32x17x32xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[FLIPPED]] {{\[\[}}0, 1], {{\[}}2], {{\[}}3, 4]] output_shape {{\[}}4, 8, 17, 1, 32] : tensor<32x17x32xf32> into tensor<4x8x17x1x32xf32>
// CHECK: return %[[EXPANDED]] : tensor<4x8x17x1x
func.func @flip(%arg0: tensor<4x8x17x1x32xf32>) -> (tensor<4x8x17x1x32xf32>) {
    %0 = hfusion.flip %arg0 : tensor<4x8x17x1x32xf32> flip_axis = 2 -> tensor<4x8x17x1x32xf32>
    return %0 : tensor<4x8x17x1x32xf32>
}

// -----

// CHECK-LABEL: func.func @reduce_with_index_case(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<1x8x4xf32>, %[[ARG1:.*]]: tensor<1x8x4xi32>)
// CHECK: %[[COLLAPSED_I32:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], {{\[}}2]] : tensor<1x8x4xi32> into tensor<8x4xi32>
// CHECK: %[[COLLAPSED_F32:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], {{\[}}2]] : tensor<1x8x4xf32> into tensor<8x4xf32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1x8xf32>
// CHECK: %[[COLLAPSED_OUT0:.*]] = tensor.collapse_shape %[[EMPTY0]] {{\[\[}}0, 1]] : tensor<1x8xf32> into tensor<8xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1x8xi32>
// CHECK: %[[COLLAPSED_OUT1:.*]] = tensor.collapse_shape %[[EMPTY1]] {{\[\[}}0, 1]] : tensor<1x8xi32> into tensor<8xi32>
// CHECK: %[[RES:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%[[COLLAPSED_F32]], %[[COLLAPSED_I32]] : tensor<8x4xf32>, tensor<8x4xi32>) outs(%[[COLLAPSED_OUT0]], %[[COLLAPSED_OUT1]] : tensor<8xf32>, tensor<8xi32>) dimensions = [1] -> tensor<8xf32>, tensor<8xi32>
// CHECK: %[[EXPANDED0:.*]] = tensor.expand_shape %[[RES]]#0 {{\[\[}}0, 1]] output_shape {{\[}}1, 8] : tensor<8xf32> into tensor<1x8xf32>
// CHECK: %[[EXPANDED1:.*]] = tensor.expand_shape %[[RES]]#1 {{\[\[}}0, 1]] output_shape {{\[}}1, 8] : tensor<8xi32> into tensor<1x8xi32>
// CHECK: return %[[EXPANDED0]], %[[EXPANDED1]] : tensor<1x8xf32>, tensor<1x8xi32>
func.func @reduce_with_index_case(%arg0: tensor<1x8x4xf32>, %arg1: tensor<1x8x4xi32>) -> (tensor<1x8xf32>, tensor<1x8xi32>) {
  %0 = tensor.empty() : tensor<1x8xf32>
  %1 = tensor.empty() : tensor<1x8xi32>
  %2:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : tensor<1x8x4xf32>, tensor<1x8x4xi32>) outs(%0, %1 : tensor<1x8xf32>, tensor<1x8xi32>) dimensions = [2]  -> tensor<1x8xf32>, tensor<1x8xi32>
  return %2#0, %2#1 : tensor<1x8xf32>, tensor<1x8xi32>
}

// -----

// CHECK-LABEL: func.func @flatten_interleave(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4x8x17x1xf32>, %[[ARG1:.*]]: tensor<4x8x17x1xf32>)
// CHECK: %[[COLLAPSED_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1, 2, 3]] : tensor<4x8x17x1xf32> into tensor<544xf32>
// CHECK: %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1, 2, 3]] : tensor<4x8x17x1xf32> into tensor<544xf32>
// CHECK: %[[INTERLEAVED:.*]] = hfusion.interleave %[[COLLAPSED_0]], %[[COLLAPSED_1]] : tensor<544xf32>, tensor<544xf32> -> tensor<1088xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INTERLEAVED]] {{\[\[}}0, 1, 2, 3]] output_shape {{\[}}4, 8, 17, 2] : tensor<1088xf32> into tensor<4x8x17x2xf32>
// CHECK: return %[[EXPANDED]] : tensor<4x8x17x2xf32>
func.func @flatten_interleave(%arg0: tensor<4x8x17x1xf32>, %arg1: tensor<4x8x17x1xf32>) -> (tensor<4x8x17x2xf32>) {
    %0 = hfusion.interleave %arg0, %arg1 : tensor<4x8x17x1xf32>, tensor<4x8x17x1xf32> -> tensor<4x8x17x2xf32>
    return %0 : tensor<4x8x17x2xf32>
}

// -----

// REGBASE-LABEL: embedding_and_fill(
// REGBASE: hfusion.embedding_gather
// REGBASE-SAME: -> tensor<1x50x8xf32>
func.func @embedding_and_fill(
    %arg3: memref<?xf32>,
    %reshape: tensor<1x50xi64>,
    %23: i64,
    %6: i64,
    %7: i64
) -> (tensor<1x50x8xf32>, tensor<1x8xf32>) {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c8_i64 = arith.constant 8 : i64
  %c1353406_i64 = arith.constant 1353406 : i64
  %24 = tensor.empty() : tensor<1x50x8xf32>
  %25 = hfusion.embedding_gather
    ins(%arg3 : memref<?xf32>,
        %reshape : tensor<1x50xi64>,
        %c1353406_i64 : i64,
        [%23, %c0_i64, %c0_i64 : i64, i64, i64],
        [%6, %7, %c8_i64 : i64, i64, i64])
    outs(%24 : tensor<1x50x8xf32>) -> tensor<1x50x8xf32>
  %26 = tensor.empty() : tensor<1x8xf32>
  %27 = linalg.fill ins(%cst_0 : f32) outs(%26 : tensor<1x8xf32>) -> tensor<1x8xf32>
  return %25, %27 : tensor<1x50x8xf32>, tensor<1x8xf32>
}


// -----

// REGBASE-LABEL: func.func @tensor_insert_unit(
// REGBASE-SAME:                  %[[ARG0:.*]]: f32,
// REGBASE-SAME:                  %[[ARG1:.*]]: tensor<f32>
// REGBASE-NEXT:    tensor.insert %[[ARG0]] into %[[ARG1]][] : tensor<f32>
func.func @tensor_insert_unit(%arg0: f32, %arg1: tensor<f32>) -> tensor<f32> {
    %inserted = tensor.insert %arg0 into %arg1[] : tensor<f32>
    return %inserted : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @linalg_trans_2d_1(
// CHECK-NOT: collapse_shape
func.func @linalg_trans_2d_1(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) permutation = [1, 0]
  return %1 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_trans_2d_2(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1]] : tensor<16x1xf32> into tensor<16xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1]] : tensor<16x1xf32> into tensor<16xf32>
// CHECK: linalg.transpose ins({{.*}} tensor<16xf32>) outs({{.*}} tensor<16xf32>) permutation = {{\[}}0]
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1]] output_shape {{\[}}16, 1] : tensor<16xf32> into tensor<16x1xf32>
func.func @linalg_trans_2d_2(%arg0: tensor<16x1xf32>) -> tensor<16x1xf32> {
  %0 = tensor.empty() : tensor<16x1xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<16x1xf32>) outs(%0 : tensor<16x1xf32>) permutation = [0, 1]
  return %1 : tensor<16x1xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_trans_3d_1(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<1x16x8xf32> into tensor<1x128xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2]] : tensor<16x8x1xf32> into tensor<128x1xf32>
// CHECK: {{.*}} = linalg.transpose ins({{.*}} : tensor<1x128xf32>) outs({{.*}} : tensor<128x1xf32>) permutation = {{\[}}1, 0]
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2]] output_shape {{\[}}16, 8, 1] : tensor<128x1xf32> into tensor<16x8x1xf32>
func.func @linalg_trans_3d_1(%arg0: tensor<1x16x8xf32>) -> tensor<16x8x1xf32> {
  %0 = tensor.empty() : tensor<16x8x1xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<1x16x8xf32>) outs(%0 : tensor<16x8x1xf32>) permutation = [1, 2, 0]
  return %1 : tensor<16x8x1xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_trans_3d_2(
// CHECK-NOT: collapse_shape
func.func @linalg_trans_3d_2(%arg0: tensor<1x16x8xf32>) -> tensor<16x1x8xf32> {
  %0 = tensor.empty() : tensor<16x1x8xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<1x16x8xf32>) outs(%0 : tensor<16x1x8xf32>) permutation = [1, 0, 2]
  return %1 : tensor<16x1x8xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_trans_3d_3(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2]] : tensor<10x15x30xf32> into tensor<150x30xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] : tensor<30x10x15xf32> into tensor<30x150xf32>
// CHECK: linalg.transpose ins({{.*}} : tensor<150x30xf32>) outs({{.*}} : tensor<30x150xf32>) permutation = {{\[}}1, 0]
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1, 2]] output_shape {{\[}}30, 10, 15] : tensor<30x150xf32> into tensor<30x10x15xf32>
func.func @linalg_trans_3d_3(%arg0: tensor<10x15x30xf32>) -> tensor<30x10x15xf32> {
  %0 = tensor.empty() : tensor<30x10x15xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<10x15x30xf32>) outs(%0 : tensor<30x10x15xf32>) permutation = [2, 0, 1]
  return %1 : tensor<30x10x15xf32>
}

// -----

// CHECK-LABEL: func.func @triton_trans_4d_1(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2, 3]] : tensor<2x2x2x2xf32> into tensor<2x8xf32>
// CHECK: linalg.transpose ins({{.*}} : tensor<8x2xf32>) outs({{.*}} : tensor<2x8xf32>) permutation = {{\[}}1, 0]
// CHECK: memref.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2, 3]] : memref<2x2x2x2xf32, strided<[8, 4, 2, 1]>> into memref<2x8xf32, strided<[8, 1]>>
// CHECK: bufferization.materialize_in_destination {{.*}} in writable {{.*}} : (tensor<2x8xf32>, memref<2x8xf32, strided<{{\[}}8, 1]>>) -> ()
func.func @triton_trans_4d_1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 2, 2, 2], strides: [8, 4, 2, 1] : memref<?xf32> to memref<2x2x2x2xf32, strided<[8, 4, 2, 1]>>
  %alloc = memref.alloc() : memref<2x2x2x2xf32>
  memref.copy %reinterpret_cast, %alloc : memref<2x2x2x2xf32, strided<[8, 4, 2, 1]>> to memref<2x2x2x2xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x2x2x2xf32>
  %1 = tensor.empty() : tensor<2x2x2x2xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x2x2x2xf32>) outs(%1 : tensor<2x2x2x2xf32>) permutation = [3, 0, 1, 2]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 2, 2, 2], strides: [8, 4, 2, 1] : memref<?xf32> to memref<2x2x2x2xf32, strided<[8, 4, 2, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<2x2x2x2xf32>, memref<2x2x2x2xf32, strided<[8, 4, 2, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @triton_trans_4d_2(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1, 2], {{\[}}3]] : tensor<2x3x5x4xf32> into tensor<2x15x4xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3]] : tensor<2x4x3x5xf32> into tensor<2x4x15xf32>
// CHECK: linalg.transpose ins({{.*}} : tensor<2x15x4xf32>) outs({{.*}} : tensor<2x4x15xf32>) permutation = {{\[}}0, 2, 1]
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], {{\[}}1], {{\[}}2, 3]] output_shape {{\[}}2, 4, 3, 5] : tensor<2x4x15xf32> into tensor<2x4x3x5xf32>
func.func @triton_trans_4d_2(%arg0: tensor<2x3x5x4xf32>) -> tensor<2x4x3x5xf32> {
  %0 = tensor.empty() : tensor<2x4x3x5xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<2x3x5x4xf32>) outs(%0 : tensor<2x4x3x5xf32>) permutation = [0, 3, 1, 2]
  return %1 : tensor<2x4x3x5xf32>
}

// -----

// CHECK-LABEL: func.func @triton_trans_4d_3(
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2, 3]] : tensor<2x3x5x4xf32> into tensor<6x20xf32>
// CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0, 1], {{\[}}2, 3]] : tensor<5x4x2x3xf32> into tensor<20x6xf32>
// CHECK: linalg.transpose ins({{.*}} : tensor<6x20xf32>) outs({{.*}} : tensor<20x6xf32>) permutation = {{\[}}1, 0]
// CHECK: tensor.expand_shape {{.*}} {{\[\[}}0, 1], {{\[}}2, 3]] output_shape {{\[}}5, 4, 2, 3] : tensor<20x6xf32> into tensor<5x4x2x3xf32>
func.func @triton_trans_4d_3(%arg0: tensor<2x3x5x4xf32>) -> tensor<5x4x2x3xf32> {
  %0 = tensor.empty() : tensor<5x4x2x3xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<2x3x5x4xf32>) outs(%0 : tensor<5x4x2x3xf32>) permutation = [2, 3, 0, 1]
  return %1 : tensor<5x4x2x3xf32>
}

// -----
// CHECK-LABEL: @test_compare_select_2D(
// CHECK: memref.copy %[[COPY_0:.*]], %[[COPY_0:.*]] : memref<512xf32, strided<[1], offset: ?>> to memref<512xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape [[VAL_1:.*]] {{\[\[}}0, 1]] : tensor<16x32xi1> into tensor<512xi1>
// CHECK: %[[VAL_2:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%[[VAL_3:.*]], %[[VAL_4:.*]] : tensor<512xf32>, tensor<512xf32>) outs(%[[VAL_0]] : tensor<512xi1>) -> tensor<512xi1>
// CHECK: %[[VAL_5:.*]] = tensor.collapse_shape %[[VAL_6:.*]] {{\[\[}}0, 1]] : tensor<16x32xf32> into tensor<512xf32>
// CHECK: %[[VAL_7:.*]] = hfusion.select ins(%[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] : tensor<512xi1>, tensor<512xf32>, tensor<512xf32>) outs(%[[VAL_5]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK: bufferization.materialize_in_destination %[[VAL_11:.*]] in writable %[[VAL_12:.*]] : (tensor<512xf32>, memref<512xf32, strided<[1], offset: ?>>) -> ()
func.func @test_compare_select_2D(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32 = arith.constant 32 : index
  %c16_i32 = arith.constant 16 : i32
  %0 = arith.muli %arg8, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %1, %c32 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [16, 32], strides: [32, 1] : memref<?xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
  %alloc = memref.alloc() : memref<16x32xf32>
  memref.copy %reinterpret_cast, %alloc : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [16, 32], strides: [32, 1] : memref<?xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<16x32xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32>
  %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x32xf32>
  %5 = tensor.empty() : tensor<16x32xi1>
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%3, %4 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%5 : tensor<16x32xi1>) -> tensor<16x32xi1>
  %7 = tensor.empty() : tensor<16x32xf32>
  %8 = hfusion.select ins(%6, %3, %4 : tensor<16x32xi1>, tensor<16x32xf32>, tensor<16x32xf32>) outs(%7 : tensor<16x32xf32>) -> tensor<16x32xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [16, 32], strides: [32, 1] : memref<?xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_2 : (tensor<16x32xf32>, memref<16x32xf32, strided<[32, 1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_compare_select_3D(
// CHECK: memref.copy %[[COPY_0:.*]], %[[COPY_0:.*]] : memref<15872xf32, strided<[1], offset: ?>> to memref<15872xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape [[VAL_1:.*]] {{\[\[}}0, 1, 2]] : tensor<16x31x32xi1> into tensor<15872xi1>
// CHECK: %[[VAL_2:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%[[VAL_3:.*]], %[[VAL_4:.*]] : tensor<15872xf32>, tensor<15872xf32>) outs(%[[VAL_0]] : tensor<15872xi1>) -> tensor<15872xi1>
// CHECK: %[[VAL_5:.*]] = tensor.collapse_shape [[VAL_6:.*]] {{\[\[}}0, 1, 2]] : tensor<16x31x32xf32> into tensor<15872xf32>
// CHECK: %[[VAL_7:.*]] = hfusion.select ins(%[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] : tensor<15872xi1>, tensor<15872xf32>, tensor<15872xf32>) outs(%[[VAL_5]] : tensor<15872xf32>) -> tensor<15872xf32>
// CHECK: bufferization.materialize_in_destination %[[VAL_11:.*]] in writable %[[VAL_12:.*]] : (tensor<15872xf32>, memref<15872xf32, strided<[1], offset: ?>>) -> ()
func.func @test_compare_select_3D(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c992 = arith.constant 992 : index
  %c16_i32 = arith.constant 16 : i32
  %0 = arith.muli %arg8, %c16_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %1, %c992 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [16, 31, 32], strides: [992, 32, 1] : memref<?xf32> to memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>>
  %alloc = memref.alloc() : memref<16x31x32xf32>
  memref.copy %reinterpret_cast, %alloc : memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>> to memref<16x31x32xf32>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<16x31x32xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [16, 31, 32], strides: [992, 32, 1] : memref<?xf32> to memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<16x31x32xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>> to memref<16x31x32xf32>
  %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x31x32xf32>
  %5 = tensor.empty() : tensor<16x31x32xi1>
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%3, %4 : tensor<16x31x32xf32>, tensor<16x31x32xf32>) outs(%5 : tensor<16x31x32xi1>) -> tensor<16x31x32xi1>
  %7 = tensor.empty() : tensor<16x31x32xf32>
  %8 = hfusion.select ins(%6, %3, %4 : tensor<16x31x32xi1>, tensor<16x31x32xf32>, tensor<16x31x32xf32>) outs(%7 : tensor<16x31x32xf32>) -> tensor<16x31x32xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [16, 31, 32], strides: [992, 32, 1] : memref<?xf32> to memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>>
  bufferization.materialize_in_destination %8 in writable %reinterpret_cast_2 : (tensor<16x31x32xf32>, memref<16x31x32xf32, strided<[992, 32, 1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_cumsum_5D_dim0(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0], {{\[}}1, 2, 3, 4]] : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> into memref<8x1024xi64, strided<[1024, 1]>>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_2:.*]] : memref<8x1024xi64, strided<[1024, 1]>> to memref<8x1024xi64>
// CHECK: %[[VAL_3:.*]] = hfusion.cumsum %[[VAL_4:.*]] : tensor<8x1024xi64> cum_dims = [0] reverse = false -> tensor<8x1024xi64>
// CHECK: bufferization.materialize_in_destination %[[VAL_3]] in writable %[[VAL_5:.*]] : (tensor<8x1024xi64>, memref<8x1024xi64, strided<[1024, 1]>>) -> ()
func.func @test_cumsum_5D_dim0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  %alloc = memref.alloc() : memref<8x4x4x4x16xi64>
  memref.copy %reinterpret_cast, %alloc : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> to memref<8x4x4x4x16xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x4x4x4x16xi64>
  %1 = hfusion.cumsum %0 : tensor<8x4x4x4x16xi64> cum_dims = [0] reverse = false -> tensor<8x4x4x4x16xi64>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast_0 : (tensor<8x4x4x4x16xi64>, memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_cumsum_5D_dim2(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1], {{\[}}2], {{\[}}3, 4]] : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> into memref<32x4x64xi64, strided<[256, 64, 1]>>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_2:.*]] : memref<32x4x64xi64, strided<[256, 64, 1]>> to memref<32x4x64xi64>
// CHECK: %[[VAL_3:.*]] = hfusion.cumsum %[[VAL_4:.*]] : tensor<32x4x64xi64> cum_dims = [1] reverse = false -> tensor<32x4x64xi64>
// CHECK: bufferization.materialize_in_destination %[[VAL_3]] in writable %[[VAL_4:.*]] : (tensor<32x4x64xi64>, memref<32x4x64xi64, strided<[256, 64, 1]>>) -> ()
func.func @test_cumsum_5D_dim2(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  %alloc = memref.alloc() : memref<8x4x4x4x16xi64>
  memref.copy %reinterpret_cast, %alloc : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> to memref<8x4x4x4x16xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x4x4x4x16xi64>
  %1 = hfusion.cumsum %0 : tensor<8x4x4x4x16xi64> cum_dims = [2] reverse = false -> tensor<8x4x4x4x16xi64>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast_0 : (tensor<8x4x4x4x16xi64>, memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_cumsum_5D_dim4(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1, 2, 3], {{\[}}4]] : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> into memref<512x16xi64, strided<[16, 1]>>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_2:.*]] : memref<512x16xi64, strided<[16, 1]>> to memref<512x16xi64>
// CHECK: %[[VAL_3:.*]] = hfusion.cumsum %[[VAL_4:.*]] : tensor<512x16xi64> cum_dims = [1] reverse = false -> tensor<512x16xi64>
// CHECK: bufferization.materialize_in_destination %[[VAL_3]] in writable %[[VAL_5:.*]] : (tensor<512x16xi64>, memref<512x16xi64, strided<[16, 1]>>) -> ()
// -----// IR Dump Before FlattenOps (hfusion-flatten-ops) //----- //
func.func @test_cumsum_5D_dim4(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  %alloc = memref.alloc() : memref<8x4x4x4x16xi64>
  memref.copy %reinterpret_cast, %alloc : memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>> to memref<8x4x4x4x16xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x4x4x4x16xi64>
  %1 = hfusion.cumsum %0 : tensor<8x4x4x4x16xi64> cum_dims = [4] reverse = false -> tensor<8x4x4x4x16xi64>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 4, 4, 4, 16], strides: [1024, 256, 64, 16, 1] : memref<?xi64> to memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>
  bufferization.materialize_in_destination %1 in writable %reinterpret_cast_0 : (tensor<8x4x4x4x16xi64>, memref<8x4x4x4x16xi64, strided<[1024, 256, 64, 16, 1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: @test_cumsum_5D_dim0_with_unit(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1, 2, 3, 4]] : memref<8x1x1x1x1xf16, strided<[1, 1, 1, 1, 1]>> into memref<8xf16, strided<[1]>>
// CHECK: memref.copy %[[VAL_0]], %[[VAL_2:.*]] : memref<8xf16, strided<[1]>> to memref<8xf16>
// CHECK: %[[VAL_3:.*]] = hfusion.cumsum %[[VAL_4:.*]] : tensor<8xf32> cum_dims = [0] reverse = false -> tensor<8xf32>
// CHECK: bufferization.materialize_in_destination %[[VAL_5:.*]] in writable %[[VAL_6:.*]] : (tensor<8xf16>, memref<8xf16, strided<[1]>>) -> ()
func.func @test_cumsum_5D_dim0_with_unit(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1] : memref<?xf16> to memref<8x1x1x1x1xf16, strided<[1, 1, 1, 1, 1]>>
  %alloc = memref.alloc() : memref<8x1x1x1x1xf16>
  memref.copy %reinterpret_cast, %alloc : memref<8x1x1x1x1xf16, strided<[1, 1, 1, 1, 1]>> to memref<8x1x1x1x1xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x1x1x1x1xf16>
  %1 = tensor.empty() : tensor<8x1x1x1x1xf32>
  %2 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<8x1x1x1x1xf16>) outs(%1 : tensor<8x1x1x1x1xf32>) -> tensor<8x1x1x1x1xf32>
  %3 = hfusion.cumsum %2 : tensor<8x1x1x1x1xf32> cum_dims = [0] reverse = false -> tensor<8x1x1x1x1xf32>
  %4 = tensor.empty() : tensor<8x1x1x1x1xf16>
  %5 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<8x1x1x1x1xf32>) outs(%4 : tensor<8x1x1x1x1xf16>) -> tensor<8x1x1x1x1xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1] : memref<?xf16> to memref<8x1x1x1x1xf16, strided<[1, 1, 1, 1, 1]>>
  bufferization.materialize_in_destination %5 in writable %reinterpret_cast_0 : (tensor<8x1x1x1x1xf16>, memref<8x1x1x1x1xf16, strided<[1, 1, 1, 1, 1]>>) -> ()
  return
}

//-----
// CHECK-LABEL: @test_subview_static_shape(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1]] : memref<8x16xf32, strided<[32, 2]>> into memref<128xf32, strided<[2]>>
// CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]][0] [20] [4] : memref<128xf32, strided<[2]>> to memref<20xf32, strided<[8]>>
func.func @test_subview_static_shape( %arg0: memref<8x16xf32, strided<[32, 2]>>) -> tensor<5x4xf32> {
    %subview = memref.subview %arg0[0, 0] [5, 4] [1, 4] : memref<8x16xf32, strided<[32, 2]>> to memref<5x4xf32, strided<[32, 8]>>
    %tensor = bufferization.to_tensor %subview : memref<5x4xf32, strided<[32, 8]>>
    return %tensor : tensor<5x4xf32>
}

//-----
// CHECK-LABEL: @test_subview_static_shape_unit(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1, 2, 3]] : memref<4x12x2x10xf32, strided<[240, 20, 10, 1]>> into memref<960xf32, strided<[1]>>
// CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]][0] [4] [20] : memref<960xf32, strided<[1]>> to memref<4xf32, strided<[20]>>
func.func @test_subview_static_shape_unit( %arg0: memref<4x12x2x10xf32, strided<[240, 20 ,10 ,1]>>) -> tensor<1x4x1x1xf32> {
    %subview = memref.subview %arg0[0, 0, 0, 0] [1, 4, 1, 1] [1, 1, 1, 1] : memref<4x12x2x10xf32, strided<[240, 20 ,10 ,1]>> to memref<1x4x1x1xf32, strided<[240, 20 ,10 ,1]>>
    %tensor = bufferization.to_tensor %subview : memref<1x4x1x1xf32, strided<[240, 20 ,10 ,1]>>
    return %tensor : tensor<1x4x1x1xf32>
}


//-----
// CHECK-LABEL: @test_subview_static_shape_lower1(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1, 2]] : memref<4x12x10xf32, strided<[120, 10, 1]>> into memref<480xf32, strided<[1]>>
// CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]][0] [20] [2] : memref<480xf32, strided<[1]>> to memref<20xf32, strided<[2]>>
func.func @test_subview_static_shape_lower1( %arg0: memref<4x12x10xf32, strided<[120, 10, 1]>>) -> tensor<1x4x5xf32> {
%1 = memref.subview %arg0[0, 0, 0] [1, 4, 5] [1, 1, 2] : memref<4x12x10xf32, strided<[120, 10, 1]>> to memref<1x4x5xf32, strided<[120, 10, 2]>>
  %2 = bufferization.to_tensor %1 : memref<1x4x5xf32, strided<[120, 10, 2]>>
  return %2 : tensor<1x4x5xf32>
}

//-----
// CHECK-LABEL: @test_subview_static_shape_lower2(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0], {{\[}}1, 2]] : memref<4x12x10xf32, strided<[120, 10, 1]>> into memref<4x120xf32, strided<[120, 1]>>
// CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]][0, 0] [1, 20] [1, 2] : memref<4x120xf32, strided<[120, 1]>> to memref<20xf32, strided<[2]>>
func.func @test_subview_static_shape_lower2( %arg0: memref<4x12x10xf32, strided<[120, 10, 1]>>) -> tensor<4x5xf32> {
%1 = memref.subview %arg0[0, 0, 0] [1, 4, 5] [1, 1, 2] : memref<4x12x10xf32, strided<[120, 10, 1]>> to memref<4x5xf32, strided<[10, 2]>>
  %2 = bufferization.to_tensor %1 : memref<4x5xf32, strided<[10, 2]>>
  return %2 : tensor<4x5xf32>
}

//-----
// CHECK-LABEL: @test_subview_dim4_dyn(
// CHECK: %[[VAL_0:.*]] = memref.collapse_shape %[[VAL_1:.*]] {{\[}}{{\[}}0, 1, 2, 3], {{\[}}4]] : memref<16x4x4x4x8xf32, strided<[512, 128, 32, 8, 1], offset: ?>> into memref<1024x8xf32, strided<[8, 1], offset: ?>>
// CHECK: %[[VAL_2:.*]] = memref.subview %[[VAL_0]][0, 0] [1024, {{.*}}] [1, 1] : memref<1024x8xf32, strided<[8, 1], offset: ?>> to memref<1024x?xf32, strided<[8, 1], offset: ?>>
func.func @test_subview_dim4_dyn(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.muli %arg12, %c32_i32 : i32
  scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg15, %c16_i32 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %3, %c512 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [16, 4, 4, 4, 8], strides: [512, 128, 32, 8, 1] : memref<?xf32> to memref<16x4x4x4x8xf32, strided<[512, 128, 32, 8, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x4x4x4x8xf32>
    %5 = arith.index_cast %arg8 : i32 to index
    %6 = arith.maxsi %5, %c0 : index
    %7 = arith.minsi %6, %c8 : index
    %8 = arith.cmpi slt, %7, %c8 : index
    scf.if %8 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<16x4x4x4x8xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0, 0, 0, 0, 0] [16, 4, 4, 4, %7] [1, 1, 1, 1, 1] : memref<16x4x4x4x8xf32, strided<[512, 128, 32, 8, 1], offset: ?>> to memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0, 0, 0, 0] [16, 4, 4, 4, %7] [1, 1, 1, 1, 1] : memref<16x4x4x4x8xf32> to memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1]>>
    memref.copy %subview, %subview_0 : memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1], offset: ?>> to memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1]>>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<16x4x4x4x8xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [16, 4, 4, 4, 8], strides: [512, 128, 32, 8, 1] : memref<?xf32> to memref<16x4x4x4x8xf32, strided<[512, 128, 32, 8, 1], offset: ?>>
    %extracted_slice = tensor.extract_slice %9[0, 0, 0, 0, 0] [16, 4, 4, 4, %7] [1, 1, 1, 1, 1] : tensor<16x4x4x4x8xf32> to tensor<16x4x4x4x?xf32>
    %subview_2 = memref.subview %reinterpret_cast_1[0, 0, 0, 0, 0] [16, 4, 4, 4, %7] [1, 1, 1, 1, 1] : memref<16x4x4x4x8xf32, strided<[512, 128, 32, 8, 1], offset: ?>> to memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<16x4x4x4x?xf32>, memref<16x4x4x4x?xf32, strided<[512, 128, 32, 8, 1], offset: ?>>) -> ()
  }
  return
}

//-----

// CSE-LABEL:   func.func @tensor_insert_01(
// CSE-SAME:                         %[[ARG0:.*]]: tensor<3x4xf32>,
// CSE: %[[OFFSET:.*]] = arith.constant 10 : index
// CSE: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1{{\]\]}}
// CSE: %[[INSERTED:.*]] = tensor.insert {{.*}} into %[[FLATTENED]][%[[OFFSET]]]
// CSE: %[[EXPANDED:.*]] = tensor.expand_shape %[[INSERTED]] {{\[\[}}0, 1{{\]\]}} output_shape [3, 4]
// CSE: return %[[EXPANDED]]
func.func @tensor_insert_01(%arg0: tensor<3x4xf32>, %arg1: f32) -> tensor<3x4xf32> {
    %c2 = arith.constant 2 : index
    %inserted = tensor.insert %arg1 into %arg0[%c2, %c2] : tensor<3x4xf32>
    return %inserted : tensor<3x4xf32>
}

// -----

// CSE-LABEL:   func.func @tensor_insert_02(
// CSE-SAME:                         %[[ARG0:.*]]: tensor<4x4x4xf16>
// CSE: %[[OFFSET:.*]] = arith.constant 6 : index
// CSE: %[[FLATTENED:.*]] = tensor.collapse_shape {{.*}} {{\[\[}}0, 1, 2]] : tensor<4x4x4xf16> into tensor<64xf16>
// CSE: %[[INSERTED:.*]] = tensor.insert {{.*}} into %[[FLATTENED]][%[[OFFSET]]]
// CSE: %[[EXPANDED:.*]] = tensor.expand_shape %[[INSERTED]] {{\[\[}}0, 1, 2{{\]\]}} output_shape [4, 4, 4]
// CSE: return %[[EXPANDED]]
func.func @tensor_insert_02(%arg0: tensor<4x4x4xf16>) -> tensor<4x4x4xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f2 = arith.constant 1.0 : f16
    %2 = tensor.insert %f2 into %arg0[%c0, %c1, %c2]: tensor<4x4x4xf16>
    return %2 : tensor<4x4x4xf16>
}

// -----

// CSE-LABEL: func.func @tensor_extract(
// CSE-SAME:                         %[[ARG0:.*]]: tensor<4x4x4xf16>
// CSE: %[[OFFSET:.*]] = arith.constant 33 : index
// CSE: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]] : tensor<4x4x4xf16> into tensor<64xf16>
// CSE: %[[EXTRACTED:.*]] = tensor.extract %[[FLATTENED]][%[[OFFSET]]] : tensor<64xf16>
func.func @tensor_extract(%arg0: tensor<4x4x4xf16>) -> f16 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %2 = tensor.extract %arg0[%c2, %c0, %c1] : tensor<4x4x4xf16>
    return %2 : f16
}

// -----

// CHECK-LABEL: func.func @linalg_elemwise_unary(
// CHECK:                         %[[ARG0:.*]]: tensor<4x4x4xf16>,
// CHECK:                         %[[ARG1:.*]]: tensor<4x4x4xf16>
// CHECK: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]] : tensor<4x4x4xf16> into tensor<64xf16>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[RES]] {{\[}}[0, 1, 2]] output_shape [4, 4, 4] : tensor<64xf16> into tensor<4x4x4xf16>
func.func @linalg_elemwise_unary(%arg0: tensor<4x4x4xf16>, %arg1: tensor<4x4x4xf16>) -> tensor<4x4x4xf16> {
    %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg0 : tensor<4x4x4xf16>) outs(%arg1 : tensor<4x4x4xf16>) -> tensor<4x4x4xf16>
    return %0 : tensor<4x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @linalg_elemwise_binary(
// CHECK:                         %[[ARG0:[^:]+]]: tensor<32x7xf32>,
// CHECK:                         %[[ARG1:.*]]: tensor<32x7xf32>,
// CHECK:                         %[[ARG2:.*]]: tensor<32x7xf32>
// CHECK: %[[FLATTENED_0:.*]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<32x7xf32> into tensor<224xf32>
// CHECK: %[[FLATTENED_1:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<32x7xf32> into tensor<224xf32>
// CHECK: %[[FLATTENED_2:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<32x7xf32> into tensor<224xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[RES]] {{\[}}[0, 1]] output_shape [32, 7] : tensor<224xf32> into tensor<32x7xf32>
func.func @linalg_elemwise_binary(%arg0: tensor<32x7xf32>, %arg1: tensor<32x7xf32>, %arg2: tensor<32x7xf32>) -> tensor<32x7xf32> {
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<32x7xf32>, tensor<32x7xf32>) outs(%arg2: tensor<32x7xf32>) -> tensor<32x7xf32>
    return %0 : tensor<32x7xf32>
}

// -----
// CHECK-LABEL: triton_scope
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL1:.*]], %[[VAL2:.*]] : tensor<1x64xf32>, tensor<1x64xf32>)
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL:.*]] : tensor<1x64xf32>)
func.func @triton_scope(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg4: memref<?xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 5.000000e-01 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
  scope.scope : () -> () {
    %2 = linalg.matmul {input_precison = "ieee"} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%2 : tensor<128x128xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    scope.return
  } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, noinline}
  %3 = tensor.empty() : tensor<1x64xf32>
  %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
  scope.scope : () -> () {
    %memspacecast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
    %5 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
    %8 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg3 = %4) -> (tensor<1x64xf32>)  : i32 {
      %6 = arith.index_cast %arg2 : i32 to index
      %extracted_slice = tensor.extract_slice %5[%6, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
      %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %arg3 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
      scf.yield %7 : tensor<1x64xf32>
    }
    %10 = scope.scope : () -> (tensor<1x64xf32>) {
      %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%8 : tensor<1x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
      scope.return %9 : tensor<1x64xf32>
    } {noinline, outline = true, vector_mode = "simd"}
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1]>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1]>>) -> ()
    scope.return
  } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, noinline}
  return
}

// -----

// CHECK-LABEL: test_subview_dyn_offset(
// CHECK: %[[ARG0:.*]] = memref.collapse_shape %arg0 {{\[}}{{\[}}0], {{\[}}1, 2, 3]] : memref<8x8x16x16xf16> into memref<8x2048xf16>
// CHECK: %[[ARG1:.*]] = memref.subview %[[ARG0]][0, %[[ARG2:.*]]] [8, 1024] [1, 1] : memref<8x2048xf16> to memref<8x1024xf16, strided<[2048, 1], offset: ?>>
func.func @test_subview_dyn_offset( %arg0: memref<8x8x16x16xf16>) -> tensor<8x4x16x16xf16> {
  %0 = arith.constant 4 : index
  %subview = memref.subview %arg0[0, %0, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xf16> to memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>>
  %2 = bufferization.to_tensor %subview : memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>>
  return %2 : tensor<8x4x16x16xf16>
}

// -----
// CHECK-LABEL: func.func @simple_3d_insert_fixed_slice_kernel_01(
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0] [2, 60] [2, 2] : tensor<2x60xf32> into tensor<4x120xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INSERTED:.*]] {{\[}}[0], [1, 2]] output_shape [4, 12, 10] : tensor<4x120xf32> into tensor<4x12x10xf32>
func.func @simple_3d_insert_fixed_slice_kernel_01() -> tensor<4x12x10xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x12x10xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x12x10xf32>) -> tensor<4x12x10xf32>
    %2 = tensor.empty() : tensor<2x12x5xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x12x5xf32>) -> tensor<2x12x5xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0] [2, 12, 5] [2, 1, 2] : tensor<2x12x5xf32> into tensor<4x12x10xf32>
    return %inserted_slice :  tensor<4x12x10xf32>
}

// -----
// CHECK-LABEL: func.func @simple_3d_insert_fixed_slice_kernel_02(
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0, 0] [2, 4, 5] [2, 3, 2] : tensor<2x4x5xf32> into tensor<4x12x10xf32>
func.func @simple_3d_insert_fixed_slice_kernel_02() -> tensor<4x12x10xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x12x10xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x12x10xf32>) -> tensor<4x12x10xf32>
    %2 = tensor.empty() : tensor<2x4x5xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0] [2, 4, 5] [2, 3, 2] : tensor<2x4x5xf32> into tensor<4x12x10xf32>
    return %inserted_slice :  tensor<4x12x10xf32>
}


// -----
// CHECK-LABEL: func.func @simple_3d_insert_fixed_slice_kernel_03(
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice {{.*}} into {{.*}}[0] [20] [2] : tensor<20xf32> into tensor<480xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INSERTED:.*]] {{\[}}[0, 1, 2]] output_shape [4, 12, 10] : tensor<480xf32> into tensor<4x12x10xf32>
func.func @simple_3d_insert_fixed_slice_kernel_03() -> tensor<4x12x10xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x12x10xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x12x10xf32>) -> tensor<4x12x10xf32>
    %2 = tensor.empty() : tensor<1x4x5xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x4x5xf32>) -> tensor<1x4x5xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0] [1, 4, 5] [1, 1, 2] : tensor<1x4x5xf32> into tensor<4x12x10xf32>
    return %inserted_slice :  tensor<4x12x10xf32>
}

// -----
// CHECK-LABEL: func.func @simple_3d_extract_fixed_slice_kernel_01(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0] [2, 60] [2, 2] : tensor<4x120xf32> to tensor<2x60xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED:.*]] {{\[}}[0], [1, 2]] output_shape [2, 12, 5] : tensor<2x60xf32> into tensor<2x12x5xf32>
func.func @simple_3d_extract_fixed_slice_kernel_01() -> tensor<2x12x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<4x12x10xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x12x10xf32>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<4x12x10xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0] [2, 12, 5] [2, 1, 2] : tensor<4x12x10xf32> to tensor<2x12x5xf32>
    return %extracted_slice :  tensor<2x12x5xf32>
}

// -----
// CHECK-LABEL: func.func @simple_3d_extract_fixed_slice_kernel_02(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0, 0, 0] [2, 4, 5] [2, 3, 2] : tensor<4x12x10xf32> to tensor<2x4x5xf32>
func.func @simple_3d_extract_fixed_slice_kernel_02() -> tensor<2x4x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<4x12x10xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x12x10xf32>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<4x12x10xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0] [2, 4, 5] [2, 3, 2] : tensor<4x12x10xf32> to tensor<2x4x5xf32>
    return %extracted_slice :  tensor<2x4x5xf32>
}


// -----
// CHECK-LABEL: func.func @simple_3d_extract_fixed_slice_kernel_03(
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice {{.*}}[0] [20] [2] : tensor<480xf32> to tensor<20xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %extracted_slice {{\[}}[0, 1, 2]] output_shape [1, 4, 5] : tensor<20xf32> into tensor<1x4x5xf32>
func.func @simple_3d_extract_fixed_slice_kernel_03() -> tensor<1x4x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<4x12x10xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x12x10xf32>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<4x12x10xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0] [1, 4, 5] [1, 1, 2] : tensor<4x12x10xf32> to tensor<1x4x5xf32>
    return %extracted_slice :  tensor<1x4x5xf32>
}

// -----
// CHECK-LABEL: func.func @dynamic_extract_fixed_slice_kernel_01(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x12x10xbf16>,
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[ARG0:.*]][0, 0, 0] [2, 5, {{.*}}] [2, 1, 2] : tensor<4x12x10xbf16> to tensor<2x5x?xbf16>
func.func @dynamic_extract_fixed_slice_kernel_01(%arg0: tensor<4x12x10xbf16>, %arg1: i32) -> tensor<2x5x?xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [2, 5, %0] [2, 1, 2] : tensor<4x12x10xbf16> to tensor<2x5x?xbf16>
  return %extracted_slice : tensor<2x5x?xbf16>
}

// -----
// CHECK-LABEL: func.func @dynamic_extract_fixed_slice_kernel_02(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x12x10xbf16>,
// CHECK: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0:.*]] {{\[}}[0], [1, 2]] : tensor<4x12x10xbf16> into tensor<4x120xbf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[FLATTENED:.*]][0, 0] [2, {{.*}}] [2, 2] : tensor<4x120xbf16> to tensor<2x?xbf16>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED:.*]] {{\[}}[0], [1, 2]] output_shape [2, %0, 5] : tensor<2x?xbf16> into tensor<2x?x5xbf16>
func.func @dynamic_extract_fixed_slice_kernel_02(%arg0: tensor<4x12x10xbf16>, %arg1: i32) -> tensor<2x?x5xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [2, %0, 5] [2, 1, 2] : tensor<4x12x10xbf16> to tensor<2x?x5xbf16>
  return %extracted_slice : tensor<2x?x5xbf16>
}

// -----
// CHECK-LABEL: func.func @dynamic_extract_fixed_slice_kernel_03(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x12x10xbf16>,
// CHECK: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0:.*]] {{\[}}[0, 1, 2]] : tensor<4x12x10xbf16> into tensor<480xbf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[FLATTENED:.*]][0] [{{.*}}] [2] : tensor<480xbf16> to tensor<?xbf16>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED:.*]] {{\[}}[0, 1, 2]] output_shape [%0, 12, 5] : tensor<?xbf16> into tensor<?x12x5xbf16>
func.func @dynamic_extract_fixed_slice_kernel_03(%arg0: tensor<4x12x10xbf16>, %arg1: i32) -> tensor<?x12x5xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [%0, 12 , 5] [1, 1, 2] : tensor<4x12x10xbf16> to tensor<?x12x5xbf16>
  return %extracted_slice : tensor<?x12x5xbf16>
}

// -----
// CHECK-LABEL: func.func @dynamic_extract_fixed_slice_kernel_04(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x?x10xbf16>,
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[ARG0:.*]][0, 0, 0] [2, {{.*}}, 5] [2, 3, 2] : tensor<4x?x10xbf16> to tensor<2x?x5xbf16>
func.func @dynamic_extract_fixed_slice_kernel_04(%arg0: tensor<4x?x10xbf16>, %arg1: i32) -> tensor<2x?x5xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [2, %0, 5] [2, 3, 2] : tensor<4x?x10xbf16> to tensor<2x?x5xbf16>
  return %extracted_slice : tensor<2x?x5xbf16>
}

// -----
// CHECK-LABEL: func.func @unit_extract_fixed_slice_kernel_01(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x12x10xbf16>,
// CHECK: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0:.*]] {{\[}}[0, 1, 2]] : tensor<4x12x10xbf16> into tensor<480xbf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[FLATTENED:.*]][0] [20] [2] : tensor<480xbf16> to tensor<20xbf16>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED:.*]] {{\[}}[0, 1, 2]] output_shape [1, 4, 5] : tensor<20xbf16> into tensor<1x4x5xbf16>
func.func @unit_extract_fixed_slice_kernel_01(%arg0: tensor<4x12x10xbf16>, %arg1: i32) -> tensor<1x4x5xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 4, 5] [1, 1, 2] : tensor<4x12x10xbf16> to tensor<1x4x5xbf16>
  return %extracted_slice : tensor<1x4x5xbf16>
}

// -----
// CHECK-LABEL: func.func @unit_extract_fixed_slice_kernel_02(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<4x12x10xbf16>,
// CHECK: %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG0:.*]] {{\[}}[0], [1, 2]] : tensor<4x12x10xbf16> into tensor<4x120xbf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[FLATTENED:.*]][0, 0] [4, 5] [1, 2] : tensor<4x120xbf16> to tensor<4x5xbf16>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED:.*]] {{\[}}[0], [1, 2]] output_shape [4, 1, 5] : tensor<4x5xbf16> into tensor<4x1x5xbf16>
func.func @unit_extract_fixed_slice_kernel_02(%arg0: tensor<4x12x10xbf16>, %arg1: i32) -> tensor<4x1x5xbf16> {
  %0 = arith.index_cast %arg1 : i32 to index
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [4, 1, 5] [1, 1, 2] : tensor<4x12x10xbf16> to tensor<4x1x5xbf16>
  return %extracted_slice : tensor<4x1x5xbf16>
}

// -----

// CSE-LABEL: func.func @buf_to_memref_hivm_copy
// CSE: bufferization.to_memref %[[VAL1:.*]] : memref<8x1024xf16, #hivm.address_space<ub>>
// CSE: hivm.hir.copy ins(%[[VAL2:.*]] : memref<8x1024xf16, #hivm.address_space<ub>>) outs(%[[SUBVIEW:.*]] : memref<8x1024xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @buf_to_memref_hivm_copy(%arg0 : tensor<8x4x16x16xf16>, %arg1 : index) {
    %alloc = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %subview = memref.subview %alloc[0, %arg1, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %empty = tensor.empty() : tensor<8x4x16x16xf16>
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg0 : tensor<8x4x16x16xf16>) outs(%empty : tensor<8x4x16x16xf16>) -> tensor<8x4x16x16xf16>
    %src = bufferization.to_memref %1 : memref<8x4x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%src : memref<8x4x16x16xf16, #hivm.address_space<ub>>) outs(%subview : memref<8x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// CSE-LABEL: func.func @memref_memory_space_cast
// CSE: memref.memory_space_cast %[[ALLOC:.*]] : memref<8192xf32, #hivm.address_space<ub>> to memref<8192xf32>
func.func @memref_memory_space_cast() -> tensor<4x16x8x16xf32> {
  %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
  %memspacecast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
  %expand_shape = memref.expand_shape %memspacecast [[0, 1], [2, 3]] output_shape [4, 16, 8, 16] : memref<64x128xf32> into memref<4x16x8x16xf32>
  %0 = bufferization.to_tensor %expand_shape restrict writable : memref<4x16x8x16xf32>
  %empty = tensor.empty() : tensor<4x16x8x16xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%0 : tensor<4x16x8x16xf32>) outs(%empty : tensor<4x16x8x16xf32>) -> tensor<4x16x8x16xf32>
  return %1 : tensor<4x16x8x16xf32>
}

// CHECK-LABEL: test_extract_slice_dyn_offset(
// CHECK: %[[ARG0:.*]] = tensor.collapse_shape %arg0 {{\[}}{{\[}}0], {{\[}}1, 2, 3]] : tensor<8x8x16x16xf16> into tensor<8x2048xf16>
// CHECK: %[[ARG1:.*]] = tensor.extract_slice %[[ARG0]][0, %[[ARG2:.*]]] [8, 1024] [1, 1] : tensor<8x2048xf16> to tensor<8x1024xf16>
func.func @test_extract_slice_dyn_offset( %arg0: tensor<8x8x16x16xf16>) -> tensor<8x4x16x16xf16> {
  %0 = arith.constant 4 : index
  %extract_slice = tensor.extract_slice %arg0[0, %0, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] : tensor<8x8x16x16xf16> to tensor<8x4x16x16xf16>
  return %extract_slice : tensor<8x4x16x16xf16>
}

// -----

// CHECK-LABEL: test_insert_slice_dyn_offset(
// CHECK: %[[ARG0:.*]] = tensor.insert_slice %[[ARG1:.*]] into %[[ARG2:.*]][0, %[[ARG3:.*]]] [8, 1024] [1, 1] : tensor<8x1024xf16> into tensor<8x2048xf16>
func.func @test_insert_slice_dyn_offset( %arg0: tensor<8x4x16x16xf16>, %arg1: tensor<8x8x16x16xf16>) -> tensor<8x8x16x16xf16> {
  %0 = arith.constant 4 : index
  %inserted_slice = tensor.insert_slice %arg0 into %arg1[0, %0, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] : tensor<8x4x16x16xf16> into tensor<8x8x16x16xf16>
  return %inserted_slice : tensor<8x8x16x16xf16>
}

// -----

// CSE-LABEL: tensor_extract_slice_different_region
// CSE: %[[BRC:.*]] = linalg.broadcast ins(%[[ARG0:.*]] : tensor<2048xf32>) outs(%[[VAL0:.*]] : tensor<16x2048xf32>) dimensions = [0]
// CSE: tensor.extract_slice %[[BRC]]
// CSE-SAME: tensor<16x2048xf32> to tensor<?x2048xf32>
func.func @tensor_extract_slice_different_region(%arg0 : tensor<2048xf32>, %arg1 : i1, %arg2 : index, %arg3 : index, %arg4 : tensor<16x2048xf32>) -> tensor<16x2048xf32> {
  %0 = tensor.empty() : tensor<16x2048xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<16x2048xf32>) dimensions = [0]
  %ret = scf.if %arg1 -> (tensor<16x2048xf32>) {
    scf.yield %broadcasted : tensor<16x2048xf32>
  } else {
    %extracted_slice = tensor.extract_slice %broadcasted[%arg2, 0] [%arg3, 2048] [1, 1] : tensor<16x2048xf32> to tensor<?x2048xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %arg4[%arg2, 0] [%arg3, 2048] [1, 1] : tensor<?x2048xf32> into tensor<16x2048xf32>
    scf.yield %inserted_slice : tensor<16x2048xf32>
  }
  return %ret : tensor<16x2048xf32>
}

// -----

// CHECK-LABEL: indirect_store
// CHECK: %[[COLLAPSED_INS:.*]] = tensor.collapse_shape %[[INS:.*]] {{\[\[0, 1\]\]}} : tensor<24x32xi32> into tensor<768xi32>
// CHECK: %[[COLLAPSED_OFFSET:.*]] = tensor.collapse_shape %[[OFFSET:.*]] {{\[\[0, 1\]\]}} : tensor<24x32xi64> into tensor<768xi64>
// CHECK: %[[COLLAPSED_MASK:.*]] = tensor.collapse_shape %[[MASK:.*]] {{\[\[0, 1\]\]}} : tensor<24x32xi8> into tensor<768xi8>
// CHECK: hfusion.indirect_store ins(%[[COLLAPSED_INS]] : tensor<768xi32>, %[[COLLAPSED_OFFSET]] : tensor<768xi64>, %[[COLLAPSED_MASK]] : tensor<768xi8>) outs(%[[ARG:.*]] : memref<?xi32>)
func.func @indirect_store(%arg0: memref<?xi32>){
  %ins = tensor.empty() : tensor<24x32xi32>
  %offset = tensor.empty() : tensor<24x32xi64>
  %mask = tensor.empty() : tensor<24x32xi8>
  hfusion.indirect_store ins(%ins : tensor<24x32xi32>, %offset : tensor<24x32xi64>, %mask : tensor<24x32xi8>) outs(%arg0 : memref<?xi32>)
  return
}

// -----

// CHECK-LABEL: while_loop_yield_mismatch
// CHECK: scf.yield %arg2, %5 : tensor<128xi1>, i1
func.func @while_loop_yield_mismatch(%arg0: tensor<128xi1>, %arg1: i1) -> tensor<128xi1> {
  %true = arith.constant true
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<128xi32>
  %res = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (tensor<128xi1>, i1) -> tensor<128xi1> {
    %condition = arith.xori %arg3, %true : i1
    scf.condition(%condition) %arg2 : tensor<128xi1>
  } do {
  ^bb0(%arg2_do: tensor<128xi1>):
    %cst_true = arith.constant true
    %57 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%arg2_do : tensor<128xi1>) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
    %58 = bufferization.alloc_tensor() : tensor<i32>
    %59 = linalg.fill ins(%c0_i32 : i32) outs(%58 : tensor<i32>) -> tensor<i32>
    %reduced = linalg.reduce ins(%57 : tensor<128xi32>) outs(%59 : tensor<i32>) dimensions = [0]
      (%in: i32, %init: i32) {
        %61 = arith.addi %in, %init : i32
        linalg.yield %61 : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %60 = arith.cmpi eq, %extracted, %c128_i32 : i32
    scf.yield %arg2_do, %60 : tensor<128xi1>, i1
  }
  return %res : tensor<128xi1>
}
