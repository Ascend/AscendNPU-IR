// RUN: bishengir-opt -convert-linalg-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_reluDh(f16) -> f16 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  %ret = linalg.map { func.call {callee = @__hmf_reluDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----
// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_sqrtf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0 : tensor<256xf32>) -> tensor<256xf32> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  %ret = linalg.map { func.call {callee = @__hmf_sqrtf} } ins(%arg0 : tensor<256xf32>) outs(%arg0 : tensor<256xf32>)
  return %ret : tensor<256xf32>
}

// -----
// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_fabsf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_fabs
func.func @test_fabs(%arg0 : tensor<256xf32>) -> tensor<256xf32> {
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
  %ret = linalg.map  { func.call {callee = @__hmf_fabsf} } ins(%arg0 : tensor<256xf32>) outs(%arg0 : tensor<256xf32>)
  return %ret : tensor<256xf32>
}

// -----
// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_expDh(f16) -> f16 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0 : tensor<256xf16>) -> tensor<256xf16> {
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  %ret = linalg.map { func.call {callee = @__hmf_expDh} } ins(%arg0 : tensor<256xf16>) outs(%arg0 : tensor<256xf16>)
  return %ret : tensor<256xf16>
}

// -----
// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_logDh(f16) -> f16 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_log
func.func @test_log(%arg0 : tensor<256xf16>) -> tensor<256xf16> {
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  %ret = linalg.map { func.call {callee = @__hmf_logDh} } ins(%arg0 : tensor<256xf16>) outs(%arg0 : tensor<256xf16>)
  return %ret : tensor<256xf16>
}


// -----
// COM: the referenced ocl function should be removed by following symbol-dce pass
func.func private @__hmf_rsqrtf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @test_rsqrtf
func.func @test_rsqrtf(%arg0 : tensor<256xf32>) -> tensor<256xf32> {
  // CHECK:       %[[RET:.*]] =  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  %ret = linalg.map { func.call {callee = @__hmf_rsqrtf} } ins(%arg0 : tensor<256xf32>) outs(%arg0 : tensor<256xf32>)
  return %ret : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @test_recipf
func.func private @__hmf_recipf(f32) -> f32 attributes {llvm.readnone}
func.func @test_recipf(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK:       %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:       %[[RET0:.*]] = tensor.empty() : tensor<32xf32>
  // CHECK:       %[[RET1:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[RET0]] : tensor<32xf32>) -> tensor<32xf32>
  // CHECK:       %[[RET3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  %1 =  linalg.map { func.call {callee = @__hmf_recipf} } ins(%arg0 : tensor<32xf32>) outs(%arg0 : tensor<32xf32>)
  return %1 : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @test_recipDh
func.func private @__hmf_recipDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_recipDh(%arg0 : tensor<32xf16>) -> tensor<32xf16> {
  // CHECK:       %[[CST:.*]] = arith.constant 1.000000e+00 : f16
  // CHECK:       %[[RET0:.*]] = tensor.empty() : tensor<32xf16>
  // CHECK:       %[[RET1:.*]] = linalg.fill ins(%[[CST]] : f16) outs(%[[RET0]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK:       %[[RET3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  %1 =  linalg.map { func.call {callee = @__hmf_recipDh} } ins(%arg0 : tensor<32xf16>) outs(%arg0 : tensor<32xf16>)
  return %1 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @test_log1pDh
func.func private @__hmf_log1pDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_log1pDh(%arg0 : tensor<32xf16>) -> tensor<32xf16> {
  // CHECK:   %[[RET:.*]] =  hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>}
  %1 =  linalg.map { func.call {callee = @__hmf_log1pDh} } ins(%arg0 : tensor<32xf16>) outs(%arg0 : tensor<32xf16>)
  return %1 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_rnDh
func.func private @__hmf_sqrt_rnDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_sqrt_rnDh(%arg0 : tensor<32xf16>) -> tensor<32xf16> {
  // CHECK:   %[[RET:.*]] =  hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  %1 =  linalg.map { func.call {callee = @__hmf_sqrt_rnDh} } ins(%arg0 : tensor<32xf16>) outs(%arg0 : tensor<32xf16>)
  return %1 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @test_tanDh
func.func private @__hmf_tanDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_tanDh(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>}
  %ret = linalg.map { func.call {callee = @__hmf_tanDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----

// CHECK-LABEL: func.func @test_tanhDh
func.func private @__hmf_tanhDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_tanhDh(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  %ret = linalg.map { func.call {callee = @__hmf_tanhDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----

// CHECK-LABEL: func.func @test_atanDh
func.func private @__hmf_atanDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_atanDh(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atan>}
  %ret = linalg.map { func.call {callee = @__hmf_atanDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----

// CHECK-LABEL: func.func @test_ilogbDh
func.func private @__hmf_ilogbDh(f16) -> f16 attributes {llvm.readnone}
func.func @test_ilogbDh(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<ilogb>}
  %ret = linalg.map { func.call {callee = @__hmf_ilogbDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----

// CHECK-LABEL: func.func @test_ldexpDh
func.func private @__hmf_ldexpDh(f16, f16) -> f16 attributes {llvm.readnone}
func.func @test_ldexpDh(%arg0 : tensor<6x6xf16>, %arg1 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>}
  %ret = linalg.map { func.call {callee = @__hmf_ldexpDh} } ins(%arg0 , %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// -----


// CHECK-LABEL: func.func @test_atomic_add
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_add(%arg0 : memref<?xf32> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xf32>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xf32, strided<[1]>>
  // CHECK:       hfusion.atomic_rmw ins(%[[UB_MEMREF:.*]] : memref<256xf32, strided<[1]>>) outs(%[[GM_MEMREF:.*]] : memref<256xf32, strided<[1], offset: ?>>) atomic_kind = <add>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xf32, strided<[1], offset: ?>>, memref<256xf32, strided<[1]>>) outs(%reinterpret_cast : memref<256xf32, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_atomic_cas
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_cas(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>, %arg2 : tensor<256xi16>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
  %3 = bufferization.to_memref %arg2 : memref<256xi16, strided<[1]>>
  // CHECK:       hfusion.atomic_cas ins(%[[CMP:.*]], %[[VAL:.*]] : memref<256xi16, strided<[1]>>, memref<256xi16, strided<[1]>>) outs(%[[OUT:.*]] : memref<256xi16, strided<[1], offset: ?>>)
  linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2, %3 : memref<256xi16, strided<[1], offset: ?>>, memref<256xi16, strided<[1]>>, memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
    ^bb0(%in: i16, %in_9: i16, %in_10: i16, %out: i16):
      %8 = arith.cmpi eq, %in, %in_9 : i16
      %9 = arith.select %8, %in_10, %in : i16
      linalg.yield %9 : i16
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_atomic_xchg
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_xchg(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
  // CHECK:       hfusion.atomic_xchg ins(%[[VAL:.*]] : memref<256xi16, strided<[1]>>) outs(%[[OUT:.*]] : memref<256xi16, strided<[1], offset: ?>>)
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xi16, strided<[1], offset: ?>>, memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: i16, %in_0: i16, %out: i16):
      %3 = arith.xori %in, %in_0 : i16
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
  //CHECK:  %[[REDUCED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%[[INPUT0:.*]] : tensor<256x64xf32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
  %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1]  {reduce_mode = "max_with_index", tie_break_left = "true"}
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

// CHECK-LABEL: func.func @test_reduce_with_index_with_index_input
func.func @test_reduce_with_index_with_index_input(%arg0 : tensor<256x64xf32>, %arg1 : tensor<256x64xi32>) -> tensor<256xf32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<256xf32>
  %1 = tensor.empty() : tensor<256xi32>
  annotation.mark %arg1 {UseIndexInput} : tensor<256x64xi32>
  //CHECK:  %[[REDUCED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%[[INPUT0:.*]], %[[INPUT1:.*]] : tensor<256x64xf32>, tensor<256x64xi32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
  %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1]  {reduce_mode = "max_with_index", tie_break_left = "true"}
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

// CHECK-LABEL: func.func @test_is_inf
func.func private @__hmf_isinff(f32) -> i1 attributes {llvm.readnone}
func.func @test_is_inf(%arg0 : tensor<32xf32>, %arg1 : tensor<32xi1>) -> tensor<32xi1> {
  // CHECK:       %[[RET1:.*]] = hfusion.isinf  %[[ARG0:.*]] : tensor<32xf32> -> tensor<32xi1> 
  %ret =  linalg.map { func.call {callee = @__hmf_isinff} } ins(%arg0 : tensor<32xf32>) outs(%arg1 : tensor<32xi1>)
  return %ret : tensor<32xi1>
}

// -----

// CHECK-LABEL: func.func @test_is_nan
func.func private @__hmf_isnanf(f32) -> i1 attributes {llvm.readnone}
func.func @test_is_nan(%arg0 : tensor<32xf32>, %arg1 : tensor<32xi1>) -> tensor<32xi1> {
  // CHECK:       %[[RET1:.*]] = hfusion.isnan  %[[ARG0:.*]] : tensor<32xf32> -> tensor<32xi1> 
  %ret =  linalg.map { func.call {callee = @__hmf_isnanf} } ins(%arg0 : tensor<32xf32>) outs(%arg1 : tensor<32xi1>)
  return %ret : tensor<32xi1>
}

// CHECK-LABEL: func.func @test_powf
func.func private @__hmf_powf(f16, f16) -> f16 attributes {llvm.readnone}
func.func @test_powf(%arg0 : tensor<6x6xf16>, %arg1 : tensor<6x6xf16>) -> tensor<6x6xf16> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
  %ret = linalg.map { func.call {callee = @__hmf_powf} } ins(%arg0 , %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
  return %ret : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_powi
func.func private @__hmf_powi(i32, i32) -> i32 attributes {llvm.readnone}
func.func @test_powi(%arg0 : tensor<32xi32>, %arg1 : tensor<32xi32>) -> tensor<32xi32> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>}
  %ret = linalg.map { func.call {callee = @__hmf_powi} } ins(%arg0 , %arg1 : tensor<32xi32>, tensor<32xi32>) outs(%arg0 : tensor<32xi32>)
  return %ret : tensor<32xi32>
}

// -----

// CHECK-LABEL: func.func @test_tanhf
func.func private @__hmf_tanhf(f32) -> f32 attributes {llvm.readnone}
func.func @test_tanhf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  %ret = linalg.map { func.call {callee = @__hmf_tanhf} } ins(%arg0 : tensor<6x6xf32>) outs(%arg0 : tensor<6x6xf32>)
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_tanf
func.func private @__hmf_tanf(f32) -> f32 attributes {llvm.readnone}
func.func @test_tanf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>}
  %ret = linalg.map { func.call {callee = @__hmf_tanf} } ins(%arg0 : tensor<6x6xf32>) outs(%arg0 : tensor<6x6xf32>)
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_atomic_max_ui8
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_max_ui8(%arg0 : memref<?xi8> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi8>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xi8, strided<[1]>>
  // CHECK:       hfusion.atomic_rmw ins(%[[UB_MEMREF:.*]] : memref<256xi8, strided<[1]>>) outs(%[[GM_MEMREF:.*]] : memref<256xi8, strided<[1], offset: ?>>) atomic_kind = <umax>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xi8, strided<[1], offset: ?>>, memref<256xi8, strided<[1]>>) outs(%reinterpret_cast : memref<256xi8, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "umax", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: i8, %in_0: i8, %out: i8):
      %3 = arith.maxui %in, %in_0 : i8
      linalg.yield %3 : i8
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_atomic_min_ui8
#map = affine_map<(d0) -> (d0)>
func.func @test_atomic_min_ui8(%arg0 : memref<?xi8> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi8>) {
  %0 = arith.constant 256 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8, strided<[1], offset: ?>>
  %2 = bufferization.to_memref %arg1 : memref<256xi8, strided<[1]>>
  // CHECK:       hfusion.atomic_rmw ins(%[[UB_MEMREF:.*]] : memref<256xi8, strided<[1]>>) outs(%[[GM_MEMREF:.*]] : memref<256xi8, strided<[1], offset: ?>>) atomic_kind = <umin>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%reinterpret_cast, %2 : memref<256xi8, strided<[1], offset: ?>>, memref<256xi8, strided<[1]>>) outs(%reinterpret_cast : memref<256xi8, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
    ^bb0(%in: i8, %in_0: i8, %out: i8):
      %3 = arith.minui %in, %in_0 : i8
      linalg.yield %3 : i8
    }
  return
}
