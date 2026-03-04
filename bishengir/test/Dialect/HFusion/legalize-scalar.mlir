// RUN: bishengir-opt -hfusion-legalize-scalar %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_scalar_add_bf16
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.addf {{.*}}, {{.*}} : tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_add_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 5.000000e+00 : bf16
  %0 = arith.addf %arg3, %cst : bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_div_bf16
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.divf {{.*}}, {{.*}} : tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_div_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 5.000000e+00 : bf16
  %0 = arith.divf %cst, %arg3 : bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_mul_bf16
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.mulf {{.*}}, {{.*}} : tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_mul_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 5.000000e+00 : bf16
  %0 = arith.mulf %arg3, %cst : bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_rem_f16
// CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
// CHECK: arith.remf {{.*}}, {{.*}} : tensor<1xf16>
// CHECK: tensor.extract {{.*}} : tensor<1xf16>
func.func @test_scalar_rem_f16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: f16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 5.000000e+00 : f16
  %0 = arith.remf %cst, %arg3 : f16
  %1 = tensor.empty() : tensor<6x5x15xf16>
  %2 = linalg.fill ins(%0 : f16) outs(%1 : tensor<6x5x15xf16>) -> tensor<6x5x15xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xf16> to memref<6x5x15xf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xf16>, memref<6x5x15xf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_sqrt_bf16
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: math.sqrt {{.*}} : tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_sqrt_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = math.sqrt %arg3 : bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_sub_bf16
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.subf {{.*}}, {{.*}} : tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_sub_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 5.000000e+00 : bf16
  %0 = arith.subf %cst, %arg3 : bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----
 
// CHECK-LABEL: func.func @test_scalar_bf16_i8(
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.fptosi {{.*}} : tensor<1xbf16> to tensor<1xi32>
// CHECK: arith.trunci {{.*}} : tensor<1xi32> to tensor<1xi8>
// CHECK: tensor.extract {{.*}} : tensor<1xi8>
func.func @test_scalar_bf16_i8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.fptosi %arg3 : bf16 to i8
  %1 = tensor.empty() : tensor<6x5x15xi8>
  %2 = linalg.fill ins(%0 : i8) outs(%1 : tensor<6x5x15xi8>) -> tensor<6x5x15xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xi8> to memref<6x5x15xi8, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xi8>, memref<6x5x15xi8, strided<[75, 15, 1]>>) -> ()
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_scalar_i8_bf16(
// CHECK: tensor.from_elements {{.*}} : tensor<1xi8>
// CHECK: arith.sitofp {{.*}} : tensor<1xi8> to tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_i8_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: i8, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.sitofp %arg3 : i8 to bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_scalar_i32_bf16(
// CHECK: tensor.from_elements {{.*}} : tensor<1xi32>
// CHECK: arith.sitofp {{.*}} : tensor<1xi32> to tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_i32_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.sitofp %arg3 : i32 to bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_scalar_u8_bf16(
// CHECK: tensor.from_elements {{.*}} : tensor<1xi8>
// CHECK: arith.uitofp {{.*}} : tensor<1xi8> to tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_u8_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: i8, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.uitofp %arg3 : i8 to bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}
 
// -----
 
// CHECK-LABEL: func.func @test_scalar_u32_bf16(
// CHECK: tensor.from_elements {{.*}} : tensor<1xi32>
// CHECK: arith.uitofp {{.*}} : tensor<1xi32> to tensor<1xbf16>
// CHECK: tensor.extract {{.*}} : tensor<1xbf16>
func.func @test_scalar_u32_bf16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.uitofp %arg3 : i32 to bf16
  %1 = tensor.empty() : tensor<6x5x15xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<6x5x15xbf16>) -> tensor<6x5x15xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xbf16> to memref<6x5x15xbf16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xbf16>, memref<6x5x15xbf16, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_bf16_u8(
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.fptosi {{.*}} : tensor<1xbf16> to tensor<1xi32>
// CHECK: arith.trunci {{.*}} : tensor<1xi32> to tensor<1xi8>
// CHECK: tensor.extract {{.*}} : tensor<1xi8>
func.func @test_scalar_bf16_u8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.fptoui %arg3 : bf16 to i8
  %1 = tensor.empty() : tensor<6x5x15xi8>
  %2 = linalg.fill ins(%0 : i8) outs(%1 : tensor<6x5x15xi8>) -> tensor<6x5x15xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xi8> to memref<6x5x15xi8, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xi8>, memref<6x5x15xi8, strided<[75, 15, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_scalar_bf16_u16(
// CHECK: tensor.from_elements {{.*}} : tensor<1xbf16>
// CHECK: arith.fptosi {{.*}} : tensor<1xbf16> to tensor<1xi32>
// CHECK: arith.trunci {{.*}} : tensor<1xi32> to tensor<1xi16>
// CHECK: tensor.extract {{.*}} : tensor<1xi16>
func.func @test_scalar_bf16_u16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: bf16, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.fptoui %arg3 : bf16 to i16
  %1 = tensor.empty() : tensor<6x5x15xi16>
  %2 = linalg.fill ins(%0 : i16) outs(%1 : tensor<6x5x15xi16>) -> tensor<6x5x15xi16>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 5, 15], strides: [75, 15, 1] : memref<?xi16> to memref<6x5x15xi16, strided<[75, 15, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<6x5x15xi16>, memref<6x5x15xi16, strided<[75, 15, 1]>>) -> ()
  return
}