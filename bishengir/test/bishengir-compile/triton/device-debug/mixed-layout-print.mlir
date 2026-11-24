// RUN: split-file %s %t
// RUN: bishengir-compile %t/batch.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-tuning-mode=true --only-run-hivm-pipeline=true --enable-layout-optimization=false --mlir-print-ir-after=hivm-insert-load-store-for-mix-cv -o %t/batch-v1.out > %t/batch-v1.log 2>&1
// RUN: FileCheck %s --check-prefix=BATCH --implicit-check-not="error:" --implicit-check-not="hivm.hir.l12ub" < %t/batch-v1.log
// RUN: FileCheck %s --check-prefix=LOWERED < %t/batch-v1.out
// RUN: bishengir-compile %t/batch.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-tuning-mode=true --only-run-hivm-pipeline=true --enable-layout-optimization=true --mlir-print-ir-after=hivm-insert-load-store-for-mix-cv,hivm-insert-convert-layout -o %t/batch-fallback.out > %t/batch-fallback.log 2>&1
// RUN: FileCheck %s --check-prefix=BATCH --implicit-check-not="error:" --implicit-check-not="InsertConvertLayout" < %t/batch-fallback.log
// RUN: bishengir-compile %t/matmul.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-tuning-mode=true --only-run-hivm-pipeline=true --enable-layout-optimization=false -o %t/matmul-v1.out > %t/matmul-v1.log 2>&1
// RUN: FileCheck %s --check-prefix=CLEAN --allow-empty --implicit-check-not="error:" < %t/matmul-v1.log
// RUN: bishengir-compile %t/vector.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-tuning-mode=true --only-run-hivm-pipeline=true --enable-layout-optimization=false -o %t/vector-v1.out > %t/vector-v1.log 2>&1
// RUN: FileCheck %s --check-prefix=VECTOR < %t/vector-v1.out
// RUN: bishengir-compile %t/vector.mlir --target=Ascend950PR_9579 --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-tuning-mode=true --only-run-hivm-pipeline=true --enable-layout-optimization=true -o %t/vector-v2.out > %t/vector-v2.log 2>&1
// RUN: FileCheck %s --check-prefix=VECTOR < %t/vector-v2.out


// LOWERED-LABEL: llvm.func @kernel_mix_aic
// LOWERED: llvm.call @nd2nz_bfloat16_t
// LOWERED-LABEL: llvm.func @kernel_mix_aiv
// LOWERED: llvm.call @load_gm_to_ubuf_1d_bfloat16_t
// LOWERED: llvm.call @print_4d_bfloat16_t_ubuf.vector
// LOWERED: llvm.call @print_4d_bfloat16_t_ubuf.vector
// LOWERED: llvm.call @print_4d_float_ubuf.vector

// Two print shapes share one 128 KiB snapshot. Duplicating this load would
// need 256 KiB of UB, exceeding the target's 248 KiB capacity.
// VECTOR-LABEL: llvm.func @kernel(
// VECTOR-NOT: llvm.call @load_gm_to_ubuf
// VECTOR: %[[UB:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<6>
// VECTOR-NEXT: llvm.call @load_gm_to_ubuf_1d_bfloat16_t({{.*}}, %[[UB]], %[[UB]],
// VECTOR-NOT: llvm.call @load_gm_to_ubuf
// VECTOR: llvm.call @print_4d_bfloat16_t_ubuf({{.*}}, %[[UB]], %[[UB]],
// VECTOR-NOT: llvm.call @load_gm_to_ubuf
// VECTOR: llvm.call @print_1d_bfloat16_t_ubuf({{.*}}, %[[UB]], %[[UB]],
// VECTOR-NOT: llvm.call @load_gm_to_ubuf

// Reconstructed from the rank-4 print / rank-3 batch-matmul report. Batch
// matmul retains its existing layout fallback. Both batched and ordinary
// matmul printing must keep the logical debug snapshot separate from Cube data.
// CLEAN: [INFO] Using merged native A5 regbase pipeline
// BATCH-LABEL: IR Dump After InsertLoadStoreForMixCV
// BATCH: hivm.hir.load {{.*}}core_type = <VECTOR>
// BATCH: hivm.hir.batchMmadL1
// BATCH: hivm.hir.debug {{.*}}memscope = #hivm.address_space<ub>{{.*}}prefix = " x: "{{.*}}tcoretype = #hivm.tcore_type<VECTOR>{{.*}} : tensor<2x2x32x32xbf16>
// BATCH: hivm.hir.debug {{.*}}memscope = #hivm.address_space<ub>{{.*}}prefix = " y: "{{.*}} : tensor<2x2x32x32xbf16>
// BATCH: hivm.hir.debug {{.*}}memscope = #hivm.address_space<ub>{{.*}}prefix = " z: "{{.*}} : tensor<2x2x32x32xf32>

//--- batch.mlir
module {
  func.func private @triton_print(tensor<2x2x32x32xbf16>) attributes {hex = false, prefix = " x: "}
  func.func private @triton_print_0(tensor<2x2x32x32xbf16>) attributes {hex = false, prefix = " y: "}
  func.func private @triton_print_1(tensor<2x2x32x32xf32>) attributes {hex = false, prefix = " z: "}
  func.func @kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant dense<4096> : tensor<1xi64>
    %cst_0 = arith.constant dense<[4, 32, 32]> : tensor<3xi64>
    %cst_1 = arith.constant dense<[2, 2, 32, 32]> : tensor<4xi64>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x32x32xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [4096], strides: [1] : memref<?xbf16> to memref<4096xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<4096xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4096xbf16, strided<[1]>> to memref<4096xbf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<4096xbf16>
    %reshape = tensor.reshape %2(%cst_1) : (tensor<4096xbf16>, tensor<4xi64>) -> tensor<2x2x32x32xbf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4096], strides: [1] : memref<?xbf16> to memref<4096xbf16, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<4096xbf16>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<4096xbf16, strided<[1]>> to memref<4096xbf16>
    %3 = bufferization.to_tensor %alloc_4 restrict writable : memref<4096xbf16>
    %reshape_5 = tensor.reshape %3(%cst_1) : (tensor<4096xbf16>, tensor<4xi64>) -> tensor<2x2x32x32xbf16>
    %reshape_6 = tensor.reshape %2(%cst_0) : (tensor<4096xbf16>, tensor<3xi64>) -> tensor<4x32x32xbf16>
    %reshape_7 = tensor.reshape %3(%cst_0) : (tensor<4096xbf16>, tensor<3xi64>) -> tensor<4x32x32xbf16>
    %4 = linalg.batch_matmul {input_precision = "ieee"} ins(%reshape_6, %reshape_7 : tensor<4x32x32xbf16>, tensor<4x32x32xbf16>) outs(%1 : tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %reshape_8 = tensor.reshape %4(%cst_1) : (tensor<4x32x32xf32>, tensor<4xi64>) -> tensor<2x2x32x32xf32>
    call @triton_print(%reshape) : (tensor<2x2x32x32xbf16>) -> ()
    call @triton_print_0(%reshape_5) : (tensor<2x2x32x32xbf16>) -> ()
    call @triton_print_1(%reshape_8) : (tensor<2x2x32x32xf32>) -> ()
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    %reshape_10 = tensor.reshape %4(%cst) : (tensor<4x32x32xf32>, tensor<1xi64>) -> tensor<4096xf32>
    bufferization.materialize_in_destination %reshape_10 in writable %reinterpret_cast_9 : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
    return
  }
}

//--- matmul.mlir
module {
  func.func private @triton_print(tensor<1x1x32x32xbf16>) attributes {hex = false, prefix = " x: "}
  func.func private @triton_print_0(tensor<1x1x32x32xbf16>) attributes {hex = false, prefix = " y: "}
  func.func private @triton_print_1(tensor<1x1x32x32xf32>) attributes {hex = false, prefix = " z: "}
  func.func @kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant dense<1024> : tensor<1xi64>
    %cst_0 = arith.constant dense<[32, 32]> : tensor<2xi64>
    %cst_1 = arith.constant dense<[1, 1, 32, 32]> : tensor<4xi64>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<1024xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024xbf16, strided<[1]>> to memref<1024xbf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
    %reshape = tensor.reshape %2(%cst_1) : (tensor<1024xbf16>, tensor<4xi64>) -> tensor<1x1x32x32xbf16>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<1024xbf16>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<1024xbf16, strided<[1]>> to memref<1024xbf16>
    %3 = bufferization.to_tensor %alloc_4 restrict writable : memref<1024xbf16>
    %reshape_5 = tensor.reshape %3(%cst_1) : (tensor<1024xbf16>, tensor<4xi64>) -> tensor<1x1x32x32xbf16>
    %reshape_6 = tensor.reshape %2(%cst_0) : (tensor<1024xbf16>, tensor<2xi64>) -> tensor<32x32xbf16>
    %reshape_7 = tensor.reshape %3(%cst_0) : (tensor<1024xbf16>, tensor<2xi64>) -> tensor<32x32xbf16>
    %4 = linalg.matmul {input_precision = "ieee"} ins(%reshape_6, %reshape_7 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reshape_8 = tensor.reshape %4(%cst_1) : (tensor<32x32xf32>, tensor<4xi64>) -> tensor<1x1x32x32xf32>
    call @triton_print(%reshape) : (tensor<1x1x32x32xbf16>) -> ()
    call @triton_print_0(%reshape_5) : (tensor<1x1x32x32xbf16>) -> ()
    call @triton_print_1(%reshape_8) : (tensor<1x1x32x32xf32>) -> ()
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1]>>
    %reshape_10 = tensor.reshape %4(%cst) : (tensor<32x32xf32>, tensor<1xi64>) -> tensor<1024xf32>
    bufferization.materialize_in_destination %reshape_10 in writable %reinterpret_cast_9 : (tensor<1024xf32>, memref<1024xf32, strided<[1]>>) -> ()
    return
  }
}

//--- vector.mlir
module {
  func.func private @triton_print(tensor<32x2x32x32xbf16>) attributes {hex = false, prefix = " x: "}
  func.func private @triton_print_flat(tensor<65536xbf16>) attributes {hex = false, prefix = "flat"}
  func.func @kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %in = memref.reinterpret_cast %arg2 to offset: [0], sizes: [65536], strides: [1] : memref<?xbf16> to memref<65536xbf16, strided<[1]>>
    %buf = memref.alloc() : memref<65536xbf16>
    memref.copy %in, %buf : memref<65536xbf16, strided<[1]>> to memref<65536xbf16>
    %v = bufferization.to_tensor %buf restrict writable : memref<65536xbf16>
    %shape = arith.constant dense<[32, 2, 32, 32]> : tensor<4xi64>
    %p = tensor.reshape %v(%shape) : (tensor<65536xbf16>, tensor<4xi64>) -> tensor<32x2x32x32xbf16>
    call @triton_print(%p) : (tensor<32x2x32x32xbf16>) -> ()
    call @triton_print_flat(%v) : (tensor<65536xbf16>) -> ()
    return
  }
}
