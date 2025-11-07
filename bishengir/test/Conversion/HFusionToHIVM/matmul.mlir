// RUN: bishengir-opt -convert-hfusion-to-hivm="mm-map-mode=macro_instr" -canonicalize-ext %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline="enable-triton-kernel-compile=true" -canonicalize-ext %s -split-input-file -verify-diagnostics | FileCheck %s
// -----
// CHECK-LABEL: test_mmadL1_no_loop
// CHECK-DAG: %[[STUB_0:.*]] = arith.constant 0 : index
// CHECK-DAG: %true = arith.constant true
// CHECK: %[[ALLOC_A:.*]] = memref.alloc() : memref<256x128xf16>
// CHECK: %[[TENSOR_A:.*]] = bufferization.to_tensor %[[ALLOC_A]] restrict writable : memref<256x128xf16>
// CHECK: %[[ALLOC_B:.*]] = memref.alloc() : memref<128x256xf16>
// CHECK: %[[TENSOR_B:.*]] = bufferization.to_tensor %[[ALLOC_B]] restrict writable : memref<128x256xf16>
// CHECK: %[[ALLOC_C:.*]] = memref.alloc() : memref<256x256xf32>
// CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<256x256xf32>
// CHECK: %[[RET1:.*]] = hivm.hir.mmadL1 ins(%[[TENSOR_A]], %[[TENSOR_B]], %true, %[[STUB_0]], %[[STUB_0]], %[[STUB_0]] :
// CHECK-SAME:                                tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
// CHECK-SAME:                          outs(%[[INIT1]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK: bufferization.materialize_in_destination %[[RET1]] in restrict writable %[[ALLOC_C]]
// CHECK: %[[ALLOC_A_T:.*]] = memref.alloc() : memref<128x256xf16>
// CHECK: %[[TENSOR_A_T:.*]] = bufferization.to_tensor %[[ALLOC_A_T]] restrict writable : memref<128x256xf16>
// CHECK: %[[INIT2:.*]] = tensor.empty() : tensor<256x256xf32>
// CHECK: %[[RET2:.*]] = hivm.hir.mmadL1 {a_transpose} ins(%[[TENSOR_A_T]], %[[TENSOR_B]], %true, %[[STUB_0]], %[[STUB_0]], %[[STUB_0]] :
// CHECK-SAME:               tensor<128x256xf16>, tensor<128x256xf16>, i1, index, index, index)
// CHECK-SAME:                                        outs(%[[INIT2]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK: bufferization.materialize_in_destination %[[RET2]] in restrict writable %[[ALLOC_C]]
// CHECK: return
// CHECK: }
func.func @test_mmadL1_no_loop() {
  %cst = arith.constant 0.000000e+00 : f32

  %ma = memref.alloc() : memref<256x128xf16>
  %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<256x128xf16>

  %mb = memref.alloc() : memref<128x256xf16>
  %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<128x256xf16>

  %mc = tensor.empty() : tensor<256x256xf32>
  %mc_fill = linalg.fill ins(%cst : f32) outs(%mc : tensor<256x256xf32>) -> tensor<256x256xf32>
  %dst = memref.alloc() : memref<256x256xf32>
  %ret = linalg.matmul ins(%ma_tensor, %mb_tensor : tensor<256x128xf16>, tensor<128x256xf16>)
                       outs(%mc_fill: tensor<256x256xf32>) -> tensor<256x256xf32>
  bufferization.materialize_in_destination %ret in restrict writable
    %dst : (tensor<256x256xf32>, memref<256x256xf32>) -> ()

  %ma_transpose = memref.alloc() : memref<128x256xf16>
  %ma_transpose_tensor = bufferization.to_tensor %ma_transpose restrict writable : memref<128x256xf16>

  %ma_transpose_init = tensor.empty() : tensor<256x128xf16>
  %ma_transpose_res = linalg.transpose ins(%ma_transpose_tensor : tensor<128x256xf16>)
                                       outs(%ma_transpose_init : tensor<256x128xf16>) permutation = [1, 0]
  %ret1 = linalg.matmul ins(%ma_transpose_res, %mb_tensor : tensor<256x128xf16>, tensor<128x256xf16>)
                        outs(%mc_fill: tensor<256x256xf32>) -> tensor<256x256xf32>
  bufferization.materialize_in_destination %ret1 in restrict writable
    %dst : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
  return
}

// -----
// CHECK-LABEL: test_mmadL1_with_k_init
// CHECK-NOT: linalg.matmul
func.func @test_mmadL1_with_k_init() -> tensor<256x256xf32> {
  %mc = tensor.empty() : tensor<256x256xf32>
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NOT: linalg.fill
  %mc_fill = linalg.fill ins(%cst : f32) outs(%mc : tensor<256x256xf32>) -> tensor<256x256xf32>
  %start = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 128 : index
  %scf_ret1 = scf.for %arg0 = %start to %end step %step iter_args(%arg = %mc_fill) -> (tensor<256x256xf32>) {
    %scf_ret = scf.for %arg1 = %start to %end step %step iter_args(%arg2 = %arg) -> (tensor<256x256xf32>) {
      %ma = memref.alloc() : memref<256x128xf16>
      %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<256x128xf16>
      %mb = memref.alloc() : memref<128x256xf16>
      %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<128x256xf16>
      // CHECK: %[[COND1:.*]] = arith.cmpi eq
      // CHECK: %[[COND2:.*]] = arith.cmpi eq
      // CHECK: %[[INIT:.*]] = arith.andi %[[COND1]], %[[COND2]] : i1
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins({{.*}}, {{.*}}, %[[INIT]], {{.*}}, {{.*}}, {{.*}} : tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
      // CHECK-SAME:                           outs({{.*}} : tensor<256x256xf32>) -> tensor<256x256xf32>
      %ret = linalg.matmul ins(%ma_tensor, %mb_tensor : tensor<256x128xf16>, tensor<128x256xf16>)
                           outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
      scf.yield %ret : tensor<256x256xf32>
    }
    scf.yield %scf_ret : tensor<256x256xf32>
  }
  return %scf_ret1 : tensor<256x256xf32>
}

// -----
// CHECK-LABEL: func.func @test_MmadL1_real_init(
// CHECK-SAME:    %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK: %[[STUB_0:.*]] = arith.constant 0 : index
// CHECK: %[[INIT_FLAG:.*]] = arith.constant false
// CHECK: %[[REAL_INIT:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_8:.*]] = hivm.hir.mmadL1 ins(%[[VAL_5]], %[[VAL_7]], %[[INIT_FLAG]], %[[STUB_0]], %[[STUB_0]], %[[STUB_0]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%[[REAL_INIT]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %[[VAL_8]] : tensor<16x16xf32>
// CHECK: }
func.func @test_MmadL1_real_init(%arg1:memref<16x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %mc = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf32>
  %ma = memref.alloc() : memref<16x16xf16>
  %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<16x16xf16>
  %mb = memref.alloc() : memref<16x16xf16>
  %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<16x16xf16>
  %ret = linalg.matmul ins(%ma_tensor, %mb_tensor : tensor<16x16xf16>, tensor<16x16xf16>)
                             outs(%mc: tensor<16x16xf32>) -> tensor<16x16xf32>
  return %ret : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_batchMmadL1
func.func @test_batchMmadL1() -> tensor<2x256x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %mc = tensor.empty() : tensor<2x256x256xf32>
  // CHECK-NOT: linalg.fill
  %mc_fill = linalg.fill ins(%cst : f32) outs(%mc : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  %ma = memref.alloc() : memref<2x256x128xf16>
  %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<2x256x128xf16>
  %mb = memref.alloc() : memref<2x128x256xf16>
  %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<2x128x256xf16>
  // CHECK-DAG: %[[INIT:.*]] = arith.constant true
  // CHECK-DAG: %[[MA:.*]] = bufferization.to_tensor{{.*}}memref<2x256x128xf16>
  // CHECK-DAG: %[[MB:.*]] = bufferization.to_tensor{{.*}}memref<2x128x256xf16>
  // CHECK: hivm.hir.batchMmadL1 ins(%[[MA]], %[[MB]], %[[INIT]]
  %ret = linalg.batch_matmul ins(%ma_tensor, %mb_tensor : tensor<2x256x128xf16>, tensor<2x128x256xf16>)
                             outs(%mc_fill: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>

  return %ret : tensor<2x256x256xf32>
}


// -----
// CHECK-LABEL: func.func @test_enable_hf32(
// CHECK-SAME:    %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
func.func @test_enable_hf32(%arg1:memref<16x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %mc = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf32>
  %ma = memref.alloc() : memref<16x16xf32>
  %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<16x16xf32>
  %mb = memref.alloc() : memref<16x16xf32>
  %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<16x16xf32>
  // CHECK: hivm.hir.mmadL1 {enable_HF32}
  %ret = linalg.matmul{input_precision = "hf32"}  ins(%ma_tensor, %mb_tensor : tensor<16x16xf32>, tensor<16x16xf32>)
                             outs(%mc: tensor<16x16xf32>) -> tensor<16x16xf32>
  return %ret : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_batchMmadL1_with_transpose
func.func @test_batchMmadL1_with_transpose() -> tensor<2x256x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %mc = tensor.empty() : tensor<2x256x256xf32>
  %mc_fill = linalg.fill ins(%cst : f32) outs(%mc : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  %ma0 = memref.alloc() : memref<2x128x256xf16>
  %ma0_tensor = bufferization.to_tensor %ma0 restrict writable : memref<2x128x256xf16>
  %ma0_transpose_init = tensor.empty() : tensor<2x256x128xf16>
  %ma0_transpose_res = linalg.transpose ins(%ma0_tensor : tensor<2x128x256xf16>)
                                       outs(%ma0_transpose_init : tensor<2x256x128xf16>) permutation = [0, 2, 1]
  %mb = memref.alloc() : memref<2x128x256xf16>
  %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<2x128x256xf16>
  // CHECK: hivm.hir.batchMmadL1 {a_transpose}
  %ret0 = linalg.batch_matmul ins(%ma0_transpose_res, %mb_tensor : tensor<2x256x128xf16>, tensor<2x128x256xf16>)
                             outs(%mc_fill: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>

  %ma1 = memref.alloc() : memref<128x256x2xf16>
  %ma1_tensor = bufferization.to_tensor %ma1 restrict writable : memref<128x256x2xf16>
  %ma1_transpose_init = tensor.empty() : tensor<2x256x128xf16>
  %ma1_transpose_res = linalg.transpose ins(%ma1_tensor : tensor<128x256x2xf16>)
                                       outs(%ma1_transpose_init : tensor<2x256x128xf16>) permutation = [2, 1, 0]
  // CHECK-NOT: hivm.hir.batchMmadL1 {a_transpose}
  %ret1 = linalg.batch_matmul ins(%ma1_transpose_res, %mb_tensor : tensor<2x256x128xf16>, tensor<2x128x256xf16>)
                             outs(%mc_fill: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  
  %res_empty = tensor.empty() : tensor<2x256x256xf32>
  %res =  linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%ret1, %ret0 : tensor<2x256x256xf32>, tensor<2x256x256xf32>) outs(%res_empty : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  return  %res : tensor<2x256x256xf32>
}