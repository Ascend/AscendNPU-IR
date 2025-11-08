// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv %s -split-input-file -verify-diagnostics --canonicalize-ext | FileCheck %s

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_vector(
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf16>, %[[ARG1:.*]]: memref<?xi8>) {
func.func @insert_load_between_fixpipe_and_vector(%arg0 : memref<?xf16>, %arg1 : memref<?xi8>) {
  %cst_1 = arith.constant 2.000000e+00 : f16
  %reinterpret_cast_fixpipe_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  %fixpipe_tmp0_tensor = bufferization.to_tensor %reinterpret_cast_fixpipe_0 restrict writable : memref<16x16xf16, strided<[16, 1], offset: 0>>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[VAL2:.*]] = hivm.hir.fixpipe {enable_nz2nd} ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[VAL3:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[VAL4:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf16>) outs(%[[VAL3]] : tensor<16x16xf16>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  // CHECK: %[[VAL5:.*]] = hivm.hir.vmul ins(%[[VAL4]], %{{.*}} : tensor<16x16xf16>, f16) outs(%{{.*}} : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = hivm.hir.fixpipe {enable_nz2nd} ins(%1 : tensor<16x16xf32>)
                               outs(%fixpipe_tmp0_tensor : tensor<16x16xf16>) -> tensor<16x16xf16>
  %4 = hivm.hir.vmul ins(%3, %cst_1 : tensor<16x16xf16>, f16) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [ 1] : memref<?xi8> to memref<512xi8, strided<[1], offset: 0>>
  %cst0 = arith.constant 0 : index
  %view = memref.view %reinterpret_cast_0[%cst0][] : memref<512xi8, strided<[1], offset: 0>> to memref<16x16xf16>
  bufferization.materialize_in_destination %4 in writable %view : (tensor<16x16xf16>, memref<16x16xf16>) -> ()
  return
}

// -----
// CHECK-LABEL: @insert_store_between_vector_and_load(
func.func @insert_store_between_vector_and_load(%arg0 : memref<?xf32>) {
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %0 = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[VAL1:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<16x16xf32>, tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK: %[[VAL2:.*]] = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[VAL3:.*]] = hivm.hir.store ins(%[[VAL1]] : tensor<16x16xf32>) outs(%[[VAL2]] : tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK: %[[VAL4:.*]] = hivm.hir.load ins(%[[VAL3]] : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
  %3 = hivm.hir.vmul ins(%2, %2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = tensor.empty() : tensor<16x16xf32>
  %5 = hivm.hir.load ins(%3 : tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  bufferization.materialize_in_destination %5 in writable %reinterpret_cast_0 : (tensor<16x16xf32>, memref<16x16xf32, strided<[16, 1], offset: 0>>) -> ()
  return
}


// -----
// CHECK-LABEL: @insert_load_store_between_vector_and_cube
// CHECK: %[[VAL1:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<16x16xf32>, f32) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[VAL2:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[VAL3:.*]] = hivm.hir.store ins(%[[VAL1]] : tensor<16x16xf32>) outs(%[[VAL2]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[VAL4:.*]] = hivm.hir.load ins(%[[VAL3]] : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf32>
func.func @insert_load_store_between_vector_and_cube(%arg0 : memref<?xf32>) {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = hivm.hir.vmul ins(%2, %cst_1 : tensor<16x16xf32>, f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = tensor.empty() : tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%3, %3, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  bufferization.materialize_in_destination %5 in writable %reinterpret_cast_0 : (tensor<16x16xf32>, memref<16x16xf32, strided<[16, 1], offset: 0>>) -> ()
  %6 = tensor.empty() : tensor<16x16xf32>
  %7 = hivm.hir.vmul ins(%3, %cst_1 : tensor<16x16xf32>, f32) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [1024], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 1024>>
  bufferization.materialize_in_destination %7 in writable %reinterpret_cast_1 : (tensor<16x16xf32>, memref<16x16xf32, strided<[16, 1], offset: 1024>>) -> ()
  return
}


// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_madl1
func.func @insert_load_between_fixpipe_and_madl1(%arg0 : memref<?xf32>, %arg1 : memref<?xf16>) {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = tensor.empty() : tensor<16x16xf32>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%4, %4, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  %7 = bufferization.to_tensor %6 restrict writable :memref<16x16xf16, strided<[16, 1], offset: 0>>
  %8 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
        ins(%5 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[F_VAL:.*]] = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK: %[[VAL1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[VAL2:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>) outs(%[[VAL1]] : tensor<16x16xf16>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  // CHECK: %[[VAL3:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[VAL4:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>) outs(%[[VAL3]] : tensor<16x16xf16>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  %9 = hivm.hir.mmadL1 ins(%8, %8, %init_condition, %c16, %c16, %c16 :
                                 tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
                           outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>

  hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
       ins(%9 : tensor<16x16xf32>) outs(%6 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  return
}


// -----
// CHECK-LABEL: @fixpipe_with_loop
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x64xf32>, %[[ARG1:.*]]: tensor<128x64xf32>) -> tensor<128x64xf32> {
module {
  func.func @fixpipe_with_loop(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 3.200000e+01 : f32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = hivm.hir.fixpipe {enable_nz2nd} ins(%arg0 : tensor<128x64xf32>) outs(%arg1 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c8_i32 iter_args(%arg3 = %0) -> (tensor<128x64xf32>)  : i32 {
      %2 = tensor.empty() : tensor<128x64xf32>
      %3 = hivm.hir.load ins(%arg3 : tensor<128x64xf32>) outs(%2 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %4 = tensor.empty() : tensor<128x64xf32>
      %5 = hivm.hir.vadd ins(%3, %cst : tensor<128x64xf32>, f32) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
      // CHECK: %[[VAL7:.*]] = hivm.hir.store ins(%{{.*}} : tensor<128x64xf32>) outs(%{{.*}} : tensor<128x64xf32>) -> tensor<128x64xf32>
      // CHECK: scf.yield %[[VAL7]] : tensor<128x64xf32>
      scf.yield %5 : tensor<128x64xf32>
    }
    return %1 : tensor<128x64xf32>
  }
}

// -----
func.func @fixpipe_with_multiple_user(%arg0 : memref<?xf32>, %arg1 : memref<?xf16>, %arg2 : memref<?xf16>) {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = tensor.empty() : tensor<16x16xf32>
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%4, %4, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  %7 = bufferization.to_tensor %6 restrict writable :memref<16x16xf16, strided<[16, 1], offset: 0>>
  %8 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
        ins(%5 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[F_VAL:.*]] = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK-DAG: %[[USE_FIRST:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>)
  // CHECK-DAG: %[[USE_SECOND:.*]] = hivm.hir.load ins(%[[F_VAL]] : tensor<16x16xf16>)
  // CHECK: hivm.hir.store ins(%[[USE_FIRST]] : tensor<16x16xf16>)
  // CHECK: hivm.hir.mmadL1 ins(%[[USE_SECOND]]
  %9 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  hivm.hir.store ins(%8 : tensor<16x16xf16>) outs(%9 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  %10 = tensor.empty() : tensor<16x16xf16>
  %11 = hivm.hir.mmadL1 ins(%8, %10, %init_condition, %c16, %c16, %c16 :
                                 tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
                           outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
       ins(%11 : tensor<16x16xf32>) outs(%6 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  return
}

// -----
// CHECK-LABEL: @insert_load_between_discrete_load_and_madl1
func.func @insert_load_between_discrete_load_and_madl1(%arg0 : memref<?xf32>, %arg1 : memref<?xf16>) {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = tensor.empty() : tensor<16x16xf32>
  %idx = tensor.empty() : tensor<16x16xi64>
  %2 = scf.for %arg25 = %c0 to %c16 step %c1 iter_args(%arg26 = %1) -> (tensor<16x16xf32>) {
    %3 = scf.for %arg27 = %c0 to %c16 step %c1 iter_args(%arg28 = %arg26) -> (tensor<16x16xf32>) {
      %extracted = tensor.extract %idx[%arg25, %arg27] : tensor<16x16xi64>
      %129 = arith.index_cast %extracted : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%129], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %130 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %131 = tensor.empty() : tensor<1x1xf32>
      %132 = hivm.hir.vbrc ins(%130 : f32) outs(%131 : tensor<1x1xf32>) -> tensor<1x1xf32>
      %inserted_slice = tensor.insert_slice %131 into %arg28[%arg25, %arg27] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
      }
    scf.yield %3 : tensor<16x16xf32>
  } {ExtractedLoadOrStore}
  // CHECK: %[[VAL0:.*]] = hivm.hir.store ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK: %[[VAL1:.*]] = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[VAL2:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf32>
  // CHECK: %[[VAL3:.*]] = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.mmadL1 ins(%[[VAL2]], %[[VAL3]]
  %4 = tensor.empty() : tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%2, %4, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
  hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
       ins(%5 : tensor<16x16xf32>) outs(%6 : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  return
}

// -----
// CHECK-LABEL: @insert_store_load_between_implicit_transposeb_and_mmad
func.func @insert_store_load_between_implicit_transposeb_and_mmad(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  // CHECK: %[[TENSORB:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<16x16xf16>
  // CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[STORE:.*]] = hivm.hir.store ins(%[[TENSORB:.*]] : tensor<16x16xf16>) outs(%[[EMPTY0:.*]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%[[STORE:.*]] : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  annotation.mark %1 {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_mmad
func.func @insert_load_between_fixpipe_and_mmad(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {enable_nz2nd} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf32>
  %4 = hivm.hir.mmadL1 ins(%0, %2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %4 : tensor<16x16xf32>
}


// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_vector
func.func @insert_load_between_fixpipe_and_vector(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {enable_nz2nd} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf16>
  %4 = hivm.hir.vexp ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %4 : tensor<16x16xf16>
}

// -----
// CHECK-LABEL: @insert_load_between_fixpipe_and_tensor_extract
func.func @insert_load_between_fixpipe_and_tensor_extract(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> f16 {
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.fixpipe {enable_nz2nd} ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  %3 = tensor.extract %2[%c0, %c0] : tensor<16x16xf16>
  return %3 : f16
}

// -----
// CHECK-LABEL: @insert_load_between_store_and_vector
func.func @insert_load_between_store_and_vector(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = hivm.hir.store ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%{{.*}} : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) init_out_buffer = false  may_implicit_transpose_with_last_axis = false -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x16xf16>
  %4 = hivm.hir.vexp ins(%2 : tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %4 : tensor<16x16xf16>
}

// -----
// CHECK-LABEL: @insert_load_between_vector_and_load
func.func @insert_load_between_vector_and_load(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
  %1 = tensor.empty() : tensor<16x16xf16>
  %2 = hivm.hir.vexp ins(%0 : tensor<16x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK: %[[LOAD:.*]] = hivm.hir.store ins(%{{.*}} : tensor<16x16xf16>) outs(%[[EMPTY1:.*]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %4 = hivm.hir.load ins(%3 : tensor<16x16xf16>) outs(%2 : tensor<16x16xf16>) init_out_buffer = false -> tensor<16x16xf16>
  return %4 : tensor<16x16xf16>
}
// -----
// CHECK-LABEL: @collapse
func.func @collapse(%arg0: memref<2x8x16xf16>, %arg1: memref<16x16xf16>) -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x8x16xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
  %2 = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape
  %collapsed = tensor.collapse_shape %0 [[0, 1], [2]] : tensor<2x8x16xf16> into tensor<16x16xf16>
  // CHECK: %[[STORE:.*]] = hivm.hir.store ins(%[[COLLAPSE]] : tensor<16x16xf16>)
  // CHECK: %[[LOAD:.*]] =  hivm.hir.load ins(%[[STORE]] : tensor<16x16xf16>)
  annotation.mark %collapsed {maybeUnCollapsibleReshape} : tensor<16x16xf16>
  %3 = hivm.hir.mmadL1 ins(%collapsed, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}
