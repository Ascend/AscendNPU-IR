// RUN: bishengir-opt --normalize-vector --split-input-file %s | FileCheck %s

func.func @vector_mask_same_multi_reduce_f32_add(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_f32_add
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <add>

func.func @vector_mask_diff_multi_reduce_f32_add(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <add>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_f32_add
// CHECK:	%[[FALSE:.+]] = arith.constant dense<0.000000e+00>
// CHECK: 	%[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:	%[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:	%[[TRUE:.+]] = vector.transfer_read
// CHECK:	%[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:	vector.reduction <add>

func.func @vector_mask_same_multi_reduce_f32_mul(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = vector.mask %0 { vector.multi_reduction <mul>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_f32_mul
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <mul>

func.func @vector_mask_diff_multi_reduce_f32_mul(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <mul>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_f32_mul
// CHECK:       %[[FALSE:.+]] = arith.constant dense<1.000000e+00>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <mul>

func.func @vector_mask_same_multi_reduce_f32_minnumf(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = vector.mask %0 { vector.multi_reduction <minnumf>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_f32_minnumf
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <minnumf>

func.func @vector_mask_diff_multi_reduce_f32_minnumf(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <minnumf>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_f32_minnumf
// CHECK:       %[[FALSE:.+]] = arith.constant dense<0x7F800000>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <minnumf>

func.func @vector_mask_same_multi_reduce_f32_maxnumf(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = vector.mask %0 { vector.multi_reduction <maxnumf>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_f32_maxnumf
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <maxnumf>

func.func @vector_mask_diff_multi_reduce_f32_maxnumf(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <maxnumf>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<f32>
  vector.transfer_write %5, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_f32_maxnumf
// CHECK:       %[[FALSE:.+]] = arith.constant dense<0xFF800000>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <maxnumf>

func.func @vector_mask_same_multi_reduce_i32_add(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_i32_add
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <add>

func.func @vector_mask_diff_multi_reduce_i32_add(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <add>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_i32_add
// CHECK:	%[[FALSE:.+]] = arith.constant dense<0>
// CHECK: 	%[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:	%[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:	%[[TRUE:.+]] = vector.transfer_read
// CHECK:	%[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:	vector.reduction <add>

func.func @vector_mask_same_multi_reduce_i32_mul(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  %4 = vector.mask %0 { vector.multi_reduction <mul>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_i32_mul
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <mul>

func.func @vector_mask_diff_multi_reduce_i32_mul(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <mul>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_i32_mul
// CHECK:       %[[FALSE:.+]] = arith.constant dense<1>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <mul>

func.func @vector_mask_same_multi_reduce_i32_minsi(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  %4 = vector.mask %0 { vector.multi_reduction <minsi>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_i32_minsi
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <minsi>

func.func @vector_mask_diff_multi_reduce_i32_minsi(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <minsi>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_i32_minsi
// CHECK:       %[[FALSE:.+]] = arith.constant dense<2147483647>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <minsi>

func.func @vector_mask_same_multi_reduce_i32_maxsi(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  %4 = vector.mask %0 { vector.multi_reduction <maxsi>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_i32_maxsi
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <maxsi>

func.func @vector_mask_diff_multi_reduce_i32_maxsi(%arg0: memref<64xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %m0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %cst : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  // CHECK-NOT: vector.extractelement
  %4 = vector.mask %m0 { vector.multi_reduction <maxsi>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_diff_multi_reduce_i32_maxsi
// CHECK:       %[[FALSE:.+]] = arith.constant dense<-2147483648>
// CHECK:       %[[READMASK:.+]] = vector.constant_mask [32]
// CHECK:       %[[SELMASK:.+]] = vector.constant_mask [16]
// CHECK:       %[[TRUE:.+]] = vector.transfer_read
// CHECK:       %[[SEL:.+]] = arith.select %[[SELMASK]], %[[TRUE]], %[[FALSE]]
// CHECK:       vector.reduction <maxsi>

func.func @vector_mask_same_multi_reduce_nontrivial_brc(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %5 = vector.broadcast %4 : f32 to vector<64xf32>
  vector.transfer_write %5, %arg2[%c0] : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: @vector_mask_same_multi_reduce_nontrivial_brc
// CHECK-NOT: vector.extractelement
// CHECK: vector.reduction <add>
// CHECK: %[[VAL:.*]] = builtin.unrealized_conversion_cast %{{.*}} : f32 to vector<f32>
// CHECK: vector.broadcast %[[VAL]] : vector<f32> to vector<64xf32>