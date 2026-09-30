// RUN: bishengir-opt -split-input-file -rewrite-slice-op-to-memory-ops %s | FileCheck %s

// CHECK-LABEL: @extractSlice
// CHECK-NOT: tensor.extract_slice
// CHECK-DAG: %[[RANGE_6:.*]] = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
// CHECK-DAG: %[[RANGE_2:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<6> : tensor<2xi32>
// CHECK-DAG: %[[MUL:.*]] = arith.muli %[[RANGE_2]], %[[CST_6]] : tensor<2xi32>
// CHECK-DAG: %[[EXPAND_6:.*]] = tt.expand_dims %[[RANGE_6]] {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
// CHECK-DAG: %[[EXPAND_2:.*]] = tt.expand_dims %[[MUL]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG: %[[BROADCAST_6:.*]] = tt.broadcast %[[EXPAND_6]] : tensor<1x6xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[BROADCAST_2:.*]] = tt.broadcast %[[EXPAND_2]] : tensor<2x1xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BROADCAST_6]], %[[BROADCAST_2]] : tensor<2x6xi32>
// CHECK-DAG: %[[SPLAT:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[PTR:.*]] = tt.addptr %[[SPLAT]], %[[ADD]] : tensor<2x6x!tt.ptr<f32>>, tensor<2x6xi32>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR]] : tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[LOAD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: tt.store %{{.*}}, %[[TRUNC]] : tensor<2x6x!tt.ptr<f16>>
tt.func @extractSlice(%src: !tt.ptr<f32>, %dst: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %2 = arith.constant dense<6> : tensor<4xi32>
  %3 = arith.muli %1, %2 : tensor<4xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<4x6xi32>
  %7 = tt.broadcast %5 : tensor<4x1xi32> -> tensor<4x6xi32>
  %8 = arith.addi %6, %7 : tensor<4x6xi32>
  %9 = tt.splat %src : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  %load = tt.load %10 : tensor<4x6x!tt.ptr<f32>>
  %trunc = arith.truncf %load : tensor<4x6xf32> to tensor<4x6xf16>

  %11 = tensor.extract_slice %trunc[0, 0] [2, 6] [1, 1] : tensor<4x6xf16> to tensor<2x6xf16>

  %12 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %13 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %14 = arith.constant dense<6> : tensor<2xi32>
  %15 = arith.muli %13, %14 : tensor<2xi32>
  %16 = tt.expand_dims %12 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %17 = tt.expand_dims %15 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %18 = tt.broadcast %16 : tensor<1x6xi32> -> tensor<2x6xi32>
  %19 = tt.broadcast %17 : tensor<2x1xi32> -> tensor<2x6xi32>
  %20 = arith.addi %18, %19 : tensor<2x6xi32>
  %21 = tt.splat %dst : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %22 = tt.addptr %21, %20 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  tt.store %22, %11 : tensor<2x6x!tt.ptr<f16>>

  tt.return
}

// -----

// CHECK-LABEL: @extractSliceBinaryOp
// CHECK-NOT: tensor.extract_slice
// CHECK-DAG: %[[RANGE_6:.*]] = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
// CHECK-DAG: %[[RANGE_2:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<6> : tensor<2xi32>
// CHECK-DAG: %[[MUL:.*]] = arith.muli %[[RANGE_2]], %[[CST_6]] : tensor<2xi32>
// CHECK-DAG: %[[EXPAND_6:.*]] = tt.expand_dims %[[RANGE_6]] {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
// CHECK-DAG: %[[EXPAND_2:.*]] = tt.expand_dims %[[MUL]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG: %[[BROADCAST_6:.*]] = tt.broadcast %[[EXPAND_6]] : tensor<1x6xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[BROADCAST_2:.*]] = tt.broadcast %[[EXPAND_2]] : tensor<2x1xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[ADD_BROADCASTS:.*]] = arith.addi %[[BROADCAST_6]], %[[BROADCAST_2]] : tensor<2x6xi32>
// CHECK-DAG: %[[SPLAT:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[PTR:.*]] = tt.addptr %[[SPLAT]], %[[ADD_BROADCASTS]] : tensor<2x6x!tt.ptr<f32>>, tensor<2x6xi32>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR]] : tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[LOAD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[TRUNC]], %[[TRUNC]] : tensor<2x6xf16>
// CHECK: tt.store %{{.*}}, %[[ADD]] : tensor<2x6x!tt.ptr<f16>>
tt.func @extractSliceBinaryOp(%src: !tt.ptr<f32>, %dst: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %2 = arith.constant dense<6> : tensor<4xi32>
  %3 = arith.muli %1, %2 : tensor<4xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<4x6xi32>
  %7 = tt.broadcast %5 : tensor<4x1xi32> -> tensor<4x6xi32>
  %8 = arith.addi %6, %7 : tensor<4x6xi32>
  %9 = tt.splat %src : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  %load = tt.load %10 : tensor<4x6x!tt.ptr<f32>>
  %trunc = arith.truncf %load : tensor<4x6xf32> to tensor<4x6xf16>
  %add = arith.addf %trunc, %trunc : tensor<4x6xf16>

  %11 = tensor.extract_slice %add[0, 0] [2, 6] [1, 1] : tensor<4x6xf16> to tensor<2x6xf16>

  %12 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %13 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %14 = arith.constant dense<6> : tensor<2xi32>
  %15 = arith.muli %13, %14 : tensor<2xi32>
  %16 = tt.expand_dims %12 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %17 = tt.expand_dims %15 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %18 = tt.broadcast %16 : tensor<1x6xi32> -> tensor<2x6xi32>
  %19 = tt.broadcast %17 : tensor<2x1xi32> -> tensor<2x6xi32>
  %20 = arith.addi %18, %19 : tensor<2x6xi32>
  %21 = tt.splat %dst : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %22 = tt.addptr %21, %20 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  tt.store %22, %11 : tensor<2x6x!tt.ptr<f16>>

  tt.return
}

// -----

// CHECK-LABEL: @extractSliceTensorPtr
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK: %[[SRC_PTR:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}}[%[[C2_I32]], %{{.*}}] {{.*}} : <tensor<2x6xf32>>
// CHECK: %[[LOAD:.*]] = tt.load %[[SRC_PTR]] : !tt.ptr<tensor<2x6xf32>>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[LOAD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: tt.store %{{.*}}, %[[TRUNC]] : !tt.ptr<tensor<2x6xf16>>
tt.func @extractSliceTensorPtr(%src: !tt.ptr<f32>, %dst: !tt.ptr<f16>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32
  %4 = arith.constant 2 : i64

  %srcPtr = tt.make_tensor_ptr %src, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  %load = tt.load %srcPtr : !tt.ptr<tensor<4x6xf32>>
  %trunc = arith.truncf %load : tensor<4x6xf32> to tensor<4x6xf16>

  %res = tensor.extract_slice %trunc[2, 0] [2, 6] [1, 1] : tensor<4x6xf16> to tensor<2x6xf16>

  %dstPtr = tt.make_tensor_ptr %dst, [%4, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x6xf16>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<2x6xf16>>

  tt.return
}

// -----

// CHECK-LABEL: @extractSliceTensorPtrOffset
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK: %[[OFFSET:.*]] = arith.addi %{{.*}}, %[[C2_I32]] : i32
// CHECK: %[[SRC_PTR:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}}[%[[OFFSET]], %{{.*}}] {{.*}} : <tensor<2x6xf32>>
// CHECK: %[[LOAD:.*]] = tt.load %[[SRC_PTR]] : !tt.ptr<tensor<2x6xf32>>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[LOAD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: tt.store %{{.*}}, %[[TRUNC]] : !tt.ptr<tensor<2x6xf16>>
tt.func @extractSliceTensorPtrOffset(%src: !tt.ptr<f32>, %dst: !tt.ptr<f16>, %offset: i32) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32
  %4 = arith.constant 2 : i64

  %srcPtr = tt.make_tensor_ptr %src, [%0, %1], [%1, %2], [%offset, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  %load = tt.load %srcPtr : !tt.ptr<tensor<4x6xf32>>
  %trunc = arith.truncf %load : tensor<4x6xf32> to tensor<4x6xf16>

  %res = tensor.extract_slice %trunc[2, 0] [2, 6] [1, 1] : tensor<4x6xf16> to tensor<2x6xf16>

  %dstPtr = tt.make_tensor_ptr %dst, [%4, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x6xf16>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<2x6xf16>>

  tt.return
}

// -----
// CHECK-LABEL: @extractSlicePreserveMask
// CHECK-NOT: tensor.extract_slice
// CHECK-DAG: %[[RANGE_6:.*]] = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
// CHECK-DAG: %[[RANGE_2:.*]] = tt.make_range {end = 4 : i32, start = 2 : i32} : tensor<2xi32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<6> : tensor<2xi32>
// CHECK-DAG: %[[CST_5:.*]] = arith.constant dense<5> : tensor<6xi32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<3> : tensor<2xi32>
// CHECK-DAG: %[[MUL:.*]] = arith.muli %[[RANGE_2]], %[[CST_6]] : tensor<2xi32>
// CHECK-DAG: %[[EXPAND_6:.*]] = tt.expand_dims %[[RANGE_6]] {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
// CHECK-DAG: %[[EXPAND_2:.*]] = tt.expand_dims %[[MUL]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG: %[[BROADCAST_6:.*]] = tt.broadcast %[[EXPAND_6]] : tensor<1x6xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[BROADCAST_2:.*]] = tt.broadcast %[[EXPAND_2]] : tensor<2x1xi32> -> tensor<2x6xi32>
// CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BROADCAST_6]], %[[BROADCAST_2]] : tensor<2x6xi32>
// CHECK-DAG: %[[SPLAT:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[PTR:.*]] = tt.addptr %[[SPLAT]], %[[ADD]] : tensor<2x6x!tt.ptr<f32>>, tensor<2x6xi32>
// CHECK-DAG: %[[MASK_RANGE_2:.*]] = tt.make_range {end = 4 : i32, start = 2 : i32} : tensor<2xi32>
// CHECK-DAG: %[[CMP_2:.*]] = arith.cmpi slt, %[[MASK_RANGE_2]], %[[CST_3]] : tensor<2xi32>
// CHECK-DAG: %[[MASK_EXPAND_2:.*]] = tt.expand_dims %[[CMP_2]] {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
// CHECK-DAG: %[[MASK_BROADCAST_2:.*]] = tt.broadcast %[[MASK_EXPAND_2]] : tensor<2x1xi1> -> tensor<2x6xi1>
// CHECK-DAG: %[[CMP_6:.*]] = arith.cmpi slt, %[[RANGE_6]], %[[CST_5]] : tensor<6xi32>
// CHECK-DAG: %[[MASK_EXPAND_6:.*]] = tt.expand_dims %[[CMP_6]] {axis = 0 : i32} : tensor<6xi1> -> tensor<1x6xi1>
// CHECK-DAG: %[[MASK_BROADCAST_6:.*]] = tt.broadcast %[[MASK_EXPAND_6]] : tensor<1x6xi1> -> tensor<2x6xi1>
// CHECK-DAG: %[[MASK:.*]] = arith.andi %[[MASK_BROADCAST_2]], %[[MASK_BROADCAST_6]] : tensor<2x6xi1>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR]], %[[MASK]] : tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[LOAD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: tt.store %{{.*}}, %[[TRUNC]] : tensor<2x6x!tt.ptr<f16>>
tt.func @extractSlicePreserveMask(%src: !tt.ptr<f32>, %dst: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %2 = arith.constant dense<6> : tensor<4xi32>
  %3 = arith.muli %1, %2 : tensor<4xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<4x6xi32>
  %7 = tt.broadcast %5 : tensor<4x1xi32> -> tensor<4x6xi32>
  %8 = arith.addi %6, %7 : tensor<4x6xi32>
  %cst_3 = arith.constant dense<3> : tensor<4xi32>
  %cst_5 = arith.constant dense<5> : tensor<6xi32>
  %cmp1 = arith.cmpi slt, %1, %cst_3 : tensor<4xi32>
  %cmp2 = arith.cmpi slt, %0, %cst_5 : tensor<6xi32>
  %exp1 = tt.expand_dims %cmp1 {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %exp2 = tt.expand_dims %cmp2 {axis = 0 : i32} : tensor<6xi1> -> tensor<1x6xi1>
  %broadcast1 = tt.broadcast %exp1 : tensor<4x1xi1> -> tensor<4x6xi1>
  %broadcast2 = tt.broadcast %exp2 : tensor<1x6xi1> -> tensor<4x6xi1>
  %mask = arith.andi %broadcast1, %broadcast2 : tensor<4x6xi1>
  %9 = tt.splat %src : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  %load = tt.load %10, %mask : tensor<4x6x!tt.ptr<f32>>
  %trunc = arith.truncf %load : tensor<4x6xf32> to tensor<4x6xf16>

  %11 = tensor.extract_slice %trunc[2, 0] [2, 6] [1, 1] : tensor<4x6xf16> to tensor<2x6xf16>

  %12 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %13 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %14 = arith.constant dense<6> : tensor<2xi32>
  %15 = arith.muli %13, %14 : tensor<2xi32>
  %16 = tt.expand_dims %12 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %17 = tt.expand_dims %15 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %18 = tt.broadcast %16 : tensor<1x6xi32> -> tensor<2x6xi32>
  %19 = tt.broadcast %17 : tensor<2x1xi32> -> tensor<2x6xi32>
  %20 = arith.addi %18, %19 : tensor<2x6xi32>
  %21 = tt.splat %dst : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %22 = tt.addptr %21, %20 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  tt.store %22, %11 : tensor<2x6x!tt.ptr<f16>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSlice
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : tensor<2x6x!tt.ptr<f16>>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : tensor<4x6x!tt.ptr<f16>>
// CHECK: %[[EXT2:.*]] = arith.extf %[[LOAD2]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[MASK_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[IN_SLICE:.*]] = arith.andi %[[LT]], %[[GE]] : tensor<4xi1>
// CHECK: %[[IN_SLICE_2D:.*]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x6xi1>
// CHECK: %[[STORE_MASK:.*]] = arith.xori %[[IN_SLICE_2D]], %{{.*}} : tensor<4x6xi1>
// CHECK: tt.store %{{.*}}, %[[EXT2]], %[[STORE_MASK]] : tensor<4x6x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %[[EXT1]] : tensor<2x6x!tt.ptr<f32>>
tt.func @insertSlice(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %2 = arith.constant dense<6> : tensor<2xi32>
  %3 = arith.muli %1, %2 : tensor<2xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<2x6xi32>
  %7 = tt.broadcast %5 : tensor<2x1xi32> -> tensor<2x6xi32>
  %8 = arith.addi %6, %7 : tensor<2x6xi32>
  %9 = tt.splat %src1 : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %10 = tt.addptr %9, %8 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  %load1 = tt.load %10 : tensor<2x6x!tt.ptr<f16>>

  %11 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %13 = arith.constant dense<6> : tensor<4xi32>
  %14 = arith.muli %12, %13 : tensor<4xi32>
  %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %16 = tt.expand_dims %14 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %17 = tt.broadcast %15 : tensor<1x6xi32> -> tensor<4x6xi32>
  %18 = tt.broadcast %16 : tensor<4x1xi32> -> tensor<4x6xi32>
  %19 = arith.addi %17, %18 : tensor<4x6xi32>
  %20 = tt.splat %src2 : !tt.ptr<f16> -> tensor<4x6x!tt.ptr<f16>>
  %21 = tt.addptr %20, %19 : tensor<4x6x!tt.ptr<f16>>, tensor<4x6xi32>
  %load2 = tt.load %21 : tensor<4x6x!tt.ptr<f16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[1, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %22 = tt.splat %dst : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %23 = tt.addptr %22, %19 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  tt.store %23, %res : tensor<4x6x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSlice
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : tensor<2x6x!tt.ptr<f16>>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : tensor<4x6x!tt.ptr<f16>>
// CHECK: %[[EXT2:.*]] = arith.extf %[[LOAD2]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[MASK_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[IN_SLICE:.*]] = arith.andi %[[LT]], %[[GE]] : tensor<4xi1>
// CHECK: %[[IN_SLICE_2D:.*]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x6xi1>
// CHECK: %[[STORE_MASK:.*]] = arith.xori %[[IN_SLICE_2D]], %{{.*}} : tensor<4x6xi1>
// CHECK: tt.store %{{.*}}, %[[EXT2]], %[[STORE_MASK]] : tensor<4x6x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %[[EXT1]] : tensor<2x6x!tt.ptr<f32>>
// CHECK: %[[ADD2:.*]] = arith.addf %[[EXT2]], %[[EXT2]] : tensor<4x6xf32>
// CHECK: %[[ADD1:.*]] = arith.addf %[[EXT1]], %[[EXT1]] : tensor<2x6xf32>
// CHECK: %[[MASK_RANGE2:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LT2:.*]] = arith.cmpi slt, %[[MASK_RANGE2]], %{{.*}} : tensor<4xi32>
// CHECK: %[[GE2:.*]] = arith.cmpi sge, %[[MASK_RANGE2]], %{{.*}} : tensor<4xi32>
// CHECK: %[[IN_SLICE2:.*]] = arith.andi %[[LT2]], %[[GE2]] : tensor<4xi1>
// CHECK: %[[IN_SLICE_2D2:.*]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x6xi1>
// CHECK: %[[STORE_MASK2:.*]] = arith.xori %[[IN_SLICE_2D2]], %{{.*}} : tensor<4x6xi1>
// CHECK: tt.store %{{.*}}, %[[ADD2]], %[[STORE_MASK2]] : tensor<4x6x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %[[ADD1]] : tensor<2x6x!tt.ptr<f32>>
// CHECK-NOT: tt.store
tt.func @insertSliceDoubleStore(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst1: !tt.ptr<f32>, %dst2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %2 = arith.constant dense<6> : tensor<2xi32>
  %3 = arith.muli %1, %2 : tensor<2xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<2x6xi32>
  %7 = tt.broadcast %5 : tensor<2x1xi32> -> tensor<2x6xi32>
  %8 = arith.addi %6, %7 : tensor<2x6xi32>
  %9 = tt.splat %src1 : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %10 = tt.addptr %9, %8 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  %load1 = tt.load %10 : tensor<2x6x!tt.ptr<f16>>

  %11 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %13 = arith.constant dense<6> : tensor<4xi32>
  %14 = arith.muli %12, %13 : tensor<4xi32>
  %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %16 = tt.expand_dims %14 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %17 = tt.broadcast %15 : tensor<1x6xi32> -> tensor<4x6xi32>
  %18 = tt.broadcast %16 : tensor<4x1xi32> -> tensor<4x6xi32>
  %19 = arith.addi %17, %18 : tensor<4x6xi32>
  %20 = tt.splat %src2 : !tt.ptr<f16> -> tensor<4x6x!tt.ptr<f16>>
  %21 = tt.addptr %20, %19 : tensor<4x6x!tt.ptr<f16>>, tensor<4x6xi32>
  %load2 = tt.load %21 : tensor<4x6x!tt.ptr<f16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[1, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %22 = tt.splat %dst1 : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %23 = tt.addptr %22, %19 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  tt.store %23, %res : tensor<4x6x!tt.ptr<f32>>

  %24 = tt.splat %dst2 : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %25 = tt.addptr %24, %19 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  %res2 = arith.addf %res, %res : tensor<4x6xf32>
  tt.store %25, %res2 : tensor<4x6x!tt.ptr<f32>>

  tt.return
}

// -----

// Verify that adjacent static intervals [1, 4) and [4, 6) are merged before
// constructing the mask for the original store.
// CHECK-LABEL: @insertSliceMergeStaticIntervals
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[ORIGINAL_PTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<8x6x!tt.ptr<f32>>, tensor<8x6xi32>
// CHECK-NEXT: %[[MASK_RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: %[[UPPER:.*]] = arith.constant dense<6> : tensor<8xi32>
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[MASK_RANGE]], %[[UPPER]] : tensor<8xi32>
// CHECK: %[[LOWER:.*]] = arith.constant dense<1> : tensor<8xi32>
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[MASK_RANGE]], %[[LOWER]] : tensor<8xi32>
// CHECK: %[[MERGED_INTERVAL:.*]] = arith.andi %[[LT]], %[[GE]] : tensor<8xi1>
// CHECK-NOT: arith.ori
// CHECK: %[[MERGED_INTERVAL_2D:.*]] = tt.broadcast %{{.*}} : tensor<8x1xi1> -> tensor<8x6xi1>
// CHECK: %[[STORE_MASK:.*]] = arith.xori %[[MERGED_INTERVAL_2D]], %{{.*}} : tensor<8x6xi1>
// CHECK: tt.store %{{.*}}, %{{.*}}, %[[STORE_MASK]] : tensor<8x6x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<3x6x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<2x6x!tt.ptr<f32>>
tt.func @insertSliceMergeStaticIntervals(%dst: !tt.ptr<f32>) {
  %base = arith.constant dense<0.0> : tensor<8x6xf16>
  %slice1 = arith.constant dense<1.0> : tensor<3x6xf16>
  %slice2 = arith.constant dense<2.0> : tensor<2x6xf16>

  %inserted_slice1 = tensor.insert_slice %slice1 into %base[1, 0] [3, 6] [1, 1] : tensor<3x6xf16> into tensor<8x6xf16>
  %inserted_slice2 = tensor.insert_slice %slice2 into %inserted_slice1[4, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<8x6xf16>
  %res = arith.extf %inserted_slice2 : tensor<8x6xf16> to tensor<8x6xf32>

  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %2 = arith.constant dense<6> : tensor<8xi32>
  %3 = arith.muli %1, %2 : tensor<8xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<8x6xi32>
  %7 = tt.broadcast %5 : tensor<8x1xi32> -> tensor<8x6xi32>
  %8 = arith.addi %6, %7 : tensor<8x6xi32>
  %9 = tt.splat %dst : !tt.ptr<f32> -> tensor<8x6x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<8x6x!tt.ptr<f32>>, tensor<8x6xi32>
  tt.store %10, %res : tensor<8x6x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSliceDynamicOffset
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : tensor<2x6x!tt.ptr<f16>>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : tensor<4x6x!tt.ptr<f16>>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[INDEX_OFFSET:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[OFFSET_END:.*]] = arith.addi %[[INDEX_OFFSET]], %{{.*}} : i32
// CHECK: %[[END_SPLAT:.*]] = tt.splat %[[OFFSET_END]] : i32 -> tensor<4xi32>
// CHECK: %[[LT:.*]] = arith.cmpi slt, %{{.*}}, %[[END_SPLAT]] : tensor<4xi32>
// CHECK: %[[START_SPLAT:.*]] = tt.splat %{{.*}} : i32 -> tensor<4xi32>
// CHECK: %[[GE:.*]] = arith.cmpi sge, %{{.*}}, %[[START_SPLAT]] : tensor<4xi32>
// CHECK: %[[STORE_MASK:.*]] = arith.xori %{{.*}}, %{{.*}} : tensor<4x6xi1>
// CHECK: tt.store %{{.*}}, %{{.*}}, %[[STORE_MASK]] : tensor<4x6x!tt.ptr<f32>>
// CHECK: %[[SLICE_OFFSET:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<2xi32>
// CHECK: tt.store %{{.*}}, %[[EXT1]] : tensor<2x6x!tt.ptr<f32>>
tt.func @insertSliceDynamicOffset(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst: !tt.ptr<f32>, %offset: index) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %2 = arith.constant dense<6> : tensor<2xi32>
  %3 = arith.muli %1, %2 : tensor<2xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<2x6xi32>
  %7 = tt.broadcast %5 : tensor<2x1xi32> -> tensor<2x6xi32>
  %8 = arith.addi %6, %7 : tensor<2x6xi32>
  %9 = tt.splat %src1 : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %10 = tt.addptr %9, %8 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  %load1 = tt.load %10 : tensor<2x6x!tt.ptr<f16>>

  %11 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %13 = arith.constant dense<6> : tensor<4xi32>
  %14 = arith.muli %12, %13 : tensor<4xi32>
  %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %16 = tt.expand_dims %14 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %17 = tt.broadcast %15 : tensor<1x6xi32> -> tensor<4x6xi32>
  %18 = tt.broadcast %16 : tensor<4x1xi32> -> tensor<4x6xi32>
  %19 = arith.addi %17, %18 : tensor<4x6xi32>
  %20 = tt.splat %src2 : !tt.ptr<f16> -> tensor<4x6x!tt.ptr<f16>>
  %21 = tt.addptr %20, %19 : tensor<4x6x!tt.ptr<f16>>, tensor<4x6xi32>
  %load2 = tt.load %21 : tensor<4x6x!tt.ptr<f16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[%offset, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %22 = tt.splat %dst : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %23 = tt.addptr %22, %19 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  tt.store %23, %res : tensor<4x6x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSliceTensorPtr
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<2x6xf16>>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<4x6xf16>>
// CHECK: %[[EXT2:.*]] = arith.extf %[[LOAD2]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: tt.store %{{.*}}, %[[EXT2]] : !tt.ptr<tensor<4x6xf32>>
// CHECK: %[[SLICE_DST:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}} : <tensor<2x6xf32>>
// CHECK: tt.store %[[SLICE_DST]], %[[EXT1]] : !tt.ptr<tensor<2x6xf32>>
tt.func @insertSliceTensorPtr(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32
  %4 = arith.constant 2 : i64

  %src1Ptr = tt.make_tensor_ptr %src1, [%4, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x6xf16>>
  %src2Ptr = tt.make_tensor_ptr %src2, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf16>>

  %load1 = tt.load %src1Ptr : !tt.ptr<tensor<2x6xf16>>
  %load2 = tt.load %src2Ptr : !tt.ptr<tensor<4x6xf16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[0, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %dstPtr = tt.make_tensor_ptr %dst, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<4x6xf32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSliceTensorPtrDoubleStore
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<2x6xf16>>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<4x6xf16>>
// CHECK: %[[EXT2:.*]] = arith.extf %[[LOAD2]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: tt.store %{{.*}}, %[[EXT2]] : !tt.ptr<tensor<4x6xf32>>
// CHECK: %[[SLICE_DST:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}} : <tensor<2x6xf32>>
// CHECK: tt.store %[[SLICE_DST]], %[[EXT1]] : !tt.ptr<tensor<2x6xf32>>
// CHECK: %[[ADD2:.*]] = arith.addf %[[EXT2]], %[[EXT2]] : tensor<4x6xf32>
// CHECK: %[[ADD1:.*]] = arith.addf %[[EXT1]], %[[EXT1]] : tensor<2x6xf32>
// CHECK: tt.store %{{.*}}, %[[ADD2]] : !tt.ptr<tensor<4x6xf32>>
// CHECK: %[[SLICE_DST2:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}} : <tensor<2x6xf32>>
// CHECK: tt.store %[[SLICE_DST2]], %[[ADD1]] : !tt.ptr<tensor<2x6xf32>>
tt.func @insertSliceTensorPtrDoubleStore(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst1: !tt.ptr<f32>, %dst2: !tt.ptr<f32>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32
  %4 = arith.constant 2 : i64

  %src1Ptr = tt.make_tensor_ptr %src1, [%4, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x6xf16>>
  %src2Ptr = tt.make_tensor_ptr %src2, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf16>>

  %load1 = tt.load %src1Ptr : !tt.ptr<tensor<2x6xf16>>
  %load2 = tt.load %src2Ptr : !tt.ptr<tensor<4x6xf16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[0, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %dst1Ptr = tt.make_tensor_ptr %dst1, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  tt.store %dst1Ptr, %res : !tt.ptr<tensor<4x6xf32>>

  %res2 = arith.addf %res, %res : tensor<4x6xf32>
  %dst2Ptr = tt.make_tensor_ptr %dst2, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  tt.store %dst2Ptr, %res2 : !tt.ptr<tensor<4x6xf32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSlicePreserveMask
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[CST_3_2:.*]] = arith.constant dense<3> : tensor<2xi32>
// CHECK: %[[CST_3_4:.*]] = arith.constant dense<3> : tensor<4xi32>
// CHECK: %[[CST_5:.*]] = arith.constant dense<5> : tensor<6xi32>
// CHECK: %[[LOAD1:.*]] = tt.load %{{.*}} : tensor<2x6x!tt.ptr<f16>>
// CHECK: %[[OLD_RANGE_6:.*]] = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
// CHECK: %[[OLD_RANGE_4:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LOAD2:.*]] = tt.load %{{.*}} : tensor<4x6x!tt.ptr<f16>>
// CHECK: %[[EXT2:.*]] = arith.extf %[[LOAD2]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK: %[[EXT1:.*]] = arith.extf %[[LOAD1]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[OLD_CMP_6:.*]] = arith.cmpi slt, %[[OLD_RANGE_6]], %[[CST_5]] : tensor<6xi32>
// CHECK: %[[OLD_CMP_4:.*]] = arith.cmpi slt, %[[OLD_RANGE_4]], %[[CST_3_4]] : tensor<4xi32>
// CHECK: %[[OLD_EXPAND_6:.*]] = tt.expand_dims %[[OLD_CMP_6]] {axis = 0 : i32} : tensor<6xi1> -> tensor<1x6xi1>
// CHECK: %[[OLD_EXPAND_4:.*]] = tt.expand_dims %[[OLD_CMP_4]] {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
// CHECK: %[[OLD_BROADCAST_6:.*]] = tt.broadcast %[[OLD_EXPAND_6]] : tensor<1x6xi1> -> tensor<4x6xi1>
// CHECK: %[[OLD_BROADCAST_4:.*]] = tt.broadcast %[[OLD_EXPAND_4]] : tensor<4x1xi1> -> tensor<4x6xi1>
// CHECK: %[[OLD_MASK:.*]] = arith.andi %[[OLD_BROADCAST_6]], %[[OLD_BROADCAST_4]] : tensor<4x6xi1>
// CHECK: %[[MASK_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[MASK_RANGE]], %{{.*}} : tensor<4xi32>
// CHECK: %[[IN_SLICE:.*]] = arith.andi %[[LT]], %[[GE]] : tensor<4xi1>
// CHECK: %[[IN_SLICE_2D:.*]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x6xi1>
// CHECK: %[[INSERT_MASK:.*]] = arith.xori %[[IN_SLICE_2D]], %{{.*}} : tensor<4x6xi1>
// CHECK: %[[STORE_MASK:.*]] = arith.andi %[[INSERT_MASK]], %[[OLD_MASK]] : tensor<4x6xi1>
// CHECK: tt.store %{{.*}}, %[[EXT2]], %[[STORE_MASK]] : tensor<4x6x!tt.ptr<f32>>
// CHECK: %[[CMP_6:.*]] = arith.cmpi slt, %[[OLD_RANGE_6]], %[[CST_5]] : tensor<6xi32>
// CHECK: %[[EXPAND_6:.*]] = tt.expand_dims %[[CMP_6]] {axis = 0 : i32} : tensor<6xi1> -> tensor<1x6xi1>
// CHECK: %[[BROADCAST_6:.*]] = tt.broadcast %[[EXPAND_6]] : tensor<1x6xi1> -> tensor<2x6xi1>
// CHECK: %[[RANGE_2:.*]] = tt.make_range {end = 3 : i32, start = 1 : i32} : tensor<2xi32>
// CHECK: %[[CMP_2:.*]] = arith.cmpi slt, %[[RANGE_2]], %[[CST_3_2]] : tensor<2xi32>
// CHECK: %[[EXPAND_2:.*]] = tt.expand_dims %[[CMP_2]] {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
// CHECK: %[[BROADCAST_2:.*]] = tt.broadcast %[[EXPAND_2]] : tensor<2x1xi1> -> tensor<2x6xi1>
// CHECK: %[[MASK:.*]] = arith.andi %[[BROADCAST_6]], %[[BROADCAST_2]] : tensor<2x6xi1>
// CHECK: tt.store %{{.*}}, %[[EXT1]], %[[MASK]] : tensor<2x6x!tt.ptr<f32>>
tt.func @insertSlicePreserveMask(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %2 = arith.constant dense<6> : tensor<2xi32>
  %3 = arith.muli %1, %2 : tensor<2xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %6 = tt.broadcast %4 : tensor<1x6xi32> -> tensor<2x6xi32>
  %7 = tt.broadcast %5 : tensor<2x1xi32> -> tensor<2x6xi32>
  %8 = arith.addi %6, %7 : tensor<2x6xi32>
  %9 = tt.splat %src1 : !tt.ptr<f16> -> tensor<2x6x!tt.ptr<f16>>
  %10 = tt.addptr %9, %8 : tensor<2x6x!tt.ptr<f16>>, tensor<2x6xi32>
  %load1 = tt.load %10 : tensor<2x6x!tt.ptr<f16>>

  %11 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %13 = arith.constant dense<6> : tensor<4xi32>
  %14 = arith.muli %12, %13 : tensor<4xi32>
  %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %16 = tt.expand_dims %14 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %17 = tt.broadcast %15 : tensor<1x6xi32> -> tensor<4x6xi32>
  %18 = tt.broadcast %16 : tensor<4x1xi32> -> tensor<4x6xi32>
  %19 = arith.addi %17, %18 : tensor<4x6xi32>
  %20 = tt.splat %src2 : !tt.ptr<f16> -> tensor<4x6x!tt.ptr<f16>>
  %21 = tt.addptr %20, %19 : tensor<4x6x!tt.ptr<f16>>, tensor<4x6xi32>
  %load2 = tt.load %21 : tensor<4x6x!tt.ptr<f16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[1, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
  %res = arith.extf %inserted_slice : tensor<4x6xf16> to tensor<4x6xf32>

  %22 = tt.splat %dst : !tt.ptr<f32> -> tensor<4x6x!tt.ptr<f32>>
  %23 = tt.addptr %22, %19 : tensor<4x6x!tt.ptr<f32>>, tensor<4x6xi32>
  %cst_5 = arith.constant dense<5> : tensor<6xi32>
  %cst_3 = arith.constant dense<3> : tensor<4xi32>
  %cmp_5 = arith.cmpi slt, %11, %cst_5 : tensor<6xi32>
  %cmp_3 = arith.cmpi slt, %12, %cst_3 : tensor<4xi32>
  %expand_5 = tt.expand_dims %cmp_5 {axis = 0 : i32} : tensor<6xi1> -> tensor<1x6xi1>
  %expand_3 = tt.expand_dims %cmp_3 {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %broadcast_5 = tt.broadcast %expand_5 : tensor<1x6xi1> -> tensor<4x6xi1>
  %broadcast_3 = tt.broadcast %expand_3 : tensor<4x1xi1> -> tensor<4x6xi1>
  %mask = arith.andi %broadcast_5, %broadcast_3 : tensor<4x6xi1>
  tt.store %23, %res, %mask : tensor<4x6x!tt.ptr<f32>>

  tt.return
}
// -----

// CHECK-LABEL: @splitSlice
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[FIRST_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[FIRST_LOAD:.*]] = tt.load %{{.*}} : tensor<12x4x!tt.ptr<f16>>
// CHECK: %[[FIRST_EXT:.*]] = arith.extf %[[FIRST_LOAD]] : tensor<12x4xf16> to tensor<12x4xf32>
// CHECK: %[[SECOND_RANGE:.*]] = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
// CHECK: %[[SECOND_LOAD:.*]] = tt.load %{{.*}} : tensor<12x4x!tt.ptr<f16>>
// CHECK: %[[SECOND_EXT:.*]] = arith.extf %[[SECOND_LOAD]] : tensor<12x4xf16> to tensor<12x4xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[FIRST_EXT]], %[[SECOND_EXT]] : tensor<12x4xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[FIRST_EXT]], %[[SECOND_EXT]] : tensor<12x4xf32>
// CHECK: %[[ADD_TRUNC:.*]] = arith.truncf %[[ADD]] : tensor<12x4xf32> to tensor<12x4xf16>
// CHECK: %[[MUL_TRUNC:.*]] = arith.truncf %[[MUL]] : tensor<12x4xf32> to tensor<12x4xf16>
// CHECK: tt.store %{{.*}}, %[[ADD_TRUNC]] : tensor<12x4x!tt.ptr<f16>>
// CHECK: tt.store %{{.*}}, %[[MUL_TRUNC]] : tensor<12x4x!tt.ptr<f16>>
tt.func @splitSlice(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %1 = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32>
  %2 = arith.constant dense<8> : tensor<12xi32>
  %3 = arith.muli %1, %2 : tensor<12xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<12xi32> -> tensor<12x1xi32>
  %6 = tt.broadcast %4 : tensor<1x8xi32> -> tensor<12x8xi32>
  %7 = tt.broadcast %5 : tensor<12x1xi32> -> tensor<12x8xi32>
  %8 = arith.addi %6, %7 : tensor<12x8xi32>
  %9 = tt.splat %src : !tt.ptr<f16> -> tensor<12x8x!tt.ptr<f16>>
  %10 = tt.addptr %9, %8 : tensor<12x8x!tt.ptr<f16>>, tensor<12x8xi32>
  %load = tt.load %10 : tensor<12x8x!tt.ptr<f16>>

  %extf = arith.extf %load : tensor<12x8xf16> to tensor<12x8xf32>
  %slice1 = tensor.extract_slice %extf[0, 0] [12, 4] [1, 1] : tensor<12x8xf32> to tensor<12x4xf32>
  %slice2 = tensor.extract_slice %extf[0, 4] [12, 4] [1, 1] : tensor<12x8xf32> to tensor<12x4xf32>

  %res1 = arith.addf %slice1, %slice2 : tensor<12x4xf32>
  %res2 = arith.mulf %slice1, %slice2 : tensor<12x4xf32>

  %dest = arith.constant dense<0.0> : tensor<12x8xf32>
  %inserted_slice1 = tensor.insert_slice %res1 into %dest[0, 0] [12, 4] [1, 1] : tensor<12x4xf32> into tensor<12x8xf32>
  %inserted_slice2 = tensor.insert_slice %res2 into %inserted_slice1[0, 4] [12, 4] [1, 1] : tensor<12x4xf32> into tensor<12x8xf32>
  %res = arith.truncf %inserted_slice2 : tensor<12x8xf32> to tensor<12x8xf16>

  %11 = tt.splat %dst : !tt.ptr<f16> -> tensor<12x8x!tt.ptr<f16>>
  %12 = tt.addptr %11, %8 : tensor<12x8x!tt.ptr<f16>>, tensor<12x8xi32>
  tt.store %12, %res : tensor<12x8x!tt.ptr<f16>>
  tt.return
}

// -----

// CHECK-LABEL: @splitSliceTensorPtr
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[FIRST_SRC:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}}[%{{.*}}, %{{.*}}] {{.*}} : <tensor<2x6xf16>>
// CHECK: %[[FIRST_LOAD:.*]] = tt.load %[[FIRST_SRC]] : !tt.ptr<tensor<2x6xf16>>
// CHECK: %[[FIRST_EXT:.*]] = arith.extf %[[FIRST_LOAD]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[SECOND_SRC:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}}[%{{.*}}, %{{.*}}] {{.*}} : <tensor<2x6xf16>>
// CHECK: %[[SECOND_LOAD:.*]] = tt.load %[[SECOND_SRC]] : !tt.ptr<tensor<2x6xf16>>
// CHECK: %[[SECOND_EXT:.*]] = arith.extf %[[SECOND_LOAD]] : tensor<2x6xf16> to tensor<2x6xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[FIRST_EXT]], %[[SECOND_EXT]] : tensor<2x6xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[FIRST_EXT]], %[[SECOND_EXT]] : tensor<2x6xf32>
// CHECK: %[[ADD_TRUNC:.*]] = arith.truncf %[[ADD]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: %[[MUL_TRUNC:.*]] = arith.truncf %[[MUL]] : tensor<2x6xf32> to tensor<2x6xf16>
// CHECK: %[[FIRST_DST:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}} : <tensor<2x6xf16>>
// CHECK: tt.store %[[FIRST_DST]], %[[ADD_TRUNC]] : !tt.ptr<tensor<2x6xf16>>
// CHECK: %[[SECOND_DST:.*]] = tt.make_tensor_ptr %{{.*}}, {{.*}} : <tensor<2x6xf16>>
// CHECK: tt.store %[[SECOND_DST]], %[[MUL_TRUNC]] : !tt.ptr<tensor<2x6xf16>>
tt.func @splitSliceTensorPtr(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32

  %srcPtr = tt.make_tensor_ptr %src, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf16>>
  %load = tt.load %srcPtr : !tt.ptr<tensor<4x6xf16>>
  %extf = arith.extf %load : tensor<4x6xf16> to tensor<4x6xf32>

  %slice1 = tensor.extract_slice %extf[0, 0] [2, 6] [1, 1] : tensor<4x6xf32> to tensor<2x6xf32>
  %slice2 = tensor.extract_slice %extf[2, 0] [2, 6] [1, 1] : tensor<4x6xf32> to tensor<2x6xf32>

  %res1 = arith.addf %slice1, %slice2 : tensor<2x6xf32>
  %res2 = arith.mulf %slice1, %slice2 : tensor<2x6xf32>

  %dest = arith.constant dense<0.0> : tensor<4x6xf32>
  %inserted_slice1 = tensor.insert_slice %res1 into %dest[0, 0] [2, 6] [1, 1] : tensor<2x6xf32> into tensor<4x6xf32>
  %inserted_slice2 = tensor.insert_slice %res2 into %inserted_slice1[2, 0] [2, 6] [1, 1] : tensor<2x6xf32> into tensor<4x6xf32>
  %res = arith.truncf %inserted_slice2 : tensor<4x6xf32> to tensor<4x6xf16>

  %dstPtr = tt.make_tensor_ptr %dst, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf16>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<4x6xf16>>

  tt.return
}

// -----

// CHECK-LABEL: tt.func @noChangeExtract
// CHECK-SAME: (%[[SRC1:.*]]: !tt.ptr<f32>, %[[SRC2:.*]]: !tt.ptr<f32>, %[[DST:.*]]: !tt.ptr<f32>) {
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i64
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[SRC1_PTR:.*]] = tt.make_tensor_ptr %[[SRC1]], [%[[C4]], %[[C4]]], [%[[C4]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
// CHECK-NEXT:    %[[SRC2_PTR:.*]] = tt.make_tensor_ptr %[[SRC1]], [%[[C4]], %[[C4]]], [%[[C4]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
// CHECK-NEXT:    %[[LOAD1:.*]] = tt.load %[[SRC1_PTR]] : !tt.ptr<tensor<4x4xf32>>
// CHECK-NEXT:    %[[LOAD2:.*]] = tt.load %[[SRC2_PTR]] : !tt.ptr<tensor<4x4xf32>>
// CHECK-NEXT:    %[[DOT:.*]] = tt.dot %[[LOAD1]], %[[LOAD2]], %[[ZERO]] : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
// CHECK-NEXT:    %[[FIRST_HALF:.*]] = tensor.extract_slice %[[DOT]][0, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:    %[[SECOND_HALF:.*]] = tensor.extract_slice %[[DOT]][2, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:    %[[RES:.*]] = arith.addf %[[FIRST_HALF]], %[[SECOND_HALF]] : tensor<2x4xf32>
// CHECK-NEXT:    %[[DST_PTR:.*]] = tt.make_tensor_ptr %[[DST]], [%[[C4]], %[[C4]]], [%[[C4]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<2x4xf32>>
// CHECK-NEXT:    tt.store %[[DST_PTR]], %[[RES]] : !tt.ptr<tensor<2x4xf32>>
// CHECK-NEXT:    tt.return
// CHECK-NEXT:  }
tt.func @noChangeExtract(%src1: !tt.ptr<f32>, %src2: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32

  %src1Ptr = tt.make_tensor_ptr %src1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x4xf32>>
  %src2Ptr = tt.make_tensor_ptr %src1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x4xf32>>
  %load1 = tt.load %src1Ptr : !tt.ptr<tensor<4x4xf32>>
  %load2 = tt.load %src2Ptr : !tt.ptr<tensor<4x4xf32>>
  %cst_0 = arith.constant dense<0.0> : tensor<4x4xf32>
  %dot = tt.dot %load1, %load2, %cst_0 : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>

  %firstHalf = tensor.extract_slice %dot[0, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
  %secondHalf = tensor.extract_slice %dot[2, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>

  %res = arith.addf %firstHalf, %secondHalf : tensor<2x4xf32>

  %dstPtr = tt.make_tensor_ptr %dst, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x4xf32>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<2x4xf32>>
  tt.return
}

// -----

// CHECK-LABEL: tt.func @noChangeInsert
// CHECK-SAME: (%[[SRC1:.*]]: !tt.ptr<f16>, %[[SRC2:.*]]: !tt.ptr<f16>, %[[DST:.*]]: !tt.ptr<f32>) {
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i64
// CHECK-NEXT:    %[[C6:.*]] = arith.constant 6 : i64
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i64
// CHECK-NEXT:    %[[SRC1_PTR:.*]] = tt.make_tensor_ptr %[[SRC1]], [%[[C2]], %[[C6]]], [%[[C6]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<2x6xf16>>
// CHECK-NEXT:    %[[SRC2_PTR:.*]] = tt.make_tensor_ptr %[[SRC2]], [%[[C4]], %[[C6]]], [%[[C6]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<4x6xf16>>
// CHECK-NEXT:    %[[LOAD1:.*]] = tt.load %[[SRC1_PTR]] : !tt.ptr<tensor<2x6xf16>>
// CHECK-NEXT:    %[[LOAD2:.*]] = tt.load %[[SRC2_PTR]] : !tt.ptr<tensor<4x6xf16>>
// CHECK-NEXT:    %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[LOAD1]] into %[[LOAD2]][0, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>
// CHECK-NEXT:    %[[ADD:.*]] = arith.addf %[[LOAD2]], %[[INSERTED_SLICE]] : tensor<4x6xf16>
// CHECK-NEXT:    %[[RES:.*]] = arith.extf %[[ADD]] : tensor<4x6xf16> to tensor<4x6xf32>
// CHECK-NEXT:    %[[DST_PTR:.*]] = tt.make_tensor_ptr %[[DST]], [%[[C4]], %[[C6]]], [%[[C6]], %[[C1]]], [%[[C0]], %[[C0]]] {order = array<i32: 1, 0>} : <tensor<4x6xf32>>
// CHECK-NEXT:    tt.store %[[DST_PTR]], %[[RES]] : !tt.ptr<tensor<4x6xf32>>
// CHECK-NEXT:    tt.return
// CHECK-NEXT:  }
tt.func @noChangeInsert(%src1: !tt.ptr<f16>, %src2: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 6 : i64
  %2 = arith.constant 1 : i64
  %3 = arith.constant 0 : i32
  %4 = arith.constant 2 : i64

  %src1Ptr = tt.make_tensor_ptr %src1, [%4, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x6xf16>>
  %src2Ptr = tt.make_tensor_ptr %src2, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf16>>

  %load1 = tt.load %src1Ptr : !tt.ptr<tensor<2x6xf16>>
  %load2 = tt.load %src2Ptr : !tt.ptr<tensor<4x6xf16>>

  %inserted_slice = tensor.insert_slice %load1 into %load2[0, 0] [2, 6] [1, 1] : tensor<2x6xf16> into tensor<4x6xf16>

  %add = arith.addf %load2, %inserted_slice : tensor<4x6xf16>
  %res = arith.extf %add : tensor<4x6xf16> to tensor<4x6xf32>

  %dstPtr = tt.make_tensor_ptr %dst, [%0, %1], [%1, %2], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<4x6xf32>>
  tt.store %dstPtr, %res : !tt.ptr<tensor<4x6xf32>>
  tt.return
}
