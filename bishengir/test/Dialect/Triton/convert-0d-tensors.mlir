// RUN: bishengir-opt -convert-zero-dim-tensors -split-input-file %s | FileCheck %s

// CHECK: @tensorOps
// CHECK-SAME: (%[[SCALAR:.*]]: f32)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[INDEX_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[INSERT:.*]] = tensor.insert %[[SCALAR]] into %[[EMPTY]][%[[INDEX_ZERO]]] : tensor<1xf32>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[INSERT]] : tensor<1xf32> to tensor<1xf16>
// CHECK: %[[INDEX_ZERO_2:.*]] = arith.constant 0 : index
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[TRUNC]][%[[INDEX_ZERO_2]]] : tensor<1xf16>
// CHECK: tt.return %[[EXTRACT]] : f16
tt.func @tensorOps(%scalar: f32) -> f16 {
  %0 = tensor.empty() : tensor<f32>
  %1 = tensor.insert %scalar into %0[] : tensor<f32>
  %2 = arith.truncf %1 : tensor<f32> to tensor<f16>
  %3 = tensor.extract %2[] : tensor<f16>
  tt.return %3 : f16
}

// -----

// CHECK: @forBlockArg
// CHECK-SAME: (%[[SCALAR:.*]]: f32, %[[PTR:.*]]: !tt.ptr<f32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[INDEX_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[START:.*]] = tensor.insert %[[SCALAR]] into %[[EMPTY]][%[[INDEX_ZERO]]] : tensor<1xf32>
// CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ITER_ARG:.*]] = %[[START]]) -> (tensor<1xf32>) {
// CHECK: %[[CST_ONE:.*]] = arith.constant dense<1.0{{.*}}0> : tensor<1xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[ITER_ARG]], %[[CST_ONE]] : tensor<1xf32>
// CHECK: scf.yield %[[ADD]] : tensor<1xf32>
// CHECK: %[[DST_PTR:.*]] = tt.splat %[[PTR]] : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: tt.store %[[DST_PTR]], %[[FOR]] : tensor<1x!tt.ptr<f32>>
tt.func @forBlockArg(%initScalar: f32, %storePtr: !tt.ptr<f32>, %lb: index, %ub: index, %step: index) {
  %0 = tensor.empty() : tensor<f32>
  %1 = tensor.insert %initScalar into %0[] : tensor<f32>

  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg4 = %1) -> (tensor<f32>) {
    %cst = arith.constant dense<1.0> : tensor<f32>
    %next_tensor = arith.addf %arg4, %cst : tensor<f32>
    scf.yield %next_tensor : tensor<f32>
  }

  %dstPtr = tt.splat %storePtr : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  tt.store %dstPtr, %result : tensor<!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @ifResult
// CHECK-SAME: (%[[SRC_PTR_1:.*]]: !tt.ptr<f16>, %[[SRC_PTR_2:.*]]: !tt.ptr<i32>, %[[DST_PTR_1:.*]]: !tt.ptr<bf16>, %[[DST_PTR_2:.*]]: !tt.ptr<f32>, %[[COND:.*]]: i1)
// CHECK: %[[IF_RES:.*]] = scf.if %[[COND]] -> (tensor<1xf16>) {
// CHECK: %[[TRUE_EMPTY:.*]] = tensor.empty() : tensor<1xf16>
// CHECK: %[[TRUE_LOAD:.*]] = tt.load %[[SRC_PTR_1]] : !tt.ptr<f16>
// CHECK: %[[TRUE_INDEX_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[TRUE_INSERT:.*]] = tensor.insert %[[TRUE_LOAD]] into %[[TRUE_EMPTY]][%[[TRUE_INDEX_ZERO]]] : tensor<1xf16>
// CHECK: scf.yield %[[TRUE_INSERT]] : tensor<1xf16>
// CHECK: } else {
// CHECK: %[[FALSE_SPLAT:.*]] = tt.splat %[[SRC_PTR_2]] : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
// CHECK: %[[FALSE_LOAD:.*]] = tt.load %[[FALSE_SPLAT]] : tensor<1x!tt.ptr<i32>>
// CHECK: %[[FALSE_CAST:.*]] = arith.sitofp %[[FALSE_LOAD]] : tensor<1xi32> to tensor<1xf16>
// CHECK: scf.yield %[[FALSE_CAST]] : tensor<1xf16>
// CHECK: scf.if %[[COND]] {
// CHECK: %[[TRUE_EXTF:.*]] = arith.extf %[[IF_RES]] : tensor<1xf16> to tensor<1xf32>
// CHECK: %[[TRUE_TRUNCF:.*]] = arith.truncf %[[TRUE_EXTF]] : tensor<1xf32> to tensor<1xbf16>
// CHECK: %[[TRUE_SPLAT:.*]] = tt.splat %[[DST_PTR_1]] : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>>
// CHECK: tt.store %[[TRUE_SPLAT]], %[[TRUE_TRUNCF]] : tensor<1x!tt.ptr<bf16>>
// CHECK: } else {
// CHECK: %[[FALSE_EXTF:.*]] = arith.extf %[[IF_RES]] : tensor<1xf16> to tensor<1xf32>
// CHECK: %[[FALSE_INDEX_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[FALSE_EXTRACT:.*]] = tensor.extract %[[FALSE_EXTF]][%[[FALSE_INDEX_ZERO]]] : tensor<1xf32>
// CHECK: tt.store %[[DST_PTR_2]], %[[FALSE_EXTRACT]] : !tt.ptr<f32>
tt.func @ifResult(%src1: !tt.ptr<f16>, %src2: !tt.ptr<i32>, %dst1: !tt.ptr<bf16>, %dst2: !tt.ptr<f32>, %cond: i1) {
  %res = scf.if %cond -> (tensor<f16>) {
    %0 = tensor.empty() : tensor<f16>
    %load = tt.load %src1 : !tt.ptr<f16>
    %insert = tensor.insert %load into %0[] : tensor<f16>
    scf.yield %insert : tensor<f16>
  } else {
    %loadPtr = tt.splat %src2 : !tt.ptr<i32> -> tensor<!tt.ptr<i32>>
    %load = tt.load %loadPtr : tensor<!tt.ptr<i32>>
    %return = arith.sitofp %load : tensor<i32> to tensor<f16>
    scf.yield %return : tensor<f16>
  }

  scf.if %cond {
    %extf = arith.extf %res : tensor<f16> to tensor<f32>
    %storeVal = arith.truncf %extf : tensor<f32> to tensor<bf16>
    %storePtr = tt.splat %dst1 : !tt.ptr<bf16> -> tensor<!tt.ptr<bf16>>
    tt.store %storePtr, %storeVal : tensor<!tt.ptr<bf16>>
    scf.yield
  } else {
    %extf = arith.extf %res : tensor<f16> to tensor<f32>
    %storeVal = tensor.extract %extf[] : tensor<f32>
    tt.store %dst2, %storeVal : !tt.ptr<f32>
    scf.yield
  }

  tt.return
}

// -----

// CHECK-LABEL: @memoryOps
// CHECK-SAME: (%[[SRC_PTR:.*]]: !tt.ptr<f32>, %[[DST_PTR:.*]]: !tt.ptr<f32>, %[[SRC_OFF:.*]]: i32, %[[DST_OFF:.*]]: i32)
// CHECK: %[[SRC_PTR_SPLAT:.*]] = tt.splat %[[SRC_PTR]] : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[SRC_OFF_SPLAT:.*]] = tt.splat %[[SRC_OFF]] : i32 -> tensor<1xi32>
// CHECK: %[[LOAD_PTR:.*]] = tt.addptr %[[SRC_PTR_SPLAT]], %[[SRC_OFF_SPLAT]] : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
// CHECK: %[[LOAD:.*]] = tt.load %[[LOAD_PTR]] : tensor<1x!tt.ptr<f32>>
// CHECK: %[[DST_PTR_SPLAT:.*]] = tt.splat %[[DST_PTR]] : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[DST_ENLARGE:.*]] = tt.reshape %[[DST_PTR_SPLAT]] : tensor<1x!tt.ptr<f32>> -> tensor<1x1x1x!tt.ptr<f32>>
// CHECK: %[[DST_SHRINK:.*]] = tt.reshape %[[DST_ENLARGE]] : tensor<1x1x1x!tt.ptr<f32>> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[DST_OFF_SPLAT:.*]] = tt.splat %[[DST_OFF]] : i32 -> tensor<1xi32>
// CHECK: %[[DST_OFF_BROADCAST:.*]] = tt.broadcast %[[DST_OFF_SPLAT]] : tensor<1xi32> -> tensor<1xi32>
// CHECK: %[[STORE_PTR:.*]] = tt.addptr %[[DST_SHRINK]], %[[DST_OFF_BROADCAST]] : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
// CHECK: tt.store %[[STORE_PTR]], %[[LOAD]] : tensor<1x!tt.ptr<f32>>
tt.func @memoryOps(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %srcOffset: i32, %dstOffset: i32) {
  %srcSplat = tt.splat %src : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  %srcOffsetSplat = tt.splat %srcOffset : i32 -> tensor<i32>
  %loadPtr = tt.addptr %srcSplat, %srcOffsetSplat : tensor<!tt.ptr<f32>>, tensor<i32>
  %load = tt.load %loadPtr : tensor<!tt.ptr<f32>>

  %dstSplat = tt.splat %dst : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  %dstReshape0 = tt.reshape %dstSplat : tensor<!tt.ptr<f32>> -> tensor<1x1x1x!tt.ptr<f32>>
  %dstReshape1 = tt.reshape %dstReshape0 : tensor<1x1x1x!tt.ptr<f32>> -> tensor<!tt.ptr<f32>>
  %dstOffsetSplat = tt.splat %dstOffset : i32 -> tensor<i32>
  %dstOffsetBroadcast = tt.broadcast %dstOffsetSplat : tensor<i32> -> tensor<i32>
  %storePtr = tt.addptr %dstReshape1, %dstOffsetBroadcast : tensor<!tt.ptr<f32>>, tensor<i32>
  tt.store %storePtr, %load : tensor<!tt.ptr<f32>>
  tt.return
}