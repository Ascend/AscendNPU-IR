// RUN: bishengir-opt -hfusion-generic-unroller %s | FileCheck %s

// CHECK-LABEL: func.func @argmax_uint_unroll
func.func @argmax_uint_unroll(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %c0_i16 = arith.constant 0 : i16
  %0 = tensor.empty() : tensor<2xi32>
  %1 = hfusion.arange offset[%c0] strides[%c1] outs(%0 : tensor<2xi32>) -> tensor<2xi32>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2], strides: [1] : memref<?xi16> to memref<2xi16, strided<[1]>>
  %alloc = memref.alloc() : memref<2xi16>
  memref.copy %reinterpret_cast, %alloc : memref<2xi16, strided<[1]>> to memref<2xi16>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<2xi16>
  %3 = tensor.empty() : tensor<i16>
  %4 = linalg.fill ins(%c0_i16 : i16) outs(%3 : tensor<i16>) -> tensor<i16>
  %5 = tensor.empty() : tensor<i32>
  %6 = linalg.fill ins(%c2147483647_i32 : i32) outs(%5 : tensor<i32>) -> tensor<i32>
  %7:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = true} <max> ins(%2, %1 : tensor<2xi16>, tensor<2xi32>) outs(%4, %6 : tensor<i16>, tensor<i32>) dimensions = [0]  -> tensor<i16>, tensor<i32>
  %extracted = tensor.extract %7#1[] : tensor<i32>
  %8 = tensor.empty() : tensor<1xi32>
  %inserted = tensor.insert %extracted into %8[%c0] : tensor<1xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
  return
}

// CHECK-NOT: hfusion.reduce_with_index

// CHECK: %[[C2:.*]] = arith.constant 2 : index

// CHECK: %[[INIT_VAL:.*]] = tensor.extract %[[SRC:.*]][%[[C0:.*]]] : tensor<2xi16>
// CHECK: %[[INIT_IDX:.*]] = tensor.extract %[[IDX:.*]][%[[C0]]] : tensor<2xi32>

// CHECK: %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C1:.*]] to %[[C2]] step %[[C1]]
// CHECK-SAME: iter_args(%[[ACC:.*]] = %[[INIT_VAL]], %[[ACCIDX:.*]] = %[[INIT_IDX]]) -> (i16, i32) {

// CHECK: %[[ELEM:.*]] = tensor.extract %[[SRC]][%[[IV]]] : tensor<2xi16>
// CHECK: %[[EIDX:.*]] = tensor.extract %[[IDX]][%[[IV]]] : tensor<2xi32>
// CHECK: %[[CMP:.*]] = arith.cmpi ugt, %[[ELEM]], %[[ACC]] : i16
// CHECK: %[[NEWACC:.*]] = arith.select %[[CMP]], %[[ELEM]], %[[ACC]] : i16
// CHECK: %[[NEWIDX:.*]] = arith.select %[[CMP]], %[[EIDX]], %[[ACCIDX]] : i32
// CHECK: scf.yield %[[NEWACC]], %[[NEWIDX]] : i16, i32
// CHECK: }

// CHECK: tensor.insert %[[FOR]]#1 into %{{.*}}[] : tensor<i32>
