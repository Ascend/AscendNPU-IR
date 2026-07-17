// RUN: not bishengir-compile %s \
// RUN:   --enable-auto-multi-buffer=True \
// RUN:   --enable-hfusion-compile=true \
// RUN:   --enable-triton-kernel-compile=true \
// RUN:   --mlir-print-ir-after-failure 2>&1 | FileCheck %s

// CHECK-COUNT-1: // -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
// CHECK-NOT: // -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
// CHECK: error: ub overflow
// CHECK: [ERROR] Failed to run BiShengIR pipeline

#loc = loc("print_failure_ir_final_retry.py":35:0)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @triton_add(%arg0: memref<?xi8> loc("print_failure_ir_final_retry.py":35:0), %arg1: memref<?xi8> loc("print_failure_ir_final_retry.py":35:0), %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("print_failure_ir_final_retry.py":35:0), %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("print_failure_ir_final_retry.py":35:0), %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} loc("print_failure_ir_final_retry.py":35:0), %arg5: i32 loc("print_failure_ir_final_retry.py":35:0), %arg6: i32 loc("print_failure_ir_final_retry.py":35:0), %arg7: i32 loc("print_failure_ir_final_retry.py":35:0), %arg8: i32 loc("print_failure_ir_final_retry.py":35:0), %arg9: i32 loc("print_failure_ir_final_retry.py":35:0), %arg10: i32 loc("print_failure_ir_final_retry.py":35:0)) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c32768_i32 = arith.constant 32768 : i32 loc(#loc1)
    %c20480_i32 = arith.constant 20480 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc3)
    %c2_i32 = arith.constant 2 : i32 loc(#loc3)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %0 = arith.muli %arg8, %c32768_i32 : i32 loc(#loc1)
    scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %1 = arith.muli %arg11, %c20480_i32 : i32 loc(#loc2)
      %2 = arith.addi %0, %1 : i32 loc(#loc4)
      %3 = arith.index_cast %2 : i32 to index loc(#loc4)
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [20480], strides: [1] : memref<?xf32> to memref<20480xf32, strided<[1], offset: ?>> loc(#loc5)
      %alloc = memref.alloc() : memref<20480xf32> loc(#loc6)
      memref.copy %reinterpret_cast, %alloc : memref<20480xf32, strided<[1], offset: ?>> to memref<20480xf32> loc(#loc6)
      %4 = bufferization.to_tensor %alloc restrict writable : memref<20480xf32> loc(#loc6)
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [20480], strides: [1] : memref<?xf32> to memref<20480xf32, strided<[1], offset: ?>> loc(#loc7)
      %alloc_1 = memref.alloc() : memref<20480xf32> loc(#loc8)
      memref.copy %reinterpret_cast_0, %alloc_1 : memref<20480xf32, strided<[1], offset: ?>> to memref<20480xf32> loc(#loc8)
      %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<20480xf32> loc(#loc8)
      %6 = arith.addf %4, %5 : tensor<20480xf32> loc(#loc9)
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%3], sizes: [20480], strides: [1] : memref<?xf32> to memref<20480xf32, strided<[1], offset: ?>> loc(#loc10)
      bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<20480xf32>, memref<20480xf32, strided<[1], offset: ?>>) -> () loc(#loc11)
    } loc(#loc3)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("print_failure_ir_final_retry.py":36:32)
#loc2 = loc("print_failure_ir_final_retry.py":41:31)
#loc3 = loc("print_failure_ir_final_retry.py":39:23)
#loc4 = loc("print_failure_ir_final_retry.py":41:23)
#loc5 = loc("print_failure_ir_final_retry.py":42:34)
#loc6 = loc("print_failure_ir_final_retry.py":42:39)
#loc7 = loc("print_failure_ir_final_retry.py":43:34)
#loc8 = loc("print_failure_ir_final_retry.py":43:39)
#loc9 = loc("print_failure_ir_final_retry.py":44:22)
#loc10 = loc("print_failure_ir_final_retry.py":45:29)
#loc11 = loc("print_failure_ir_final_retry.py":45:40)
