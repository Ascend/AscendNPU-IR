// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main %s | FileCheck %s

module {
    func.func @kernel(%arg0: memref<?x5xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<?x5xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<5xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<5xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}, %arg4: memref<5xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0 = arith.constant 0.: f32
        linalg.fill ins(%c0: f32) outs(%arg2: memref<5xf32>)
        linalg.reduce {arith.addf} ins(%arg0: memref<?x5xf32>) outs(%arg2: memref<5xf32>) dimensions = [0]
        linalg.fill ins(%c0: f32) outs(%arg3: memref<5xf32>)
        linalg.reduce {arith.addf} ins(%arg1: memref<?x5xf32>) outs(%arg3: memref<5xf32>) dimensions = [0]
        linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %arg3: memref<5xf32>, memref<5xf32>) outs(%arg4: memref<5xf32>)
        func.return
    }
}

// CHECK-LABEL:   func.func private @closeFileHandle(!llvm.ptr)
// CHECK-LABEL:   func.func private @printDataF32(!llvm.ptr, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-LABEL:   func.func private @getFileHandle(!llvm.ptr) -> !llvm.ptr
// CHECK-LABEL:   func.func private @getDataF32(memref<*xf32>) attributes {llvm.emit_c_interface}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:                    %[[VAL_3:[^:]*]]: index) {
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_2]]) : memref<?x5xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_3]]) : memref<?x5xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<5xf32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<5xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<5xf32>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_4]] : memref<?x5xf32> to memref<*xf32>
// CHECK:           %[[VAL_10:.*]] = memref.cast %[[VAL_5]] : memref<?x5xf32> to memref<*xf32>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_6]] : memref<5xf32> to memref<*xf32>
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_7]] : memref<5xf32> to memref<*xf32>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_8]] : memref<5xf32> to memref<*xf32>
// CHECK:           call @getDataF32(%[[VAL_9]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_10]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_11]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_12]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_13]]) : (memref<*xf32>) -> ()
// CHECK:           %[[VAL_14:.*]] = call @getFileHandle(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF32(%[[VAL_14]], %[[VAL_9]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_14]], %[[VAL_10]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_14]], %[[VAL_11]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_14]], %[[VAL_12]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_14]], %[[VAL_13]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_14]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_9]] : memref<*xf32> to memref<?x5xf32>
// CHECK:           %[[VAL_16:.*]] = memref.cast %[[VAL_10]] : memref<*xf32> to memref<?x5xf32>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_11]] : memref<*xf32> to memref<5xf32>
// CHECK:           %[[VAL_18:.*]] = memref.cast %[[VAL_12]] : memref<*xf32> to memref<5xf32>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_13]] : memref<*xf32> to memref<5xf32>
// CHECK:           call @kernel(%[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (memref<?x5xf32>, memref<?x5xf32>, memref<5xf32>, memref<5xf32>, memref<5xf32>) -> ()
// CHECK:           %[[VAL_20:.*]] = memref.cast %[[VAL_17]] : memref<5xf32> to memref<*xf32>
// CHECK:           %[[VAL_21:.*]] = memref.cast %[[VAL_18]] : memref<5xf32> to memref<*xf32>
// CHECK:           %[[VAL_22:.*]] = memref.cast %[[VAL_19]] : memref<5xf32> to memref<*xf32>
// CHECK:           %[[VAL_23:.*]] = call @getFileHandle(%[[VAL_1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF32(%[[VAL_23]], %[[VAL_20]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_23]], %[[VAL_21]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_23]], %[[VAL_22]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_23]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
