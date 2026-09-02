// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main %s | FileCheck %s

module {
    func.func @kernel(%arg0: memref<?x5xbf16>, %arg1: memref<?x5xbf16>) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0 = arith.constant 0.: bf16
        %1 = memref.alloc() : memref<5xbf16>
        linalg.fill ins(%c0: bf16) outs(%1: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg0: memref<?x5xbf16>) outs(%1: memref<5xbf16>) dimensions = [0]
        %2 = memref.alloc() : memref<5xbf16>
        linalg.fill ins(%c0: bf16) outs(%2: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg1: memref<?x5xbf16>) outs(%2: memref<5xbf16>) dimensions = [0]
        %3 = memref.alloc() : memref<5xbf16>
        linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %2: memref<5xbf16>, memref<5xbf16>) outs(%3: memref<5xbf16>)
        func.return %1, %2, %3 :memref<5xbf16>, memref<5xbf16>, memref<5xbf16>
    }
}

// CHECK-LABEL:   func.func private @closeFileHandle(!llvm.ptr)
// CHECK-LABEL:   func.func private @printDataBF16(!llvm.ptr, memref<*xbf16>) attributes {llvm.emit_c_interface}
// CHECK-LABEL:   func.func private @getFileHandle(!llvm.ptr) -> !llvm.ptr
// CHECK-LABEL:   func.func private @getDataBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:                    %[[VAL_3:[^:]*]]: index) {
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_2]]) : memref<?x5xbf16>
// CHECK:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_3]]) : memref<?x5xbf16>
// CHECK:           %[[VAL_6:.*]] = memref.cast %[[VAL_4]] : memref<?x5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_5]] : memref<?x5xbf16> to memref<*xbf16>
// CHECK:           call @getDataBF16(%[[VAL_6]]) : (memref<*xbf16>) -> ()
// CHECK:           call @getDataBF16(%[[VAL_7]]) : (memref<*xbf16>) -> ()
// CHECK:           %[[VAL_8:.*]] = call @getFileHandle(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataBF16(%[[VAL_8]], %[[VAL_6]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_8]], %[[VAL_7]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_8]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_6]] : memref<*xbf16> to memref<?x5xbf16>
// CHECK:           %[[VAL_10:.*]] = memref.cast %[[VAL_7]] : memref<*xbf16> to memref<?x5xbf16>
// CHECK:           %[[VAL_11:.*]]:3 = call @kernel(%[[VAL_9]], %[[VAL_10]]) : (memref<?x5xbf16>, memref<?x5xbf16>) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>)
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_11]]#0 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_11]]#1 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_11]]#2 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_15:.*]] = call @getFileHandle(%[[VAL_1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataBF16(%[[VAL_15]], %[[VAL_12]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_15]], %[[VAL_13]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_15]], %[[VAL_14]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_15]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
