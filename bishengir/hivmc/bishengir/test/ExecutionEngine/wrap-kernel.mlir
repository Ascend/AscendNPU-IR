// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main %s | FileCheck %s

module {
    func.func @kernel(%arg0: tensor<?x5xf16>, %arg1: tensor<?x5xf16>) -> (tensor<5xf16>, tensor<5xf16>, tensor<5xf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0 = arith.constant 0.: f16
        %00 = tensor.empty() : tensor<5xf16>
        %000 = linalg.fill ins(%c0: f16) outs(%00: tensor<5xf16>) -> tensor<5xf16>
        %1 = linalg.reduce {arith.addf} ins(%arg0: tensor<?x5xf16>) outs(%000: tensor<5xf16>) dimensions = [0]
        %01 = tensor.empty() : tensor<5xf16>
        %001 = linalg.fill ins(%c0: f16) outs(%01: tensor<5xf16>) -> tensor<5xf16>
        %2 = linalg.reduce {arith.addf} ins(%arg1: tensor<?x5xf16>) outs(%001: tensor<5xf16>) dimensions = [0]
        %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %2: tensor<5xf16>, tensor<5xf16>) outs(%001: tensor<5xf16>) -> tensor<5xf16>
        func.return %1, %2, %3 :tensor<5xf16>, tensor<5xf16>, tensor<5xf16>
    }
}

// CHECK-LABEL:   func.func private @closeFileHandle(!llvm.ptr)
// CHECK-LABEL:   func.func private @printDataF16(!llvm.ptr, memref<*xf16>) attributes {llvm.emit_c_interface}
// CHECK-LABEL:   func.func private @getFileHandle(!llvm.ptr) -> !llvm.ptr
// CHECK-LABEL:   func.func private @getDataF16(memref<*xf16>) attributes {llvm.emit_c_interface}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:                    %[[VAL_3:[^:]*]]: index) {
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_2]]) : memref<?x5xf16>
// CHECK:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_3]]) : memref<?x5xf16>
// CHECK:           %[[VAL_6:.*]] = memref.cast %[[VAL_4]] : memref<?x5xf16> to memref<*xf16>
// CHECK:           %[[VAL_7:.*]] = memref.cast %[[VAL_5]] : memref<?x5xf16> to memref<*xf16>
// CHECK:           call @getDataF16(%[[VAL_6]]) : (memref<*xf16>) -> ()
// CHECK:           call @getDataF16(%[[VAL_7]]) : (memref<*xf16>) -> ()
// CHECK:           %[[VAL_8:.*]] = call @getFileHandle(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF16(%[[VAL_8]], %[[VAL_6]]) : (!llvm.ptr, memref<*xf16>) -> ()
// CHECK:           call @printDataF16(%[[VAL_8]], %[[VAL_7]]) : (!llvm.ptr, memref<*xf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_8]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_6]] : memref<*xf16> to memref<?x5xf16>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<?x5xf16>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_7]] : memref<*xf16> to memref<?x5xf16>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<?x5xf16>
// CHECK:           %[[VAL_13:.*]]:3 = call @kernel(%[[VAL_10]], %[[VAL_12]]) : (tensor<?x5xf16>, tensor<?x5xf16>) -> (tensor<5xf16>, tensor<5xf16>, tensor<5xf16>)
// CHECK:           %[[VAL_14:.*]] = tensor.cast %[[VAL_13]]#0 : tensor<5xf16> to tensor<*xf16>
// CHECK:           %[[VAL_15:.*]] = bufferization.to_memref %[[VAL_14]] : memref<*xf16>
// CHECK:           %[[VAL_16:.*]] = tensor.cast %[[VAL_13]]#1 : tensor<5xf16> to tensor<*xf16>
// CHECK:           %[[VAL_17:.*]] = bufferization.to_memref %[[VAL_16]] : memref<*xf16>
// CHECK:           %[[VAL_18:.*]] = tensor.cast %[[VAL_13]]#2 : tensor<5xf16> to tensor<*xf16>
// CHECK:           %[[VAL_19:.*]] = bufferization.to_memref %[[VAL_18]] : memref<*xf16>
// CHECK:           %[[VAL_20:.*]] = call @getFileHandle(%[[VAL_1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF16(%[[VAL_20]], %[[VAL_15]]) : (!llvm.ptr, memref<*xf16>) -> ()
// CHECK:           call @printDataF16(%[[VAL_20]], %[[VAL_17]]) : (!llvm.ptr, memref<*xf16>) -> ()
// CHECK:           call @printDataF16(%[[VAL_20]], %[[VAL_19]]) : (!llvm.ptr, memref<*xf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_20]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
