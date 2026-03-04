// REQUIRES: execution-engine
// RUN: bishengir-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false --enable-cpu-runner-after=cse,12,wrapper-name=main_wrapper --enable-manage-host-resources %s | FileCheck %s

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

// CHECK:      llvm.func @main_wrapper