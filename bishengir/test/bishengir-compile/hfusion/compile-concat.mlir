// RUN: bishengir-compile --enable-lir-compile=false --enable-hfusion-compile=true --enable-hivm-compile=true %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @concat_compile(%arg0: tensor<3x256x12288xi64>, %arg1: tensor<3x256x1024xi64>) -> tensor<3x256x13312xi64> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %concat = tensor.concat dim(2) %arg0, %arg1 : (tensor<3x256x12288xi64>, tensor<3x256x1024xi64>) -> tensor<3x256x13312xi64>
    return %concat : tensor<3x256x13312xi64>
  }
}