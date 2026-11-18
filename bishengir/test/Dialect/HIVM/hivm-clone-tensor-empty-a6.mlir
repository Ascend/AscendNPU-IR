// RUN: bishengir-opt %s -hivm-clone-tensor-empty | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend960DT_968B7">} {
  func.func private @vf(tensor<16xf32>) -> tensor<16xf32> attributes {hivm.vector_function}
  func.func private @ordinary(tensor<16xf32>) -> tensor<16xf32>

  // CHECK-LABEL: func.func @clone_shared_empty
  func.func @clone_shared_empty() {
    // CHECK: %[[SHARED:.*]] = tensor.empty() : tensor<16xf32>
    %shared = tensor.empty() : tensor<16xf32>
    // CHECK: %[[FIRST:.*]] = tensor.empty() : tensor<16xf32>
    // CHECK-NEXT: %{{.*}} = call @vf(%[[FIRST]]) {hivm.vector_function}
    %first = func.call @vf(%shared) {hivm.vector_function} : (tensor<16xf32>) -> tensor<16xf32>
    // CHECK: %[[SECOND:.*]] = tensor.empty() : tensor<16xf32>
    // CHECK-NEXT: %{{.*}} = call @vf(%[[SECOND]]) {hivm.vector_function}
    %second = func.call @vf(%shared) {hivm.vector_function} : (tensor<16xf32>) -> tensor<16xf32>
    // CHECK-NEXT: %{{.*}} = call @ordinary(%[[SHARED]])
    %ordinary = func.call @ordinary(%shared) : (tensor<16xf32>) -> tensor<16xf32>
    return
  }
}
