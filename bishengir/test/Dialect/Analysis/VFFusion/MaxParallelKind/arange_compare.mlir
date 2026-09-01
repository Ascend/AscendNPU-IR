// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// A single-use arange feeding a compare is forced into the compare's group.
// This keeps the generated range inside the VF even when a copy separates the
// original producer from the rest of the consumer chain.

// CHECK-LABEL: func.func private @arange_compare_across_copy_fused_0(
// CHECK: hfusion.arange
// CHECK: hfusion.compare {{.*}}vlt
// CHECK: hfusion.compare {{.*}}vne
// CHECK: hfusion.elemwise_binary {{.*}}vand
// CHECK: return

// CHECK-LABEL: func.func @arange_compare_across_copy(
// CHECK: memref.copy
// CHECK-NOT: hfusion.arange
// CHECK: call @arange_compare_across_copy_fused_0

module {
  func.func @arange_compare_across_copy(
      %arg0: memref<16384xf16>, %bound: i32) -> tensor<16384xi1> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rangeInit = tensor.empty() : tensor<16384xi32>
    %maskInit = tensor.empty() : tensor<16384xi1>
    %range = hfusion.arange offset[%c0] strides[%c1]
        outs(%rangeInit : tensor<16384xi32>) -> tensor<16384xi32>

    %alloc = memref.alloc() : memref<16384xf16>
    memref.copy %arg0, %alloc : memref<16384xf16> to memref<16384xf16>
    %data = bufferization.to_tensor %alloc restrict writable
        : memref<16384xf16>
    %valid = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
        ins(%range, %bound : tensor<16384xi32>, i32)
        outs(%maskInit : tensor<16384xi1>) -> tensor<16384xi1>
    %isnan = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
        ins(%data, %data : tensor<16384xf16>, tensor<16384xf16>)
        outs(%maskInit : tensor<16384xi1>) -> tensor<16384xi1>
    %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
        ins(%valid, %isnan : tensor<16384xi1>, tensor<16384xi1>)
        outs(%maskInit : tensor<16384xi1>) -> tensor<16384xi1>
    return %result : tensor<16384xi1>
  }
}
