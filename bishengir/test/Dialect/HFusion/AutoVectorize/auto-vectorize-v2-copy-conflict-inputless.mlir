// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true enable-vf-stack-limit=false" -outline-vector-function | FileCheck %s

// CHECK-LABEL: func.func @inputless_generator_across_copy_outlined_vf_0
// CHECK: vector.step
// CHECK: arith.index_cast
// CHECK: vector.transfer_read {{.*}} : tensor<64xf16>
// CHECK: arith.cmpi slt
// CHECK-LABEL: func.func @inputless_generator_across_copy(
// CHECK: memref.copy
// CHECK-NOT: linalg.generic
// CHECK: func.call @inputless_generator_across_copy_outlined_vf_0

#map = affine_map<(d0) -> (d0)>

module {
  func.func @inputless_generator_across_copy(
      %arg0: memref<16384xf16, strided<[1]>>) -> tensor<16384xi1>
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c8193_i32 = arith.constant 8193 : i32
    %idx_init = tensor.empty() : tensor<16384xi32>
    %idx = linalg.generic {
        indexing_maps = [#map], iterator_types = ["parallel"]}
        outs(%idx_init : tensor<16384xi32>) {
    ^bb0(%out: i32):
      %i = linalg.index 0 : index
      %i32 = arith.index_cast %i : index to i32
      linalg.yield %i32 : i32
    } -> tensor<16384xi32>

    %alloc = memref.alloc() : memref<16384xf16>
    memref.copy %arg0, %alloc
        : memref<16384xf16, strided<[1]>> to memref<16384xf16>
    %data = bufferization.to_tensor %alloc restrict writable
        : memref<16384xf16>
    %mask_init = tensor.empty() : tensor<16384xi1>
    %mask = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%idx, %data : tensor<16384xi32>, tensor<16384xf16>)
        outs(%mask_init : tensor<16384xi1>) {
    ^bb0(%in_idx: i32, %in_data: f16, %out: i1):
      %valid = arith.cmpi slt, %in_idx, %c8193_i32 : i32
      %isnan = arith.cmpf une, %in_data, %in_data : f16
      %result = arith.andi %valid, %isnan : i1
      linalg.yield %result : i1
    } -> tensor<16384xi1>
    return %mask : tensor<16384xi1>
  }
}
