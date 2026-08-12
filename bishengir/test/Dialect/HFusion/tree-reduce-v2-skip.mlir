// RUN: bishengir-opt --tree-reduce-v2="enable-ar enable-ra" %s | FileCheck %s

// Test that TreeReduceV2 skips modules containing functions with hivm.part_of_mix attribute.
// The vector.multi_reduction should NOT be transformed.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  // CHECK-LABEL: func.func @cv_part_of_mix_case
  // CHECK: vector.multi_reduction <add>
  func.func @cv_part_of_mix_case(%arg0: tensor<128x384xf32>, %arg6: tensor<384xf32>) -> tensor<384xf32> attributes {hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg7 = %c0 to %c128 step %c1 iter_args(%arg8 = %arg6) -> (tensor<384xf32>) {
      %1 = scf.for %arg9 = %c0 to %c384 step %c64 iter_args(%arg10 = %arg8) -> (tensor<384xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg7, %arg9] [1, 64] [1, 1] : tensor<128x384xf32> to tensor<1x64xf32>
        %extracted_slice_0 = tensor.extract_slice %arg10[%arg9] [64] [1] : tensor<384xf32> to tensor<64xf32>
        %cst = arith.constant 0.000000e+00 : f32
        %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst : tensor<1x64xf32>, vector<1x64xf32>
        %cst_4 = arith.constant 0.000000e+00 : f32
        %3 = vector.transfer_read %extracted_slice_0[%c0], %cst_4 : tensor<64xf32>, vector<64xf32>
        %4 = vector.multi_reduction <add>, %2, %3 [0] : vector<1x64xf32> to vector<64xf32>
        %5 = vector.transfer_write %4, %extracted_slice_0[%c0] : vector<64xf32>, tensor<64xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg10[%arg9] [64] [1] : tensor<64xf32> into tensor<384xf32>
        scf.yield %inserted_slice : tensor<384xf32>
      }
      scf.yield %1 : tensor<384xf32>
    }
    return %0 : tensor<384xf32>
  }

  // This function has hivm.part_of_mix -> the entire module should be skipped.
  func.func @cv_part_of_mix_helper() attributes {hivm.part_of_mix} {
    return
  }
}
