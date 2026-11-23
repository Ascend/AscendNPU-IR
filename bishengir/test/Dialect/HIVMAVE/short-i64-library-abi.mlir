// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 --hivm-vectorize --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --hivm-vectorize=phase=prepare-buffers --hivm-infer-mem-scope --normalize-vector --hivm-vectorize=phase=backend --convert-vector-to-hivmave --convert-arith-to-hivmave --analyze-vector-layout --ave-normalize-ops --remove-vector-layout-attr -convert-hivmave-to-std --verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: @short_i64_load_fused_0
  // CHECK: llvm.alloca
  // CHECK: call @vload_NORM_int64_t_{{(unalign_)?}}rank1
  // CHECK: call @_mlir_ciface_cast_int64_t_to_int32_t
  func.func @short_i64_load(%src: tensor<4xi64>, %init: tensor<4xi32>) -> tensor<4xi32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %result = hivm.hir.vcast ins(%src : tensor<4xi64>) outs(%init : tensor<4xi32>)
        -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}
