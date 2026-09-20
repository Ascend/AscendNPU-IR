// RUN: bishengir-opt %s --convert-hivm-to-tritongpu | FileCheck %s --check-prefix=STAGE2
// RUN: bishengir-opt %s --convert-hivm-to-tritongpu --rewrite-slice-op-to-triton | FileCheck %s --check-prefix=PIPELINE

// Stage 2 clones the body while converting func.func to tt.func. Keep tensor
// ops that survived stage 1 legal while their containing function is rebuilt.
//
// STAGE2-LABEL: tt.func @tensor_insert_dynamic_memref(
// STAGE2: tensor.insert
// STAGE2-NOT: func.func
//
// The later Triton slice rewrite consumes the tensor op; declaring it legal in
// stage 2 must not leave an unlowered operation at the downstream handoff.
// PIPELINE-LABEL: tt.func @tensor_insert_dynamic_memref(
// PIPELINE-NOT: tensor.insert
// PIPELINE: tt.splat
// PIPELINE: tt.store
// PIPELINE-NOT: tensor.insert

module {
  func.func @tensor_insert_dynamic_memref(
      %arg0: memref<?xf32>, %out: memref<1xf32>, %value: f32, %index: index)
      attributes {no_inline, outline, vector_function,
                  vf_mode = #hivm.vf_mode<SIMT>} {
    %empty = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %value into %empty[%index] : tensor<1xf32>
    hivm.hir.local_store ins(%out : memref<1xf32>,
                             %inserted : tensor<1xf32>)
    return
  }
}
