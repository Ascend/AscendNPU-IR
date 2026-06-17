// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion="enable-vf-stack-limit=false" | FileCheck %s --check-prefix=NO-LIMIT
// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion="enable-vf-stack-limit=true" | FileCheck %s --check-prefix=LIMIT

// NO-LIMIT-LABEL: func.func @vf_stack_limit_splits_chain
// NO-LIMIT: linalg.generic
// NO-LIMIT-NOT: linalg.generic

// LIMIT-LABEL: func.func @vf_stack_limit_splits_chain
// LIMIT: linalg.generic
// LIMIT: linalg.generic

module {
  func.func @vf_stack_limit_splits_chain(
      %arg0: tensor<512xi64>, %arg1: tensor<512xi64>,
      %arg2: tensor<512xi64>, %arg3: tensor<512xi64>,
      %arg4: tensor<512xi64>, %arg5: tensor<512xi64>,
      %arg6: tensor<512xi64>) -> tensor<512xi64>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<512xi64>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg1 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
        ins(%1, %arg2 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%2, %arg3 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
        ins(%3, %arg4 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%4, %arg5 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
        ins(%5, %arg6 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%6, %arg0 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
        ins(%7, %arg1 : tensor<512xi64>, tensor<512xi64>)
        outs(%0 : tensor<512xi64>) -> tensor<512xi64>
    return %8 : tensor<512xi64>
  }
}
