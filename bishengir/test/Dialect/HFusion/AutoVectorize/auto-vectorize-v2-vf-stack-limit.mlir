// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true enable-vf-stack-limit=false emit-transform-sequence=true" | FileCheck %s --check-prefix=FUSE
// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true enable-vf-stack-limit=true emit-transform-sequence=true" | FileCheck %s --check-prefix=LIMIT

// FUSE-LABEL: func.func @vf_stack_limit_multi_consumer
// FUSE: hivm.hir.vgather
// FUSE-LABEL: transform.sequence
// FUSE: annotate {{.*}} "outlined-loop-target-1"
// FUSE: transform.structured.match attributes {"hfusion-auto-vectorize-target-1"}
// FUSE: transform.structured.match attributes {"outlined-loop-target-1"}
// FUSE: transform.structured.fuse_into_containing_op
// FUSE-NOT: "outlined-loop-target-2"

// LIMIT-LABEL: func.func @vf_stack_limit_multi_consumer
// LIMIT: hivm.hir.vgather
// LIMIT-LABEL: transform.sequence
// LIMIT-NOT: transform.structured.fuse_into_containing_op
// LIMIT: annotate {{.*}} "outlined-loop-target-1"
// LIMIT-NOT: transform.structured.fuse_into_containing_op
// LIMIT: annotate {{.*}} "outlined-loop-target-2"
// LIMIT-NOT: transform.structured.fuse_into_containing_op

#map = affine_map<(d0) -> (d0)>

module {
  func.func @vf_stack_limit_multi_consumer(
      %arg0: tensor<512xf32>, %arg1: tensor<512xf32>,
      %arg2: tensor<512xf32>, %arg3: tensor<512xf32>,
      %arg4: tensor<512xf32>, %arg5: tensor<512xf32>,
      %arg6: tensor<512xf32>, %arg7: tensor<512xf32>,
      %arg8: tensor<512xf32>, %arg9: tensor<512xf32>,
      %arg10: tensor<512xf32>, %arg11: tensor<512xf32>,
      %arg12: tensor<512xf32>, %arg13: tensor<512xf32>,
      %arg14: tensor<512xf32>, %arg15: tensor<512xf32>,
      %arg16: tensor<512xf32>, %arg17: tensor<512xf32>,
      %arg18: tensor<512xf32>, %arg19: tensor<512xf32>,
      %arg20: tensor<512xf32>, %arg21: tensor<512xf32>,
      %arg22: tensor<512xf32>, %arg23: tensor<512xf32>,
      %arg24: tensor<512xf32>, %consumer_arg: tensor<512xf32>,
      %idx: tensor<512xi32>) -> (tensor<512xf32>, tensor<512xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %init = tensor.empty() : tensor<512xf32>
    %producer = linalg.generic {
        indexing_maps = [
          #map, #map, #map, #map, #map, #map, #map, #map, #map, #map,
          #map, #map, #map, #map, #map, #map, #map, #map, #map, #map,
          #map, #map, #map, #map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7,
            %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14,
            %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21,
            %arg22, %arg23, %arg24 :
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>, tensor<512xf32>, tensor<512xf32>,
            tensor<512xf32>)
        outs(%init : tensor<512xf32>) {
    ^bb0(%in0: f32, %in1: f32, %in2: f32, %in3: f32, %in4: f32,
         %in5: f32, %in6: f32, %in7: f32, %in8: f32, %in9: f32,
         %in10: f32, %in11: f32, %in12: f32, %in13: f32, %in14: f32,
         %in15: f32, %in16: f32, %in17: f32, %in18: f32, %in19: f32,
         %in20: f32, %in21: f32, %in22: f32, %in23: f32, %in24: f32,
         %out: f32):
      %0 = arith.addf %in0, %in1 : f32
      %1 = arith.addf %0, %in2 : f32
      %2 = arith.addf %1, %in3 : f32
      %3 = arith.addf %2, %in4 : f32
      %4 = arith.addf %3, %in5 : f32
      %5 = arith.addf %4, %in6 : f32
      %6 = arith.addf %5, %in7 : f32
      %7 = arith.addf %6, %in8 : f32
      %8 = arith.addf %7, %in9 : f32
      %9 = arith.addf %8, %in10 : f32
      %10 = arith.addf %9, %in11 : f32
      %11 = arith.addf %10, %in12 : f32
      %12 = arith.addf %11, %in13 : f32
      %13 = arith.addf %12, %in14 : f32
      %14 = arith.addf %13, %in15 : f32
      %15 = arith.addf %14, %in16 : f32
      %16 = arith.addf %15, %in17 : f32
      %17 = arith.addf %16, %in18 : f32
      %18 = arith.addf %17, %in19 : f32
      %19 = arith.addf %18, %in20 : f32
      %20 = arith.addf %19, %in21 : f32
      %21 = arith.addf %20, %in22 : f32
      %22 = arith.addf %21, %in23 : f32
      %23 = arith.addf %22, %in24 : f32
      linalg.yield %23 : f32
    } -> tensor<512xf32>
    %consumer = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%producer, %consumer_arg : tensor<512xf32>, tensor<512xf32>)
        outs(%init : tensor<512xf32>) -> tensor<512xf32>
    %gather_init = tensor.empty() : tensor<512xf32>
    %gather = hivm.hir.vgather ins(%producer : tensor<512xf32>)
                               indices(%idx : tensor<512xi32>)
                               outs(%gather_init : tensor<512xf32>)
                               -> tensor<512xf32>
    return %consumer, %gather : tensor<512xf32>, tensor<512xf32>
  }
}
