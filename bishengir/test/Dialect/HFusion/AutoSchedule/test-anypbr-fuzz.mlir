// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="max-buffer-count-tuning=-10" -split-input-file -debug-only="hfusion-auto-schedule,fusible-producer-analyzer" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG-10
// RUN: bishengir-opt %s -hfusion-auto-schedule="max-buffer-count-tuning=100" -split-input-file -debug-only="hfusion-auto-schedule,fusible-producer-analyzer" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG-100
// RUN: bishengir-opt %s -hfusion-auto-schedule="max-buffer-count-tuning=500" -split-input-file -debug-only="hfusion-auto-schedule,fusible-producer-analyzer" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG-500

// Anchor is:          [256,   16,    100]
// Tiling factors are: [%arg5, %arg6, %arg7] in kernel function

// CHECK-DEBUG-10:  hfusion.load {__intermediate_producer__, __reduction0_axis2_fusible_producer__}
// CHECK-DEBUG-10:  linalg.broadcast {{.*}}  {__intermediate_producer__, __reduction0_axis0_fusible_producer__, __reduction0_axis2_fusible_producer__}
// CHECK-DEBUG-10:  linalg.broadcast {{.*}}  {__intermediate_producer__, __reduction0_axis0_fusible_producer__, __reduction0_axis2_fusible_producer__}
// CHECK-DEBUG-10:  linalg.reduce {{.*}} {__intermediate_producer__, __reduction0__}
// CHECK-DEBUG-10:  hfusion.store
//
// When max-buffer-count-tuning=-10, tiling case is 0. When we tile reduction, we only need to tile axis 0. Axis 2 is full load.
// CHECK-DEBUG-10:  %[[TILING_FACTOR_0:.*]] = transform.func.get_func_argument {{.*}}[5]
// CHECK-DEBUG-10:  transform.structured.tile_reduction_using_for {{.*}} by tile_sizes = [%[[TILING_FACTOR_0]], 0, 0]

// When max-buffer-count-tuning=100, tiling case is 1. When we tile reduction, the tiling factor for axis 0 is 1. And axis 2 is full load.
// CHECK-DEBUG-100:  transform.structured.tile_reduction_using_for {{.*}} by tile_sizes = [1, 0, 0]

// When max-buffer-count-tuning=500, tiling case is 2. When we tile reduction, the tiling factor for axis 0 is 1. And we tile axis 2 by tiling factor.
// CHECK-DEBUG-500:  %[[TILING_FACTOR_2:.*]] = transform.func.get_func_argument {{.*}}[7]
// CHECK-DEBUG-500:  transform.structured.tile_reduction_using_for {{.*}} by tile_sizes = [1, 0, %[[TILING_FACTOR_2]]]
module {
  func.func @foo(%arg0: tensor<100xf32>, %arg1: tensor<256x100xf32>, %arg2: tensor<16xf32>) -> (tensor<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<256x100xf32>
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<100xf32>) outs(%0 : tensor<256x100xf32>) dimensions = [0]
    %1 = tensor.empty() : tensor<256x16x100xf32>
    %broadcasted1 = linalg.broadcast ins(%broadcasted : tensor<256x100xf32>) outs(%1 : tensor<256x16x100xf32>) dimensions = [1]
    %2 = tensor.empty() : tensor<16xf32>
    %reduced = linalg.reduce { arith.addf } ins(%broadcasted1 : tensor<256x16x100xf32>) outs(%2 : tensor<16xf32>) dimensions = [0, 2]
    return %reduced : tensor<16xf32>
  }
}

// -----

// Anchor is:          [256,   16,    100]
// Tiling factors are: [%arg6, %arg7, %arg8] in kernel function

// CHECK-DEBUG-500:  hfusion.load {__intermediate_producer__, __reduction0_axis0_fusible_producer__, __result1_axis0_fusible_producer__}
// CHECK-DEBUG-500:  linalg.broadcast {{.*}}  {__intermediate_producer__, __reduction0_axis0_fusible_producer__, __reduction0_axis2_fusible_producer__, __result1_axis0_fusible_producer__, __result1_axis2_fusible_producer__}
// CHECK-DEBUG-500:  linalg.broadcast {{.*}}  {__intermediate_producer__, __reduction0_axis0_fusible_producer__, __reduction0_axis2_fusible_producer__, __result1_axis0_fusible_producer__, __result1_axis2_fusible_producer__}
// CHECK-DEBUG-500:  hfusion.store
// CHECK-DEBUG-500:  linalg.reduce {{.*}} {__intermediate_producer__, __reduction0__}
// CHECK-DEBUG-500:  hfusion.store
//
// When max-buffer-count-tuning=500, tiling case is 2. 
//
// Tiling factor for output tensor<16xf32> is [1] because tiling key = 2.
// CHECK-DEBUG-500:  %[[TILING_FACTOR_2:.*]] = transform.func.get_func_argument {{.*}}[8]
// CHECK-DEBUG-500:  %[[OUTPUT_0:.*]] = transform.structured.match attributes {hfusion.return_operand_num = 0 : i64}
// CHECK-DEBUG-500:  transform.structured.tile_using_for %[[OUTPUT_0]] tile_sizes [1] interchange = [0]
// Tiling factor for output tensor<256x16x100xf32> is [0, 1, 0] because we only tile the "strictly parallel" axis the first time.
// CHECK-DEBUG-500:  %[[OUTPUT_1:.*]] = transform.structured.match attributes {hfusion.return_operand_num = 1 : i64}
// CHECK-DEBUG-500:  transform.structured.tile_using_for %[[OUTPUT_1]] tile_sizes [0, 1, 0] interchange = [0, 1, 2]
// When we tile reduction, the tiling factor for axis 0 is 1. And we tile axis 2 by tiling factor.
// CHECK-DEBUG-500:  transform.structured.tile_reduction_using_for {{.*}} by tile_sizes = [1, 0, %[[TILING_FACTOR_2]]]
module {
  func.func @foo(%arg0: tensor<256xf32>, %arg1: tensor<256x100xf32>, %arg2: tensor<16xf32>) -> (tensor<16xf32>, tensor<256x16x100xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<256x100xf32>
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<256xf32>) outs(%0 : tensor<256x100xf32>) dimensions = [1]
    %1 = tensor.empty() : tensor<256x16x100xf32>
    %broadcasted1 = linalg.broadcast ins(%broadcasted : tensor<256x100xf32>) outs(%1 : tensor<256x16x100xf32>) dimensions = [1]
    %2 = tensor.empty() : tensor<16xf32>
    %reduced = linalg.reduce { arith.addf } ins(%broadcasted1 : tensor<256x16x100xf32>) outs(%2 : tensor<16xf32>) dimensions = [0, 2]
    return %reduced, %broadcasted1 : tensor<16xf32>, tensor<256x16x100xf32>
  }
}
