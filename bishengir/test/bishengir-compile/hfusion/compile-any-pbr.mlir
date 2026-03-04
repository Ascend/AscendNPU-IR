// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true %s | FileCheck %s

// -----
// CHECK: define dso_local void @test_tile_last_reduce_last
func.func @test_tile_last_reduce_last(%arg0: tensor<1024xf32>, %arg1: tensor<1024x10240xf32>, %arg2: tensor<1024x10240xf32>, %arg3: tensor<1024x10240xf32>) -> tensor<1024xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %1 = tensor.empty() : tensor<1024x10240xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<1024x10240xf32>, tensor<1024x10240xf32>) outs(%arg3: tensor<1024x10240xf32>) -> tensor<1024x10240xf32>
  %4 = tensor.empty() : tensor<1024xf32>
  %sum = linalg.reduce {arith.addf} ins(%3 : tensor<1024x10240xf32>) 
                                    outs(%4 : tensor<1024xf32>) dimensions = [1]
  %5 = tensor.empty() : tensor<1024xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %sum : tensor<1024xf32>, tensor<1024xf32>) 
                                                                  outs(%5: tensor<1024xf32>) -> tensor<1024xf32>
  return %6 : tensor<1024xf32>
}

// -----
// CHECK: define dso_local void @test_tile_first_reduce_last
func.func @test_tile_first_reduce_last(%arg0: tensor<40960xf32>, %arg1: tensor<40960x1024xf32>, %arg2: tensor<40960x1024xf32>, %arg3: tensor<40960x1024xf32>) -> tensor<40960xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %1 = tensor.empty() : tensor<40960x1024xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<40960x1024xf32>, tensor<40960x1024xf32>) outs(%arg3: tensor<40960x1024xf32>) -> tensor<40960x1024xf32>
  %4 = tensor.empty() : tensor<40960xf32>
  %sum = linalg.reduce {arith.addf} ins(%3 : tensor<40960x1024xf32>) 
                                    outs(%4 : tensor<40960xf32>) dimensions = [1]
  %5 = tensor.empty() : tensor<40960xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %sum : tensor<40960xf32>, tensor<40960xf32>) 
                                                                  outs(%5: tensor<40960xf32>) -> tensor<40960xf32>
  return %6 : tensor<40960xf32>
}

// -----
// CHECK: define dso_local void @test_tile_first_reduce_first
func.func @test_tile_first_reduce_first(%arg0: tensor<256xf32>, %arg1: tensor<16x256xf32>, %arg2: tensor<16x256xf32>, %arg3: tensor<16x256xf32>) -> tensor<256xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %1 = tensor.empty() : tensor<16x256xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<16x256xf32>, tensor<16x256xf32>) outs(%arg3: tensor<16x256xf32>) -> tensor<16x256xf32>
  %4 = tensor.empty() : tensor<256xf32>
  %sum = linalg.reduce {arith.addf} ins(%3 : tensor<16x256xf32>) 
                                    outs(%4 : tensor<256xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<256xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %sum : tensor<256xf32>, tensor<256xf32>) 
                                                             outs(%5: tensor<256xf32>) -> tensor<256xf32>
  return %6 : tensor<256xf32>
}

// -----
// CHECK: define dso_local void @test_tile_first_reduce_mid
func.func @test_tile_first_reduce_mid(%arg0: tensor<64x100xf32>, %arg1: tensor<64x640x100xf32>, 
                                      %arg2: tensor<64x640x100xf32>, %arg3: tensor<64x640x100xf32>) -> tensor<64x100xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %1 = tensor.empty() : tensor<64x640x100xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<64x640x100xf32>, tensor<64x640x100xf32>) 
                                                             outs(%arg3: tensor<64x640x100xf32>) -> tensor<64x640x100xf32>
  %4 = tensor.empty() : tensor<64x100xf32>
  %sum = linalg.reduce {arith.addf} ins(%arg1 : tensor<64x640x100xf32>) 
                                    outs(%4 : tensor<64x100xf32>) dimensions = [1]
  %5 = tensor.empty() : tensor<64x100xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %sum : tensor<64x100xf32>, tensor<64x100xf32>) 
                                                                  outs(%5: tensor<64x100xf32>) -> tensor<64x100xf32>
  return %6 : tensor<64x100xf32>
}
