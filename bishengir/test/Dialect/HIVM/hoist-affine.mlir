// RUN: bishengir-opt %s -test-hivm-transform-patterns=test-hoist-affine -allow-unregistered-dialect -split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s0, s1)>
module {
  func.func @foo(%arg0: index) {
  // CHECK: affine.apply
  // CHECK: affine.min
  // CHECK: cf.br
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = affine.apply #map()[%arg0]
    %1 = affine.min #map1()[%arg0, %0]
    "some_use"(%1) : (index) -> ()
    return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
  // CHECK: affine.apply
  // CHECK: cf.br
    cf.br ^bb1(%arg1: index)
  ^bb1(%0: index):
    %index = "some_op"(%0) : (index) -> (index)
    // CHECK: affine.apply
    // CHECK: "some_use"
    // CHECK: "some_use"
    %1 = affine.apply #map()[%index]
    %2 = affine.apply #map()[%arg0]
    "some_use"(%1) : (index) -> ()
    "some_use"(%2) : (index) -> ()
    return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
     %index = "some_op"(%0) : (index) -> (index)
     // CHECK: affine.apply
     cf.br ^bb2(%arg1: index)
    ^bb2(%1: index):
      %2 = affine.apply #map()[%index]
      "some_use"(%2) : (index) -> ()
  return
  }
}

// -----

#map = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
     %index = "some_op"(%0) : (index) -> (index)
     // CHECK: affine.apply
     cf.br ^bb2
    ^bb2():
      %2 = affine.apply #map()[%index, %arg1]
      "some_use"(%2) : (index) -> ()
  return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
    %1 = "some_op"(%0) : (index) -> (index)
    // CHECK: affine.apply
    // CHECK: affine.apply
    %2 = affine.apply #map()[%1]
    "some_use"(%2) : (index) -> ()
    %3 = affine.apply #map()[%1]
    "some_use"(%3) : (index) -> ()
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_0(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]], %[[VAL_2]])
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_4]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_5]]{{\[}}%[[VAL_3]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_6]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_0(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%offset0, %offset1)
    %extracted_slice = tensor.extract_slice %2[%3] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_1(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.minsi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_4]], %[[VAL_2]])
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_6]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_7]]{{\[}}%[[VAL_5]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_8]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_1(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %c16 = arith.constant 16 : index
    %offset2 = arith.minsi %offset0, %c16 : index
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%offset2, %offset1)
    %extracted_slice = tensor.extract_slice %2[%3] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = affine.max #[[$ATTR_2]](%[[VAL_1]], %[[VAL_2]])
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_4]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_5]]{{\[}}%[[VAL_3]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_6]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_2(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %max = affine.max affine_map<(d0, d1) -> (d0, d1)>(%offset0, %offset1)
    %extracted_slice = tensor.extract_slice %2[%max] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_3(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.minsi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = affine.max #[[$ATTR_3]](%[[VAL_4]], %[[VAL_2]])
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_6]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_7]]{{\[}}%[[VAL_5]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_8]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_3(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %c16 = arith.constant 16 : index
    %offset2 = arith.minsi %offset0, %c16 : index
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %max = affine.max affine_map<(d0, d1) -> (d0, d1)>(%offset2, %offset1)
    %extracted_slice = tensor.extract_slice %2[%max] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_4(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = affine.min #[[$ATTR_4]](%[[VAL_1]], %[[VAL_2]])
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_5:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_4]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_5]]{{\[}}%[[VAL_3]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_6]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_4(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %min = affine.min affine_map<(d0, d1) -> (d0, d1)>(%offset0, %offset1)
    %extracted_slice = tensor.extract_slice %2[%min] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_5(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.minsi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = affine.min #[[$ATTR_5]](%[[VAL_4]], %[[VAL_2]])
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_7:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_6]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_7]]{{\[}}%[[VAL_5]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           return %[[VAL_8]] : tensor<16xf32>
// CHECK:         }
func.func @test_hoistaffine_5(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<16xf32> {
    %c16 = arith.constant 16 : index
    %offset2 = arith.minsi %offset0, %c16 : index
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %min = affine.min affine_map<(d0, d1) -> (d0, d1)>(%offset2, %offset1)
    %extracted_slice = tensor.extract_slice %2[%min] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    return %extracted_slice: tensor<16xf32>
}

// -----
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_hoistaffine_6(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<64xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: index,
// CHECK-SAME:                                  %[[VAL_2:.*]]: index) -> tensor<8xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.minsi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = affine.min #[[$ATTR_6]](%[[VAL_4]], %[[VAL_1]])
// CHECK:           %[[VAL_6:.*]] = affine.min #[[$ATTR_6]](%[[VAL_4]], %[[VAL_2]])
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_8:.*]] = hivm.hir.vln ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_7]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_8]]{{\[}}%[[VAL_6]]] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_9]]{{\[}}%[[VAL_5]]] [8] [1] : tensor<16xf32> to tensor<8xf32>
// CHECK:           return %[[VAL_10]] : tensor<8xf32>
// CHECK:         }
func.func @test_hoistaffine_6(%arg0: tensor<64xf32>, %offset0: index, %offset1: index) -> tensor<8xf32> {
    %c16 = arith.constant 16 : index
    %offset2 = arith.minsi %offset0, %c16 : index
    %min_1 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%offset2, %offset0)
    %1 = tensor.empty() : tensor<64xf32>
    %2 = hivm.hir.vln ins(%arg0: tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %min = affine.min affine_map<(d0, d1) -> (d0, d1)>(%offset2, %offset1)
    %extracted_slice = tensor.extract_slice %2[%min] [16] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<16xf32>
    %extracted_slice_1 = tensor.extract_slice %extracted_slice[%min_1] [8] [1] : tensor<16xf32> to tensor<8xf32>
    return %extracted_slice_1: tensor<8xf32>
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 64)>
// CHECK-LABEL:   func.func @test_hoistaffine_7(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<512xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]] {
// CHECK:             %[[VAL_5:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_4]]]
// CHECK:             %[[VAL_6:.*]] = tensor.extract_slice %[[ARG0]]{{\[}}%[[VAL_5]]] [256] [1] {to_be_bubbled_slice} : tensor<512xf32> to tensor<256xf32>
// CHECK:             %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_0]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (tensor<256xf32>) {
// CHECK:               %[[VAL_10:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_8]]]
// CHECK:               %[[VAL_11:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[VAL_4]], %[[VAL_8]]]
// CHECK:               %[[VAL_12:.*]] = tensor.empty() : tensor<512xf32>
// CHECK:               %[[VAL_13:.*]] = tensor.insert_slice %[[VAL_9]] into %[[VAL_12]]{{\[}}%[[VAL_5]]] [256] [1] : tensor<256xf32> into tensor<512xf32>
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_11]]] [32] [1] : tensor<512xf32> to tensor<32xf32>
// CHECK:               %[[VAL_15:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:               %[[VAL_16:.*]] = hivm.hir.vln ins(%[[VAL_14]] : tensor<32xf32>) outs(%[[VAL_15]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:               %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_9]]{{\[}}%[[VAL_10]]] [32] [1] : tensor<32xf32> into tensor<256xf32>
// CHECK:               scf.yield %[[VAL_17]] : tensor<256xf32>
// CHECK:             }
// CHECK:             hivm.hir.store ins(%[[VAL_7]] : tensor<256xf32>) outs(%[[ARG1]] : memref<256xf32>) {tiled_op}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
// CHECK:           return
// CHECK:         }
func.func @test_hoistaffine_7(%arg0: tensor<512xf32>, %arg1: memref<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<64xf32>
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %1 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg2]
    %2 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg2]
    %extracted_slice = tensor.extract_slice %arg0[%2] [256] [1] {to_be_bubbled_slice} : tensor<512xf32> to tensor<256xf32>
    %3 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %extracted_slice) -> (tensor<256xf32>) {
      %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg3] 
      %5 = tensor.empty() : tensor<512xf32>
      %inserted_slice = tensor.insert_slice %arg4 into %5[%2] [256] [1] : tensor<256xf32> into tensor<512xf32>
      %6 = affine.apply affine_map<()[s0, s1] -> (s0 * 32 + s1 * 64)>()[%arg2, %arg3]
      %extracted_slice_0 = tensor.extract_slice %inserted_slice[%6] [32] [1] : tensor<512xf32> to tensor<32xf32>
      %7 = tensor.empty() : tensor<32xf32>
      %8 = hivm.hir.vln ins(%extracted_slice_0 : tensor<32xf32>) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
      %inserted_slice_1 = tensor.insert_slice %8 into %arg4[%4] [32] [1] : tensor<32xf32> into tensor<256xf32>
      scf.yield %inserted_slice_1 : tensor<256xf32>
    }
    hivm.hir.store ins(%3 : tensor<256xf32>) outs(%arg1 : memref<256xf32>) {tiled_op}
  } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return
}