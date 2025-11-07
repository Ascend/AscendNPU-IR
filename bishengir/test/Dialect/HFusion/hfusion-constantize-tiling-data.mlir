// RUN: bishengir-opt -hfusion-constantize-tiling-data --cse --split-input-file %s -verify-diagnostics -allow-unregistered-dialect | FileCheck %s
// RUN: bishengir-opt -hfusion-constantize-tiling-data --cse -canonicalize-ext --split-input-file %s -verify-diagnostics -allow-unregistered-dialect | FileCheck %s --check-prefix=CANON
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // CHECK-LABEL: func.func @calculate_tiling(
  // CHECK-NOT: arith.constant 42
  // CHECK: return
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i64, i64, i64
  }
  // CHECK-LABEL: func.func @dynamic_foo_tiling_case_1(
  // CHECK-SAME: hacc.arg_type = #hacc.arg_type<tiling_key>
  // CHECK-SAME: hacc.arg_type = #hacc.arg_type<tiling_data>
  // CHECK-SAME: hacc.arg_type = #hacc.arg_type<tiling_data>
  // CHECK-NOT: hacc.arg_type = #hacc.arg_type<tiling_data>
  // CHECK: arith.constant 42 : i64
  // CHECK: return
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg0 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %arg2 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.sitofp %arg3 : i64 to f32
      %8 = arith.divf %6, %7 : f32
      %9 = arith.sitofp %arg4 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // CHECK-LABEL: func.func @calculate_tiling(
  // CHECK-NOT: arith.constant 42
  // CHECK: return
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i64, i64, i64
  }
  // CHECK-LABEL: func.func @dynamic_foo_tiling_case_1(
  // CHECK: arith.constant 42 : i64
  // CHECK: return
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg0 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %arg2 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.sitofp %arg3 : i64 to f32
      %8 = arith.divf %6, %7 : f32
      %9 = arith.sitofp %arg4 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @dynamic_foo
  // CHECK: %[[RES:.*]]:3 = call @calculate_tiling(
  // CHECK: return
  func.func @dynamic_foo(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) {
    %res:4 = call @calculate_tiling(%arg2) : (tensor<?xf32>) -> (i64, i64, i64, i64)
    %resFin = call @dynamic_foo_tiling_case_1(%res#0, %arg2, %res#1, %res#2, %res#3) : (i64, tensor<?xf32>, i64, i64, i64) -> tensor<?xf32>
    return
  }
}


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // CHECK-LABEL: func.func @calculate_tiling(
  // CHECK-NOT: arith.constant 42
  // CHECK: return
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i64, i64, i64
  }
  // CHECK-LABEL: func.func @dynamic_foo_tiling_case_1(
  // CHECK-NOT: arith.constant 42 : i64
  // CHECK: return
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg4 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %arg4 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.sitofp %arg4 : i64 to f32
      %8 = arith.divf %6, %7 : f32
      %9 = arith.sitofp %arg4 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @dynamic_foo
  // CHECK: %[[RES:.*]]:3 = call @calculate_tiling(%[[INPUT:.*]]) : (tensor<?xf32>) -> (i64, i64, i64)
  // CHECK: call @dynamic_foo_tiling_case_1
  // CHECK-SAME: %[[RES]]#0
  // CHECK-SAME: %[[INPUT]]
  // CHECK-SAME: %[[RES]]#1
  // CHECK-SAME: %[[RES]]#2
  // CHECK: return
  func.func @dynamic_foo(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) {
    %res:4 = call @calculate_tiling(%arg2) : (tensor<?xf32>) -> (i64, i64, i64, i64)
    %resFin = call @dynamic_foo_tiling_case_1(%res#0, %arg2, %res#1, %res#2, %res#3) : (i64, tensor<?xf32>, i64, i64, i64) -> tensor<?xf32>
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // CHECK-LABEL: func.func @calculate_tiling(
  // CHECK-NOT: arith.constant 42
  // CHECK: return
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i64, i64, i64
  }
  // CHECK-LABEL: func.func @dynamic_foo_tiling_case_1(
  // CHECK-NOT: arith.constant 42 : i64
  // CHECK: return
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg4 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %arg4 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.sitofp %arg4 : i64 to f32
      %8 = arith.divf %6, %7 : f32
      %9 = arith.sitofp %arg4 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @dynamic_foo_tiling_case_2(
  // CHECK: arith.constant 42 : i64
  // CHECK: return
  func.func @dynamic_foo_tiling_case_2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg0 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %arg2 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.sitofp %arg2 : i64 to f32
      %8 = arith.divf %6, %7 : f32
      %9 = arith.sitofp %arg2 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @dynamic_foo
  // CHECK: %[[RES:.*]]:3 = call @calculate_tiling(
  // CHECK: = call @dynamic_foo_tiling_case_1(
  // CHECK-SAME: %[[RES]]#0
  // CHECK-SAME: %[[RES]]#1
  // CHECK: = call @dynamic_foo_tiling_case_2(
  // CHECK-SAME: %[[RES]]#0
  // CHECK-SAME: %[[RES]]#1
  // CHECK: return
  func.func @dynamic_foo(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) {
    %res:4 = call @calculate_tiling(%arg2) : (tensor<?xf32>) -> (i64, i64, i64, i64)
    %resFin = call @dynamic_foo_tiling_case_1(%res#0, %arg2, %res#1, %res#2, %res#3) : (i64, tensor<?xf32>, i64, i64, i64) -> tensor<?xf32>
    %resFin2 = call @dynamic_foo_tiling_case_2(%res#0, %arg2, %res#1, %res#2, %res#3) : (i64, tensor<?xf32>, i64, i64, i64) -> tensor<?xf32>
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i64, i64, i64
  }
  // expected-error@below {{Non i64 device tiling data args}}
  func.func @dynamic_foo_tiling_case_1(%arg0: i32 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // expected-error@below {{Non i64 calculate tiling return type}}
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i32 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i32

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64, %triple_size_i64 : i64, i32, i64, i64
  }
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // expected-error@below {{Calc tiling order and usage inconsistency}}
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42_i64 = arith.constant 42 : i64

    // Get the size of the input tensor
    %size = tensor.dim %arg0, %c0 : tensor<?xf32>
    %size_i64 = arith.index_cast %size : index to i64
    %c53_i64 = arith.constant 53 : i64

    // Calculate double and triple of the size
    %double_size = arith.muli %size, %c1 : index
    %double_size_i64 = arith.index_cast %double_size : index to i64
    %triple_size = arith.addi %double_size, %size : index
    %triple_size_i64 = arith.index_cast %triple_size : index to i64

    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, %size] : tensor<?xf32> into tensor<?x1xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?x1xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %size_i64, %c42_i64,  %double_size_i64: i64, i64, i64
  }
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                       %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?xf32>)
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      linalg.yield %1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

module {
  func.func @calc_tiling(%arg0: i64) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}) {
    %zack = arith.constant 4 : i64
    return %zack : i64
  }

  // CHECK-LABEL: func.func @host(
  // CHECK: call @device(
  // CHECK: return
  func.func @host(%hello: i64) {
    %res = call @calc_tiling(%hello) : (i64) -> i64
    %what = call @device(%res, %hello) : (i64, i64) -> (i64)
    return
  }

  func.func private @external_func() -> i1

  // arguments get emptied
  // CHECK-LABEL: func.func @device(%arg0: i64)
  // CHECK: arith.constant 4 : i64
  // CHECK: return
  // CANON-LABEL: func.func @device(%arg0: i64)
  // CANON-NOT: scf.if
  // CANON: arith.constant 4 : i64
  // CANON: arith.subi
  // CANON: arith.divsi
  // CANON: arith.remsi
  // CANON: return
  func.func @device(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}, %arg1 : i64) -> i64
  attributes {hacc.tiling_function = #hacc.tiling_function<@calc_tiling>} {
    %c0_i64 = arith.constant 0 : i64

    // Use an external function call
    %external_cond = call @external_func() : () -> i1

    // Combine conditions
    %condition = arith.cmpi eq, %arg0, %c0_i64 : i64
    %final_cond = arith.andi %external_cond, %condition : i1
    // Use the combined condition with a result
    %result = scf.if %condition -> (i64) {
      %c1_i64 = arith.constant 1 : i64
      %sum = arith.addi %c1_i64, %arg1 : i64
      %product = arith.muli %sum, %arg0 : i64
      %if_result = arith.andi %product, %c1_i64 : i64
      scf.yield %if_result : i64
    } else {
      %diff = arith.subi %arg0, %arg1 : i64
      %quotient = arith.divsi %diff, %arg1 : i64
      %else_result = arith.remsi %quotient, %diff : i64
      scf.yield %else_result : i64
    }
    return %result : i64
  }
}

// -----

module {
  // CHECK-LABEL: func.func @forward_0_tiling_func(
  // CHECK: return
  func.func @forward_0_tiling_func(%arg0: tensor<20xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  // CHECK-LABEL: func.func @forward_0_0(
  // CHECK: return
  func.func @forward_0_0(%arg0: tensor<20xf32>,
                         %arg1: tensor<2x20xf32>,
                         %arg2: tensor<2x20xf32>,
                         %arg3: tensor<2x20xf32>,
                         %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}) -> tensor<2x20xf32>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@forward_0_tiling_func>} {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<2x10xf32>
    %1 = tensor.empty() : tensor<2x20xf32>
    return %1 : tensor<2x20xf32>
  }
  // CHECK-LABEL: func.func @forward_1_tiling_func(
  // CHECK: return
  func.func @forward_1_tiling_func(%arg0: tensor<20xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  // CHECK-LABEL: func.func @forward_1_0(
  // CHECK: return
  func.func @forward_1_0(%arg0: tensor<20xf32>,
                         %arg1: tensor<2x20xf32>,
                         %arg2: tensor<2x20xf32>,
                         %arg3: tensor<2x20xf32>,
                         %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}) -> tensor<2x20xf32>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@forward_1_tiling_func>} {
    %0 = tensor.empty() : tensor<2x10xf32>
    %1 = tensor.empty() : tensor<2x20xf32>
    return %1 : tensor<2x20xf32>
  }
  // CHECK-LABEL: func.func @forward
  // CHECK: = call @forward_0_0(
  // CHECK: = call @forward_1_0(
  // CHECK: return
  func.func @forward(%arg0: tensor<2x10xf32>,
                     %arg1: tensor<2x20xf32>,
                     %arg2: tensor<20x10xf32>,
                     %arg3: tensor<20xf32>,
                     %arg4: tensor<20x20xf32>,
                     %arg5: tensor<20xf32>,
                     %arg6: tensor<10x20xf32>,
                     %arg7: tensor<10xf32>,
                     %arg8: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                     %arg9: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<2x20xf32>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
    %0 = tensor.empty() : tensor<2x20xf32>
    %1 = linalg.matmul_transpose_b ins(%arg0, %arg2 : tensor<2x10xf32>, tensor<20x10xf32>) outs(%0 : tensor<2x20xf32>) -> tensor<2x20xf32>
    %2 = tensor.empty() : tensor<2x20xf32>
    %3 = call @forward_0_0(%arg3, %1, %arg1, %2, %arg8) : (tensor<20xf32>, tensor<2x20xf32>, tensor<2x20xf32>, tensor<2x20xf32>, i64) -> tensor<2x20xf32>
    %4 = tensor.empty() : tensor<2x20xf32>
    %5 = linalg.matmul_transpose_b ins(%3, %arg4 : tensor<2x20xf32>, tensor<20x20xf32>) outs(%4 : tensor<2x20xf32>) -> tensor<2x20xf32>
    %6 = tensor.empty() : tensor<2x20xf32>
    %7 = call @forward_1_0(%arg5, %5, %arg1, %6, %arg9) : (tensor<20xf32>, tensor<2x20xf32>, tensor<2x20xf32>, tensor<2x20xf32>, i64) -> tensor<2x20xf32>
    return %3 : tensor<2x20xf32>
  }
  // CHECK-LABEL: func.func @caller
  // CHECK: = call @forward(
  // CHECK: return
  func.func @caller(%arg0: tensor<2x10xf32>,
                    %arg1: tensor<2x20xf32>,
                    %arg2: tensor<20x10xf32>,
                    %arg3: tensor<20xf32>,
                    %arg4: tensor<20x20xf32>,
                    %arg5: tensor<20xf32>,
                    %arg6: tensor<10x20xf32>,
                    %arg7: tensor<10xf32>,
                    %arg8: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                    %arg9: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<2x20xf32>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %3 = call @forward(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<2x10xf32>, tensor<2x20xf32>, tensor<20x10xf32>, tensor<20xf32>, tensor<20x20xf32>, tensor<20xf32>, tensor<10x20xf32>, tensor<10xf32>, i64, i64) -> tensor<2x20xf32>
    return %3 : tensor<2x20xf32>
  }
}


// -----

module {
  func.func @tiling_func(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) ->
                                                       (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %ret0 = "some_calculation"() : () -> i64
    %ret1 = "some_calculation"() : () -> i64
    // CHECK-NOT: arith.constant
    %ret2 = arith.constant 42: i64
    %ret3 = "some_calculation"() : () -> i64
    return %ret0, %ret1, %ret2, %ret3 : i64, i64, i64, i64
  }
  func.func @device_kernel_tiling_0(%arg0: tensor<?x1xf16>,
                                    %arg1: tensor<1x?xf16>,
                                    %arg2: tensor<?x?xf16>,
                                    %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                    %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                    %arg5: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                    %arg6: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@tiling_func>} {
    // CHECK: %[[C42:.*]] = arith.constant 42 : i64
    "some_use"(%arg3) : (i64) -> ()
    "some_use"(%arg4) : (i64) -> ()
    // CHECK: "some_use"(%[[C42]]) : (i64) -> ()
    "some_use"(%arg5) : (i64) -> ()
    "some_use"(%arg6) : (i64) -> ()
    %ret0 = "some_op"() : () -> tensor<?x?xf16>
    return %ret0 : tensor<?x?xf16>
  }
  func.func @device_kernel_tiling_1(%arg0: tensor<?x1xf16>,
                                    %arg1: tensor<1x?xf16>,
                                    %arg2: tensor<?x?xf16>,
                                    %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                    %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                    %arg5: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                                    %arg6: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@tiling_func>} {
    // CHECK: %[[C42:.*]] = arith.constant 42 : i64
    "some_use"(%arg3) : (i64) -> ()
    "some_use"(%arg4) : (i64) -> ()
    // CHECK: "some_use"(%[[C42]]) : (i64) -> ()
    "some_use"(%arg5) : (i64) -> ()
    "some_use"(%arg6) : (i64) -> ()
    %ret0 = "some_op"() : () -> tensor<?x?xf16>
    return %ret0 : tensor<?x?xf16>
  }
  func.func @main(%arg0: tensor<?x1xf16>,
                  %arg1: tensor<1x?xf16>,
                  %arg2: tensor<?x?xf16>,
                  %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                  %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                  %arg5: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                  %arg6: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: %[[C42:.*]] = arith.constant 42 : i64
    %0 = arith.index_castui %arg3 : i64 to index
    // CHECK: "some_use"(%[[C42]]) : (i64) -> ()
    "some_use"(%arg5) : (i64) -> ()
    %1 = scf.index_switch %0 -> tensor<?x?xf16>
    case 1 {
      %2 = func.call @device_kernel_tiling_1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>, i64, i64, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    case 0 {
      %2 = func.call @device_kernel_tiling_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>, i64, i64, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    default {
      %false = arith.constant false
      cf.assert %false, "Invalid tiling key"
      %2 = ub.poison : tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    return %1 : tensor<?x?xf16>
  }
}

// -----

module {
  func.func @tiling_func(%arg0: tensor<?xf16>) ->
  (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
   i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
   i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    // When the tiling key is constant, but some tiling data is not constant,
    // tiling key cannot be constantized.
    // CHECK: arith.constant
    %ret0 = arith.constant 1 : i64
    // CHECK: "some_calculation"
    %ret1 = "some_calculation"() : () -> i64
    // CHECK-NOT: arith.constant
    %ret2 = arith.constant 1 : i64
    return %ret0, %ret1, %ret2 : i64, i64, i64
  }
  func.func @device_func(
    %arg0: tensor<?xf16>,
    %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
    %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
    %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> ()
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@tiling_func>} {
    return
  }
}
