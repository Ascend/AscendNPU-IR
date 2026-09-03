// RUN: bishengir-opt -hfusion-pack-tiling-data %s --split-input-file -allow-unregistered-dialect | FileCheck %s
// RUN: bishengir-opt -hfusion-pack-tiling-data="pack-tiling-key=false" %s --split-input-file -allow-unregistered-dialect | FileCheck %s -check-prefix=CHECK-NOT-PACK-TILING-KEY
// RUN: bishengir-opt -hfusion-pack-tiling-data="emit-get-tiling-struct-size-function" --split-input-file -allow-unregistered-dialect %s | FileCheck %s -check-prefix=CHECK-EMIT-GET-TILING-SIZE
// RUN: bishengir-opt -hfusion-pack-tiling-data="include-symbols=foo0_tiling_func" %s --split-input-file -allow-unregistered-dialect | FileCheck %s -check-prefix=CHECK-INCLUDE-SYMBOL

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  // CHECK: func.func @calculate_tiling(%{{.*}}: tensor<?xf32>
  // CHECK: %[[TILING:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>})
  //
  // CHECK-NOT-PACK-TILING-KEY: func.func @calculate_tiling(%{{.*}}: tensor<?xf32>
  // CHECK-NOT-PACK-TILING-KEY:      %[[TILING_KEY:.*]]: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK-NOT-PACK-TILING-KEY-SAME: %[[TILING_STRUCT:.*]]: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>})
  func.func @calculate_tiling(%arg0: tensor<?xf32>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                        i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.muli %dim, %c1 : index
    %2 = arith.addi %1, %dim : index
    %3 = arith.index_cast %2 : index to i64
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<?x1xf32>) outs(%expanded : tensor<?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<?x1xf32>
    // CHECK: memref.store %{{.*}}, %[[TILING]]
    // CHECK: memref.store %{{.*}}, %[[TILING]]
    // CHECK-NOT-PACK-TILING-KEY: llvm.store %{{.*}}, %[[TILING_KEY]]
    // CHECK-NOT-PACK-TILING-KEY: memref.store %{{.*}}, %[[TILING_STRUCT]]
    // CHECK: return
    return %0, %3 : i64, i64
  }
  // CHECK: func.func @dynamic_foo_tiling_case_1({{.*}}: tensor<?xf32>,
  // CHECK: %[[TILING1:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?xf32>
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_KEY:.*]]: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_STRUCT:.*]]: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?xf32>
  func.func @dynamic_foo_tiling_case_1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?xf32>
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    // CHECK: memref.load %[[TILING1]]
    // CHECK: memref.load %[[TILING1]]
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg2 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.divf %4, %3 : f32
      %6 = arith.divf %5, %3 : f32
      %7 = arith.divf %6, %3 : f32
      linalg.yield %7 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK: func.func @dynamic_foo_tiling_case_2({{.*}}: tensor<?xf32>,
  // CHECK: %[[TILING2:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?xf32>
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_KEY:.*]]: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_STRUCT:.*]]: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?xf32>
  func.func @dynamic_foo_tiling_case_2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                       %arg1: tensor<?xf32>,
                                       %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?xf32>
  attributes {hacc.tiling_function = #hacc.tiling_function<@calculate_tiling>} {
    %c42_i64 = arith.constant 42 : i64
    // CHECK: memref.load %[[TILING2]]
    // CHECK: memref.load %[[TILING2]]
    %0 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in : f32
      %3 = arith.sitofp %arg0 : i64 to f32
      %4 = arith.addf %2, %3 : f32
      %5 = arith.sitofp %c42_i64 : i64 to f32
      %6 = arith.divf %4, %5 : f32
      %7 = arith.divf %6, %5 : f32
      %8 = arith.divf %7, %5 : f32
      linalg.yield %8 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// // -----

module {
  // CHECK: func.func @foo_tiling_func
  // CHECK-SAME: {{.*}}: tensor<?x1xf16>, %[[TILING_STRUCT3:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  func.func @foo_tiling_func(%arg0: tensor<?x1xf16>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                         i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %ret:2 = "some_computation"(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    return %ret#0, %ret#1 : i64, i64
  }
  // CHECK: func.func @foo_10
  // CHECK-SAME: {{.*}}: tensor<?x1xf16>, %[[TILING_STRUCT2:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  func.func @foo_10(%arg0: tensor<?x1xf16>,
                   %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                   %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} {
    %ret = "some_computation"(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
  // CHECK: func.func @foo_0
  // CHECK-SAME: {{.*}}: tensor<?x1xf16>, %[[TILING_STRUCT1:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  func.func @foo_0(%arg0: tensor<?x1xf16>,
                   %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                   %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} {
    %ret = "some_computation"(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
  // CHECK: func.func @device_caller
  // CHECK-SAME: {{.*}}: tensor<?x1xf16>
  func.func @device_caller(%arg0: tensor<?x1xf16>) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2xi64>
    // CHECK: call @foo_tiling_func({{.*}}, %[[ALLOC]]) : (tensor<?x1xf16>, memref<2xi64>) -> ()

    // CHECK-NOT-PACK-TILING-KEY: %[[ALLOC:.*]] = memref.alloc() : memref<1xi64>
    // CHECK-NOT-PACK-TILING-KEY: %[[ALLOCA:.*]] = llvm.alloca
    // CHECK-NOT-PACK-TILING-KEY: call @foo_tiling_func({{.*}}, %[[ALLOCA]], %[[ALLOC]]) : (tensor<?x1xf16>, !llvm.ptr, memref<1xi64>) -> ()
    %0:2 = call @foo_tiling_func(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    %1 = arith.index_castui %0#0 : i64 to index
    %2 = scf.index_switch %1 -> tensor<?x?xf16>
    case 10 {
      // CHECK: func.call @foo_10({{.*}}, %[[ALLOC]])
      // CHECK-NOT-PACK-TILING-KEY: call @foo_10({{.*}}, %[[ALLOCA]], %[[ALLOC]]) : (tensor<?x1xf16>, !llvm.ptr, memref<1xi64>)
      %2 = func.call @foo_10(%arg0, %0#0, %0#1) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    case 0 {
      // CHECK: func.call @foo_0({{.*}}, %[[ALLOC]])
      // CHECK-NOT-PACK-TILING-KEY: call @foo_0({{.*}}, %[[ALLOCA]], %[[ALLOC]]) : (tensor<?x1xf16>, !llvm.ptr, memref<1xi64>)
      %2 = func.call @foo_0(%arg0, %0#0, %0#1) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    default {
      %false = arith.constant false
      cf.assert %false, "Invalid tiling key"
      %posion = ub.poison : tensor<?x?xf16>
      scf.yield %posion : tensor<?x?xf16>
    }
    // CHECK-NOT: annotation
    annotation.mark %2 {hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} : tensor<?x?xf16>
    return %2 : tensor<?x?xf16>
  }
}

// -----

module {
  func.func @foo_tiling_func(%arg0: tensor<?x1xf16>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                         i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %ret:2 = "some_computation"(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    return %ret#0, %ret#1 : i64, i64
  }
  func.func @foo_1(%arg0: tensor<?x1xf16>,
                   %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                   %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} {
    %ret = "some_computation"(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
  func.func @foo_0(%arg0: tensor<?x1xf16>,
                   %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                   %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} {
    %ret = "some_computation"(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
  // CHECK: func.func @device_caller
  // CHECK: {{.*}}: tensor<?x1xf16>,
  // CHECK: %[[TILING_STRUCT0:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>},
  // CHECK: %[[TILING_STRUCT1:.*]]: memref<2xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  //
  // CHECK-NOT-PACK-TILING-KEY: func.func @device_caller
  // CHECK-NOT-PACK-TILING-KEY: {{.*}}: tensor<?x1xf16>,
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_KEY0:.*]]: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_STRUCT0:.*]]: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_KEY1:.*]]: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK-NOT-PACK-TILING-KEY: %[[TILING_STRUCT1:.*]]: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}
  func.func @device_caller(%arg0: tensor<?x1xf16>,
                           %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                           %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
                           %arg3: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                           %arg4: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> (tensor<?x?xf16>,tensor<?x?xf16>)
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = arith.index_castui %arg1 : i64 to index
    %1 = scf.index_switch %0 -> tensor<?x?xf16>
    case 1 {
      // CHECK: func.call @foo_1({{.*}}, %[[TILING_STRUCT0]])
      %2 = func.call @foo_1(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    case 0 {
      // CHECK: func.call @foo_0({{.*}}, %[[TILING_STRUCT0]])
      %2 = func.call @foo_0(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %2 : tensor<?x?xf16>
    }
    default {
      %false = arith.constant false
      cf.assert %false, "Invalid tiling key"
      %posion = ub.poison : tensor<?x?xf16>
      scf.yield %posion : tensor<?x?xf16>
    }
    // CHECK-NOT: annotation.mark
    annotation.mark %1 {hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} : tensor<?x?xf16>

    %2 = arith.index_castui %arg3 : i64 to index
    %3 = scf.index_switch %2 -> tensor<?x?xf16>
    case 1 {
      // CHECK: func.call @foo_1({{.*}}, %[[TILING_STRUCT1]])
      %4 = func.call @foo_1(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %4 : tensor<?x?xf16>
    }
    case 0 {
      // CHECK: func.call @foo_0({{.*}}, %[[TILING_STRUCT1]])
      %4 = func.call @foo_0(%arg0, %arg3, %arg4) : (tensor<?x1xf16>, i64, i64) -> tensor<?x?xf16>
      scf.yield %4 : tensor<?x?xf16>
    }
    default {
      %false = arith.constant false
      cf.assert %false, "Invalid tiling key"
      %posion = ub.poison : tensor<?x?xf16>
      scf.yield %posion : tensor<?x?xf16>
    }
    // CHECK-NOT: annotation.mark
    annotation.mark %3 {hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func>} : tensor<?x?xf16>
    return %1,%3 : tensor<?x?xf16>,tensor<?x?xf16>
  }
}

// -----

module {
  // CHECK-EMIT-GET-TILING-SIZE: func.func @foo_get_tiling_struct_size_function() -> i64
  // CHECK-EMIT-GET-TILING-SIZE: attributes {hacc.function_kind = #hacc.function_kind<HOST>
  // CHECK-EMIT-GET-TILING-SIZE:             hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
  // CHECK-EMIT-GET-TILING-SIZE:   %[[C2:.*]] = arith.constant 2 : i64
  // CHECK-EMIT-GET-TILING-SIZE:   return %[[C2]] : i64
  // CHECK-EMIT-GET-TILING-SIZE: }
  func.func @foo_tiling_function(%arg0: tensor<?x1xf16>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                             i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %ret:2 = "some_computation"(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    return %ret#0, %ret#1 : i64, i64
  }
  func.func @foo(%arg0: tensor<?x1xf16>,
                 %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                 %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_function>} {
    %ret = "some_computation"(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, i64, i64) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
}
// -----

module {
  // CHECK-EMIT-GET-TILING-SIZE: func.func @foo_get_tiling_struct_size_function() -> i64
  // CHECK-EMIT-GET-TILING-SIZE: attributes {hacc.function_kind = #hacc.function_kind<HOST>
  // CHECK-EMIT-GET-TILING-SIZE:             hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
  // CHECK-EMIT-GET-TILING-SIZE:   %[[C0:.*]] = arith.constant 0 : i64
  // CHECK-EMIT-GET-TILING-SIZE:   return %[[C0]] : i64
  // CHECK-EMIT-GET-TILING-SIZE: }
  func.func @foo_tiling_function()
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    return
  }
  func.func @foo()
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_function>} {
    return
  }
}

// -----

module {
  // CHECK-INCLUDE-SYMBOL: foo0_tiling_func
  // CHECK-INCLUDE-SYMBOL: memref
  func.func @foo0_tiling_func(%arg0: tensor<?x1xf16>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                          i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %ret:2 = "some_computation"(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    return %ret#0, %ret#1 : i64, i64
  }

  // CHECK-INCLUDE-SYMBOL: foo0
  // CHECK-INCLUDE-SYMBOL: #hacc.tiling_function<@foo0_tiling_func>
  func.func @foo0(%arg0: tensor<?x1xf16>,
                  %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                  %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo0_tiling_func>} {
    return
  }

  // CHECK-INCLUDE-SYMBOL: foo_tiling_func1
  // CHECK-INCLUDE-SYMBOL-NOT: memref
  func.func @foo_tiling_func1(%arg0: tensor<?x1xf16>) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                                                          i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %ret:2 = "some_computation"(%arg0) : (tensor<?x1xf16>) -> (i64, i64)
    return %ret#0, %ret#1 : i64, i64
  }

  // CHECK-INCLUDE-SYMBOL: foo1
  // CHECK-INCLUDE-SYMBOL: #hacc.tiling_function<@foo_tiling_func1>
  func.func @foo1(%arg0: tensor<?x1xf16>,
                  %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
                  %arg2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@foo_tiling_func1>} {
    return
  }
}

// -----

module {
  // CHECK-EMIT-GET-TILING-SIZE: func.func @dummy_get_tiling_struct_size_function
  // CHECK-EMIT-GET-TILING-SIZE:   %[[C1:.*]] = arith.constant 1 : i64
  // CHECK-EMIT-GET-TILING-SIZE:   return %[[C1]] : i64
  func.func @dummy_tiling_function(%arg0: memref<1xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    return
  }
  func.func @dummy(%arg0: memref<12xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>})
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@dummy_tiling_function>} {
    return
  }
}
