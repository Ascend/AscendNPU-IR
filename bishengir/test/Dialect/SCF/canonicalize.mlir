// RUN: bishengir-opt -canonicalize-ext -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_hoist_same_yield(
// CHECK-SAME: {{.*}}: memref<?x1xf16>, {{.*}}: memref<1x?xf16>, %[[arg2:.*]]: memref<?x?xf16>,
// CHECK-NOT: scf.yield {{.*}} : memref<?x?xf16>
// CHECK: return %[[arg2]] : memref<?x?xf16>
func.func @test_hoist_same_yield_callee_0(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) {return}
func.func @test_hoist_same_yield_callee_1(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) {return}
func.func @test_hoist_same_yield(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) -> memref<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg3[%c0] : memref<4xi64>
    %1 = arith.index_castui %0 : i64 to index
    %2 = scf.index_switch %1 -> memref<?x?xf16> 
    case 0 {
        func.call @test_hoist_same_yield_callee_0(%arg0, %arg1, %arg2, %arg3) : (memref<?x1xf16>, memref<1x?xf16>, memref<?x?xf16>, memref<4xi64>) -> ()
        scf.yield %arg2 : memref<?x?xf16>
    }
    case 1 {
        func.call @test_hoist_same_yield_callee_1(%arg0, %arg1, %arg2, %arg3) : (memref<?x1xf16>, memref<1x?xf16>, memref<?x?xf16>, memref<4xi64>) -> ()
        scf.yield %arg2 : memref<?x?xf16>
    }
    default {
        cf.assert %false, "Invalid tiling key"
        scf.yield %arg2 : memref<?x?xf16>
    }
    return %2 : memref<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @test_hoist_same_yields(
// CHECK-SAME: {{.*}}: memref<?x1xf16>, %[[arg1:.*]]: memref<1x?xf16>, %[[arg2:.*]]: memref<?x?xf16>,
// CHECK-NOT: scf.yield {{.*}}, {{.*}} : memref<1x?xf16>, memref<?x?xf16>
// CHECK: return %[[arg1]], %[[arg2]] : memref<1x?xf16>, memref<?x?xf16>
func.func @test_hoist_same_yields_callee_0(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) {return}
func.func @test_hoist_same_yields_callee_1(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) {return}
func.func @test_hoist_same_yields(%arg0: memref<?x1xf16>, %arg1: memref<1x?xf16>, %arg2: memref<?x?xf16>, %arg3: memref<4xi64>) -> (memref<1x?xf16>, memref<?x?xf16>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg3[%c0] : memref<4xi64>
    %1 = arith.index_castui %0 : i64 to index
    %2:2 = scf.index_switch %1 -> memref<1x?xf16>, memref<?x?xf16>
    case 0 {
        func.call @test_hoist_same_yields_callee_0(%arg0, %arg1, %arg2, %arg3) : (memref<?x1xf16>, memref<1x?xf16>, memref<?x?xf16>, memref<4xi64>) -> ()
        scf.yield %arg1, %arg2 : memref<1x?xf16>, memref<?x?xf16>
    }
    case 1 {
        func.call @test_hoist_same_yields_callee_1(%arg0, %arg1, %arg2, %arg3) : (memref<?x1xf16>, memref<1x?xf16>, memref<?x?xf16>, memref<4xi64>) -> ()
        scf.yield %arg1, %arg2 : memref<1x?xf16>, memref<?x?xf16>
    }
    default {
        cf.assert %false, "Invalid tiling key"
        scf.yield %arg1, %arg2 : memref<1x?xf16>, memref<?x?xf16>
    }
    return %2#0, %2#1 : memref<1x?xf16>, memref<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @move_in_tensor_cast_to_if
// CHECK: %[[RES_1:.*]]:4 = scf.if
// CHECK: %[[THEN_BRC_1:.*]] = hivm.hir.vbrc
// CHECK-DAG: %[[THEN_CAST_1:.*]] = tensor.cast %[[THEN_BRC_1]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK-DAG: %[[THEN_UNARY_1:.*]] = linalg.elemwise_unary [[DUMP:.*]] ins(%[[THEN_CAST_1]] : tensor<?x2xf32>)
// CHECK-DAG: %[[THEN_CAST_2:.*]] = tensor.cast %[[THEN_BRC_1]] : tensor<?x?xf32> to tensor<2x2xf32>
// CHECK-DAG: %[[THEN_CAST_3:.*]] = tensor.cast %[[THEN_BRC_1]] : tensor<?x?xf32> to tensor<2x2xf32>
// CHECK: scf.yield %[[THEN_CAST_2]], %[[THEN_CAST_1]], %[[THEN_CAST_3]], %[[THEN_UNARY_1]]
// CHECK: } else {
// CHECK: %[[ELSE_BRC_1:.*]] = hivm.hir.vbrc
// CHECK-DAG: %[[ELSE_UNARY_1:.*]] = linalg.elemwise_unary [[DUMP:.*]] ins(%[[ELSE_BRC_1]] : tensor<?x2xf32>)
// CHECK-DAG: %[[ELSE_CAST_1:.*]] = tensor.cast %[[ELSE_BRC_1]] : tensor<?x2xf32> to tensor<2x2xf32>
// CHECK-DAG: %[[ELSE_CAST_2:.*]] = tensor.cast %[[ELSE_BRC_1]] : tensor<?x2xf32> to tensor<2x2xf32>
// CHECK: scf.yield %[[ELSE_CAST_1]], %[[ELSE_BRC_1]], %[[ELSE_CAST_2]], %[[ELSE_UNARY_1]]
// CHECK: }
// CHECK: %[[CAST_1:.*]] = tensor.cast %[[RES_1]]
// CHECK: %[[CAST_2:.*]] = tensor.cast %[[RES_1]]
// CHECK: return %[[RES_1]]#0, %[[RES_1]]#0, %[[CAST_1]], %[[CAST_2]], %[[RES_1]]#2, %[[RES_1]]#2, %[[RES_1]]#3

func.func @move_in_tensor_cast_to_if_specified_replacement(%arg0: index, %arg1: index, %arg2: tensor<?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<?x?xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<?x2xf32>) {
    %cst = arith.constant 1.70150435 : f32
    %cst_1 = arith.constant 1.1 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %dim = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg3, %c1 : tensor<?x?xf32>

    %dim_2 = tensor.dim %arg4, %c0 : tensor<?x2xf32>

    %0 = arith.cmpi eq, %arg0, %arg1 : index
    %1, %2, %3, %4 = scf.if %0 -> (tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>) {
        %empty = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
        %empty_1 = tensor.empty(%dim) : tensor<?x2xf32>
        %2 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
        %3 = tensor.cast %2 : tensor<?x?xf32> to tensor<?x2xf32>

        %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3: tensor<?x2xf32>) outs(%empty_1: tensor<?x2xf32>) -> tensor<?x2xf32>
        scf.yield %3, %3, %3, %4 : tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>
    } else {
        %empty = tensor.empty(%dim_2) : tensor<?x2xf32>
        %empty_1 = tensor.empty(%dim_2) : tensor<?x2xf32>
        %2 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%empty : tensor<?x2xf32>) -> tensor<?x2xf32>
        
        %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%2: tensor<?x2xf32>) outs(%empty_1: tensor<?x2xf32>) -> tensor<?x2xf32>
        scf.yield %2, %2, %2, %3 : tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>, tensor<?x2xf32>
    }
    // 2 tensor cast same shape
    %5 = tensor.cast %1 : tensor<?x2xf32> to tensor<2x2xf32>
    %6 = tensor.cast %1 : tensor<?x2xf32> to tensor<2x2xf32>

    // 2 tensor cast different result shape
    %7 = tensor.cast %2 : tensor<?x2xf32> to tensor<2x2xf32>
    %8 = tensor.cast %2 : tensor<?x2xf32> to tensor<?x?xf32>

    // 2 tensor cast same shape chaining
    %9 = tensor.cast %3 : tensor<?x2xf32> to tensor<?x?xf32>
    %10 = tensor.cast %3 : tensor<?x2xf32> to tensor<?x?xf32>
    %11 = tensor.cast %9 : tensor<?x?xf32> to tensor<2x2xf32>
    %12 = tensor.cast %10 : tensor<?x?xf32> to tensor<2x2xf32>

    return %5, %6, %7, %8, %11, %12, %4 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<?x?xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<?x2xf32>
}