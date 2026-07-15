// RUN: bishengir-opt -transform-interpreter -canonicalize --split-input-file -allow-unregistered-dialect %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {transform.with_named_sequence} {
    // CHECK: func.func @test_outline_multiple_loop_into_one_function_outlined_vf_0
    // CHECK: func.func @test_outline_multiple_loop_into_one_function
    func.func @test_outline_multiple_loop_into_one_function(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> attributes {test} {
        %c0 = arith.constant 0 : index
        %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
        %0 = tensor.empty(%dim) : tensor<?xf32>
        %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) attrs = {"hfusion-auto-vectorize-target-0"} {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
            %4 = arith.addf %in, %in_0 : f32
            linalg.yield %4 : f32
        } -> tensor<?xf32>

        %2 = tensor.empty() : tensor<f32>
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%1 : tensor<?xf32>) outs(%2 : tensor<f32>)  attrs = {"hfusion-auto-vectorize-target-1"} {
        ^bb0(%in: f32, %out: f32):
            %4 = arith.maxnumf %out, %in : f32
            linalg.yield %4 : f32
        } -> tensor<f32>
        return %3 : tensor<f32>
    }

    // `transform.loops.fuse_into_containting_op` can be utilized here to fuse this two loops. For testing
    // ability of outlining multiple loops into one VF function, keep separated.
    transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.consume}) {
        %0 = transform.structured.match attributes {"hfusion-auto-vectorize-target-0"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %match = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_licm to %match : !transform.any_op
        %match_1 = transform.structured.match attributes {test} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %match_1 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
        } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.any_op
        %1 = transform.structured.match attributes {"hfusion-auto-vectorize-target-1"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op_1, %loops_1 = transform.structured.tile_using_for %1 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.structured.vectorize %tiled_linalg_op vector_sizes [64] : !transform.any_op
        transform.structured.vectorize %tiled_linalg_op_1 vector_sizes [64] : !transform.any_op

        %function, %call = transform.loop.extended_outline_op %loops, %loops_1 {func_name = "test_outline_multiple_loop_into_one_function_outlined_vf_0"} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}

// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {transform.with_named_sequence} {
    // CHECK: func.func @test_outline_multiple_loop_into_separate_functions_outlined_vf_0
    // CHECK: func.func @test_outline_multiple_loop_into_separate_functions_outlined_vf_1
    // CHECK: func.func @test_outline_multiple_loop_into_separate_functions
    func.func @test_outline_multiple_loop_into_separate_functions(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
        %c0 = arith.constant 0 : index
        %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
        %0 = tensor.empty(%dim) : tensor<?xf32>
        %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) attrs = {"hfusion-auto-vectorize-target-0"} {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
            %4 = arith.addf %in, %in_0 : f32
            linalg.yield %4 : f32
        } -> tensor<?xf32>

        // Non vectorizable ops between add and reduce
        %extracted_slice = tensor.extract_slice %1[64] [128] [1] : tensor<?xf32> to tensor<128xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice into %0[128] [128] [1] : tensor<128xf32> into tensor<?xf32>

        %2 = tensor.empty() : tensor<f32>
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%inserted_slice : tensor<?xf32>) outs(%2 : tensor<f32>)  attrs = {"hfusion-auto-vectorize-target-1"} {
        ^bb0(%in: f32, %out: f32):
            %4 = arith.maxnumf %out, %in : f32
            linalg.yield %4 : f32
        } -> tensor<f32>
        return %3 : tensor<f32>
    }

    // Since existance of non-vectorizable ops between two linalg statements, we can't fuse then together. Extract them
    // into two vf functions.
    transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.consume}) {
        %0 = transform.structured.match attributes {"hfusion-auto-vectorize-target-0"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %1 = transform.structured.match attributes {"hfusion-auto-vectorize-target-1"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op_1, %loops_1 = transform.structured.tile_using_for %1 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.structured.vectorize %tiled_linalg_op vector_sizes [64] : !transform.any_op
        transform.structured.vectorize %tiled_linalg_op_1 vector_sizes [64] : !transform.any_op

        %function, %call = transform.loop.extended_outline_op %loops {func_name = "test_outline_multiple_loop_into_separate_functions_outlined_vf_0"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %function_1, %call_1 = transform.loop.extended_outline_op %loops_1 {func_name = "test_outline_multiple_loop_into_separate_functions_outlined_vf_1"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}

// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {transform.with_named_sequence} {
    // CHECK: func.func @test_outline_multiple_loop_into_separate_functions_outlined_vf_0
    // CHECK: func.func @test_outline_multiple_loop_into_separate_functions
    func.func @test_outline_multiple_loop_into_separate_functions(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 4 : index
        %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
        %0 = tensor.empty(%dim) : tensor<?xf32>
        %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) attrs = {"hfusion-auto-vectorize-target-0"} {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
            %4 = arith.addf %in, %in_0 : f32
            linalg.yield %4 : f32
        } -> tensor<?xf32>

        // Non vectorizable ops between add and reduce
        // %extracted_slice = tensor.extract_slice %1[64] [128] [1] : tensor<?xf32> to tensor<128xf32>
        // %inserted_slice = tensor.insert_slice %extracted_slice into %0[128] [128] [1] : tensor<128xf32> into tensor<?xf32>
        %not_important = arith.addi %dim, %c64 : index
        %div = arith.divsi %dim, %c2 : index
        %ext = tensor.extract_slice %1[0][%div][1] : tensor<?xf32> to tensor<?xf32>

        %2 = tensor.empty() : tensor<f32>
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%ext : tensor<?xf32>) outs(%2 : tensor<f32>)  attrs = {"hfusion-auto-vectorize-target-1"} {
        ^bb0(%in: f32, %out: f32):
            %4 = arith.maxnumf %out, %in : f32
            linalg.yield %4 : f32
        } -> tensor<f32>
        "prevent_cse" (%not_important) : (index) -> ()
        return %3 : tensor<f32>
    }

    // Since existance of non-vectorizable ops between two linalg statements, we can't fuse then together. Extract them
    // into two vf functions.
    transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.consume}) {
        %0 = transform.structured.match attributes {"hfusion-auto-vectorize-target-0"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %1 = transform.structured.match attributes {"hfusion-auto-vectorize-target-1"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op_1, %loops_1 = transform.structured.tile_using_for %1 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.structured.vectorize %tiled_linalg_op vector_sizes [64] : !transform.any_op
        transform.structured.vectorize %tiled_linalg_op_1 vector_sizes [64] : !transform.any_op

        %function, %call = transform.loop.extended_outline_op %loops, %loops_1 {func_name = "test_outline_multiple_loop_into_separate_functions_outlined_vf_0"} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
