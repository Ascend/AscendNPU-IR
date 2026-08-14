// RUN: bishengir-opt -transform-interpreter -canonicalize -allow-unregistered-dialect %s | FileCheck %s

// ExtendedLoopOutlineOp used to build its SymbolTable *after*
// outlineSingleBlockRegion had already inserted the new func::FuncOp under
// its raw, requested name. If that name collided with an existing symbol
// (as can happen when two independent outlining passes pick the same
// naming scheme), SymbolTable's constructor asserted on the pre-existing
// duplicate instead of letting SymbolTable::insert() auto-rename it.
//
// Here both extended_outline_op calls below request the exact same
// func_name, forcing that collision directly.

// CHECK: func.func @kernel_outlined_vf_0
// CHECK: func.func @kernel_outlined_vf_0_0
// CHECK: func.func @kernel

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {transform.with_named_sequence} {
    func.func @kernel(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
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

        // Non vectorizable op between add and reduce, so the two regions
        // must be outlined into two separate functions.
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

    transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.consume}) {
        %0 = transform.structured.match attributes {"hfusion-auto-vectorize-target-0"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %1 = transform.structured.match attributes {"hfusion-auto-vectorize-target-1"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %tiled_linalg_op_1, %loops_1 = transform.structured.tile_using_for %1 tile_sizes [64] interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.structured.vectorize %tiled_linalg_op vector_sizes [64] : !transform.any_op
        transform.structured.vectorize %tiled_linalg_op_1 vector_sizes [64] : !transform.any_op

        // Both requests use the *same* func_name on purpose to force a
        // symbol-name collision between the two outlined functions.
        %function, %call = transform.loop.extended_outline_op %loops {func_name = "kernel_outlined_vf_0"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %function_1, %call_1 = transform.loop.extended_outline_op %loops_1 {func_name = "kernel_outlined_vf_0"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
