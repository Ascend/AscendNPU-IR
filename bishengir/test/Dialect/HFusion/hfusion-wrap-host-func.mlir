// RUN: bishengir-opt %s --hfusion-wrap-host-func --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-wrap-host-func=remove-unused-arguments --split-input-file | FileCheck --check-prefix=CHECK-REMOVE-UNUSED-ARGS %s

module {
  // CHECK: @test_get_tiling_struct_size_function({{.*}}
  // CHECK-REMOVE-UNUSED-ARGS: @test_get_tiling_struct_size_function()
  // CHECK: %[[RES0:.*]] = call @test_0_infer_output_shape_function_get_tiling_struct_size_function()
  // CHECK: return %[[RES0]]
  func.func @test_0_infer_output_shape_function_get_tiling_struct_size_function() -> i64
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
              hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
    %c3_i64 = arith.constant 3 : i64
    return %c3_i64 : i64
  }
  // CHECK: @test_tiling_function
  func.func @test_0_tiling_function(%arg0: tensor<1x?xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
                                    %arg1: tensor<?x1xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
                                    %arg2: tensor<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
                                    %arg3: memref<3xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>})
  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    return
  }
  // CHECK: @test_infer_output_shape_function
  // CHECK: call @test_0_infer_output_shape_function
  func.func @test_0_infer_output_shape_function(
    %arg0: tensor<1x?xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
    %arg1: tensor<?x1xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
    %arg2: tensor<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> tensor<2xindex>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
              hacc.host_func_type = #hacc.host_func_type<infer_output_shape_function>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x?xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?x1xf16>
    %from_elements = tensor.from_elements %dim_0, %dim : tensor<2xindex>
    return %from_elements : tensor<2xindex>
  }
  func.func @test_0_1(%arg0: tensor<1x?xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
                      %arg1: tensor<?x1xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
                      %arg2: tensor<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
                      %arg3: memref<3xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
              hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@test_0_infer_output_shape_function_get_tiling_struct_size_function>,
              hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@test_0_infer_output_shape_function>,
              hacc.tiling_function = #hacc.tiling_function<@test_0_tiling_function>,
              hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    return %arg2 : tensor<?x?xf16>
  }
  func.func @test_0_0(%arg0: tensor<1x?xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
                      %arg1: tensor<?x1xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
                      %arg2: tensor<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
                      %arg3: memref<3xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
              hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@test_0_infer_output_shape_function_get_tiling_struct_size_function>,
              hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@test_0_infer_output_shape_function>,
              hacc.tiling_function = #hacc.tiling_function<@test_0_tiling_function>,
              hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    return %arg2 : tensor<?x?xf16>
  }
  func.func @test(%arg0: tensor<?x1xf16>,
                  %arg1: tensor<1x?xf16>,
                  %arg2: tensor<?x?xf16>,
                  %arg3: memref<3xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg3[%c0] : memref<3xi64>
    %1 = arith.index_castui %0 : i64 to index
    %2 = scf.index_switch %1 -> tensor<?x?xf16>
    case 1 {
      %3 = func.call @test_0_1(%arg1, %arg0, %arg2, %arg3) : (tensor<1x?xf16>, tensor<?x1xf16>, tensor<?x?xf16>, memref<3xi64>) -> tensor<?x?xf16>
      scf.yield %3 : tensor<?x?xf16>
    }
    case 0 {
      %3 = func.call @test_0_0(%arg1, %arg0, %arg2, %arg3) : (tensor<1x?xf16>, tensor<?x1xf16>, tensor<?x?xf16>, memref<3xi64>) -> tensor<?x?xf16>
      scf.yield %3 : tensor<?x?xf16>
    }
    default {
      scf.yield %arg2 : tensor<?x?xf16>
    }
    return %2 : tensor<?x?xf16>
  }
}


