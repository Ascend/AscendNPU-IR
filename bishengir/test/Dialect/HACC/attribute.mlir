// RUN: bishengir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test InferOutputShapeFunctionAttr
//===----------------------------------------------------------------------===//

// CHECK: func.func @device_func0
// CHECK: hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@host_infer_output_shape_function>
func.func @device_func0() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@host_infer_output_shape_function>}{
    return
}

//===----------------------------------------------------------------------===//
// Test InferWorkspaceShapeFunctionAttr
//===----------------------------------------------------------------------===//

// CHECK: func.func @device_func1
// CHECK: hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@host_infer_workspace_shape_function>
func.func @device_func1() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@host_infer_workspace_shape_function>}{
    return
}

//===----------------------------------------------------------------------===//
// Test HACC_KernelArgTypeAttr 
//===----------------------------------------------------------------------===//

func.func @kernel_arg_type() {
  "test.kernel_arg_type"() {
  // CHECK: #hacc.arg_type<ffts_base_address>
  ffts_base_address        = #hacc.arg_type<ffts_base_address>,
  // CHECK: #hacc.arg_type<input>
  input                    = #hacc.arg_type<input>,
  // CHECK: #hacc.arg_type<input_and_output>
  input_and_output         = #hacc.arg_type<input_and_output>,
  // CHECK: #hacc.arg_type<mesh_arg>
  mesh_arg                 = #hacc.arg_type<mesh_arg>,
  // CHECK: #hacc.arg_type<output>
  output                   = #hacc.arg_type<output>,
  // CHECK: #hacc.arg_type<tiling_data>
  tiling_data              = #hacc.arg_type<tiling_data>,
  // CHECK: #hacc.arg_type<tiling_key>
  tiling_key               = #hacc.arg_type<tiling_key>,
  // CHECK: #hacc.arg_type<tiling_struct>
  tiling_struct            = #hacc.arg_type<tiling_struct>,
  // CHECK: #hacc.arg_type<workspace>
  workspace                = #hacc.arg_type<workspace>
  } : () -> ()

  return
}
