// RUN: bishengir-opt %s --vf-fusion="fusion-mode=ub-aware-op" | FileCheck %s
//
// Regression test: UB-aware VFFusion must not crash on vector.transfer_read.
//
// vector.transfer_read implements DestinationStyleOpInterface but its
// getDpsInitsMutable() returns an empty range (0 inits, 1 result).
// Without the bounds guard in canReuseInputForOutput, the estimator calls
// getTiedOpOperand(result#0) which indexes into the empty inits range,
// triggering:  Assertion `index < length && "index is out of bounds"' failed.
//
// The crash requires:
//   1. vector.transfer_write fused with its producer into one group, then
//   2. vector.transfer_read (consuming the write's tensor result) tries to
//      merge into the group.  Its vector result is used outside the merged
//      group, making it an external output evaluated by canReuseInputForOutput.
//
// CHECK: func.func

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"UB_SIZE", 32768 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {

func.func @test_no_crash_transfer_read_dps(
    %input: tensor<64xf32>, %dest: tensor<64xf32>
) -> tensor<64xf32> attributes {hivm.vector_function, outline = true, vector_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    // vector.transfer_read from tensor: 1 result, 0 DPS inits.
    %vec_in = vector.transfer_read %input[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %empty = tensor.empty() : tensor<64xf32>
    // vector.transfer_write: 1 result, 1 DPS init (%empty).
    // Fuses with transfer_read above into one group.
    %written = vector.transfer_write %vec_in, %empty[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    // Second transfer_read merges into the group above; its vector result
    // is used by the final transfer_write (outside the merged group),
    // making it an external output that canReuseInputForOutput evaluates.
    %readback = vector.transfer_read %written[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %result = vector.transfer_write %readback, %dest[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %result : tensor<64xf32>
}

} // module
