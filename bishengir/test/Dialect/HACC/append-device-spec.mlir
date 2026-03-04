// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B1 -split-input-file | FileCheck --check-prefix=910B1 %s
// RUN: bishengir-opt %s --hacc-append-device-spec -split-input-file | FileCheck --check-prefix=UNKNOWN %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9589 -split-input-file | FileCheck --check-prefix=910_9589 %s


// 910B1: dlti.target_system_spec = #dlti.target_system_spec<"NPU"
// 910B1-SAME: "VECTOR_CORE_COUNT", 48
// 910B1-SAME: "UB_SIZE", 1572864
// 910B1-SAME: "L0C_SIZE", 1048576
// 910_9589: "VECTOR_CORE_COUNT", 64
// 910_9589-SAME: "UB_SIZE", 2031616
// 910_9589-SAME: "L0C_SIZE", 2097152
module {

}

// -----

// expected-warning@+1 {{Overwriting the target by the pass option...}}
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {

}

// -----

// expected-warning@+1 {{Overwriting the device spec...}}
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<"NPU" :
    #hacc.target_device_spec<
      #dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>
    >
  >} {

}

// -----

// UNKNOWN: dlti.target_system_spec = #dlti.target_system_spec<"NPU"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {

}