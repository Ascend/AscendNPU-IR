// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B1 -split-input-file | FileCheck --check-prefix=910B1 %s
// RUN: bishengir-opt %s --hacc-append-device-spec -split-input-file | FileCheck --check-prefix=UNKNOWN %s
// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend950PR_9589 custom-aic-number=5 custom-aiv-number=10" -split-input-file | FileCheck --check-prefix=OVERRIDE %s
// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend950 custom-aic-number=8 custom-aiv-number=16" -split-input-file | FileCheck --check-prefix=GENERIC950 %s
// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend910B1 custom-aic-number=10" -split-input-file | FileCheck --check-prefix=ONLY_AIC %s
// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend910B1 custom-aiv-number=16" -split-input-file | FileCheck --check-prefix=ONLY_AIV %s
// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend910B1 custom-aic-number=1000 custom-aiv-number=1000" -split-input-file | FileCheck --check-prefix=NO_LIMIT %s


// 910B1: dlti.target_system_spec = #dlti.target_system_spec<"NPU"
// 910B1-SAME: "VECTOR_CORE_COUNT", 48
// 910B1-SAME: "UB_SIZE", 1572864
// 910B1-SAME: "L0C_SIZE", 1048576
// OVERRIDE: "AI_CORE_COUNT", 5
// OVERRIDE-SAME: "CUBE_CORE_COUNT", 5
// OVERRIDE-SAME: "VECTOR_CORE_COUNT", 10
// GENERIC950: "AI_CORE_COUNT", 8
// GENERIC950-SAME: "CUBE_CORE_COUNT", 8
// GENERIC950-SAME: "VECTOR_CORE_COUNT", 16
// GENERIC950-SAME: "UB_SIZE", 2031616
// GENERIC950-SAME: "L0C_SIZE", 2097152
// GENERIC950-SAME: "MINIMAL_D_CACHE_SIZE", 262144
// GENERIC950-SAME: "ARCH", "dav-c310"
// ONLY_AIC: "AI_CORE_COUNT", 10
// ONLY_AIC-SAME: "CUBE_CORE_COUNT", 10
// ONLY_AIC-SAME: "VECTOR_CORE_COUNT", 48
// ONLY_AIV: "AI_CORE_COUNT", 24
// ONLY_AIV-SAME: "CUBE_CORE_COUNT", 24
// ONLY_AIV-SAME: "VECTOR_CORE_COUNT", 16
// NO_LIMIT: "AI_CORE_COUNT", 1000
// NO_LIMIT-SAME: "CUBE_CORE_COUNT", 1000
// NO_LIMIT-SAME: "VECTOR_CORE_COUNT", 1000
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
