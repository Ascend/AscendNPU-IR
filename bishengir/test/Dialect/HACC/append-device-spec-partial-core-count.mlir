// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend950 custom-aic-number=8" -verify-diagnostics

// expected-error@below {{invalid core counts for target 'Ascend950': AI_CORE_COUNT=8, CUBE_CORE_COUNT=8, VECTOR_CORE_COUNT=0}}
module {

}
