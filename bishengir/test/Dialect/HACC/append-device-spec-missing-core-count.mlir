// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950 -verify-diagnostics

// expected-error@below {{invalid core counts for target 'Ascend950': AI_CORE_COUNT=0, CUBE_CORE_COUNT=0, VECTOR_CORE_COUNT=0}}
module {

}
