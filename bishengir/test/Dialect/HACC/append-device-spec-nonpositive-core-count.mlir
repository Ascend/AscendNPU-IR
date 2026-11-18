// RUN: bishengir-opt %s --hacc-append-device-spec="target=Ascend910B1 custom-aic-number=-5 custom-aiv-number=10" -verify-diagnostics

// expected-error@below {{invalid core counts for target 'Ascend910B1': AI_CORE_COUNT=-5, CUBE_CORE_COUNT=-5, VECTOR_CORE_COUNT=10}}
module {

}
