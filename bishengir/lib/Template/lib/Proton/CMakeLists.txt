set(PROTON_AIV_SOURCES
    ProtonGetSysCnt.cpp
    ProtonCircularStore.cpp
)

set(PROTON_AIC_SOURCES
    ProtonGetSysCnt.cpp
    ProtonCircularStore.cpp
)

# AIV (Vector) - compiled without PROTON_AIC, so PROTON_CORE_ATTR/PROTON_FN_NAME
# resolve to the plain __aiv__ / non-suffixed names.
compile_cpp_to_bc("aiv" "Ascend910B" ${PROTON_AIV_SOURCES})
compile_cpp_to_bc("aiv" "Ascend910D" ${PROTON_AIV_SOURCES})

# AIC (Cube) - same source files, compiled with -DPROTON_AIC so the same
# functions become __aicore__ and get the _cube name suffix.
set(EXTRA_BISHENG_DEFINES -DPROTON_AIC)
compile_cpp_to_bc("aic" "Ascend910B" ${PROTON_AIC_SOURCES})
compile_cpp_to_bc("aic" "Ascend910D" ${PROTON_AIC_SOURCES})
unset(EXTRA_BISHENG_DEFINES)
