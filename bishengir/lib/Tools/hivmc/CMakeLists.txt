add_bishengir_library(BiShengIRHIVMCLib
    HIVMCConfig.cpp
    HIVMCMain.cpp
    Utility.cpp
    AdapterSanitizer.cpp
    BishengLlcBridge.cpp
    BishengAsyncRuntime.cpp

    ADDITIONAL_HEADER_DIRS
    ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/hivmc

    LINK_LIBS PUBLIC
    BiShengIRHIVMCLibA3
    BiShengIRHIVMCLibA5
    BiShengIRVersionLib
    BiShengIRPass
    BiShengIRTransform
)

# Vendor in-process compiler service header (bisheng-llc.h).
# BISHENG_LLC_INCLUDE_DIR is defined in bishengir/lib/Tools/CMakeLists.txt
# (above all add_subdirectory calls) so every consumer sees the same value.
if(BISHENG_LLC_INCLUDE_DIR)
  target_include_directories(BiShengIRHIVMCLib PRIVATE ${BISHENG_LLC_INCLUDE_DIR})
endif()
# dlmopen/dlsym/dlinfo for the v2 bridge.
target_link_libraries(BiShengIRHIVMCLib PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(A5)
add_subdirectory(A3)
