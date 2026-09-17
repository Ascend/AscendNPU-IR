add_bishengir_library(BiShengIRHIVMCLib
    HIVMCConfig.cpp
    HIVMCMain.cpp
    Utility.cpp
    AdapterSanitizer.cpp

    ADDITIONAL_HEADER_DIRS
    ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/hivmc

    LINK_LIBS PUBLIC
    BiShengIRHIVMCLibA3
    BiShengIRHIVMCLibA5
    BiShengIRVersionLib
    BiShengIRPass
    BiShengIRTransform
)

add_subdirectory(A5)
add_subdirectory(A3)
