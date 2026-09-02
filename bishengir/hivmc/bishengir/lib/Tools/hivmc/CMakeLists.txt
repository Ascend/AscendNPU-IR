get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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
    ${translation_libs}
    ${conversion_libs}
    ${dialect_libs}
    ${extension_libs}
    BiShengIRVersionLib
    BiShengIRPass
    BishengIRTransform
)

add_subdirectory(A5)
add_subdirectory(A3)
