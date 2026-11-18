add_bishengir_conversion_library(BiShengIRProtonToHIVM
    ProtonToHIVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${BISHENGIR_MAIN_INCLUDE_DIR}

    DEPENDS
    BiShengIRConversionPassIncGen
    ProtonAttrDefsIncGen
    ProtonTableGen

    LINK_COMPONENTS
    Support

    LINK_LIBS PUBLIC
    BiShengIRHACCDialect
    BiShengIRHACCUtils
    BiShengIRHIVMDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    ProtonIR
)
