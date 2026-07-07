add_bishengir_conversion_library(MLIRVectorToHIVMAVE
  VectorToHIVMAVE.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToHIVMAVE

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAVEDialect
  MLIRVectorDialect
  MLIRHIVMDialect
  MLIRHIVMUtils
)
