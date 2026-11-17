add_bishengir_dialect_library(BiShengIRBMUDialect
  BMU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/BMU

  DEPENDS
  BiShengIRBMUIncGen
  BiShengIRBMUAttributesIncGen
  BiShengIRBMUOpsIncGen
  BiShengIRBMUIntrinOpsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
)
