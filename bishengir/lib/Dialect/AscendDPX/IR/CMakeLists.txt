add_bishengir_dialect_library(MLIRAscendDPXDialect
  AscendDPXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/AscendDPX

  DEPENDS
  MLIRAscendDPXIncGen
  MLIRAscendDPXEnumsIncGen
  MLIRAscendDPXAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  TritonGPUIR
)