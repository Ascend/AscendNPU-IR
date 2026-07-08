add_bishengir_conversion_library(MLIRAnalyzeArithVectorMask
  AnalyzeVectorLayout.cpp
  RemoveVectorLayoutAttr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/AVE

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMathDialect
  MLIRHIVMDialect
  MLIRHIVMUtils
)
