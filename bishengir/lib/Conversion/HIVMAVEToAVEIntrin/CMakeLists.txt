add_bishengir_conversion_library(MLIRHIVMAVEToAVEIntrin
  HIVMAVEToAVEIntrin.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HIVMAVEToAVEIntrin

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRAffineDialect
  MLIRAnalysis
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRValueBoundsOpInterface
  MLIRAVEDialect
  MLIRHIVMRegbaseIntrinsDialect
  MLIRHIVMRegbaseIntrinsUtils
)
