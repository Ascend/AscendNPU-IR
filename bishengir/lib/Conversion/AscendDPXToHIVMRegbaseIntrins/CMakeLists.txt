add_bishengir_conversion_library(AscendDPXToHIVMRegbaseIntrins
  AscendDPXToHIVMRegbaseIntrins.cpp
  AscendDPXMathOpsLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AscendDPXToHIVMRegbaseIntrins

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRFuncDialect
  BiShengIRAscendDPXDialect
  MLIRHIVMRegbaseIntrinsDialect
  MLIRHIVMRegbaseIntrinsUtils
)
