set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  )

add_bishengir_conversion_library(BiShengIRHIVMToTritonGPU
  IS_OPEN_SOURCE

  BufferizationToTriton.cpp
  ReinterpretCastOpReplacementPattern.cpp
  FuncToTriton.cpp
  HIVMToTritonGPU.cpp
  HIVMToTriton.cpp
  HIVMToArith.cpp
  HIVMToTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HIVMToTritonGPU

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
  TritonIR
)
