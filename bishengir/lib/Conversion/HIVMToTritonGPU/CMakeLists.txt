set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  )

add_bishengir_conversion_library(BiShengIRHIVMToTritonGPU
  IS_OPEN_SOURCE

  BufferizationToTriton.cpp
  AffineToTriton.cpp
  MemrefToTriton.cpp
  FuncToTriton.cpp
  HIVMToTritonGPU.cpp
  HIVMToTriton.cpp
  HIVMToArith.cpp
  TensorToTriton.cpp
  HIVMToMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HIVMToTritonGPU

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRAffineDialect
  MLIRAffineUtils
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
