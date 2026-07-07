add_bishengir_dialect_library(AscendDPXTransforms
  DivOptimization.cpp
  HoistCallScalarToCaller.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/AscendDPX/Transforms

  DEPENDS
  BiShengIRAscendDPXTransformsIncGen
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRAscendDPXDialect
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRLLVMDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)