add_bishengir_library(BiShengIRLLVMTransforms
  LLVMParameterPacking.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/LLVMIR/Transforms

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRSCFTransformOps
  MLIRHIVMTransformOps
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRLLVMDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
