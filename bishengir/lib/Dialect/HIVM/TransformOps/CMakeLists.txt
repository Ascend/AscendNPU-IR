add_bishengir_dialect_library(MLIRHIVMTransformOps
  HIVMTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/TransformOps

  DEPENDS
  MLIRHIVMTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRHIVMDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRHIVMUtils
  )