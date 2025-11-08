add_bishengir_dialect_library(BiShengIRHIVMTransformOps
  HIVMTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/TransformOps

  DEPENDS
  BiShengIRHIVMTransformOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  BiShengIRHIVMUtils
  )