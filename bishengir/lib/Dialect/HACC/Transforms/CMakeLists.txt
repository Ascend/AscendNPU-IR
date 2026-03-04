add_bishengir_dialect_library(MLIRHACCTransforms
  AppendDeviceSpec.cpp
  RenameFunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC/Transforms

  DEPENDS
  MLIRHACCTransformsIncGen
  BiShengIRNPUTargetSpecIncGen

  LINK_LIBS PUBLIC
  MLIRHACCDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
