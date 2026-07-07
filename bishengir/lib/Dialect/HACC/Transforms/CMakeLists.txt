add_bishengir_dialect_library(BiShengIRHACCTransforms
  AppendDeviceSpec.cpp
  RenameFunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC/Transforms

  DEPENDS
  BiShengIRHACCTransformsIncGen
  BiShengIRNPUTargetSpecIncGen

  LINK_LIBS PUBLIC
  BiShengIRHACCDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
