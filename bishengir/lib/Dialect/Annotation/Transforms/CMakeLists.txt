add_bishengir_dialect_library(BiShengIRAnnotationTransforms
  BufferizableOpInterfaceImpl.cpp
  Lowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Annotation/Transforms

  DEPENDS
  BiShengIRAnnotationTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  BiShengIRAnnotationDialect
  MLIRPass
  MLIRMemRefDialect
)
