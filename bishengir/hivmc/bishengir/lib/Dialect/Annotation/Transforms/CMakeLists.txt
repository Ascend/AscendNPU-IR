add_bishengir_dialect_library(MLIRAnnotationTransforms
  BufferizableOpInterfaceImpl.cpp
  Lowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Annotation/Transforms

  DEPENDS
  MLIRAnnotationTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRAnnotationDialect
  MLIRPass
  MLIRMemRefDialect
  )
