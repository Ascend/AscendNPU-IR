add_bishengir_dialect_library(BiShengIRAnnotationDialect
  AnnotationDialect.cpp
  AnnotationOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Annotation

  DEPENDS
  BiShengIRAnnotationOpsIncGen
  MLIRAnnotationAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
