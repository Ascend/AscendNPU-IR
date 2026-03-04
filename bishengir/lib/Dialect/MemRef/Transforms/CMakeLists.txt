add_bishengir_library(BishengIRMemRefTransforms
  BindBuffer.cpp
  FoldAllocReshape.cpp
  DeadStoreElimination.cpp
  RemoveRedundantCopy.cpp
  PropagateReshape/PropagateMemrefCollapseDown.cpp
  PropagateReshape/PropagateMemrefExpandUp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/MemRef/Transforms

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnnotationDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)