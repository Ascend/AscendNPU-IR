add_bishengir_library(BiShengIRMemRefTransforms
  FoldAllocReshape.cpp
  DeadStoreElimination.cpp
  RemoveRedundantCopy.cpp
  PropagateReshape/PropagateMemrefCollapseDown.cpp
  PropagateReshape/PropagateMemrefExpandUp.cpp
  PropagateReshape/SwapMemrefCollapseExpand.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/MemRef/Transforms

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)