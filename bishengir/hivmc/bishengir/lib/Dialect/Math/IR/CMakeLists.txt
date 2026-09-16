add_bishengir_dialect_library(MLIRMathExtDialect
  MathExtOps.cpp
  MathExtDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Math

  DEPENDS
  MLIRMathExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMDialect
)
