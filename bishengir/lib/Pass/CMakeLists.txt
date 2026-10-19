add_bishengir_library(BiShengIRPass
  PassExecutionPolicy.cpp
  PassManager.cpp
  PassManagerOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Pass

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
