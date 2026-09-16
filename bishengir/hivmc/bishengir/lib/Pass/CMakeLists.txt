add_bishengir_library(BiShengIRPass
  PassManager.cpp
  PassManagerOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Pass

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  )
