add_bishengir_conversion_library(BishengIRTorchToSymbol
  IS_OPEN_SOURCE
 
  TorchToSymbol.cpp
 
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TorchToSymbol
  ${TORCH_MLIR_INCLUDE_DIRS}
 
  DEPENDS
  BiShengIRSymbolIncGen
 
  LINK_LIBS PUBLIC
  BiShengIRSymbolDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRMemRefDialect
  MLIRMLProgramDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  TorchMLIRConversionPasses
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
)