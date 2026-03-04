add_bishengir_conversion_library(BishengIRTorchToHFusion
  IS_OPEN_SOURCE

  TorchToHFusion.cpp
  Elementwise.cpp
  Reduction.cpp
  DataMovement.cpp
  Uncategorized.cpp
  TensorConstructors.cpp
  Utils.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TorchToHFusion
  ${TORCH_MLIR_INCLUDE_DIRS}

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRHFusionDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRHFusionUtils
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRMLProgramDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTosaDialect
  MLIRSupport
  TorchMLIRTMTensorDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRConversionPasses
  TorchMLIRTMTensorPasses
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchPasses
  TorchMLIRTorchOnnxToTorch
)
