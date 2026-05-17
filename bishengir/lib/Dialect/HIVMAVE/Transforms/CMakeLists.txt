add_bishengir_dialect_library(MLIRHIVMAVETransforms
  AnnotateDistOpLayout.cpp
  AppendVectorLayout.cpp
  ComplexReductionIntermediateLowering.cpp
  EliminateVectorLayout.cpp
  I1opSoftImpl.cpp
  LegalizeOptHIVMAVE.cpp
  PLTToPLTM.cpp
  ProcessVsstb.cpp
  ProcessMembar.cpp
  OptimizeReductionLoop.cpp
  ReplaceWithVectorScalar.cpp
  ScalarBroadcastToVLoad.cpp
  SIMTGridBlockPartition.cpp
  SIMTLoopNestsScheduling.cpp
  CombineAVEOPs.cpp
  HoistVstas.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMAVE/Transforms

  DEPENDS
  MLIRAVETransformsIncGen
  MLIRAVEIncGen
  MLIRAVEEnumsIncGen
  MLIRAVEAttributesIncGen
  MLIRAVEOpsIncGen
  MLIRAVETypesIncGen
  MLIRAVETransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAVEDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRDialectUtils
  MLIRHIVMDialect
  MLIRHIVMUtils
  MLIRIR
  MLIRTransformDialect
  BiShengIRDialectUtils
  )
