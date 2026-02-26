file(GLOB HIVM_FLATTEN ${CMAKE_CURRENT_SOURCE_DIR}/FlattenInterface/*.cpp)
file(GLOB HIVM_STRUCTURE ${CMAKE_CURRENT_SOURCE_DIR}/StructuredOpInterface/*.cpp)
file(GLOB HIVM_AGGREGATED ${CMAKE_CURRENT_SOURCE_DIR}/BiShengIRAggregatedOpInterface/*.cpp)
file(GLOB HIVM_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/ExtraBufferOpInterface/*.cpp)
file(GLOB HIVM_IMPL_SCALAR ${CMAKE_CURRENT_SOURCE_DIR}/ImplByScalarOpInterface/*.cpp)
file(GLOB HIVM_INFER_CORE_TYPE ${CMAKE_CURRENT_SOURCE_DIR}/InferCoreTypeInterface/*.cpp)
file(GLOB HIVM_LAYOUT ${CMAKE_CURRENT_SOURCE_DIR}/OpLayoutInterface/*.cpp)
file(GLOB HIVM_PIPE ${CMAKE_CURRENT_SOURCE_DIR}/OpPipeInterface/*.cpp)
file(GLOB HIVM_UNIT_FLAG ${CMAKE_CURRENT_SOURCE_DIR}/HIVMUnitFlagEnabledInterface/*.cpp)
file(GLOB HIVM_VECTOR_BASE ${CMAKE_CURRENT_SOURCE_DIR}/VectorOpBase/*.cpp)

add_bishengir_dialect_library(BiShengIRHIVMDialect
  ${HIVM_FLATTEN}
  ${HIVM_STRUCTURE}
  ${HIVM_AGGREGATED}
  ${HIVM_BUFFER}
  ${HIVM_IMPL_SCALAR}
  ${HIVM_INFER_CORE_TYPE}
  ${HIVM_LAYOUT}
  ${HIVM_PIPE}
  ${HIVM_UNIT_FLAG}
  ${HIVM_VECTOR_BASE}
  HIVMCanonicalizations.cpp
  HIVMDialect.cpp
  HIVMDMAOps.cpp
  HIVMImpl.cpp
  HIVMInterfaces.cpp
  HIVMMacroOps.cpp
  HIVMOps.cpp
  HIVMSynchronizationOps.cpp
  HIVMTraits.cpp
  HIVMVectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  DEPENDS
  BiShengIRHIVMIncGen
  BiShengIRHIVMAttributesIncGen
  BiShengIRHIVMDMAOpsIncGen
  BiShengIRHIVMInterfacesIncGen
  BiShengIRHIVMMacroOpsIncGen
  BiShengIRHIVMVectorOpsIncGen
  BiShengIRHIVMSynchronizationOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
  MLIRSideEffectInterfaces
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRTensorDialect
  MLIRViewLikeInterface
  BiShengIRDialectUtils
  BiShengIRHACCDialect
)
