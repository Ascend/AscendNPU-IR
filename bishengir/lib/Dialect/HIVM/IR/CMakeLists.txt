file(GLOB HIVM_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/bishengir/lib/Dialect/HIVM/IR/CMakeLists.txt/*.cpp)
file(GLOB HIVM_FLATTEN ${CMAKE_CURRENT_SOURCE_DIR}/FlattenInterface/*.cpp)
file(GLOB HIVM_STRUCTURE ${CMAKE_CURRENT_SOURCE_DIR}/StructuredOpInterface/*.cpp)
file(GLOB HIVM_AGGREGATED ${CMAKE_CURRENT_SOURCE_DIR}/BiShengIRAggregatedOpInterface/*.cpp)
file(GLOB HIVM_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/ExtraBufferOpInterface/*.cpp)
file(GLOB HIVM_IMPL_SCALAR ${CMAKE_CURRENT_SOURCE_DIR}/ImplByScalarOpInterface/*.cpp)
file(GLOB HIVM_INFER_CORE_TYPE ${CMAKE_CURRENT_SOURCE_DIR}/InferCoreTypeInterface/*.cpp)
file(GLOB HIVM_IMPL_SCALAR ${CMAKE_CURRENT_SOURCE_DIR}/ImplByScalarOpInterface/*.cpp)
file(GLOB HIVM_INFER_CORE_TYPE ${CMAKE_CURRENT_SOURCE_DIR}/InferCoreTypeInterface/*.cpp)
file(GLOB HIVM_CANONICALIZATIONS ${CMAKE_CURRENT_SOURCE_DIR}/Canonicalizations/*.cpp)

add_bishengir_dialect_library(MLIRHIVMDialect
  ${HIVM_FLATTEN}
  ${HIVM_STRUCTURE}
  ${HIVM_AGGREGATED}
  ${HIVM_BUFFER}
  ${HIVM_IMPL_SCALAR}
  ${HIVM_INFER_CORE_TYPE}
  ${HIVM_CANONICALIZATIONS}
  HIVM.cpp
  HIVMCanonicalizations.cpp
  HIVMDMAOps.cpp
  HIVMImpl.cpp
  HIVMInterfaces.cpp
  HIVMIntrinOps.cpp
  HIVMMacroOps.cpp
  HIVMSynchronizationOps.cpp
  HIVMTraits.cpp
  HIVMVectorOps.cpp
  Vectorize/VectorizeElementwise.cpp
  Vectorize/VectorizeReduce.cpp
  Vectorize/Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  DEPENDS
  MLIRHIVMIncGen
  MLIRHIVMAttributesIncGen
  MLIRHIVMDMAOpsIncGen
  MLIRHIVMInterfacesIncGen
  MLIRHIVMIntrinOpsIncGen
  MLIRHIVMMacroOpsIncGen
  MLIRHIVMVectorOpsIncGen
  MLIRHIVMSynchronizationOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAnnotationDialect
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefExtDialect
  MLIRSideEffectInterfaces
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRTensorDialect
  MLIRViewLikeInterface
  MLIRHACCUtils
  BiShengIRDialectUtils
)
