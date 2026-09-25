# Loop-shape fingerprints consumed by CV / HFusion heuristics are generated at
# build time. Declare the dependency before creating the dialect library so its
# object library also waits for the generated include.
set(SHAPE_REGISTRY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build-tools/coverage/shape_registry.py")
if(NOT EXISTS "${SHAPE_REGISTRY}" AND DEFINED LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR)
  set(SHAPE_REGISTRY
      "${LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR}/../build-tools/coverage/shape_registry.py")
endif()
find_package(Python3 QUIET COMPONENTS Interpreter)
set(SKEW_SHAPE_TABLE ${CMAKE_CURRENT_BINARY_DIR}/SkewShapeTable.inc)
set(SKEW_SHAPE_TABLE_TARGET)
if(Python3_Interpreter_FOUND)
  add_custom_command(
    OUTPUT ${SKEW_SHAPE_TABLE}
    COMMAND ${Python3_EXECUTABLE} ${SHAPE_REGISTRY} --emit-cxx ${SKEW_SHAPE_TABLE}
    DEPENDS ${SHAPE_REGISTRY}
    COMMENT "Generating loop-shape fingerprints"
    VERBATIM)
  add_custom_target(BiShengIRSkewShapeTable DEPENDS ${SKEW_SHAPE_TABLE})
  set(SKEW_SHAPE_TABLE_TARGET BiShengIRSkewShapeTable)
else()
  file(WRITE ${SKEW_SHAPE_TABLE}
       "namespace {\nconstexpr unsigned kSpanKeyCount = 0;\nconstexpr uint64_t kSpanKeys[kSpanKeyCount] = {0ULL};\n}\n")
endif()

file(GLOB HIVM_FLATTEN ${CMAKE_CURRENT_SOURCE_DIR}/FlattenInterface/*.cpp)
file(GLOB HIVM_STRUCTURE ${CMAKE_CURRENT_SOURCE_DIR}/StructuredOpInterface/*.cpp)
file(GLOB HIVM_AGGREGATED ${CMAKE_CURRENT_SOURCE_DIR}/BiShengIRAggregatedOpInterface/*.cpp)
file(GLOB HIVM_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/ExtraBufferOpInterface/*.cpp)
file(GLOB HIVM_IMPL_SCALAR ${CMAKE_CURRENT_SOURCE_DIR}/ImplByScalarOpInterface/*.cpp)
file(GLOB HIVM_INFER_CORE_TYPE ${CMAKE_CURRENT_SOURCE_DIR}/InferCoreTypeInterface/*.cpp)
file(GLOB HIVM_LAYOUT ${CMAKE_CURRENT_SOURCE_DIR}/OpLayoutInterface/*.cpp)
file(GLOB HIVM_LIBRARY_FUNCTION ${CMAKE_CURRENT_SOURCE_DIR}/LibraryFunctionOpInterface/*.cpp)
file(GLOB HIVM_PIPE ${CMAKE_CURRENT_SOURCE_DIR}/OpPipeInterface/*.cpp)
file(GLOB HIVM_VECTOR_BASE ${CMAKE_CURRENT_SOURCE_DIR}/VectorOpBase/*.cpp)
file(GLOB HIVM_VECTORIZE ${CMAKE_CURRENT_SOURCE_DIR}/Vectorize/*.cpp)
file(GLOB HIVM_CONTRACT ${CMAKE_CURRENT_SOURCE_DIR}/Contracts/*.cpp)
file(GLOB HIVM_CANONICALIZATIONS ${CMAKE_CURRENT_SOURCE_DIR}/Canonicalizations/*.cpp)
file(GLOB HIVM_CUSTOM_OP ${CMAKE_CURRENT_SOURCE_DIR}/CustomOp/*.cpp)
file(GLOB HIVM_VECTORIZE ${CMAKE_CURRENT_SOURCE_DIR}/Vectorize/*.cpp)

add_bishengir_dialect_library(BiShengIRHIVMDialect
  ${HIVM_FLATTEN}
  ${HIVM_STRUCTURE}
  ${HIVM_AGGREGATED}
  ${HIVM_BUFFER}
  ${HIVM_IMPL_SCALAR}
  ${HIVM_INFER_CORE_TYPE}
  ${HIVM_LAYOUT}
  ${HIVM_LIBRARY_FUNCTION}
  ${HIVM_PIPE}
  ${HIVM_VECTOR_BASE}
  ${HIVM_VECTORIZE}
  ${HIVM_CANONICALIZATIONS}
  ${HIVM_CUSTOM_OP}
  ${HIVM_VECTORIZE}
  HIVMCanonicalizations.cpp
  HIVMDialect.cpp
  HIVMDialectExtension.cpp
  HIVMDMAOps.cpp
  HIVMImpl.cpp
  HIVMInterfaces.cpp
  HIVMIntrinOps.cpp
  ${HIVM_CONTRACT}

  HIVMMacroOps.cpp
  HIVMOps.cpp
  HIVMProtonOps.cpp
  HIVMSynchronizationOps.cpp
  HIVMTraits.cpp
  HIVMVectorOps.cpp
  StoreLikeOpInterface.cpp
  ../Utils/Utils.cpp
  ../Utils/ShapeRegistry.cpp
  ${SKEW_SHAPE_TABLE}
  ../Utils/RegbaseUtils.cpp
  ../Utils/MultiBufferLoopAdapter.cpp
  ../Utils/WorklistBuilder.cpp
  ../Utils/MemrefAliasAnalysisState.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/IR/CustomOp
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils

  DEPENDS
  ${SKEW_SHAPE_TABLE_TARGET}
  BiShengIRFixpipePreQuantContractIncGen
  BiShengIRHIVMIncGen
  BiShengIRHIVMEnumsIncGen
  BiShengIRHIVMAttributesIncGen
  BiShengIRHIVMDMAOpsIncGen
  BiShengIRHIVMInterfacesIncGen
  BiShengIRHIVMIntrinOpsIncGen
  BiShengIRHIVMMacroOpsIncGen
  BiShengIRHIVMVectorOpsIncGen
  BiShengIRHIVMSynchronizationOpsIncGen
  BiShengIRScopeIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  BiShengIRScopeDialect
  BiShengIRMathExtDialect
  MLIRBufferizationDialect
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
  MLIRVectorDialect
  MLIRViewLikeInterface
  MLIRVectorDialect
  BiShengIRDialectUtils
  BiShengIRHACCDialect
  BiShengIRHACCUtils
)

target_include_directories(BiShengIRHIVMDialect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
