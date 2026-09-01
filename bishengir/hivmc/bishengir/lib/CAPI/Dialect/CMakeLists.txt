add_mlir_public_c_api_library(MLIRCAPIHIVM
  HIVM.cpp
  HIVMPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRHIVMDialect
  MLIRHIVMTransforms
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPIMesh
  Mesh.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMeshDialect
  MLIRMeshTransforms
  BiShengIRMeshTransforms
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPIHFusion
  HFusion.cpp
  HFusionPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRHFusionDialect
  MLIRHFusionTransforms
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPIAnnotation
  Annotation.cpp
  AnnotationPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRAnnotationTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRAnnotationDialect
  MLIRAnnotationTransforms
  MLIRPass
)
