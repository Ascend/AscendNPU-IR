add_mlir_public_c_api_library(BiShengIRCAPIHIVM
  HIVM.cpp
  HIVMPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  BiShengIRHIVMDialect
  BiShengIRHIVMTransforms
  MLIRPass
)

add_mlir_public_c_api_library(BiShengIRCAPIMesh
  Mesh.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  BiShengIRMeshDialect
  BiShengIRMeshTransforms
  MLIRPass
)

add_mlir_public_c_api_library(BiShengIRCAPIHFusion
  HFusion.cpp
  HFusionPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  BiShengIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  BiShengIRHFusionDialect
  BiShengIRHFusionTransforms
  MLIRPass
)

add_mlir_public_c_api_library(BiShengIRCAPIAnnotation
  Annotation.cpp
  AnnotationPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/bishengir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  BiShengIRAnnotationTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  BiShengIRAnnotationDialect
  BiShengIRAnnotationTransforms
  MLIRPass
)
