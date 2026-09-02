file(GLOB VFFUSION_ANALYZER ${CMAKE_CURRENT_SOURCE_DIR}/VFFusionAnalyzer/*.cpp)
file(GLOB VFFUSION_KIND ${CMAKE_CURRENT_SOURCE_DIR}/VFFusionKind/*.cpp)

add_bishengir_dialect_library(BiShengIRVFFusion
  ${VFFUSION_ANALYZER}
  ${VFFUSION_KIND}
  VFFusionBlock.cpp
  VFFusionPass.cpp
  VFFusionOutliner.cpp
  VFUnionFind.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRPass
)
