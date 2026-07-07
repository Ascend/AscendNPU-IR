add_bishengir_library(BiShengIRTensorUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRIR
  MLIRDialect
  MLIRPass
  BiShengIRDialectUtils
)
