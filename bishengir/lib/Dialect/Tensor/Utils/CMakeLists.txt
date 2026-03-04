add_bishengir_library(BishengIRTensorUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRIR
  MLIRDialect
  MLIRPass
  BiShengIRDialectUtils
)