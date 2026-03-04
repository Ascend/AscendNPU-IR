add_bishengir_library(BiShengIRSymbolUtils
  Utils.cpp
 
  LINK_LIBS PUBLIC
  BiShengIRSymbolDialect
  BiShengIRDialectUtils
  MLIRIR
  MLIRDialect
  MLIRPass
)
