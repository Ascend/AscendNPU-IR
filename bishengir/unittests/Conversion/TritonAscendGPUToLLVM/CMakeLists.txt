add_bishengir_unittest(BiShengIRSharedMemorySwizzleTests
  SharedMemorySwizzleTest.cpp
)

target_link_libraries(BiShengIRSharedMemorySwizzleTests PRIVATE
  TritonAscendGPUToLLVM
  TritonTools
  MLIRIR
)
