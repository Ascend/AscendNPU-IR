add_subdirectory(bishengir-compile)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  add_subdirectory(bishengir-triton-libs)
endif()

# add_subdirectory(ConfigOptions)
# add_subdirectory(Utils)
add_subdirectory(Version)
