compile_cpp_to_bc("aic" "Ascend910B")
if (BISHENGIR_DISABLE_CANN STREQUAL "OFF")
  compile_cpp_to_bc("aic" "Ascend910D")
endif()
