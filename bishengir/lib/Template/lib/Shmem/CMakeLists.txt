file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set (platform_type "Ascend910B")

foreach (cpp_with_path ${cpp_list})
    compile_shmem_to_bc(${cpp_with_path} "aic" ${platform_type})
    compile_shmem_to_bc(${cpp_with_path} "aiv" ${platform_type})
    compile_shmem_to_bc(${cpp_with_path} "mix_aiv" ${platform_type})
    compile_shmem_to_bc(${cpp_with_path} "mix_aic" ${platform_type})
endforeach()