option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")

if(CMAKE_SYSROOT)
    message("-- Using custom sysroot: ${CMAKE_SYSROOT}")
else()
    message("-- No CMAKE_SYSROOT specified. Omitting sysroot flag.")
endif()

#[[Compile single xx.cpp to xx.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "Ascend910B"]]
function(compile_single_cpp_to_bc cpp_with_path core_type platform_type)
    # Get name of xx.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif ()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-cube)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
            list(APPEND bisheng_args --cce-enable-sanitizer)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE ".cpp" ".aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-vec)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
            list(APPEND bisheng_args --cce-enable-sanitizer)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE ".cpp" ".aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (${core_type} STREQUAL "host")
        message(STATUS "Running on host platform.")
    else ()
        if (NOT ${platform_type} STREQUAL "Ascend910B" AND
            NOT ${platform_type} STREQUAL "Ascend910D")
            message(FATAL_ERROR
                    "Unsupported platform! Currently support Ascend910B and Ascend910D.")
        endif ()
    endif ()

    list(APPEND bisheng_args -std=c++17)
    if ("${platform_type}" STREQUAL "Ascend910D")
        # RegBase sources carry private C310 variants of shared headers here.
        # Keeping this path ahead of the global include path prevents the A5
        # ABI from replacing Ascend910B declarations.
        list(APPEND bisheng_args -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif ()
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -o ${bc_name})
    
    # Conditionally append sysroot if it exists
    if(CMAKE_SYSROOT)
        list(APPEND bisheng_args --sysroot=${CMAKE_SYSROOT})
    endif()

    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${BISHENG_TOOL} ${bisheng_args}
    )


    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
        message(FATAL_ERROR "Compile ${cpp_with_path} failed, please check \
${CMAKE_CURRENT_BINARY_DIR}/${bc_name} exist or not!")
    endif ()

endfunction()

function(compile_shmem_to_bc cpp_with_path core_type platform_type)
    # Get name of xx.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)
    list(APPEND bisheng_args --cce-enable-print)
    list(APPEND bisheng_args --cce-enable-sanitizer)

    set(aicore_arch "dav-c220")

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif ()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        list(APPEND bisheng_args "--cce-aicore-arch=${aicore_arch}-cube")
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE "\\.cpp" ".aic.c220.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE "\\.cpp" ".mix.aic.c220.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args "--cce-aicore-arch=${aicore_arch}-vec")
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE "\\.cpp" ".aiv.c220.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE "\\.cpp" ".mix.aiv.c220.bc" bc_name ${cpp_name})
        endif ()
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (NOT ${platform_type} STREQUAL "Ascend910B")
        message(FATAL_ERROR "Unsupported platform! Currently support Ascend910B.")
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -I ${PROJECT_SOURCE_DIR}/third-party/shmem/include/)
    list(APPEND bisheng_args -I ${PROJECT_SOURCE_DIR}/third-party/shmem/src/device/)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/)
    else ()
        message(FATAL_ERROR "Unknown architecture")
    endif ()
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface)
    list(APPEND bisheng_args -o ${bc_name})
    
    # Conditionally append sysroot if it exists
    if(CMAKE_SYSROOT)
        list(APPEND bisheng_args --sysroot=${CMAKE_SYSROOT})
    endif()

    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${BISHENG_TOOL} ${bisheng_args}
    )


    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
        message(FATAL_ERROR "Compile ${cpp_with_path} failed, please check \
${CMAKE_CURRENT_BINARY_DIR}/${bc_name} exist or not!")
    endif ()

endfunction()

#[[Compile single xx.cpp to xx_debug.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "DEBUG"]]
function(compile_single_cpp_to_bc_debug cpp_with_path core_type platform_type)
    # Get name of xx_debug.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})
    set_source_files_properties(${cpp_with_path} PROPERTIES LANGUAGE c++)

    set(bisheng_args "-O2")
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if (${core_type} STREQUAL "aic")
        string(REGEX REPLACE ".cpp" ".aic.bc" bc_name ${cpp_name})
    elseif (${core_type} STREQUAL "aiv")
        string(REGEX REPLACE ".cpp" ".aiv.bc" bc_name ${cpp_name})
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})    
    else ()
        message(FATAL_ERROR "Core type must be aic/aiv or host.")
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -DENABLE_CPU_TRACE_INTRINSIC)
    list(APPEND bisheng_args -o ${bc_name})

    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${BISHENG_TOOL} ${bisheng_args}
    )

    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
        message(FATAL_ERROR "Compile ${cpp_with_path} failed, please check \
${CMAKE_CURRENT_BINARY_DIR}/${bc_name} exist or not!")
    endif ()

endfunction()

#[[Compile multi xx.cpp to multi xx.bc
 Core_type: "aic" "aiv"
 Platform_type: "Ascend910B"]]
function(compile_cpp_to_bc)
    set (core_type " ")
    set (platform_type " ")

    set(core_type ${ARGV0})
    if(NOT ${ARGV0} STREQUAL "host")
        set(platform_type ${ARGV1})
    endif ()

    if (ARGC GREATER 2)
        list(SUBLIST ARGV 2 -1 cpp_list)
        set(explicit_cpp_list)
        foreach (cpp_source ${cpp_list})
            if (IS_ABSOLUTE "${cpp_source}")
                list(APPEND explicit_cpp_list "${cpp_source}")
            else ()
                list(APPEND explicit_cpp_list
                     "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_source}")
            endif ()
        endforeach ()
        set(cpp_list ${explicit_cpp_list})
    else ()
        file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    endif ()
    foreach (cpp_with_path ${cpp_list})
        if (ENABLE_CPU_TRACE_INTRINSIC AND
            (NOT ${core_type} STREQUAL "mix_aic") AND
            (NOT ${core_type} STREQUAL "mix_aiv"))
            compile_single_cpp_to_bc_debug(${cpp_with_path} ${core_type} "DEBUG")
        else ()
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} ${platform_type})
        endif ()
    endforeach ()
endfunction()

# Link all bitcode files of a given architecture into meta_op files.
# `arch` is the arch suffix in the .bc filenames ("" for c220, "c310" for c310).
# `output_suffix` is the suffix appended to output meta_op filenames
# ("" for c220, ".c310" for c310).
function(link_one_arch_to_meta_op arch output_suffix)
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")

    if("${arch}" STREQUAL "")
        set(aic_glob "*.aic.bc")
        set(aiv_glob "*.aiv.bc")
        set(mix_aic_glob "*.mix.aic.bc")
        set(mix_aiv_glob "*.mix.aiv.bc")
    else()
        set(aic_glob "*.aic.${arch}.bc")
        set(aiv_glob "*.aiv.${arch}.bc")
        set(mix_aic_glob "*.mix.aic.${arch}.bc")
        set(mix_aiv_glob "*.mix.aiv.${arch}.bc")
    endif()

    file(GLOB_RECURSE aic_op_list     "${CMAKE_CURRENT_BINARY_DIR}/${aic_glob}")
    file(GLOB_RECURSE aiv_op_list     "${CMAKE_CURRENT_BINARY_DIR}/${aiv_glob}")
    file(GLOB_RECURSE mix_aic_op_list "${CMAKE_CURRENT_BINARY_DIR}/${mix_aic_glob}")
    file(GLOB_RECURSE mix_aiv_op_list "${CMAKE_CURRENT_BINARY_DIR}/${mix_aiv_glob}")
    list(REMOVE_ITEM aic_op_list ${mix_aic_op_list})
    list(REMOVE_ITEM aiv_op_list ${mix_aiv_op_list})

    set(output_aic_file_name     "${BITCODE_OUTPUT_DIR}/meta_op.aic${output_suffix}.bc")
    set(output_aiv_file_name     "${BITCODE_OUTPUT_DIR}/meta_op.aiv${output_suffix}.bc")
    set(output_mix_aic_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aic${output_suffix}.bc")
    set(output_mix_aiv_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aiv${output_suffix}.bc")

    foreach(pair "aic_op_list;${output_aic_file_name};${aic_glob}"
                 "aiv_op_list;${output_aiv_file_name};${aiv_glob}"
                 "mix_aic_op_list;${output_mix_aic_file_name};${mix_aic_glob}"
                 "mix_aiv_op_list;${output_mix_aiv_file_name};${mix_aiv_glob}")
        list(GET pair 0 list_var)
        list(GET pair 1 out_file)
        list(GET pair 2 glob_pat)
        set(current_files ${${list_var}})
        if(NOT current_files)
            message(STATUS "No ${glob_pat} files found, skip linking ${out_file}")
            continue()
        endif()
        set(link_args ${current_files} "-o" ${out_file})
        string(REGEX REPLACE ";" " " link_command "${LLVM_LINK_TOOL} ${link_args}")
        message("-- command of link multi ${glob_pat} to ${out_file} is ${link_command}")
        execute_process(
                COMMAND ${LLVM_LINK_TOOL} ${link_args}
                RESULT_VARIABLE llvm_link_result
        )
        if(NOT llvm_link_result EQUAL 0)
            message(FATAL_ERROR "llvm-link failed for ${out_file} with exit code ${llvm_link_result}")
        endif()
        if (NOT EXISTS "${out_file}")
            message(FATAL_ERROR "llvm-link failed, \
please check ${out_file} exist or not!")
        endif ()
        execute_process(
                COMMAND rm ${current_files}
        )
    endforeach()
endfunction()

function(link_all_to_meta_op)
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    file(MAKE_DIRECTORY ${BITCODE_OUTPUT_DIR})

    # c220 (Ascend910B / A3): use .c220 suffix.
    link_one_arch_to_meta_op("c220" ".c220")

    # c310 (Ascend910D / A5): use .c310 suffix.
    link_one_arch_to_meta_op("c310" ".c310")

    # Link host bitcode (arch-agnostic).
    file(GLOB_RECURSE all_host_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.host.bc")
    set(output_host_file_name "${BITCODE_OUTPUT_DIR}/host.bc")
    set(host_link_args ${all_host_op_list} "-o" ${output_host_file_name})
    string(REGEX REPLACE ";" " " host_link_command "${LLVM_LINK_TOOL} ${host_link_args}")
    message("-- command of link multi .host.bc to host.bc is ${host_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${host_link_args}
            RESULT_VARIABLE host_link_result
    )
    if(NOT host_link_result EQUAL 0)
        message(FATAL_ERROR "llvm-link failed for ${output_host_file_name} with exit code ${host_link_result}")
    endif()

    if (NOT EXISTS "${output_host_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_host_file_name} exist or not!")
    endif ()

    execute_process(
            COMMAND rm ${all_host_op_list}
    )

endfunction()

if (BISHENGIR_BUILD_TEMPLATE STREQUAL "ON")
    message("-- BISHENGIR_BUILD_TEMPLATE Value: ${BISHENGIR_BUILD_TEMPLATE}")
    message("-- BISHENG_COMPILER_PATH Value: ${BISHENG_COMPILER_PATH}")

    set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(BISHENG_TOOL "${BISHENG_COMPILER_PATH}/ccec")
    set(LLVM_LINK_TOOL "${BISHENG_COMPILER_PATH}/llvm-link")
    if ((NOT EXISTS "${BISHENG_TOOL}") OR (NOT EXISTS "${LLVM_LINK_TOOL}"))
        message(FATAL_ERROR "Can not find ccec or llvm-link in ${BISHENG_COMPILER_PATH},\
please check ${BISHENG_TOOL} or ${LLVM_LINK_TOOL} exist or not!")
    endif ()

    add_subdirectory(lib)
    link_all_to_meta_op()

    install(FILES
            ${CMAKE_BINARY_DIR}/lib/meta_op.aic.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aiv.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aic.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aiv.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aic.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aiv.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aic.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aiv.c310.bc
            ${CMAKE_BINARY_DIR}/lib/host.bc
            DESTINATION lib
            COMPONENT bishengir-publish-products)
endif ()

option(ENABLE_SYCL "If enable sycl, use bishengcpp. Otherwise use bisheng." OFF)
message("-- ENABLE_SYCL Value: ${ENABLE_SYCL}")

if (ENABLE_SYCL)
    add_definitions(-DENABLE_SYCL)
endif ()
