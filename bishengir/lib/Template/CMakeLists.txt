option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")

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
        list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
        list(APPEND bisheng_args --cce-enable-print)
        list(APPEND bisheng_args --cce-enable-sanitizer)
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE ".cpp" ".aic.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aic.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
        list(APPEND bisheng_args --cce-enable-print)
        list(APPEND bisheng_args --cce-enable-sanitizer)
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE ".cpp" ".aiv.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aiv.bc" bc_name ${cpp_name})
        endif ()
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (${core_type} STREQUAL "host")
        message(STATUS "Running on host platform.")
    else ()
        if (NOT ${platform_type} STREQUAL "Ascend910B")
            message(FATAL_ERROR "Unsupported platform! Currently support Ascend910B.")
        endif ()
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
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

    file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
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

function(link_all_to_meta_op)
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    file(MAKE_DIRECTORY ${BITCODE_OUTPUT_DIR})

    file(GLOB_RECURSE all_aic_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.aic.bc")
    file(GLOB_RECURSE all_aiv_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.aiv.bc")
    file(GLOB_RECURSE all_mix_aic_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.mix.aic.bc")
    file(GLOB_RECURSE all_mix_aiv_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.mix.aiv.bc")
    list(REMOVE_ITEM all_aic_op_list ${all_mix_aic_op_list})
    list(REMOVE_ITEM all_aiv_op_list ${all_mix_aiv_op_list})
    file(GLOB_RECURSE all_host_op_list "${CMAKE_CURRENT_BINARY_DIR}/*.host.bc")
    set(output_aic_file_name "${BITCODE_OUTPUT_DIR}/meta_op.aic.bc")
    set(output_aiv_file_name "${BITCODE_OUTPUT_DIR}/meta_op.aiv.bc")
    set(output_mix_aic_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aic.bc")
    set(output_mix_aiv_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aiv.bc")
    set(output_host_file_name "${BITCODE_OUTPUT_DIR}/host.bc")

    set(aic_link_args ${all_aic_op_list} "-o" ${output_aic_file_name})
    set(aiv_link_args ${all_aiv_op_list} "-o" ${output_aiv_file_name})
    set(mix_aic_link_args ${all_mix_aic_op_list} "-o" ${output_mix_aic_file_name})
    set(mix_aiv_link_args ${all_mix_aiv_op_list} "-o" ${output_mix_aiv_file_name})
    set(host_link_args ${all_host_op_list} "-o" ${output_host_file_name})

    string(REGEX REPLACE ";" " " aic_link_command "${LLVM_LINK_TOOL} ${aic_link_args}")
    message("-- command of link multi .aic.bc to meta_op.aic.bc is ${aic_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${aic_link_args}
    )

    string(REGEX REPLACE ";" " " aiv_link_command "${LLVM_LINK_TOOL} ${aiv_link_args}")
    message("-- command of link multi .aiv.bc to meta_op.aiv.bc is ${aiv_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${aiv_link_args}
    )

    string(REGEX REPLACE ";" " " mix_aic_link_command "${LLVM_LINK_TOOL} ${mix_aic_link_args}")
    message("-- command of link multi .mix.aic.bc to meta_op.mix.aic.bc is ${mix_aic_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${mix_aic_link_args}
    )

    string(REGEX REPLACE ";" " " mix_aiv_link_command "${LLVM_LINK_TOOL} ${mix_aiv_link_args}")
    message("-- command of link multi .mix.aiv.bc to meta_op.mix.aiv.bc is ${mix_aiv_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${mix_aiv_link_args}
    )

    string(REGEX REPLACE ";" " " host_link_command "${LLVM_LINK_TOOL} ${host_link_args}")
    message("-- command of link multi .host.bc to host.bc is ${host_link_command}")
    execute_process(
            COMMAND ${LLVM_LINK_TOOL} ${host_link_args}
    )

    if (NOT EXISTS "${output_aic_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_aic_file_name} exist or not!")
    endif ()

    if (NOT EXISTS "${output_aiv_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_aiv_file_name} exist or not!")
    endif ()

    if (NOT EXISTS "${output_mix_aic_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_mix_aic_file_name} exist or not!")
    endif ()

    if (NOT EXISTS "${output_mix_aiv_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_mix_aiv_file_name} exist or not!")
    endif ()

    if (NOT EXISTS "${output_host_file_name}")
        message(FATAL_ERROR "llvm-link failed, \
please check ${output_host_file_name} exist or not!")
    endif ()

    execute_process(
            COMMAND rm ${all_aic_op_list}
            COMMAND rm ${all_aiv_op_list}
            COMMAND rm ${all_mix_aic_op_list}
            COMMAND rm ${all_mix_aiv_op_list}
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
endif ()

option(ENABLE_SYCL "If enable sycl, use bishengcpp. Otherwise use bisheng." OFF)
message("-- ENABLE_SYCL Value: ${ENABLE_SYCL}")

if (ENABLE_SYCL)
    add_definitions(-DENABLE_SYCL)
endif ()
