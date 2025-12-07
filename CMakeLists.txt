cmake_minimum_required(VERSION 4.0)
project(untitled LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)

add_library(untitled SHARED
        hook.mm
        butterflies.cpp
        butterflies.h)

target_link_libraries(untitled PRIVATE objc "-framework Metal" "-framework QuartzCore" "-framework CoreFoundation" "-framework CoreAudio" "-framework Foundation")


set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders.metal)
set(SHADER_AIR ${CMAKE_CURRENT_BINARY_DIR}/shaders.air)
set(SHADER_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/shaders.metallib)

add_custom_command(
        OUTPUT ${SHADER_AIR}
        COMMAND xcrun -sdk macosx metal -c ${SHADER_SRC} -o ${SHADER_AIR}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling ${SHADER_SRC} -> ${SHADER_AIR}"
        VERBATIM
)

add_custom_command(
        OUTPUT ${SHADER_METALLIB}
        COMMAND xcrun -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_METALLIB}
        DEPENDS ${SHADER_AIR}
        COMMENT "Linking ${SHADER_AIR} -> ${SHADER_METALLIB}"
        VERBATIM
)

add_custom_target(shaders ALL DEPENDS ${SHADER_METALLIB})
add_dependencies(untitled shaders)