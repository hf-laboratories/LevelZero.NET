cmake_minimum_required(VERSION 3.16)
project(LevelZeroShim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(L0_SHIM_NAME "LevelZeroShim")
set(L0_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/level-zero/include")

set(LEVEL_ZERO_ROOT $ENV{LEVEL_ZERO_ROOT})
set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})

set(LEVEL_ZERO_HINTS
    "${LEVEL_ZERO_ROOT}"
    "${LEVEL_ZERO_ROOT}/lib"
    "${LEVEL_ZERO_ROOT}/lib64"
    "${LEVEL_ZERO_ROOT}/bin"
    "${ONEAPI_ROOT}/level-zero/latest"
    "${ONEAPI_ROOT}/level-zero/latest/lib"
    "${ONEAPI_ROOT}/level-zero/latest/lib64"
    "${ONEAPI_ROOT}/level-zero/latest/bin"
)

if(WIN32)
    list(APPEND LEVEL_ZERO_HINTS
        "C:/Program Files (x86)/Intel/oneAPI/level-zero/latest"
        "C:/Program Files/Intel/oneAPI/level-zero/latest"
        "C:/Program Files (x86)/Intel/oneAPI/level-zero/latest/lib"
        "C:/Program Files/Intel/oneAPI/level-zero/latest/lib"
        "C:/Program Files (x86)/Intel/oneAPI/level-zero/latest/bin"
        "C:/Program Files/Intel/oneAPI/level-zero/latest/bin"
    )
endif()

find_library(LEVEL_ZERO_LOADER
    NAMES ze_loader ze_loader.lib ze_loader.dll
    HINTS ${LEVEL_ZERO_HINTS}
    PATH_SUFFIXES lib lib64 bin
)

if(NOT LEVEL_ZERO_LOADER)
    message(WARNING "Level Zero loader library not found. Building stub shim.")
    set(L0_BUILD_STUB ON)
else()
    set(L0_BUILD_STUB OFF)
endif()

add_library(${L0_SHIM_NAME} SHARED
    shim.cpp
)

target_include_directories(${L0_SHIM_NAME} PRIVATE
    ${L0_INCLUDE_DIR}
)

if(NOT L0_BUILD_STUB)
    target_link_libraries(${L0_SHIM_NAME} PRIVATE ${LEVEL_ZERO_LOADER})
else()
    target_compile_definitions(${L0_SHIM_NAME} PRIVATE L0_STUB=1)
endif()

if(WIN32)
    set_target_properties(${L0_SHIM_NAME} PROPERTIES
        PREFIX ""
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

install(TARGETS ${L0_SHIM_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES shim.h DESTINATION include)

message(STATUS "=== LevelZeroShim Build Configuration ===")
message(STATUS "Loader library: ${LEVEL_ZERO_LOADER}")
message(STATUS "Stub build: ${L0_BUILD_STUB}")
