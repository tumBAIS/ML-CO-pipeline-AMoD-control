cmake_minimum_required(VERSION 3.15)
project(sl-co-amod)
#set(CMAKE_CXX_STANDARD 17)
set(PYTHON_EXECUTABLE /usr/bin/python3.8)

#FIND_PACKAGE(pybind11 CONFIG REQUIRED)

# pybind11
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/pybind11)

pybind11_add_module (
        pybind11module
        ${PROJECT_SOURCE_DIR}/module/cplusplus_interface.cpp
)

target_include_directories (
        pybind11module
        PRIVATE ${PROJECT_SOURCE_DIR}/module
        PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(pybind11module PRIVATE lib/decimal_for_cpp/include)
target_link_libraries(pybind11module PRIVATE project_options)
target_link_libraries(pybind11module PRIVATE CONAN_PKG::boost)
target_link_libraries(pybind11module PRIVATE CONAN_PKG::spdlog)

#SET(CMAKE_BUILD_TYPE Debug)
SET(CONAN_EXTRA_REQUIRES boost/1.72.0 benchmark/1.5.0)
include(cmake/Conan.cmake)
run_conan()


# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

# enable flto for non-debug builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

if (MSVC)
else ()
    ## Flags
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    #set (CMAKE_CXX_COMPILER "/usr/bin/icpc")
    #set (CMAKE_LINKER "/usr/bin/xild")
    #set (CMAKE_AR "/usr/bin/xiar")
endif ()


file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES} include/kedgedisjoint.hpp include/kdisjoint.hpp module/cplusplus_interface.cpp include/kedgedisjoint.hpp)


if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:/Od >)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:/O2>)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-O3 -flto -frename-registers -march=skylake -DNDEBUG -s>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:-O0 -ggdb -march=native>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -flto -frename-registers -ggdb -march=native>)

    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-flto")
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PRIVATE lib/decimal_for_cpp/include)
target_link_libraries(${PROJECT_NAME} PRIVATE project_options)
target_link_libraries(${PROJECT_NAME} PRIVATE CONAN_PKG::boost)
target_link_libraries(${PROJECT_NAME} PRIVATE CONAN_PKG::spdlog)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)

if (MSVC)
else ()
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif ()



if(CONAN_CATCH2_ROOT_DEBUG)
    include(${CONAN_CATCH2_ROOT_DEBUG}/lib/cmake/Catch2/Catch.cmake)
else()
    include(${CONAN_CATCH2_ROOT}/lib/cmake/Catch2/Catch.cmake)
endif()
