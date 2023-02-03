include(CMakeParseArguments)

macro(conan_find_apple_frameworks FRAMEWORKS_FOUND FRAMEWORKS SUFFIX BUILD_TYPE)
    if(APPLE)
        if(CMAKE_BUILD_TYPE)
            set(_BTYPE ${CMAKE_BUILD_TYPE})
        elseif(NOT BUILD_TYPE STREQUAL "")
            set(_BTYPE ${BUILD_TYPE})
        endif()
        if(_BTYPE)
            if(${_BTYPE} MATCHES "Debug|_DEBUG")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "Release|_RELEASE")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "RelWithDebInfo|_RELWITHDEBINFO")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "MinSizeRel|_MINSIZEREL")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            endif()
        endif()
        foreach(_FRAMEWORK ${FRAMEWORKS})
            # https://cmake.org/pipermail/cmake-developers/2017-August/030199.html
            find_library(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND NAME ${_FRAMEWORK} PATHS ${CONAN_FRAMEWORK_DIRS${SUFFIX}} CMAKE_FIND_ROOT_PATH_BOTH)
            if(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND)
                list(APPEND ${FRAMEWORKS_FOUND} ${CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND})
            else()
                message(FATAL_ERROR "Framework library ${_FRAMEWORK} not found in paths: ${CONAN_FRAMEWORK_DIRS${SUFFIX}}")
            endif()
        endforeach()
    endif()
endmacro()


#################
###  BOOST
#################
set(CONAN_BOOST_ROOT "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4")
set(CONAN_INCLUDE_DIRS_BOOST "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/include")
set(CONAN_LIB_DIRS_BOOST "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/lib")
set(CONAN_BIN_DIRS_BOOST )
set(CONAN_RES_DIRS_BOOST )
set(CONAN_SRC_DIRS_BOOST )
set(CONAN_BUILD_DIRS_BOOST "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/")
set(CONAN_FRAMEWORK_DIRS_BOOST )
set(CONAN_LIBS_BOOST boost_contract boost_coroutine boost_fiber_numa boost_fiber boost_context boost_graph boost_iostreams boost_log_setup boost_log boost_locale boost_math_c99 boost_math_c99f boost_math_c99l boost_math_tr1 boost_math_tr1f boost_math_tr1l boost_program_options boost_random boost_regex boost_stacktrace_addr2line boost_stacktrace_backtrace boost_stacktrace_basic boost_stacktrace_noop boost_timer boost_type_erasure boost_thread boost_atomic boost_chrono boost_container boost_date_time boost_unit_test_framework boost_prg_exec_monitor boost_test_exec_monitor boost_exception boost_wave boost_filesystem boost_wserialization boost_serialization)
set(CONAN_PKG_LIBS_BOOST boost_contract boost_coroutine boost_fiber_numa boost_fiber boost_context boost_graph boost_iostreams boost_log_setup boost_log boost_locale boost_math_c99 boost_math_c99f boost_math_c99l boost_math_tr1 boost_math_tr1f boost_math_tr1l boost_program_options boost_random boost_regex boost_stacktrace_addr2line boost_stacktrace_backtrace boost_stacktrace_basic boost_stacktrace_noop boost_timer boost_type_erasure boost_thread boost_atomic boost_chrono boost_container boost_date_time boost_unit_test_framework boost_prg_exec_monitor boost_test_exec_monitor boost_exception boost_wave boost_filesystem boost_wserialization boost_serialization)
set(CONAN_SYSTEM_LIBS_BOOST dl rt pthread)
set(CONAN_FRAMEWORKS_BOOST )
set(CONAN_FRAMEWORKS_FOUND_BOOST "")  # Will be filled later
set(CONAN_DEFINES_BOOST "-DBOOST_STACKTRACE_ADDR2LINE_LOCATION=\"/usr/bin/addr2line\""
			"-DBOOST_STACKTRACE_USE_ADDR2LINE"
			"-DBOOST_STACKTRACE_USE_BACKTRACE"
			"-DBOOST_STACKTRACE_USE_NOOP")
set(CONAN_BUILD_MODULES_PATHS_BOOST )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_BOOST "BOOST_STACKTRACE_ADDR2LINE_LOCATION=\"/usr/bin/addr2line\""
			"BOOST_STACKTRACE_USE_ADDR2LINE"
			"BOOST_STACKTRACE_USE_BACKTRACE"
			"BOOST_STACKTRACE_USE_NOOP")

set(CONAN_C_FLAGS_BOOST "")
set(CONAN_CXX_FLAGS_BOOST "")
set(CONAN_SHARED_LINKER_FLAGS_BOOST "")
set(CONAN_EXE_LINKER_FLAGS_BOOST "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_BOOST_LIST "")
set(CONAN_CXX_FLAGS_BOOST_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_BOOST_LIST "")
set(CONAN_EXE_LINKER_FLAGS_BOOST_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_BOOST "${CONAN_FRAMEWORKS_BOOST}" "_BOOST" "")
# Append to aggregated values variable
set(CONAN_LIBS_BOOST ${CONAN_PKG_LIBS_BOOST} ${CONAN_SYSTEM_LIBS_BOOST} ${CONAN_FRAMEWORKS_FOUND_BOOST})


#################
###  BENCHMARK
#################
set(CONAN_BENCHMARK_ROOT "/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984")
set(CONAN_INCLUDE_DIRS_BENCHMARK "/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/include")
set(CONAN_LIB_DIRS_BENCHMARK "/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/lib")
set(CONAN_BIN_DIRS_BENCHMARK )
set(CONAN_RES_DIRS_BENCHMARK )
set(CONAN_SRC_DIRS_BENCHMARK )
set(CONAN_BUILD_DIRS_BENCHMARK "/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/")
set(CONAN_FRAMEWORK_DIRS_BENCHMARK )
set(CONAN_LIBS_BENCHMARK benchmark benchmark_main)
set(CONAN_PKG_LIBS_BENCHMARK benchmark benchmark_main)
set(CONAN_SYSTEM_LIBS_BENCHMARK pthread rt)
set(CONAN_FRAMEWORKS_BENCHMARK )
set(CONAN_FRAMEWORKS_FOUND_BENCHMARK "")  # Will be filled later
set(CONAN_DEFINES_BENCHMARK )
set(CONAN_BUILD_MODULES_PATHS_BENCHMARK )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_BENCHMARK )

set(CONAN_C_FLAGS_BENCHMARK "")
set(CONAN_CXX_FLAGS_BENCHMARK "")
set(CONAN_SHARED_LINKER_FLAGS_BENCHMARK "")
set(CONAN_EXE_LINKER_FLAGS_BENCHMARK "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_BENCHMARK_LIST "")
set(CONAN_CXX_FLAGS_BENCHMARK_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_BENCHMARK_LIST "")
set(CONAN_EXE_LINKER_FLAGS_BENCHMARK_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_BENCHMARK "${CONAN_FRAMEWORKS_BENCHMARK}" "_BENCHMARK" "")
# Append to aggregated values variable
set(CONAN_LIBS_BENCHMARK ${CONAN_PKG_LIBS_BENCHMARK} ${CONAN_SYSTEM_LIBS_BENCHMARK} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK})


#################
###  CATCH2
#################
set(CONAN_CATCH2_ROOT "/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_CATCH2 "/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_CATCH2 "/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib")
set(CONAN_BIN_DIRS_CATCH2 )
set(CONAN_RES_DIRS_CATCH2 )
set(CONAN_SRC_DIRS_CATCH2 )
set(CONAN_BUILD_DIRS_CATCH2 "/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/Catch2")
set(CONAN_FRAMEWORK_DIRS_CATCH2 )
set(CONAN_LIBS_CATCH2 )
set(CONAN_PKG_LIBS_CATCH2 )
set(CONAN_SYSTEM_LIBS_CATCH2 )
set(CONAN_FRAMEWORKS_CATCH2 )
set(CONAN_FRAMEWORKS_FOUND_CATCH2 "")  # Will be filled later
set(CONAN_DEFINES_CATCH2 )
set(CONAN_BUILD_MODULES_PATHS_CATCH2 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_CATCH2 )

set(CONAN_C_FLAGS_CATCH2 "")
set(CONAN_CXX_FLAGS_CATCH2 "")
set(CONAN_SHARED_LINKER_FLAGS_CATCH2 "")
set(CONAN_EXE_LINKER_FLAGS_CATCH2 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_CATCH2_LIST "")
set(CONAN_CXX_FLAGS_CATCH2_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_CATCH2_LIST "")
set(CONAN_EXE_LINKER_FLAGS_CATCH2_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_CATCH2 "${CONAN_FRAMEWORKS_CATCH2}" "_CATCH2" "")
# Append to aggregated values variable
set(CONAN_LIBS_CATCH2 ${CONAN_PKG_LIBS_CATCH2} ${CONAN_SYSTEM_LIBS_CATCH2} ${CONAN_FRAMEWORKS_FOUND_CATCH2})


#################
###  PYBIND11
#################
set(CONAN_PYBIND11_ROOT "/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_PYBIND11 "/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/pybind11"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(CONAN_LIB_DIRS_PYBIND11 "/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib")
set(CONAN_BIN_DIRS_PYBIND11 )
set(CONAN_RES_DIRS_PYBIND11 )
set(CONAN_SRC_DIRS_PYBIND11 )
set(CONAN_BUILD_DIRS_PYBIND11 "/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/pybind11")
set(CONAN_FRAMEWORK_DIRS_PYBIND11 )
set(CONAN_LIBS_PYBIND11 )
set(CONAN_PKG_LIBS_PYBIND11 )
set(CONAN_SYSTEM_LIBS_PYBIND11 )
set(CONAN_FRAMEWORKS_PYBIND11 )
set(CONAN_FRAMEWORKS_FOUND_PYBIND11 "")  # Will be filled later
set(CONAN_DEFINES_PYBIND11 )
set(CONAN_BUILD_MODULES_PATHS_PYBIND11 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_PYBIND11 )

set(CONAN_C_FLAGS_PYBIND11 "")
set(CONAN_CXX_FLAGS_PYBIND11 "")
set(CONAN_SHARED_LINKER_FLAGS_PYBIND11 "")
set(CONAN_EXE_LINKER_FLAGS_PYBIND11 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_PYBIND11_LIST "")
set(CONAN_CXX_FLAGS_PYBIND11_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST "")
set(CONAN_EXE_LINKER_FLAGS_PYBIND11_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_PYBIND11 "${CONAN_FRAMEWORKS_PYBIND11}" "_PYBIND11" "")
# Append to aggregated values variable
set(CONAN_LIBS_PYBIND11 ${CONAN_PKG_LIBS_PYBIND11} ${CONAN_SYSTEM_LIBS_PYBIND11} ${CONAN_FRAMEWORKS_FOUND_PYBIND11})


#################
###  DOCOPT.CPP
#################
set(CONAN_DOCOPT.CPP_ROOT "/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4")
set(CONAN_INCLUDE_DIRS_DOCOPT.CPP "/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/include")
set(CONAN_LIB_DIRS_DOCOPT.CPP "/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib")
set(CONAN_BIN_DIRS_DOCOPT.CPP )
set(CONAN_RES_DIRS_DOCOPT.CPP )
set(CONAN_SRC_DIRS_DOCOPT.CPP )
set(CONAN_BUILD_DIRS_DOCOPT.CPP "/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/"
			"/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_DOCOPT.CPP )
set(CONAN_LIBS_DOCOPT.CPP docopt)
set(CONAN_PKG_LIBS_DOCOPT.CPP docopt)
set(CONAN_SYSTEM_LIBS_DOCOPT.CPP m)
set(CONAN_FRAMEWORKS_DOCOPT.CPP )
set(CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP "")  # Will be filled later
set(CONAN_DEFINES_DOCOPT.CPP )
set(CONAN_BUILD_MODULES_PATHS_DOCOPT.CPP )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP )

set(CONAN_C_FLAGS_DOCOPT.CPP "")
set(CONAN_CXX_FLAGS_DOCOPT.CPP "")
set(CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP "")
set(CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_DOCOPT.CPP_LIST "")
set(CONAN_CXX_FLAGS_DOCOPT.CPP_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_LIST "")
set(CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP "${CONAN_FRAMEWORKS_DOCOPT.CPP}" "_DOCOPT.CPP" "")
# Append to aggregated values variable
set(CONAN_LIBS_DOCOPT.CPP ${CONAN_PKG_LIBS_DOCOPT.CPP} ${CONAN_SYSTEM_LIBS_DOCOPT.CPP} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP})


#################
###  SPDLOG
#################
set(CONAN_SPDLOG_ROOT "/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b")
set(CONAN_INCLUDE_DIRS_SPDLOG "/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/include")
set(CONAN_LIB_DIRS_SPDLOG "/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/lib")
set(CONAN_BIN_DIRS_SPDLOG )
set(CONAN_RES_DIRS_SPDLOG )
set(CONAN_SRC_DIRS_SPDLOG )
set(CONAN_BUILD_DIRS_SPDLOG "/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/")
set(CONAN_FRAMEWORK_DIRS_SPDLOG )
set(CONAN_LIBS_SPDLOG spdlogd)
set(CONAN_PKG_LIBS_SPDLOG spdlogd)
set(CONAN_SYSTEM_LIBS_SPDLOG pthread)
set(CONAN_FRAMEWORKS_SPDLOG )
set(CONAN_FRAMEWORKS_FOUND_SPDLOG "")  # Will be filled later
set(CONAN_DEFINES_SPDLOG "-DSPDLOG_COMPILED_LIB"
			"-DSPDLOG_FMT_EXTERNAL")
set(CONAN_BUILD_MODULES_PATHS_SPDLOG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_SPDLOG "SPDLOG_COMPILED_LIB"
			"SPDLOG_FMT_EXTERNAL")

set(CONAN_C_FLAGS_SPDLOG "")
set(CONAN_CXX_FLAGS_SPDLOG "")
set(CONAN_SHARED_LINKER_FLAGS_SPDLOG "")
set(CONAN_EXE_LINKER_FLAGS_SPDLOG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_SPDLOG_LIST "")
set(CONAN_CXX_FLAGS_SPDLOG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_SPDLOG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_SPDLOG "${CONAN_FRAMEWORKS_SPDLOG}" "_SPDLOG" "")
# Append to aggregated values variable
set(CONAN_LIBS_SPDLOG ${CONAN_PKG_LIBS_SPDLOG} ${CONAN_SYSTEM_LIBS_SPDLOG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG})


#################
###  ZLIB
#################
set(CONAN_ZLIB_ROOT "/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8")
set(CONAN_INCLUDE_DIRS_ZLIB "/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/include")
set(CONAN_LIB_DIRS_ZLIB "/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib")
set(CONAN_BIN_DIRS_ZLIB )
set(CONAN_RES_DIRS_ZLIB )
set(CONAN_SRC_DIRS_ZLIB )
set(CONAN_BUILD_DIRS_ZLIB "/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/")
set(CONAN_FRAMEWORK_DIRS_ZLIB )
set(CONAN_LIBS_ZLIB z)
set(CONAN_PKG_LIBS_ZLIB z)
set(CONAN_SYSTEM_LIBS_ZLIB )
set(CONAN_FRAMEWORKS_ZLIB )
set(CONAN_FRAMEWORKS_FOUND_ZLIB "")  # Will be filled later
set(CONAN_DEFINES_ZLIB )
set(CONAN_BUILD_MODULES_PATHS_ZLIB )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_ZLIB )

set(CONAN_C_FLAGS_ZLIB "")
set(CONAN_CXX_FLAGS_ZLIB "")
set(CONAN_SHARED_LINKER_FLAGS_ZLIB "")
set(CONAN_EXE_LINKER_FLAGS_ZLIB "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_ZLIB_LIST "")
set(CONAN_CXX_FLAGS_ZLIB_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST "")
set(CONAN_EXE_LINKER_FLAGS_ZLIB_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_ZLIB "${CONAN_FRAMEWORKS_ZLIB}" "_ZLIB" "")
# Append to aggregated values variable
set(CONAN_LIBS_ZLIB ${CONAN_PKG_LIBS_ZLIB} ${CONAN_SYSTEM_LIBS_ZLIB} ${CONAN_FRAMEWORKS_FOUND_ZLIB})


#################
###  BZIP2
#################
set(CONAN_BZIP2_ROOT "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef")
set(CONAN_INCLUDE_DIRS_BZIP2 "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/include")
set(CONAN_LIB_DIRS_BZIP2 "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/lib")
set(CONAN_BIN_DIRS_BZIP2 "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/bin")
set(CONAN_RES_DIRS_BZIP2 )
set(CONAN_SRC_DIRS_BZIP2 )
set(CONAN_BUILD_DIRS_BZIP2 "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/"
			"/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_BZIP2 )
set(CONAN_LIBS_BZIP2 bz2)
set(CONAN_PKG_LIBS_BZIP2 bz2)
set(CONAN_SYSTEM_LIBS_BZIP2 )
set(CONAN_FRAMEWORKS_BZIP2 )
set(CONAN_FRAMEWORKS_FOUND_BZIP2 "")  # Will be filled later
set(CONAN_DEFINES_BZIP2 )
set(CONAN_BUILD_MODULES_PATHS_BZIP2 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_BZIP2 )

set(CONAN_C_FLAGS_BZIP2 "")
set(CONAN_CXX_FLAGS_BZIP2 "")
set(CONAN_SHARED_LINKER_FLAGS_BZIP2 "")
set(CONAN_EXE_LINKER_FLAGS_BZIP2 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_BZIP2_LIST "")
set(CONAN_CXX_FLAGS_BZIP2_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST "")
set(CONAN_EXE_LINKER_FLAGS_BZIP2_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_BZIP2 "${CONAN_FRAMEWORKS_BZIP2}" "_BZIP2" "")
# Append to aggregated values variable
set(CONAN_LIBS_BZIP2 ${CONAN_PKG_LIBS_BZIP2} ${CONAN_SYSTEM_LIBS_BZIP2} ${CONAN_FRAMEWORKS_FOUND_BZIP2})


#################
###  LIBBACKTRACE
#################
set(CONAN_LIBBACKTRACE_ROOT "/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8")
set(CONAN_INCLUDE_DIRS_LIBBACKTRACE "/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/include")
set(CONAN_LIB_DIRS_LIBBACKTRACE "/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib")
set(CONAN_BIN_DIRS_LIBBACKTRACE )
set(CONAN_RES_DIRS_LIBBACKTRACE )
set(CONAN_SRC_DIRS_LIBBACKTRACE )
set(CONAN_BUILD_DIRS_LIBBACKTRACE "/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/")
set(CONAN_FRAMEWORK_DIRS_LIBBACKTRACE )
set(CONAN_LIBS_LIBBACKTRACE backtrace)
set(CONAN_PKG_LIBS_LIBBACKTRACE backtrace)
set(CONAN_SYSTEM_LIBS_LIBBACKTRACE )
set(CONAN_FRAMEWORKS_LIBBACKTRACE )
set(CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE "")  # Will be filled later
set(CONAN_DEFINES_LIBBACKTRACE )
set(CONAN_BUILD_MODULES_PATHS_LIBBACKTRACE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE )

set(CONAN_C_FLAGS_LIBBACKTRACE "")
set(CONAN_CXX_FLAGS_LIBBACKTRACE "")
set(CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE "")
set(CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBBACKTRACE_LIST "")
set(CONAN_CXX_FLAGS_LIBBACKTRACE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE "${CONAN_FRAMEWORKS_LIBBACKTRACE}" "_LIBBACKTRACE" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBBACKTRACE ${CONAN_PKG_LIBS_LIBBACKTRACE} ${CONAN_SYSTEM_LIBS_LIBBACKTRACE} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE})


#################
###  FMT
#################
set(CONAN_FMT_ROOT "/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4")
set(CONAN_INCLUDE_DIRS_FMT "/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/include")
set(CONAN_LIB_DIRS_FMT "/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib")
set(CONAN_BIN_DIRS_FMT )
set(CONAN_RES_DIRS_FMT )
set(CONAN_SRC_DIRS_FMT )
set(CONAN_BUILD_DIRS_FMT "/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/")
set(CONAN_FRAMEWORK_DIRS_FMT )
set(CONAN_LIBS_FMT fmtd)
set(CONAN_PKG_LIBS_FMT fmtd)
set(CONAN_SYSTEM_LIBS_FMT )
set(CONAN_FRAMEWORKS_FMT )
set(CONAN_FRAMEWORKS_FOUND_FMT "")  # Will be filled later
set(CONAN_DEFINES_FMT )
set(CONAN_BUILD_MODULES_PATHS_FMT )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_FMT )

set(CONAN_C_FLAGS_FMT "")
set(CONAN_CXX_FLAGS_FMT "")
set(CONAN_SHARED_LINKER_FLAGS_FMT "")
set(CONAN_EXE_LINKER_FLAGS_FMT "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_FMT_LIST "")
set(CONAN_CXX_FLAGS_FMT_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_FMT_LIST "")
set(CONAN_EXE_LINKER_FLAGS_FMT_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_FMT "${CONAN_FRAMEWORKS_FMT}" "_FMT" "")
# Append to aggregated values variable
set(CONAN_LIBS_FMT ${CONAN_PKG_LIBS_FMT} ${CONAN_SYSTEM_LIBS_FMT} ${CONAN_FRAMEWORKS_FOUND_FMT})


### Definition of global aggregated variables ###

set(CONAN_PACKAGE_NAME None)
set(CONAN_PACKAGE_VERSION None)

set(CONAN_SETTINGS_ARCH "x86_64")
set(CONAN_SETTINGS_ARCH_BUILD "x86_64")
set(CONAN_SETTINGS_BUILD_TYPE "Debug")
set(CONAN_SETTINGS_COMPILER "gcc")
set(CONAN_SETTINGS_COMPILER_LIBCXX "libstdc++11")
set(CONAN_SETTINGS_COMPILER_VERSION "9")
set(CONAN_SETTINGS_OS "Linux")
set(CONAN_SETTINGS_OS_BUILD "Linux")

set(CONAN_DEPENDENCIES boost benchmark catch2 pybind11 docopt.cpp spdlog zlib bzip2 libbacktrace fmt)
# Storing original command line args (CMake helper) flags
set(CONAN_CMD_CXX_FLAGS ${CONAN_CXX_FLAGS})

set(CONAN_CMD_SHARED_LINKER_FLAGS ${CONAN_SHARED_LINKER_FLAGS})
set(CONAN_CMD_C_FLAGS ${CONAN_C_FLAGS})
# Defining accumulated conan variables for all deps

set(CONAN_INCLUDE_DIRS "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/include"
			"/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/include"
			"/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/pybind11"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include"
			"/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/include"
			"/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/include"
			"/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/include"
			"/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/include"
			"/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/include"
			"/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/include" ${CONAN_INCLUDE_DIRS})
set(CONAN_LIB_DIRS "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/lib"
			"/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/lib"
			"/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib"
			"/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib"
			"/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/lib"
			"/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib"
			"/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/lib"
			"/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/lib"
			"/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib" ${CONAN_LIB_DIRS})
set(CONAN_BIN_DIRS "/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/bin" ${CONAN_BIN_DIRS})
set(CONAN_RES_DIRS  ${CONAN_RES_DIRS})
set(CONAN_FRAMEWORK_DIRS  ${CONAN_FRAMEWORK_DIRS})
set(CONAN_LIBS boost_contract boost_coroutine boost_fiber_numa boost_fiber boost_context boost_graph boost_iostreams boost_log_setup boost_log boost_locale boost_math_c99 boost_math_c99f boost_math_c99l boost_math_tr1 boost_math_tr1f boost_math_tr1l boost_program_options boost_random boost_regex boost_stacktrace_addr2line boost_stacktrace_backtrace boost_stacktrace_basic boost_stacktrace_noop boost_timer boost_type_erasure boost_thread boost_atomic boost_chrono boost_container boost_date_time boost_unit_test_framework boost_prg_exec_monitor boost_test_exec_monitor boost_exception boost_wave boost_filesystem boost_wserialization boost_serialization benchmark benchmark_main docopt spdlogd z bz2 backtrace fmtd ${CONAN_LIBS})
set(CONAN_PKG_LIBS boost_contract boost_coroutine boost_fiber_numa boost_fiber boost_context boost_graph boost_iostreams boost_log_setup boost_log boost_locale boost_math_c99 boost_math_c99f boost_math_c99l boost_math_tr1 boost_math_tr1f boost_math_tr1l boost_program_options boost_random boost_regex boost_stacktrace_addr2line boost_stacktrace_backtrace boost_stacktrace_basic boost_stacktrace_noop boost_timer boost_type_erasure boost_thread boost_atomic boost_chrono boost_container boost_date_time boost_unit_test_framework boost_prg_exec_monitor boost_test_exec_monitor boost_exception boost_wave boost_filesystem boost_wserialization boost_serialization benchmark benchmark_main docopt spdlogd z bz2 backtrace fmtd ${CONAN_PKG_LIBS})
set(CONAN_SYSTEM_LIBS dl rt m pthread ${CONAN_SYSTEM_LIBS})
set(CONAN_FRAMEWORKS  ${CONAN_FRAMEWORKS})
set(CONAN_FRAMEWORKS_FOUND "")  # Will be filled later
set(CONAN_DEFINES "-DSPDLOG_COMPILED_LIB"
			"-DSPDLOG_FMT_EXTERNAL"
			"-DBOOST_STACKTRACE_ADDR2LINE_LOCATION=\"/usr/bin/addr2line\""
			"-DBOOST_STACKTRACE_USE_ADDR2LINE"
			"-DBOOST_STACKTRACE_USE_BACKTRACE"
			"-DBOOST_STACKTRACE_USE_NOOP" ${CONAN_DEFINES})
set(CONAN_BUILD_MODULES_PATHS  ${CONAN_BUILD_MODULES_PATHS})
set(CONAN_CMAKE_MODULE_PATH "/home/kaijungel/.conan/data/boost/1.72.0/_/_/package/a77ca77d437f4354bdec12173c1a0c2b8b7d01f4/"
			"/home/kaijungel/.conan/data/benchmark/1.5.0/_/_/package/701cd14b2476695e09483232d7420e99459b5984/"
			"/home/kaijungel/.conan/data/catch2/2.11.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/Catch2"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/kaijungel/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib/cmake/pybind11"
			"/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/"
			"/home/kaijungel/.conan/data/docopt.cpp/0.6.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/lib/cmake"
			"/home/kaijungel/.conan/data/spdlog/1.5.0/_/_/package/9df34d57e1d38147409e41fa69c20cd1153df23b/"
			"/home/kaijungel/.conan/data/zlib/1.2.11/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/"
			"/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/"
			"/home/kaijungel/.conan/data/bzip2/1.0.8/_/_/package/1d984b241c5c01f748cafe0b37dc9567ed236fef/lib/cmake"
			"/home/kaijungel/.conan/data/libbacktrace/cci.20210118/_/_/package/23b828d52c0630e6b0b96d2945419feb7843c4f8/"
			"/home/kaijungel/.conan/data/fmt/6.1.2/_/_/package/038baac88f4c7bfa972ce5adac1616bed8fe2ef4/" ${CONAN_CMAKE_MODULE_PATH})

set(CONAN_CXX_FLAGS " ${CONAN_CXX_FLAGS}")
set(CONAN_SHARED_LINKER_FLAGS " ${CONAN_SHARED_LINKER_FLAGS}")
set(CONAN_EXE_LINKER_FLAGS " ${CONAN_EXE_LINKER_FLAGS}")
set(CONAN_C_FLAGS " ${CONAN_C_FLAGS}")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND "${CONAN_FRAMEWORKS}" "" "")
# Append to aggregated values variable: Use CONAN_LIBS instead of CONAN_PKG_LIBS to include user appended vars
set(CONAN_LIBS ${CONAN_LIBS} ${CONAN_SYSTEM_LIBS} ${CONAN_FRAMEWORKS_FOUND})


###  Definition of macros and functions ###

macro(conan_define_targets)
    if(${CMAKE_VERSION} VERSION_LESS "3.1.2")
        message(FATAL_ERROR "TARGETS not supported by your CMake version!")
    endif()  # CMAKE > 3.x
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CONAN_CMD_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CONAN_CMD_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CONAN_CMD_SHARED_LINKER_FLAGS}")


    set(_CONAN_PKG_LIBS_BOOST_DEPENDENCIES "${CONAN_SYSTEM_LIBS_BOOST} ${CONAN_FRAMEWORKS_FOUND_BOOST} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BOOST_DEPENDENCIES "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BOOST}" "${CONAN_LIB_DIRS_BOOST}"
                                  CONAN_PACKAGE_TARGETS_BOOST "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES}"
                                  "" boost)
    set(_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_BOOST_DEBUG} ${CONAN_FRAMEWORKS_FOUND_BOOST_DEBUG} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BOOST_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BOOST_DEBUG}" "${CONAN_LIB_DIRS_BOOST_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_BOOST_DEBUG "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_DEBUG}"
                                  "debug" boost)
    set(_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_BOOST_RELEASE} ${CONAN_FRAMEWORKS_FOUND_BOOST_RELEASE} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BOOST_RELEASE}" "${CONAN_LIB_DIRS_BOOST_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_BOOST_RELEASE "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELEASE}"
                                  "release" boost)
    set(_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_BOOST_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_BOOST_RELWITHDEBINFO} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BOOST_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_BOOST_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_BOOST_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" boost)
    set(_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_BOOST_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_BOOST_MINSIZEREL} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BOOST_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BOOST_MINSIZEREL}" "${CONAN_LIB_DIRS_BOOST_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_BOOST_MINSIZEREL "${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" boost)

    add_library(CONAN_PKG::boost INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::boost PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_BOOST} ${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BOOST_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_BOOST_RELEASE} ${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BOOST_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_BOOST_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BOOST_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_BOOST_MINSIZEREL} ${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BOOST_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_BOOST_DEBUG} ${_CONAN_PKG_LIBS_BOOST_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BOOST_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BOOST_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_BOOST}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_BOOST_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_BOOST_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_BOOST_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_BOOST_DEBUG}>)
    set_property(TARGET CONAN_PKG::boost PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_BOOST}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_BOOST_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_BOOST_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_BOOST_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_BOOST_DEBUG}>)
    set_property(TARGET CONAN_PKG::boost PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_BOOST_LIST} ${CONAN_CXX_FLAGS_BOOST_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_BOOST_RELEASE_LIST} ${CONAN_CXX_FLAGS_BOOST_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_BOOST_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_BOOST_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_BOOST_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_BOOST_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_BOOST_DEBUG_LIST}  ${CONAN_CXX_FLAGS_BOOST_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES "${CONAN_SYSTEM_LIBS_BENCHMARK} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BENCHMARK}" "${CONAN_LIB_DIRS_BENCHMARK}"
                                  CONAN_PACKAGE_TARGETS_BENCHMARK "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES}"
                                  "" benchmark)
    set(_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_BENCHMARK_DEBUG} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BENCHMARK_DEBUG}" "${CONAN_LIB_DIRS_BENCHMARK_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_BENCHMARK_DEBUG "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_DEBUG}"
                                  "debug" benchmark)
    set(_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_BENCHMARK_RELEASE} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BENCHMARK_RELEASE}" "${CONAN_LIB_DIRS_BENCHMARK_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_BENCHMARK_RELEASE "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELEASE}"
                                  "release" benchmark)
    set(_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_BENCHMARK_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BENCHMARK_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_BENCHMARK_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_BENCHMARK_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" benchmark)
    set(_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_BENCHMARK_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_BENCHMARK_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BENCHMARK_MINSIZEREL}" "${CONAN_LIB_DIRS_BENCHMARK_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_BENCHMARK_MINSIZEREL "${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" benchmark)

    add_library(CONAN_PKG::benchmark INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::benchmark PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_BENCHMARK} ${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BENCHMARK_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_BENCHMARK_RELEASE} ${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BENCHMARK_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_BENCHMARK_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BENCHMARK_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_BENCHMARK_MINSIZEREL} ${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BENCHMARK_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_BENCHMARK_DEBUG} ${_CONAN_PKG_LIBS_BENCHMARK_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BENCHMARK_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BENCHMARK_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::benchmark PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_BENCHMARK}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_BENCHMARK_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_BENCHMARK_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_BENCHMARK_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_BENCHMARK_DEBUG}>)
    set_property(TARGET CONAN_PKG::benchmark PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_BENCHMARK}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_BENCHMARK_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_BENCHMARK_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_BENCHMARK_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_BENCHMARK_DEBUG}>)
    set_property(TARGET CONAN_PKG::benchmark PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_BENCHMARK_LIST} ${CONAN_CXX_FLAGS_BENCHMARK_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_BENCHMARK_RELEASE_LIST} ${CONAN_CXX_FLAGS_BENCHMARK_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_BENCHMARK_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_BENCHMARK_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_BENCHMARK_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_BENCHMARK_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_BENCHMARK_DEBUG_LIST}  ${CONAN_CXX_FLAGS_BENCHMARK_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES "${CONAN_SYSTEM_LIBS_CATCH2} ${CONAN_FRAMEWORKS_FOUND_CATCH2} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_CATCH2_DEPENDENCIES "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_CATCH2}" "${CONAN_LIB_DIRS_CATCH2}"
                                  CONAN_PACKAGE_TARGETS_CATCH2 "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES}"
                                  "" catch2)
    set(_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_CATCH2_DEBUG} ${CONAN_FRAMEWORKS_FOUND_CATCH2_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_CATCH2_DEBUG}" "${CONAN_LIB_DIRS_CATCH2_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_CATCH2_DEBUG "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_DEBUG}"
                                  "debug" catch2)
    set(_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_CATCH2_RELEASE} ${CONAN_FRAMEWORKS_FOUND_CATCH2_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_CATCH2_RELEASE}" "${CONAN_LIB_DIRS_CATCH2_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_CATCH2_RELEASE "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELEASE}"
                                  "release" catch2)
    set(_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_CATCH2_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_CATCH2_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_CATCH2_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_CATCH2_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_CATCH2_RELWITHDEBINFO "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" catch2)
    set(_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_CATCH2_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_CATCH2_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_CATCH2_MINSIZEREL}" "${CONAN_LIB_DIRS_CATCH2_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_CATCH2_MINSIZEREL "${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" catch2)

    add_library(CONAN_PKG::catch2 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::catch2 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_CATCH2} ${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_CATCH2_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_CATCH2_RELEASE} ${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_CATCH2_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_CATCH2_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_CATCH2_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_CATCH2_MINSIZEREL} ${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_CATCH2_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_CATCH2_DEBUG} ${_CONAN_PKG_LIBS_CATCH2_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_CATCH2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_CATCH2_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::catch2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_CATCH2}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_CATCH2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_CATCH2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_CATCH2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_CATCH2_DEBUG}>)
    set_property(TARGET CONAN_PKG::catch2 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_CATCH2}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_CATCH2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_CATCH2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_CATCH2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_CATCH2_DEBUG}>)
    set_property(TARGET CONAN_PKG::catch2 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_CATCH2_LIST} ${CONAN_CXX_FLAGS_CATCH2_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_CATCH2_RELEASE_LIST} ${CONAN_CXX_FLAGS_CATCH2_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_CATCH2_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_CATCH2_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_CATCH2_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_CATCH2_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_CATCH2_DEBUG_LIST}  ${CONAN_CXX_FLAGS_CATCH2_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES "${CONAN_SYSTEM_LIBS_PYBIND11} ${CONAN_FRAMEWORKS_FOUND_PYBIND11} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11}" "${CONAN_LIB_DIRS_PYBIND11}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11 "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}"
                                  "" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_PYBIND11_DEBUG} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_DEBUG}" "${CONAN_LIB_DIRS_PYBIND11_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_DEBUG "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}"
                                  "debug" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_PYBIND11_RELEASE} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_RELEASE}" "${CONAN_LIB_DIRS_PYBIND11_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_RELEASE "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}"
                                  "release" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_PYBIND11_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_PYBIND11_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" pybind11)
    set(_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_PYBIND11_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_PYBIND11_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PYBIND11_MINSIZEREL}" "${CONAN_LIB_DIRS_PYBIND11_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_PYBIND11_MINSIZEREL "${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" pybind11)

    add_library(CONAN_PKG::pybind11 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_PYBIND11} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_PYBIND11_RELEASE} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_PYBIND11_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_PYBIND11_MINSIZEREL} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_PYBIND11_DEBUG} ${_CONAN_PKG_LIBS_PYBIND11_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PYBIND11_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_PYBIND11}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_PYBIND11_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_PYBIND11_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_PYBIND11_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_PYBIND11_DEBUG}>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_PYBIND11}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_PYBIND11_DEBUG}>)
    set_property(TARGET CONAN_PKG::pybind11 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_PYBIND11_LIST} ${CONAN_CXX_FLAGS_PYBIND11_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_PYBIND11_RELEASE_LIST} ${CONAN_CXX_FLAGS_PYBIND11_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_PYBIND11_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_PYBIND11_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_PYBIND11_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_PYBIND11_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_PYBIND11_DEBUG_LIST}  ${CONAN_CXX_FLAGS_PYBIND11_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES "${CONAN_SYSTEM_LIBS_DOCOPT.CPP} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_DOCOPT.CPP}" "${CONAN_LIB_DIRS_DOCOPT.CPP}"
                                  CONAN_PACKAGE_TARGETS_DOCOPT.CPP "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES}"
                                  "" docopt.cpp)
    set(_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_DOCOPT.CPP_DEBUG} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_DOCOPT.CPP_DEBUG}" "${CONAN_LIB_DIRS_DOCOPT.CPP_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_DOCOPT.CPP_DEBUG "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_DEBUG}"
                                  "debug" docopt.cpp)
    set(_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_DOCOPT.CPP_RELEASE} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_DOCOPT.CPP_RELEASE}" "${CONAN_LIB_DIRS_DOCOPT.CPP_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_DOCOPT.CPP_RELEASE "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELEASE}"
                                  "release" docopt.cpp)
    set(_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_DOCOPT.CPP_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_DOCOPT.CPP_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_DOCOPT.CPP_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_DOCOPT.CPP_RELWITHDEBINFO "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" docopt.cpp)
    set(_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_DOCOPT.CPP_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_DOCOPT.CPP_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_DOCOPT.CPP_MINSIZEREL}" "${CONAN_LIB_DIRS_DOCOPT.CPP_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_DOCOPT.CPP_MINSIZEREL "${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" docopt.cpp)

    add_library(CONAN_PKG::docopt.cpp INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::docopt.cpp PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_DOCOPT.CPP} ${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_DOCOPT.CPP_RELEASE} ${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_DOCOPT.CPP_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_DOCOPT.CPP_MINSIZEREL} ${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_DOCOPT.CPP_DEBUG} ${_CONAN_PKG_LIBS_DOCOPT.CPP_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_DOCOPT.CPP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_DOCOPT.CPP_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::docopt.cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_DOCOPT.CPP}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_DOCOPT.CPP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_DOCOPT.CPP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_DOCOPT.CPP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DOCOPT.CPP_DEBUG}>)
    set_property(TARGET CONAN_PKG::docopt.cpp PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_DOCOPT.CPP_DEBUG}>)
    set_property(TARGET CONAN_PKG::docopt.cpp PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_DOCOPT.CPP_LIST} ${CONAN_CXX_FLAGS_DOCOPT.CPP_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_DOCOPT.CPP_RELEASE_LIST} ${CONAN_CXX_FLAGS_DOCOPT.CPP_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_DOCOPT.CPP_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_DOCOPT.CPP_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_DOCOPT.CPP_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_DOCOPT.CPP_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_DOCOPT.CPP_DEBUG_LIST}  ${CONAN_CXX_FLAGS_DOCOPT.CPP_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_SPDLOG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG}" "${CONAN_LIB_DIRS_SPDLOG}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}"
                                  "" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_SPDLOG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_DEBUG} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_DEBUG}" "${CONAN_LIB_DIRS_SPDLOG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_DEBUG "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}"
                                  "debug" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_SPDLOG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_RELEASE} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_RELEASE}" "${CONAN_LIB_DIRS_SPDLOG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_RELEASE "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}"
                                  "release" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_SPDLOG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_RELWITHDEBINFO} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_SPDLOG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" spdlog)
    set(_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_SPDLOG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_SPDLOG_MINSIZEREL} CONAN_PKG::fmt")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_SPDLOG_MINSIZEREL}" "${CONAN_LIB_DIRS_SPDLOG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_SPDLOG_MINSIZEREL "${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" spdlog)

    add_library(CONAN_PKG::spdlog INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_SPDLOG} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_SPDLOG_RELEASE} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_SPDLOG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_SPDLOG_MINSIZEREL} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_SPDLOG_DEBUG} ${_CONAN_PKG_LIBS_SPDLOG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_SPDLOG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_SPDLOG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_SPDLOG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_SPDLOG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_SPDLOG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_SPDLOG_DEBUG}>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_SPDLOG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_SPDLOG_DEBUG}>)
    set_property(TARGET CONAN_PKG::spdlog PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_SPDLOG_LIST} ${CONAN_CXX_FLAGS_SPDLOG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_SPDLOG_RELEASE_LIST} ${CONAN_CXX_FLAGS_SPDLOG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_SPDLOG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_SPDLOG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_SPDLOG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_SPDLOG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_SPDLOG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_SPDLOG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES "${CONAN_SYSTEM_LIBS_ZLIB} ${CONAN_FRAMEWORKS_FOUND_ZLIB} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB}" "${CONAN_LIB_DIRS_ZLIB}"
                                  CONAN_PACKAGE_TARGETS_ZLIB "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}"
                                  "" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_ZLIB_DEBUG} ${CONAN_FRAMEWORKS_FOUND_ZLIB_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_DEBUG}" "${CONAN_LIB_DIRS_ZLIB_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_DEBUG "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}"
                                  "debug" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_ZLIB_RELEASE} ${CONAN_FRAMEWORKS_FOUND_ZLIB_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_RELEASE}" "${CONAN_LIB_DIRS_ZLIB_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_RELEASE "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}"
                                  "release" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_ZLIB_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_ZLIB_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_ZLIB_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_ZLIB_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_ZLIB_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_MINSIZEREL}" "${CONAN_LIB_DIRS_ZLIB_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_MINSIZEREL "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" zlib)

    add_library(CONAN_PKG::zlib INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_ZLIB} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_ZLIB_RELEASE} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_ZLIB_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_ZLIB_MINSIZEREL} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_ZLIB_DEBUG} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_ZLIB}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_ZLIB_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_ZLIB_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_ZLIB_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_ZLIB_DEBUG}>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_ZLIB}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_ZLIB_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_ZLIB_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_ZLIB_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_ZLIB_DEBUG}>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_ZLIB_LIST} ${CONAN_CXX_FLAGS_ZLIB_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_ZLIB_RELEASE_LIST} ${CONAN_CXX_FLAGS_ZLIB_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_ZLIB_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_ZLIB_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_ZLIB_DEBUG_LIST}  ${CONAN_CXX_FLAGS_ZLIB_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES "${CONAN_SYSTEM_LIBS_BZIP2} ${CONAN_FRAMEWORKS_FOUND_BZIP2} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2}" "${CONAN_LIB_DIRS_BZIP2}"
                                  CONAN_PACKAGE_TARGETS_BZIP2 "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}"
                                  "" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_BZIP2_DEBUG} ${CONAN_FRAMEWORKS_FOUND_BZIP2_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_DEBUG}" "${CONAN_LIB_DIRS_BZIP2_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_DEBUG "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}"
                                  "debug" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_BZIP2_RELEASE} ${CONAN_FRAMEWORKS_FOUND_BZIP2_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_RELEASE}" "${CONAN_LIB_DIRS_BZIP2_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_RELEASE "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}"
                                  "release" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_BZIP2_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_BZIP2_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_BZIP2_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_BZIP2_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_BZIP2_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_MINSIZEREL}" "${CONAN_LIB_DIRS_BZIP2_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_MINSIZEREL "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" bzip2)

    add_library(CONAN_PKG::bzip2 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_BZIP2} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_BZIP2_RELEASE} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_BZIP2_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_BZIP2_MINSIZEREL} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_BZIP2_DEBUG} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_BZIP2}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_BZIP2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_BZIP2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_BZIP2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_BZIP2_DEBUG}>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_BZIP2}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_BZIP2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_BZIP2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_BZIP2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_BZIP2_DEBUG}>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_BZIP2_LIST} ${CONAN_CXX_FLAGS_BZIP2_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_BZIP2_RELEASE_LIST} ${CONAN_CXX_FLAGS_BZIP2_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_BZIP2_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_BZIP2_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_BZIP2_DEBUG_LIST}  ${CONAN_CXX_FLAGS_BZIP2_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBBACKTRACE} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBBACKTRACE}" "${CONAN_LIB_DIRS_LIBBACKTRACE}"
                                  CONAN_PACKAGE_TARGETS_LIBBACKTRACE "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES}"
                                  "" libbacktrace)
    set(_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBBACKTRACE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBBACKTRACE_DEBUG}" "${CONAN_LIB_DIRS_LIBBACKTRACE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBBACKTRACE_DEBUG "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_DEBUG}"
                                  "debug" libbacktrace)
    set(_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBBACKTRACE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBBACKTRACE_RELEASE}" "${CONAN_LIB_DIRS_LIBBACKTRACE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBBACKTRACE_RELEASE "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELEASE}"
                                  "release" libbacktrace)
    set(_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBBACKTRACE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBBACKTRACE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBBACKTRACE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBBACKTRACE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libbacktrace)
    set(_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBBACKTRACE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBBACKTRACE_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBBACKTRACE_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBBACKTRACE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBBACKTRACE_MINSIZEREL "${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libbacktrace)

    add_library(CONAN_PKG::libbacktrace INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libbacktrace PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBBACKTRACE} ${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBBACKTRACE_RELEASE} ${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBBACKTRACE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBBACKTRACE_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBBACKTRACE_DEBUG} ${_CONAN_PKG_LIBS_LIBBACKTRACE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBBACKTRACE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBBACKTRACE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libbacktrace PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBBACKTRACE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBBACKTRACE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBBACKTRACE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBBACKTRACE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBBACKTRACE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libbacktrace PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBBACKTRACE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libbacktrace PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBBACKTRACE_LIST} ${CONAN_CXX_FLAGS_LIBBACKTRACE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBBACKTRACE_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBBACKTRACE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBBACKTRACE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBBACKTRACE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBBACKTRACE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBBACKTRACE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBBACKTRACE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBBACKTRACE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES "${CONAN_SYSTEM_LIBS_FMT} ${CONAN_FRAMEWORKS_FOUND_FMT} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT}" "${CONAN_LIB_DIRS_FMT}"
                                  CONAN_PACKAGE_TARGETS_FMT "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}"
                                  "" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_FMT_DEBUG} ${CONAN_FRAMEWORKS_FOUND_FMT_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_DEBUG}" "${CONAN_LIB_DIRS_FMT_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_FMT_DEBUG "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}"
                                  "debug" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_FMT_RELEASE} ${CONAN_FRAMEWORKS_FOUND_FMT_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_RELEASE}" "${CONAN_LIB_DIRS_FMT_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_FMT_RELEASE "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}"
                                  "release" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_FMT_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_FMT_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_FMT_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_FMT_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" fmt)
    set(_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_FMT_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_FMT_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FMT_MINSIZEREL}" "${CONAN_LIB_DIRS_FMT_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_FMT_MINSIZEREL "${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" fmt)

    add_library(CONAN_PKG::fmt INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_FMT} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_FMT_RELEASE} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_FMT_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_FMT_MINSIZEREL} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_FMT_DEBUG} ${_CONAN_PKG_LIBS_FMT_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FMT_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FMT_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_FMT}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_FMT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_FMT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_FMT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_FMT_DEBUG}>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_FMT}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_FMT_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_FMT_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_FMT_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_FMT_DEBUG}>)
    set_property(TARGET CONAN_PKG::fmt PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_FMT_LIST} ${CONAN_CXX_FLAGS_FMT_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_FMT_RELEASE_LIST} ${CONAN_CXX_FLAGS_FMT_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_FMT_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_FMT_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_FMT_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_FMT_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_FMT_DEBUG_LIST}  ${CONAN_CXX_FLAGS_FMT_DEBUG_LIST}>)

    set(CONAN_TARGETS CONAN_PKG::boost CONAN_PKG::benchmark CONAN_PKG::catch2 CONAN_PKG::pybind11 CONAN_PKG::docopt.cpp CONAN_PKG::spdlog CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::libbacktrace CONAN_PKG::fmt)

endmacro()


macro(conan_basic_setup)
    set(options TARGETS NO_OUTPUT_DIRS SKIP_RPATH KEEP_RPATHS SKIP_STD SKIP_FPIC)
    cmake_parse_arguments(ARGUMENTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(CONAN_EXPORTED)
        conan_message(STATUS "Conan: called by CMake conan helper")
    endif()

    if(CONAN_IN_LOCAL_CACHE)
        conan_message(STATUS "Conan: called inside local cache")
    endif()

    if(NOT ARGUMENTS_NO_OUTPUT_DIRS)
        conan_message(STATUS "Conan: Adjusting output directories")
        conan_output_dirs_setup()
    endif()

    if(NOT ARGUMENTS_TARGETS)
        conan_message(STATUS "Conan: Using cmake global configuration")
        conan_global_flags()
    else()
        conan_message(STATUS "Conan: Using cmake targets configuration")
        conan_define_targets()
    endif()

    if(ARGUMENTS_SKIP_RPATH)
        # Change by "DEPRECATION" or "SEND_ERROR" when we are ready
        conan_message(WARNING "Conan: SKIP_RPATH is deprecated, it has been renamed to KEEP_RPATHS")
    endif()

    if(NOT ARGUMENTS_SKIP_RPATH AND NOT ARGUMENTS_KEEP_RPATHS)
        # Parameter has renamed, but we keep the compatibility with old SKIP_RPATH
        conan_set_rpath()
    endif()

    if(NOT ARGUMENTS_SKIP_STD)
        conan_set_std()
    endif()

    if(NOT ARGUMENTS_SKIP_FPIC)
        conan_set_fpic()
    endif()

    conan_check_compiler()
    conan_set_libcxx()
    conan_set_vs_runtime()
    conan_set_find_paths()
    conan_include_build_modules()
    conan_set_find_library_paths()
endmacro()


macro(conan_set_find_paths)
    # CMAKE_MODULE_PATH does not have Debug/Release config, but there are variables
    # CONAN_CMAKE_MODULE_PATH_DEBUG to be used by the consumer
    # CMake can find findXXX.cmake files in the root of packages
    set(CMAKE_MODULE_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_MODULE_PATH})

    # Make find_package() to work
    set(CMAKE_PREFIX_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_PREFIX_PATH})

    # Set the find root path (cross build)
    set(CMAKE_FIND_ROOT_PATH ${CONAN_CMAKE_FIND_ROOT_PATH} ${CMAKE_FIND_ROOT_PATH})
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
    endif()
endmacro()


macro(conan_set_find_library_paths)
    # CMAKE_INCLUDE_PATH, CMAKE_LIBRARY_PATH does not have Debug/Release config, but there are variables
    # CONAN_INCLUDE_DIRS_DEBUG/RELEASE CONAN_LIB_DIRS_DEBUG/RELEASE to be used by the consumer
    # For find_library
    set(CMAKE_INCLUDE_PATH ${CONAN_INCLUDE_DIRS} ${CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH ${CONAN_LIB_DIRS} ${CMAKE_LIBRARY_PATH})
endmacro()


macro(conan_set_vs_runtime)
    if(CONAN_LINK_RUNTIME)
        conan_get_policy(CMP0091 policy_0091)
        if(policy_0091 STREQUAL "NEW")
            if(CONAN_LINK_RUNTIME MATCHES "MTd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
            elseif(CONAN_LINK_RUNTIME MATCHES "MDd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
            elseif(CONAN_LINK_RUNTIME MATCHES "MT")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
            elseif(CONAN_LINK_RUNTIME MATCHES "MD")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
            endif()
        else()
            foreach(flag CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE
                         CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO
                         CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL)
                if(DEFINED ${flag})
                    string(REPLACE "/MD" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
            foreach(flag CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
                if(DEFINED ${flag})
                    string(REPLACE "/MDd" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
        endif()
    endif()
endmacro()


macro(conan_flags_setup)
    # Macro maintained for backwards compatibility
    conan_set_find_library_paths()
    conan_global_flags()
    conan_set_rpath()
    conan_set_vs_runtime()
    conan_set_libcxx()
endmacro()


function(conan_message MESSAGE_OUTPUT)
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        message(${ARGV${0}})
    endif()
endfunction()


function(conan_get_policy policy_id policy)
    if(POLICY "${policy_id}")
        cmake_policy(GET "${policy_id}" _policy)
        set(${policy} "${_policy}" PARENT_SCOPE)
    else()
        set(${policy} "" PARENT_SCOPE)
    endif()
endfunction()


function(conan_find_libraries_abs_path libraries package_libdir libraries_abs_path)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAME ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${CONAN_FOUND_LIBRARY})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()
    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


function(conan_package_library_targets libraries package_libdir libraries_abs_path deps build_type package_name)
    unset(_CONAN_ACTUAL_TARGETS CACHE)
    unset(_CONAN_FOUND_SYSTEM_LIBS CACHE)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAME ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(_LIB_NAME CONAN_LIB::${package_name}_${_LIBRARY_NAME}${build_type})
            add_library(${_LIB_NAME} UNKNOWN IMPORTED)
            set_target_properties(${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${CONAN_FOUND_LIBRARY})
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIB_NAME})
            set(_CONAN_ACTUAL_TARGETS ${_CONAN_ACTUAL_TARGETS} ${_LIB_NAME})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
            set(_CONAN_FOUND_SYSTEM_LIBS "${_CONAN_FOUND_SYSTEM_LIBS};${_LIBRARY_NAME}")
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()

    # Add all dependencies to all targets
    string(REPLACE " " ";" deps_list "${deps}")
    foreach(_CONAN_ACTUAL_TARGET ${_CONAN_ACTUAL_TARGETS})
        set_property(TARGET ${_CONAN_ACTUAL_TARGET} PROPERTY INTERFACE_LINK_LIBRARIES "${_CONAN_FOUND_SYSTEM_LIBS};${deps_list}")
    endforeach()

    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


macro(conan_set_libcxx)
    if(DEFINED CONAN_LIBCXX)
        conan_message(STATUS "Conan: C++ stdlib: ${CONAN_LIBCXX}")
        if(CONAN_COMPILER STREQUAL "clang" OR CONAN_COMPILER STREQUAL "apple-clang")
            if(CONAN_LIBCXX STREQUAL "libstdc++" OR CONAN_LIBCXX STREQUAL "libstdc++11" )
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
            elseif(CONAN_LIBCXX STREQUAL "libc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
            endif()
        endif()
        if(CONAN_COMPILER STREQUAL "sun-cc")
            if(CONAN_LIBCXX STREQUAL "libCstd")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=Cstd")
            elseif(CONAN_LIBCXX STREQUAL "libstdcxx")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcxx4")
            elseif(CONAN_LIBCXX STREQUAL "libstlport")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stlport4")
            elseif(CONAN_LIBCXX STREQUAL "libstdc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcpp")
            endif()
        endif()
        if(CONAN_LIBCXX STREQUAL "libstdc++11")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        elseif(CONAN_LIBCXX STREQUAL "libstdc++")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        endif()
    endif()
endmacro()


macro(conan_set_std)
    conan_message(STATUS "Conan: Adjusting language standard")
    # Do not warn "Manually-specified variables were not used by the project"
    set(ignorevar "${CONAN_STD_CXX_FLAG}${CONAN_CMAKE_CXX_STANDARD}${CONAN_CMAKE_CXX_EXTENSIONS}")
    if (CMAKE_VERSION VERSION_LESS "3.1" OR
        (CMAKE_VERSION VERSION_LESS "3.12" AND ("${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "20" OR "${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "gnu20")))
        if(CONAN_STD_CXX_FLAG)
            conan_message(STATUS "Conan setting CXX_FLAGS flags: ${CONAN_STD_CXX_FLAG}")
            set(CMAKE_CXX_FLAGS "${CONAN_STD_CXX_FLAG} ${CMAKE_CXX_FLAGS}")
        endif()
    else()
        if(CONAN_CMAKE_CXX_STANDARD)
            conan_message(STATUS "Conan setting CPP STANDARD: ${CONAN_CMAKE_CXX_STANDARD} WITH EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS}")
            set(CMAKE_CXX_STANDARD ${CONAN_CMAKE_CXX_STANDARD})
            set(CMAKE_CXX_EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS})
        endif()
    endif()
endmacro()


macro(conan_set_rpath)
    conan_message(STATUS "Conan: Adjusting default RPATHs Conan policies")
    if(APPLE)
        # https://cmake.org/Wiki/CMake_RPATH_handling
        # CONAN GUIDE: All generated libraries should have the id and dependencies to other
        # dylibs without path, just the name, EX:
        # libMyLib1.dylib:
        #     libMyLib1.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     libMyLib0.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        #     /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
        # AVOID RPATH FOR *.dylib, ALL LIBS BETWEEN THEM AND THE EXE
        # SHOULD BE ON THE LINKER RESOLVER PATH (./ IS ONE OF THEM)
        set(CMAKE_SKIP_RPATH 1 CACHE BOOL "rpaths" FORCE)
        # Policy CMP0068
        # We want the old behavior, in CMake >= 3.9 CMAKE_SKIP_RPATH won't affect the install_name in OSX
        set(CMAKE_INSTALL_NAME_DIR "")
    endif()
endmacro()


macro(conan_set_fpic)
    if(DEFINED CONAN_CMAKE_POSITION_INDEPENDENT_CODE)
        conan_message(STATUS "Conan: Adjusting fPIC flag (${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})")
        set(CMAKE_POSITION_INDEPENDENT_CODE ${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})
    endif()
endmacro()


macro(conan_output_dirs_setup)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endmacro()


macro(conan_split_version VERSION_STRING MAJOR MINOR)
    #make a list from the version string
    string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")

    #write output values
    list(LENGTH VERSION_LIST _version_len)
    list(GET VERSION_LIST 0 ${MAJOR})
    if(${_version_len} GREATER 1)
        list(GET VERSION_LIST 1 ${MINOR})
    endif()
endmacro()


macro(conan_error_compiler_version)
    message(FATAL_ERROR "Detected a mismatch for the compiler version between your conan profile settings and CMake: \n"
                        "Compiler version specified in your conan profile: ${CONAN_COMPILER_VERSION}\n"
                        "Compiler version detected in CMake: ${VERSION_MAJOR}.${VERSION_MINOR}\n"
                        "Please check your conan profile settings (conan profile show [default|your_profile_name])\n"
                        "P.S. You may set CONAN_DISABLE_CHECK_COMPILER CMake variable in order to disable this check."
           )
endmacro()

set(_CONAN_CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR})

function(conan_get_compiler CONAN_INFO_COMPILER CONAN_INFO_COMPILER_VERSION)
    conan_message(STATUS "Current conanbuildinfo.cmake directory: " ${_CONAN_CURRENT_DIR})
    if(NOT EXISTS ${_CONAN_CURRENT_DIR}/conaninfo.txt)
        conan_message(STATUS "WARN: conaninfo.txt not found")
        return()
    endif()

    file (READ "${_CONAN_CURRENT_DIR}/conaninfo.txt" CONANINFO)

    # MATCHALL will match all, including the last one, which is the full_settings one
    string(REGEX MATCH "full_settings.*" _FULL_SETTINGS_MATCHED ${CONANINFO})
    string(REGEX MATCH "compiler=([-A-Za-z0-9_ ]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER)
        set(${CONAN_INFO_COMPILER} ${_CONAN_INFO_COMPILER} PARENT_SCOPE)
    endif()

    string(REGEX MATCH "compiler.version=([-A-Za-z0-9_.]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER_VERSION)
        set(${CONAN_INFO_COMPILER_VERSION} ${_CONAN_INFO_COMPILER_VERSION} PARENT_SCOPE)
    endif()
endfunction()


function(check_compiler_version)
    conan_split_version(${CMAKE_CXX_COMPILER_VERSION} VERSION_MAJOR VERSION_MINOR)
    if(DEFINED CONAN_SETTINGS_COMPILER_TOOLSET)
       conan_message(STATUS "Conan: Skipping compiler check: Declared 'compiler.toolset'")
       return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        # MSVC_VERSION is defined since 2.8.2 at least
        # https://cmake.org/cmake/help/v2.8.2/cmake.html#variable:MSVC_VERSION
        # https://cmake.org/cmake/help/v3.14/variable/MSVC_VERSION.html
        if(
            # 1930 = VS 17.0 (v143 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "17" AND NOT MSVC_VERSION EQUAL 1930) OR
            # 1920-1929 = VS 16.0 (v142 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "16" AND NOT((MSVC_VERSION GREATER 1919) AND (MSVC_VERSION LESS 1930))) OR
            # 1910-1919 = VS 15.0 (v141 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "15" AND NOT((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS 1920))) OR
            # 1900      = VS 14.0 (v140 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "14" AND NOT(MSVC_VERSION EQUAL 1900)) OR
            # 1800      = VS 12.0 (v120 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "12" AND NOT VERSION_MAJOR STREQUAL "18") OR
            # 1700      = VS 11.0 (v110 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "11" AND NOT VERSION_MAJOR STREQUAL "17") OR
            # 1600      = VS 10.0 (v100 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "10" AND NOT VERSION_MAJOR STREQUAL "16") OR
            # 1500      = VS  9.0 (v90 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "9" AND NOT VERSION_MAJOR STREQUAL "15") OR
            # 1400      = VS  8.0 (v80 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "8" AND NOT VERSION_MAJOR STREQUAL "14") OR
            # 1310      = VS  7.1, 1300      = VS  7.0
            (CONAN_COMPILER_VERSION STREQUAL "7" AND NOT VERSION_MAJOR STREQUAL "13") OR
            # 1200      = VS  6.0
            (CONAN_COMPILER_VERSION STREQUAL "6" AND NOT VERSION_MAJOR STREQUAL "12") )
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "gcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 5.0)
            conan_message(STATUS "Conan: Compiler GCC>=5, checking major version ${CONAN_COMPILER_VERSION}")
            conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "clang")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 8.0)
            conan_message(STATUS "Conan: Compiler Clang>=8, checking major version ${CONAN_COMPILER_VERSION}")
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "apple-clang" OR CONAN_COMPILER STREQUAL "sun-cc" OR CONAN_COMPILER STREQUAL "mcst-lcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
           conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "intel")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 19.1)
            if(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
               conan_error_compiler_version()
            endif()
        else()
            if(NOT ${VERSION_MAJOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR})
               conan_error_compiler_version()
            endif()
        endif()
    else()
        conan_message(STATUS "WARN: Unknown compiler '${CONAN_COMPILER}', skipping the version check...")
    endif()
endfunction()


function(conan_check_compiler)
    if(CONAN_DISABLE_CHECK_COMPILER)
        conan_message(STATUS "WARN: Disabled conan compiler checks")
        return()
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER_ID)
        if(DEFINED CMAKE_C_COMPILER_ID)
            conan_message(STATUS "This project seems to be plain C, using '${CMAKE_C_COMPILER_ID}' compiler")
            set(CMAKE_CXX_COMPILER_ID ${CMAKE_C_COMPILER_ID})
            set(CMAKE_CXX_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
        else()
            message(FATAL_ERROR "This project seems to be plain C, but no compiler defined")
        endif()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID)
        # This use case happens when compiler is not identified by CMake, but the compilers are there and work
        conan_message(STATUS "*** WARN: CMake was not able to identify a C or C++ compiler ***")
        conan_message(STATUS "*** WARN: Disabling compiler checks. Please make sure your settings match your environment ***")
        return()
    endif()
    if(NOT DEFINED CONAN_COMPILER)
        conan_get_compiler(CONAN_COMPILER CONAN_COMPILER_VERSION)
        if(NOT DEFINED CONAN_COMPILER)
            conan_message(STATUS "WARN: CONAN_COMPILER variable not set, please make sure yourself that "
                          "your compiler and version matches your declared settings")
            return()
        endif()
    endif()

    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL ${CMAKE_SYSTEM_NAME})
        set(CROSS_BUILDING 1)
    endif()

    # If using VS, verify toolset
    if (CONAN_COMPILER STREQUAL "Visual Studio")
        if (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "LLVM" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "llvm" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "clang" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Clang")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Clang")
        elseif (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Intel")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Intel")
        else()
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "MSVC")
        endif()

        if (NOT CMAKE_CXX_COMPILER_ID MATCHES ${EXPECTED_CMAKE_CXX_COMPILER_ID})
            message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}'. Toolset specifies compiler as '${EXPECTED_CMAKE_CXX_COMPILER_ID}' "
                                "but CMake detected '${CMAKE_CXX_COMPILER_ID}'")
        endif()

    # Avoid checks when cross compiling, apple-clang crashes because its APPLE but not apple-clang
    # Actually CMake is detecting "clang" when you are using apple-clang, only if CMP0025 is set to NEW will detect apple-clang
    elseif((CONAN_COMPILER STREQUAL "gcc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR
        (CONAN_COMPILER STREQUAL "apple-clang" AND NOT CROSS_BUILDING AND (NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")) OR
        (CONAN_COMPILER STREQUAL "clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR
        (CONAN_COMPILER STREQUAL "sun-cc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "SunPro") )
        message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}', is not the one detected by CMake: '${CMAKE_CXX_COMPILER_ID}'")
    endif()


    if(NOT DEFINED CONAN_COMPILER_VERSION)
        conan_message(STATUS "WARN: CONAN_COMPILER_VERSION variable not set, please make sure yourself "
                             "that your compiler version matches your declared settings")
        return()
    endif()
    check_compiler_version()
endfunction()


macro(conan_set_flags build_type)
    set(CMAKE_CXX_FLAGS${build_type} "${CMAKE_CXX_FLAGS${build_type}} ${CONAN_CXX_FLAGS${build_type}}")
    set(CMAKE_C_FLAGS${build_type} "${CMAKE_C_FLAGS${build_type}} ${CONAN_C_FLAGS${build_type}}")
    set(CMAKE_SHARED_LINKER_FLAGS${build_type} "${CMAKE_SHARED_LINKER_FLAGS${build_type}} ${CONAN_SHARED_LINKER_FLAGS${build_type}}")
    set(CMAKE_EXE_LINKER_FLAGS${build_type} "${CMAKE_EXE_LINKER_FLAGS${build_type}} ${CONAN_EXE_LINKER_FLAGS${build_type}}")
endmacro()


macro(conan_global_flags)
    if(CONAN_SYSTEM_INCLUDES)
        include_directories(SYSTEM ${CONAN_INCLUDE_DIRS}
                                   "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                                   "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                                   "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                                   "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    else()
        include_directories(${CONAN_INCLUDE_DIRS}
                            "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                            "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                            "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                            "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    endif()

    link_directories(${CONAN_LIB_DIRS})

    conan_find_libraries_abs_path("${CONAN_LIBS_DEBUG}" "${CONAN_LIB_DIRS_DEBUG}"
                                  CONAN_LIBS_DEBUG)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELEASE}" "${CONAN_LIB_DIRS_RELEASE}"
                                  CONAN_LIBS_RELEASE)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_RELWITHDEBINFO}"
                                  CONAN_LIBS_RELWITHDEBINFO)
    conan_find_libraries_abs_path("${CONAN_LIBS_MINSIZEREL}" "${CONAN_LIB_DIRS_MINSIZEREL}"
                                  CONAN_LIBS_MINSIZEREL)

    add_compile_options(${CONAN_DEFINES}
                        "$<$<CONFIG:Debug>:${CONAN_DEFINES_DEBUG}>"
                        "$<$<CONFIG:Release>:${CONAN_DEFINES_RELEASE}>"
                        "$<$<CONFIG:RelWithDebInfo>:${CONAN_DEFINES_RELWITHDEBINFO}>"
                        "$<$<CONFIG:MinSizeRel>:${CONAN_DEFINES_MINSIZEREL}>")

    conan_set_flags("")
    conan_set_flags("_RELEASE")
    conan_set_flags("_DEBUG")

endmacro()


macro(conan_target_link_libraries target)
    if(CONAN_TARGETS)
        target_link_libraries(${target} ${CONAN_TARGETS})
    else()
        target_link_libraries(${target} ${CONAN_LIBS})
        foreach(_LIB ${CONAN_LIBS_RELEASE})
            target_link_libraries(${target} optimized ${_LIB})
        endforeach()
        foreach(_LIB ${CONAN_LIBS_DEBUG})
            target_link_libraries(${target} debug ${_LIB})
        endforeach()
    endif()
endmacro()


macro(conan_include_build_modules)
    if(CMAKE_BUILD_TYPE)
        if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_DEBUG} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELEASE} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELWITHDEBINFO} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_MINSIZEREL} ${CONAN_BUILD_MODULES_PATHS})
        endif()
    endif()

    foreach(_BUILD_MODULE_PATH ${CONAN_BUILD_MODULES_PATHS})
        include(${_BUILD_MODULE_PATH})
    endforeach()
endmacro()


### Definition of user declared vars (user_info) ###

set(CONAN_USER_BOOST_stacktrace_addr2line_available "True")