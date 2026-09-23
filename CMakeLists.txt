cmake_minimum_required(VERSION 3.16)
project(igaos VERSION 0.2.0 LANGUAGES C CXX
        DESCRIPTION "Indigenous GPU-Accelerated Optimization Solver")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

option(IGAOS_BUILD_SHARED  "Build the shared library carrying the C ABI" ON)
option(IGAOS_BUILD_TESTS   "Build the regression suite" ON)
option(IGAOS_ENABLE_CUDA   "Compile the GPU kernels (UNTESTED -- see cuda/pdhg_kernels.cu)" OFF)
option(IGAOS_WARNINGS      "Build with the full warning set" ON)
option(IGAOS_HARDENING     "Compile and link with exploit-mitigation flags" ON)

# ---------------------------------------------------------------------------
# HARDENING
#
# This solver reads MPS files it did not write, and is meant to be embedded in
# planning systems that run inside a refinery's network.  The mitigations below
# are the ones that cost nothing measurable at -O2 and turn a memory-safety bug
# from an exploit into a crash.  They are not a substitute for the bug not
# being there -- CI runs AddressSanitizer, UndefinedBehaviorSanitizer and two
# fuzzers for that -- they are the second line for the bug nobody found.
#
# Every flag is PROBED before it is used.  A compiler that does not support one
# silently does without it rather than failing to configure, because a build
# that breaks on an unfamiliar toolchain is a build nobody adopts, and adoption
# is the whole argument for a sovereign solver.
#
# Turn the whole block off with -DIGAOS_HARDENING=OFF when measuring, so a
# benchmark number is never quietly a benchmark of the mitigations.
# ---------------------------------------------------------------------------
if(IGAOS_HARDENING)
  include(CheckCXXSourceCompiles)
  include(CheckCXXCompilerFlag)

  function(igaos_try_compile_flag flag out)
    string(MAKE_C_IDENTIFIER "IGAOS_CF_${flag}" cache_var)
    check_cxx_compiler_flag("${flag}" ${cache_var})
    if(${cache_var})
      set(_acc ${${out}})
      list(APPEND _acc "${flag}")
      set(${out} "${_acc}" PARENT_SCOPE)
    endif()
  endfunction()

  function(igaos_try_link_flag flag out)
    string(MAKE_C_IDENTIFIER "IGAOS_LF_${flag}" cache_var)
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_cxx_source_compiles("int main(){return 0;}" ${cache_var})
    if(${cache_var})
      set(_acc ${${out}})
      list(APPEND _acc "${flag}")
      set(${out} "${_acc}" PARENT_SCOPE)
    endif()
  endfunction()

  set(IGAOS_HARDEN_COMPILE "")
  set(IGAOS_HARDEN_LINK "")

  if(MSVC)
    # /GS is on by default; /guard:cf and /Qspectre are not.
    igaos_try_compile_flag("/guard:cf" IGAOS_HARDEN_COMPILE)
    igaos_try_compile_flag("/Qspectre" IGAOS_HARDEN_COMPILE)
    igaos_try_compile_flag("/sdl"      IGAOS_HARDEN_COMPILE)
    list(APPEND IGAOS_HARDEN_LINK "/guard:cf" "/DYNAMICBASE" "/NXCOMPAT" "/HIGHENTROPYVA")
  else()
    # Stack and control-flow integrity.
    igaos_try_compile_flag("-fstack-protector-strong"  IGAOS_HARDEN_COMPILE)
    igaos_try_compile_flag("-fstack-clash-protection"  IGAOS_HARDEN_COMPILE)
    igaos_try_compile_flag("-fcf-protection=full"      IGAOS_HARDEN_COMPILE)   # x86
    igaos_try_compile_flag("-mbranch-protection=standard" IGAOS_HARDEN_COMPILE) # aarch64
    igaos_try_compile_flag("-Wformat"                  IGAOS_HARDEN_COMPILE)
    igaos_try_compile_flag("-Wformat-security"         IGAOS_HARDEN_COMPILE)
    # Padding and locals initialised rather than left holding whatever was
    # there.  Cheap here because the hot loops allocate once, not per call.
    igaos_try_compile_flag("-ftrivial-auto-var-init=zero" IGAOS_HARDEN_COMPILE)

    # _FORTIFY_SOURCE needs optimisation to do anything and actively fights
    # AddressSanitizer, which reports its interposed calls as false positives.
    # So: optimised builds only, and never when a sanitizer is in the flags.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_CXX_FLAGS MATCHES "sanitize")
      list(APPEND IGAOS_HARDEN_COMPILE "-U_FORTIFY_SOURCE" "-D_FORTIFY_SOURCE=2")
    endif()

    # Linker mitigations.  The -z family is GNU ld / lld only -- Apple's ld
    # rejects them -- so each is probed rather than assumed.
    igaos_try_link_flag("-Wl,-z,relro"         IGAOS_HARDEN_LINK)
    igaos_try_link_flag("-Wl,-z,now"           IGAOS_HARDEN_LINK)
    igaos_try_link_flag("-Wl,-z,noexecstack"   IGAOS_HARDEN_LINK)
    igaos_try_link_flag("-Wl,-z,separate-code" IGAOS_HARDEN_LINK)
  endif()

  if(IGAOS_HARDEN_COMPILE)
    add_compile_options(${IGAOS_HARDEN_COMPILE})
  endif()
  if(IGAOS_HARDEN_LINK)
    add_link_options(${IGAOS_HARDEN_LINK})
  endif()
  # Position-independent executables.  CMAKE_POSITION_INDEPENDENT_CODE above
  # covers the objects; this is what makes the CLI itself relocatable so ASLR
  # applies to it and not only to the libraries it loads.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  if(NOT MSVC AND NOT APPLE)
    include(CheckPIESupported OPTIONAL RESULT_VARIABLE IGAOS_HAVE_PIE_MODULE)
    if(IGAOS_HAVE_PIE_MODULE)
      check_pie_supported(OUTPUT_VARIABLE IGAOS_PIE_WHY LANGUAGES CXX)
      if(CMAKE_CXX_LINK_PIE_SUPPORTED)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
      endif()
    endif()
  endif()

  message(STATUS "igaos hardening: compile ${IGAOS_HARDEN_COMPILE}")
  message(STATUS "igaos hardening: link    ${IGAOS_HARDEN_LINK}")
endif()

find_package(OpenMP)

set(IGAOS_SOURCES
  src/sparse.cpp
  src/model.cpp
  src/mps.cpp
  src/lu.cpp
  src/ldl.cpp
  src/simplex.cpp
  src/presolve.cpp
  src/cuts.cpp
  src/ipm.cpp
  src/miqp.cpp
  src/global.cpp
  src/expr.cpp
  src/nlp.cpp
  src/crossover.cpp
  src/pdhg.cpp
  src/solver.cpp
  src/certificate.cpp
  src/iis.cpp
  src/sensitivity.cpp
  src/capi.cpp)

# ---------------------------------------------------------------------------
# Compiled once, linked into both the static and the shared library.
# ---------------------------------------------------------------------------
add_library(igaos_objects OBJECT ${IGAOS_SOURCES})
target_include_directories(igaos_objects PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(igaos_objects PUBLIC OpenMP::OpenMP_CXX)
endif()
if(IGAOS_WARNINGS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  target_compile_options(igaos_objects PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_library(igaos_core STATIC $<TARGET_OBJECTS:igaos_objects>)
target_include_directories(igaos_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(igaos_core PUBLIC OpenMP::OpenMP_CXX)
endif()
add_library(igaos::core ALIAS igaos_core)

if(IGAOS_BUILD_SHARED)
  add_library(igaos_shared SHARED $<TARGET_OBJECTS:igaos_objects>)
  set_target_properties(igaos_shared PROPERTIES
    OUTPUT_NAME igaos
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  target_include_directories(igaos_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(igaos_shared PUBLIC OpenMP::OpenMP_CXX)
  endif()
  add_library(igaos::shared ALIAS igaos_shared)
endif()

# ---------------------------------------------------------------------------
# The GPU kernels are OFF by default and stay that way until someone with a
# device has actually run them.  See the header comment in the .cu file.
# ---------------------------------------------------------------------------
if(IGAOS_ENABLE_CUDA)
  enable_language(CUDA)
  message(WARNING
    "IGAOS_ENABLE_CUDA is ON. cuda/pdhg_kernels.cu has never been compiled or "
    "executed -- validate it against the CPU path before trusting any result.")
  add_library(igaos_cuda STATIC cuda/pdhg_kernels.cu)
  set_target_properties(igaos_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(igaos_cuda PUBLIC include)
endif()

# ---------------------------------------------------------------------------
add_executable(igaos_cli apps/igaos_cli.cpp)
set_target_properties(igaos_cli PROPERTIES OUTPUT_NAME igaos)
target_link_libraries(igaos_cli PRIVATE igaos_core)

# ---------------------------------------------------------------------------
if(IGAOS_BUILD_TESTS)
  enable_testing()
  add_executable(igaos_tests tests/test_all.cpp)
  target_link_libraries(igaos_tests PRIVATE igaos_core)
  add_test(NAME unit COMMAND igaos_tests)

  # Compiled as C and linked against the shared library, so the test exercises
  # the ABI the way a real consumer would rather than re-including the C++.
  add_executable(igaos_capi_test tests/test_capi.c)
  if(IGAOS_BUILD_SHARED)
    target_link_libraries(igaos_capi_test PRIVATE igaos_shared)
  else()
    target_link_libraries(igaos_capi_test PRIVATE igaos_core)
    set_target_properties(igaos_capi_test PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  # A C translation unit brings no implicit libm, and the test does its own
  # arithmetic on the results.
  find_library(IGAOS_LIBM m)
  if(IGAOS_LIBM)
    target_link_libraries(igaos_capi_test PRIVATE ${IGAOS_LIBM})
  endif()
  add_test(NAME c_abi COMMAND igaos_capi_test)

  # The CUDA kernels, executed on the CPU.  cuda/emulate.hpp defines the CUDA
  # built-ins in ordinary C++ so the kernel bodies in cuda/pdhg_kernels.cu --
  # the actual text, not a re-implementation -- can be run and compared against
  # src/pdhg.cpp and against the simplex.  This needs no GPU and no CUDA
  # toolkit, so it belongs in the default test run: it is the only thing in the
  # project that checks what the device path COMPUTES, and the first time it
  # ran it found defect 23.
  add_executable(igaos_cuda_emulate tools/cuda_emulate.cpp)
  target_include_directories(igaos_cuda_emulate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  target_link_libraries(igaos_cuda_emulate PRIVATE igaos_core)
  add_test(NAME cuda_kernels COMMAND igaos_cuda_emulate)

  # The nonlinear layer: exact derivatives against central differences, the
  # Hock-Schittkowski collection against its published optima, and a MINLP
  # against exhaustive enumeration.
  add_executable(igaos_nlp_check tools/nlp_check.cpp)
  target_link_libraries(igaos_nlp_check PRIVATE igaos_core)
  add_test(NAME nlp COMMAND igaos_nlp_check)

  # The Haverly pooling problems against their published global optima -- the
  # standard counterexample that a linearised blending model gets wrong.
  add_executable(igaos_pooling_check tools/pooling_check.cpp)
  target_link_libraries(igaos_pooling_check PRIVATE igaos_core)
  add_test(NAME pooling COMMAND igaos_pooling_check)

  # Not a ctest: it needs generated instances and forty local solves each.
  # See bench/results_nlp.txt for the recorded run.
  add_executable(igaos_pooling_bench tools/pooling_bench.cpp)
  target_link_libraries(igaos_pooling_bench PRIVATE igaos_core)

  # The MPS reader against hostile input.  This IS a ctest, deliberately: it is
  # the only code in the project that parses bytes nobody here produced, so the
  # check belongs in the run everybody does rather than in the run somebody
  # remembers.  It is also the cheapest way to get the sanitizers pointed at the
  # parser -- the sanitizer CI job runs ctest, so this target gets ASan and UBSan
  # over several thousand malformed files for free.
  #
  # Seeded, so a failure in CI replays from the two numbers in the log:
  #     ./build/igaos_fuzz_mps 4000 1
  add_executable(igaos_fuzz_mps tools/fuzz_mps.cpp)
  target_link_libraries(igaos_fuzz_mps PRIVATE igaos_core)
  add_test(NAME mps_fuzz COMMAND igaos_fuzz_mps 20000 1)
endif()

# ---------------------------------------------------------------------------
include(GNUInstallDirs)
set(IGAOS_INSTALL_TARGETS igaos_core igaos_cli)
if(IGAOS_BUILD_SHARED)
  list(APPEND IGAOS_INSTALL_TARGETS igaos_shared)
endif()
install(TARGETS ${IGAOS_INSTALL_TARGETS}
        EXPORT igaosTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/igaos DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT igaosTargets
        FILE igaosTargets.cmake
        NAMESPACE igaos::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/igaos)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/igaosConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/igaosConfig.cmake"
"include(CMakeFindDependencyMacro)\nfind_dependency(OpenMP)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/igaosTargets.cmake\")\n")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/igaosConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/igaosConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/igaos)
