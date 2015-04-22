#=============================================================================
# CMake configuration file for Vehicle Fording Project
#=============================================================================

cmake_minimum_required(VERSION 2.8)
project(Fording_Simple)
include(cmake/SBELUtils.cmake)
# Policy CMP0046 warning when a dependency does not exist
# According to a dev from kitware The add_dependencies
# command is for inter-target ordering dependencies, not
# file-level dependencies.
# This warning appears in Cmake 3.0
IF(CMAKE_MAJOR_VERSION EQUAL 3)
    IF(COMMAND cmake_policy)
        cmake_policy(SET CMP0046 OLD)
    ENDIF(COMMAND cmake_policy)
ENDIF(CMAKE_MAJOR_VERSION EQUAL 3)
IF(APPLE)
    set(CMAKE_MACOSX_RPATH OFF)
ENDIF()

# Location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
# Specify output locations for libraries and executables
# Set the build type depending on the platform
# ------------------------------------------------------------------------------
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------------
# Find the ChronoEngine library
# ------------------------------------------------------------------------------


FIND_PACKAGE(ChronoEngine)

IF(CHRONOENGINE_LIBRARY_DEBUG)
  MESSAGE(STATUS "ChronoEngine library (debug) found")
ENDIF()
IF(CHRONOENGINE_LIBRARY_RELEASE)
  MESSAGE(STATUS "ChronoEngine library (release) found")
ENDIF()

# Stop now if the path to the Chrono SDK is invalid or if we could not find
# at least one of the debug or release Chrono libraries.
IF(NOT EXISTS "${CH_CHRONO_SDKDIR}/physics/ChSystem.h")
  MESSAGE(FATAL_ERROR "Cannot find the Chrono SDK. Make sure CH_CHRONO_SDKDIR is set correctly.")
  RETURN()
ENDIF()
IF(NOT CHRONOENGINE_LIBRARY_DEBUG AND NOT CHRONOENGINE_LIBRARY_RELEASE)
  MESSAGE(FATAL_ERROR "Cannot find the ChronoEngine library. Check CH_LIBDIR_DEBUG and CH_LIBDIR_RELEASE.")
  RETURN()
ENDIF()

# Add paths to Chrono headers
INCLUDE_DIRECTORIES(${CHRONOENGINE_INCLUDES})

find_package(ZLIB)
# ------------------------------------------------------------------------------
# Find the ChronoParallel library
# ------------------------------------------------------------------------------

FIND_PACKAGE(ChronoParallel COMPONENTS utils)

IF(CHRONOPARALLEL_LIBRARY_DEBUG)
  MESSAGE(STATUS "ChronoParallel library (debug) found")
ENDIF()
IF(CHRONOPARALLEL_LIBRARY_RELEASE)
  MESSAGE(STATUS "ChronoParallel library (release) found")
ENDIF()

# Stop now if the path to the Chrono SDK is invalid or if we could not find
# at least one of the debug or release Chrono libraries.
IF(NOT EXISTS "${CH_PARALLEL_SDKDIR}/chrono_parallel/physics/ChSystemParallel.h")
  MESSAGE(FATAL_ERROR "Cannot find the Chrono Parallel SDK. Make sure CH_PARALLEL_SDKDIR is set correctly.")
  RETURN()
ENDIF()
IF(NOT CHRONOPARALLEL_LIBRARY_DEBUG AND NOT CHRONOPARALLEL_LIBRARY_RELEASE)
  MESSAGE(FATAL_ERROR "Cannot find the ChronoEngine library. Check CHRONOPARALLEL_LIBRARY_DEBUG and CHRONOPARALLEL_LIBRARY_RELEASE.")
  RETURN()
ENDIF()

# Add paths to Chrono headers
INCLUDE_DIRECTORIES(${CHRONOENGINE_INCLUDES})

#--------------------------------------------------------------
# Find the ChronoVehicle library

FIND_PACKAGE(ChronoVehicle)

# Stop now if we could not find the ChronoVehicle SDK
IF(NOT EXISTS "${CH_VEHICLE_SDKDIR}/subsys/ChVehicle.h")
  MESSAGE(FATAL_ERROR "Cannot find the ChronoVehicle SDK. Make sure CH_VEHICLE_SDKDIR is set correctly.")
  RETURN()
ENDIF()

# Check debug/release ChronoVehicle libraries. Stop now if neither could be found.
IF(CHRONOVEHICLE_LIBRARY_DEBUG)
  MESSAGE(STATUS "ChronoVehicle library (debug) found")
ENDIF()
IF(CHRONOVEHICLE_LIBRARY_RELEASE)
  MESSAGE(STATUS "ChronoVehicle library (release) found")
ENDIF()

IF(NOT CHRONOVEHICLE_LIBRARY_DEBUG AND NOT CHRONOVEHICLE_LIBRARY_RELEASE)
  MESSAGE(FATAL_ERROR "Cannot find the ChronoVehicle library. Check CH_VEHICLE_LIBDIR_DEBUG and CH_VEHICLE_LIBDIR_RELEASE.")
  RETURN()
ENDIF()

MESSAGE(STATUS "ChronoVehicle includes:  ${CHRONOVEHICLE_INCLUDES}")
MESSAGE(STATUS "ChronoVehicle libraries: ${CHRONOVEHICLE_LIBRARIES}")

# ------------------------------------------------------------------------------
# Dependencies for ChronoEngine_OpenGL
# ------------------------------------------------------------------------------

SET(BLAZE_DIR ""  CACHE PATH   "Where is Blaze located?")

OPTION(ENABLE_OPENGL false)

IF (ENABLE_OPENGL)
    find_package(OpenGL)
    find_package(GLM)
    find_package(GLEW)
    find_package(GLFW)

    # On windows, ask for the GLEW and GLFW DLLs so that we can copy. This is
    # optional.  If not specified, it is the user's responsibility to make them
    # available at runtime.
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        SET(GLEW_DLL  "" CACHE FILEPATH "The GLEW DLL")
        SET(GLFW_DLL  "" CACHE FILEPATH "The GLFW DLL")
    ENDIF()

    # If all dependencies are available, enable building OpenGL support.
    IF(OPENGL_FOUND AND GLM_FOUND AND GLEW_FOUND AND GLFW_FOUND)
        SET(CHRONO_PARALLEL_HAS_OPENGL "#define CHRONO_PARALLEL_HAS_OPENGL")
        SET(ENABLE_OPENGL ON)
    ELSE()
        # Unable to find everything, turn off OpenGL
        MESSAGE("Cannot build ChronoEngine_OpenGL (missing dependencies).")
        SET(CHRONO_PARALLEL_HAS_OPENGL "")
        SET(ENABLE_OPENGL OFF)
    ENDIF()
ENDIF()
# ------------------------------------------------------------------------------
# Find CUDA
# ------------------------------------------------------------------------------
#enable_cuda_support()
# ------------------------------------------------------------------------------
# Enable OpenMP
# ------------------------------------------------------------------------------
enable_openmp_support()

# ------------------------------------------------------------------------------
# Thrust
# ------------------------------------------------------------------------------
find_package(Thrust)

#Without the following define, thrust needs the cuda sdk to compile
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

#--------------------------------------------------------------
# Generate the configuration file 

SET(CHRONOVEHICLE_DATA_DIR "data/")

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_BINARY_DIR}/demos_vehicle/config.h
  )

MESSAGE(STATUS "PRJ bin dir: ${PROJECT_BINARY_DIR}")
MESSAGE(STATUS "CMK bin dir: ${CMAKE_BINARY_DIR}")


#--------------------------------------------------------------
# Set the include paths and libraries

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${CHRONOENGINE_INCLUDES}
    ${CHRONOPARALLEL_INCLUDES}
    ${CHRONOVEHICLE_INCLUDES}
    "${CH_PARALLEL_LIBDIR_RELEASE}/../" 
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_BINARY_DIR}/demos_vehicle/"
    #${GLFW_INCLUDE_DIR}
    ${THRUST_INCLUDE_DIR}
    ${BLAZE_DIR}
    )

SET(LIBRARIES
    ${CHRONOENGINE_LIBRARY}
    ${CHRONOPARALLEL_LIBRARY}
    ${CHRONOPARALLEL_UTILS_LIBRARY}
#    ${CHRONOPARALLEL_OPENGL_LIBRARY}
    ${CHRONOVEHICLE_LIBRARIES}
    )

IF (ENABLE_OPENGL)
    INCLUDE_DIRECTORIES(
        ${GLM_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIR}
        ${GLFW_INCLUDE_DIR}
        )

    SET(LIBRARIES ${LIBRARIES} ${CHRONOPARALLEL_OPENGL_LIBRARY} ${ZLIB_LIBRARY})
ENDIF()

# ------------------------------------------------------------------------------
# Automatically copy DLLs 
IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    # HACK?
    IF(CH_VEHICLE_LIBDIR_DEBUG)
        SET(CH_VEHICLE_BINDIR "${CH_VEHICLE_LIBDIR_DEBUG}/../../bin")
    ENDIF()
    IF(CH_VEHICLE_LIBDIR_RELEASE)
        SET(CH_VEHICLE_BINDIR "${CH_VEHICLE_LIBDIR_RELEASE}/../../bin")
    ENDIF()

    # Create custom target for copying DLLs; add it to the default build target
    ADD_CUSTOM_TARGET(COPY_VEHICLE_DLLS ALL)

    # Create custom command, invoked post-build to copy DLLs to the appropriate
    # directory (depending on the configuration selected at build time in VS)
    ADD_CUSTOM_COMMAND(
        TARGET COPY_VEHICLE_DLLS POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CH_VEHICLE_BINDIR}/$<CONFIGURATION>/ChronoVehicle.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIGURATION>"
        )
ENDIF()

#--------------------------------------------------------------
# Always use full RPATH (differentiating between the build and install trees)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

#--------------------------------------------------------------

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	SET (CH_LINKERFLAG_EXE  "${CH_LINKERFLAG_EXE} -framework IOKit -framework Cocoa -framework OpenGL")
ENDIF()

#--------------------------------------------------------------
# List of all executables

SET(DEMOS
    #hmmwv
    hmmwv_simple
    )

MESSAGE(STATUS "Vehicle fording programs...")

FOREACH(PROGRAM ${DEMOS})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_BUILDFLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
        )

    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})

    INSTALL(TARGETS ${PROGRAM} DESTINATION bin)

ENDFOREACH(PROGRAM)

