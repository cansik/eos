# The examples need a few additional dependencies (e.g. boost filesystem, program_options, and OpenCV highgui):

# Check installed version in order to include the correct OpenCV libraries:
# First call find_package without a version to find any OpenCV. OpenCV_VERSION_MAJOR is then defined.
find_package(OpenCV REQUIRED core)
if ("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
    message(STATUS "OpenCV 2.x detected")
    find_package(OpenCV 2.4.3 REQUIRED)
elseif ("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
    message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
    find_package(OpenCV 3 REQUIRED)
elseif ("${OpenCV_VERSION_MAJOR}$" EQUAL 4)
    message(STATUS "OpenCV 4.x detected - including imgcodecs for compatibility")
    find_package(OpenCV 4 REQUIRED)
endif ()
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if (MSVC)
    # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
    add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
    add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
    add_definitions(-D_HAS_AUTO_PTR_ETC) # Boost 1.65.1 still does not work with VS C++17 mode, this is the workaround
endif ()
find_package(Boost 1.50.0 COMPONENTS system filesystem program_options REQUIRED)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

# glfw
find_package(glfw3 3.3 REQUIRED)

# glad
include_directories(glad/include)
add_subdirectory(glad/)

# Simple model fitting (orthographic camera & shape to landmarks) example:
add_executable(fit-model-simple fit-model-simple.cpp)
target_link_libraries(fit-model-simple eos ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(fit-model-simple "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model-simple PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours:
add_executable(fit-model fit-model.cpp)
target_link_libraries(fit-model eos ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(fit-model "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Relatime Model fitting example that fits orthographic camera, shape, blendshapes, and contours:
add_executable(fit-model-realtime fit-model-realtime.cpp)
target_link_libraries(fit-model-realtime eos ${OpenCV_LIBS} ${Boost_LIBRARIES} glfw glad)
target_link_libraries(fit-model-realtime "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model-realtime PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS})

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours to multiple images:
add_executable(fit-model-multi fit-model-multi.cpp)
target_link_libraries(fit-model-multi eos ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(fit-model-multi "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model-multi PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Generate random samples from the model:
add_executable(generate-obj generate-obj.cpp)
target_link_libraries(generate-obj eos ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(generate-obj PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Install these targets:
install(TARGETS fit-model-simple DESTINATION bin)
install(TARGETS fit-model DESTINATION bin)
install(TARGETS fit-model-realtime DESTINATION bin)
install(TARGETS fit-model-multi DESTINATION bin)
install(TARGETS generate-obj DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION bin)


if (EOS_BUILD_CERES_EXAMPLE)
    # Find Ceres, for the fit-model-ceres app:
    find_package(Ceres REQUIRED)
    message(STATUS "Ceres locations: Headers: ${CERES_INCLUDE_DIRS} Library: ${CERES_LIBRARIES}")

    # Single and multi-image non-linear model fitting with Ceres example:
    add_executable(fit-model-ceres fit-model-ceres.cpp)
    target_link_libraries(fit-model-ceres eos ${CERES_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
    target_include_directories(fit-model-ceres PUBLIC ${CERES_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    install(TARGETS fit-model-ceres DESTINATION bin)
endif ()
