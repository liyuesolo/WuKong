add_definitions(-DCGAL_DISABLE_ROUNDING_MATH_CHECK)
set(DEPS Half tbb tbbmalloc tbbmalloc_proxy igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png CGAL suitesparseconfig metis)

file(GLOB HEADERS "include/*.h" "include/*.hpp" "include/Tessellation/*.h" "include/Boundary/*.h" "include/Energy/*.h" "include/ImageMatch/*.h" "include/TrajectoryOpt/*.h" "src/implot/*.h")
file(GLOB SOURCES "src/*.cpp" "src/Tessellation/*.cpp" "src/Boundary/*.cpp" "src/Energy/*.cpp" "src/ImageMatch/*.cpp" "src/TrajectoryOpt/*.cpp" "src/implot/*.cpp")

add_executable(Foam2D ${HEADERS} ${SOURCES})

file(GLOB CODEGENLIBS "deprecated/codegen/codegen_energy/lib*.so" "deprecated/codegen/codegen_imagematch/lib*.so" "deprecated/codegen/codegen_energy_areatarget/lib*.so")

find_package(IPOPT REQUIRED)
if (IPOPT_FOUND)
    message("IPOPT Found!")
    message("ipopt include dirs:" ${IPOPT_INCLUDE_DIRS})
    message("ipopt libraries:" ${IPOPT_LIBRARIES})
    add_definitions("-DUSE_IPOPT")
    include_directories(${IPOPT_INCLUDE_DIRS})
    if (IPOPT_FOUND)
        target_link_libraries(Foam2D ${IPOPT_LIBRARIES})
    endif ()
else ()
    message("IPOPT Not Found! install by \n sudo apt-get install coinor-libipopt-dev")
endif ()

find_package(MKL REQUIRED)
if (MKL_FOUND)
    message("MKL Found!")
    message("mkl include dirs:" ${MKL_INCLUDE_DIR})
    message("mkl libraries:" ${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIR})
    if (MKL_FOUND)
        target_link_libraries(Foam2D ${MKL_LIBRARIES})
    endif ()
else ()
    message("MKL Not Found!")
endif ()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Foam2D ${OpenCV_LIBS})

target_link_libraries(Foam2D ${CODEGENLIBS} ${DEPS})

add_subdirectory(src/optLib)
