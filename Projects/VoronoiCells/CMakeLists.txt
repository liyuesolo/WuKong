set(DEPS lbfgsb_solver solver boost_filesystem tbb 
    tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core 
    igl::opengl_glfw igl::opengl_glfw_imgui 
    igl::tetgen igl::triangle igl::png mkl_intel_lp64 
    mkl_sequential mkl_core cholmod_solver gmp mpfr)



file(GLOB HEADERS "include/*.h")
file(GLOB SOURCES "src/*.cpp")


# == Deps
add_subdirectory(deps/geometry-central)

add_executable(VoronoiCells ${HEADERS} ${SOURCES})

target_link_libraries(VoronoiCells ${DEPS} geometry-central)