add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DMKL_LP64)
set(DEPS lbfgsb_solver solver boost_filesystem clipper tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen igl::triangle igl::png mkl_intel_lp64 mkl_sequential mkl_core cholmod_solver)

# file(GLOB HEADERS "include/*.h" "include/autodiff/*.h" "tactile/.*h" "tactile/.*hpp" )
file(GLOB SOURCES "cpp/*.cpp")

add_executable(NeuralConstitutiveModel ${HEADERS} ${SOURCES})

target_link_libraries(NeuralConstitutiveModel ${DEPS})
