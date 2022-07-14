set(DEPS Half tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr)

file(GLOB HEADERS "include/*.h" "include/*.hpp" )
file(GLOB SOURCES "src/*.cpp" )


add_executable(Foam2D ${HEADERS} ${SOURCES})

target_link_libraries(Foam2D ${DEPS})