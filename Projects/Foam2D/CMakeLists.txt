set(DEPS Half tbb tbbmalloc tbbmalloc_proxy igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png CGAL suitesparseconfig metis)

file(GLOB HEADERS "include/*.h" "include/*.hpp")
file(GLOB SOURCES "src/*.cpp")

add_executable(Foam2D ${HEADERS} ${SOURCES})

file(GLOB CODEGENLIBS "codegen/lib*.so")

target_link_libraries(Foam2D ${CODEGENLIBS} ${DEPS})

add_subdirectory(src/optLib)
