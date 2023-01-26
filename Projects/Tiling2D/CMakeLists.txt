set(DEPS igl::mosek  poisson_disk lbfgsb_solver solver boost_filesystem clipper tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen igl::triangle igl::png mkl_intel_lp64 mkl_sequential mkl_core cholmod_solver gmp mpfr)



file(GLOB HEADERS "include/*.h" "include/autodiff/*.h" "tactile/.*h" "tactile/.*hpp" )
file(GLOB SOURCES "src/*.cpp" "tactile/*.cpp" "src/autodiff/*.cpp")

add_executable(Tiling2D ${HEADERS} ${SOURCES})
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../gmsh/build/libgmsh.so)  
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(Tiling2D ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../pytorch/)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../pytorch/build_libtorch/build/lib/)
find_package(Torch REQUIRED)
target_link_libraries(Tiling2D torch)

target_include_directories(Tiling2D PUBLIC ../../Libs/lbfgsb-Jonas)
target_include_directories(Tiling2D PUBLIC ../../Libs/spectra/include)
target_include_directories(Tiling2D PUBLIC ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/include)
target_include_directories(Tiling2D PUBLIC ${CMAKE_SOURCE_DIR}/../gmsh/api)  
target_link_libraries(Tiling2D ${DEPS})

target_include_directories(Tiling2D PUBLIC ${CMAKE_SOURCE_DIR}/Libs/cyCodeBase)
