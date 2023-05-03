set(DEPS lbfgsb_solver solver boost_filesystem clipper tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen igl::triangle igl::png mkl_intel_lp64 mkl_sequential mkl_core cholmod_solver gmp mpfr)


file(GLOB HEADERS "include/*.h" "autodiff/*.h"  )
file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")

add_executable(NeuralElement ${HEADERS} ${SOURCES})
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../gmsh/build/libgmsh.so)  
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(NeuralElement ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)

# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../pytorch/)
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../pytorch/build_libtorch/build/lib/)
# find_package(Torch REQUIRED)
# target_link_libraries(NeuralElement torch)

target_include_directories(NeuralElement PUBLIC ../../Libs/lbfgsb-Jonas)
target_include_directories(NeuralElement PUBLIC ../../Libs/spectra/include)
target_include_directories(NeuralElement PUBLIC ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/include)
target_include_directories(NeuralElement PUBLIC ${CMAKE_SOURCE_DIR}/../gmsh/api)  
target_link_libraries(NeuralElement ${DEPS})
