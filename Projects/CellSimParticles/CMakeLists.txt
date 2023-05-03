set(DEPS lbfgsb_solver cholmod_solver solver tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr igl::tetgen)

file(GLOB HEADERS "include/*.h" )
file(GLOB SOURCES "*.cpp" "src/*.cpp" "autodiff/*.cpp")


add_executable(CellSimParticles ${HEADERS} ${SOURCES})


target_link_libraries(CellSimParticles ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSimParticles ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSimParticles ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSimParticles ${CMAKE_SOURCE_DIR}/../ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSimParticles ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)


target_include_directories(CellSimParticles PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSimParticles PUBLIC ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/include)

target_include_directories(CellSimParticles PUBLIC ${CMAKE_SOURCE_DIR}/Libs/cyCodeBase)
target_link_libraries(CellSimParticles ${DEPS})
