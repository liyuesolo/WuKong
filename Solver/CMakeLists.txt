set(DEPS tbbmalloc tbbmalloc_proxy)

set(SuiteSparse_ROOT ${CMAKE_SOURCE_DIR}/../SuiteSparse-5.12.0)
include_directories(${SuiteSparse_ROOT})
  
add_definitions(-DUSE_CHOLMOD)
include_directories(${SuiteSparse_ROOT}/include)
link_directories(${SuiteSparse_ROOT}/lib)
add_library(cholmod_solver STATIC CHOLMODSolver.hpp CHOLMODSolver.cpp)
target_include_directories(cholmod_solver SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(cholmod_solver suitesparseconfig metis cholmod amd camd ccolamd colamd spqr)
set(DEPS ${DEPS} cholmod_solver)

add_library(solver STATIC MMASolver.h MMASolver.cpp)
target_link_libraries(solver ${DEPS})