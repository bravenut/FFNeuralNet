# file(GLOB_RECURSE network_sources ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
set(network_bin ${PROJECT_NAME})
add_executable(${network_bin} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(${network_bin} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_compile_definitions(${network_bin} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
