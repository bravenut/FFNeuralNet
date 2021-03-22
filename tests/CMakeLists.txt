file(GLOB_RECURSE test_sources ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

set(test_exe test_${PROJECT_NAME})
add_executable(${test_exe} ${test_sources})
target_link_libraries(${test_exe}
        ${Boost_LIBRARIES})

install(TARGETS test_${PROJECT_NAME} DESTINATION bin)

add_definitions(-DFFNET_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
