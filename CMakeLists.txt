cmake_minimum_required(VERSION 3.13.0)

project(FFNet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -std=c++2a")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ggdb -g3")

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

#--------------------------------------
# Libraries
#
# use armadillo without runtimelibrary, wrap blas and lapack
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/external/armadillo-10.3.0/include")

# include simple mnist reader
find_package(MNIST PATHS external/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

#--------------------------------------
include_directories(${MNIST_INCLUDE_DIR})
add_subdirectory(src)
add_subdirectory(tests)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_DONT_USE_WRAPPER -lblas -llapack")


# Pass MNIST data directory to network.cpp
