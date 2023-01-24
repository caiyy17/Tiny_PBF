cmake_minimum_required(VERSION 3.16)
project(Tiny_PBF)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/config.h")
option(USE_OPENCV "USE_OPENCV" ON)

# import OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

# import OpenCV
if(USE_OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_executable(${PROJECT_NAME} main.cpp)

if(USE_OPENCV)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()