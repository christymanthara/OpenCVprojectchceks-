cmake_minimum_required(VERSION 3.10)
project(BilliardAnalysis)

# Set the C++ standard 17 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Headers
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Source files
add_executable(BilliardAnalysis ${SOURCE_FILES})

# Link OpenCV libraries
target_link_libraries(BilliardAnalysis ${OpenCV_LIBS})
