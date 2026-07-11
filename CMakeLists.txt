cmake_minimum_required(VERSION 3.16)

project(person_outline_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT OpenCV_DIR)
    foreach(_opencv_dir
        /opt/homebrew/opt/opencv/lib/cmake/opencv5
        /usr/local/opt/opencv/lib/cmake/opencv5
        /opt/homebrew/opt/opencv/lib/cmake/opencv4
        /usr/local/opt/opencv/lib/cmake/opencv4)
        if(EXISTS "${_opencv_dir}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_opencv_dir}" CACHE PATH "OpenCV package directory")
            break()
        endif()
    endforeach()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn)

add_executable(person_outline
    src/main.cpp
)

target_include_directories(person_outline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(person_outline PRIVATE ${OpenCV_LIBS})

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    target_link_libraries(person_outline PRIVATE camera)
endif()

if(MSVC)
    target_compile_options(person_outline PRIVATE /W4)
else()
    target_compile_options(person_outline PRIVATE -Wall -Wextra -Wpedantic)
endif()
