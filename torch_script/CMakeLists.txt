cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_script)

set(CMAKE_BUILD_TYPE Debug)

add_executable(
    torch_script
    main.cpp
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Torch Version: ${Torch_VERSION}")

target_link_libraries(torch_script /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgcodecs.so /usr/local/lib/libopencv_imgproc.so)
target_link_libraries(torch_script /mnt/c/Users/Vishnu/Documents/libtorch/lib/libtorch_cpu.so /mnt/c/Users/Vishnu/Documents/libtorch/lib/libtorch.so /mnt/c/Users/Vishnu/Documents/libtorch/lib/libc10.so)
include_directories(/usr/local/include/opencv4)
include_directories(/mnt/c/Users/Vishnu/Documents/libtorch/include)
