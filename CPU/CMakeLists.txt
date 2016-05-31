cmake_minimum_required(VERSION 2.8)
project(ThesisCode)
find_package( OpenCV REQUIRED )
add_executable(OPENCV_HOG_CPU OPENCV_HOG_CPU.cpp)
target_link_libraries(OPENCV_HOG_CPU ${OpenCV_LIBS} )
add_executable(OPENCV_HOG_GPU OPENCV_HOG_GPU.cpp)
target_link_libraries(OPENCV_HOG_GPU ${OpenCV_LIBS} )

