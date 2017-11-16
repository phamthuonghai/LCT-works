cmake_minimum_required(VERSION 3.8)
project(fibheap)

set(CMAKE_CXX_STANDARD 11)

#set(SOURCE_FILES main.cpp)
#add_executable(fibheap ${SOURCE_FILES})
add_executable(fibheap fibheap.cpp)
add_executable(heapgen heapgen.cpp)
