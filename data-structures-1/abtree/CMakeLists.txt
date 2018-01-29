cmake_minimum_required(VERSION 3.8)
project(abtree)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp abtree.cpp abtree.h)
add_executable(abtree ${SOURCE_FILES})
add_executable(abgen abgen.c)

target_link_libraries(abgen m)
