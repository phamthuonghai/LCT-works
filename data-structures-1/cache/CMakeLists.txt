cmake_minimum_required(VERSION 3.8)
project(cache)

set(CMAKE_C_STANDARD 99)

add_executable(cache cache.c)
add_executable(cachesim cachesim.c)

target_link_libraries(cache m)
