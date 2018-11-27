cmake_minimum_required(VERSION 3.8)

project(GameOfLife LANGUAGES CUDA C)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(SOURCE_FILES src/main.c src/game.c src/cell-data.c src/cpu-frame-computer.c src/gpu-frame-computer.cu src/cuda.cu)

add_executable(gol ${SOURCE_FILES})
target_link_libraries(gol ${SDL2_LIBRARIES})
