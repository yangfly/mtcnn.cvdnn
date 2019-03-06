cmake_minimum_required(VERSION 2.8)
set(name mtcnn)
project(${name} C CXX)

list(APPEND CMAKE_CXX_FLAGS -std=c++11)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}/opencv/include
)

link_directories(${CMAKE_CURRENT_LIST_DIR}/opencv/vc15/lib)
list(APPEND MTCNN_LINKER_LIBS opencv_world342)

file(GLOB MTCNN_HDR ${CMAKE_CURRENT_LIST_DIR}/src/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)
file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
set (CONVERTER_SRC ${CMAKE_CURRENT_LIST_DIR}/src/converter.cpp)
list(REMOVE_ITEM MTCNN_SRC ${CONVERTER_SRC})

add_executable(${name} ${MTCNN_SRC} ${MTCNN_HDR})
target_link_libraries(${name} ${MTCNN_LINKER_LIBS})

add_executable(converter ${CONVERTER_SRC})
target_link_libraries(converter ${MTCNN_LINKER_LIBS})

# add_library(${name} SHARED ${MTCNN_SRC} ${MTCNN_HDR})