cmake_minimum_required(VERSION 2.8)
set(name mtcnn)
project(${name} C CXX)

list(APPEND CMAKE_CXX_FLAGS -std=c++11)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../opencv/include
)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../opencv/${SUB_FLOBER}/lib)
list(APPEND MTCNN_LINKER_LIBS opencv_world342)

file(GLOB MTCNN_HDR ${CMAKE_CURRENT_LIST_DIR}/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cc
                    ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_executable(${name} ${MTCNN_SRC} ${MTCNN_HDR})
target_link_libraries(${name} ${MTCNN_LINKER_LIBS})

# add_library(${name} SHARED ${MTCNN_SRC} ${MTCNN_HDR})