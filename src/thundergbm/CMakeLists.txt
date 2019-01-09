#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC util/*.c* updater/*.c* objective/*.c* *.c*)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/fasttree_train.cpp")

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
cuda_add_executable(fasttree-train fasttree_train.cpp)
target_link_libraries(fasttree-train ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY})
