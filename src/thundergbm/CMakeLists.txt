set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC util/*.c* builder/*.c* objective/*.c* metric/*.c* *.c*)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/thundergbm_train.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/thundergbm_predict.cpp")

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY})

cuda_add_executable(${PROJECT_NAME}-train thundergbm_train.cpp)
target_link_libraries(${PROJECT_NAME}-train ${PROJECT_NAME})

cuda_add_executable(${PROJECT_NAME}-predict thundergbm_predict.cpp)
target_link_libraries(${PROJECT_NAME}-predict ${PROJECT_NAME})

