set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC util/*.c* updater/*.c* objective/*.c* *.c*)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/thundergbm_train.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_LIST_DIR}/thundergbm_predict.cpp")

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
cuda_add_executable(thundergbm-train thundergbm_train.cpp)
target_link_libraries(thundergbm-train ${PROJECT_NAME})
cuda_add_executable(thundergbm-predict thundergbm_predict.cpp)
target_link_libraries(thundergbm-predict ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY})
