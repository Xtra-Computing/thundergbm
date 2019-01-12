add_subdirectory(googletest)

include_directories(googletest/googletest/include)
include_directories(googletest/googlemock/include)

file(GLOB TEST_SRC *)

cuda_add_executable(${PROJECT_NAME}-test ${TEST_SRC} ${COMMON_INCLUDES})
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME} gtest)

