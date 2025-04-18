cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(src)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(myprogram main.cpp)
# target_link_libraries(myprogram "${TORCH_LIBRARIES}")
# set_property(TARGET myprogram PROPERTY CXX_STANDARD 17)

# OR Method 2: Using set_target_properties
add_executable(src main.cpp)
set_target_properties(src PROPERTIES OUTPUT_NAME "myprogram")
target_link_libraries(src "${TORCH_LIBRARIES}")
set_property(TARGET src PROPERTY CXX_STANDARD 17)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)