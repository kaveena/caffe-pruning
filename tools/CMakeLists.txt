# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # caffe target already exits
  if(name MATCHES "caffe")
    set(name ${name}.bin)
  endif()

  if(name MATCHES "caffe.bin")
    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${Caffe_LINK})
    caffe_default_properties(${name})

    # set back RUNTIME_OUTPUT_DIRECTORY
    caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
    caffe_set_solution_folder(${name} tools)

    # restore output name without suffix
    set_target_properties(${name} PROPERTIES OUTPUT_NAME caffe)

    # Install
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  else()
    if(BUILD_tools)
      # target
      add_executable(${name} ${source})
      target_link_libraries(${name} ${Caffe_LINK})
      caffe_default_properties(${name})

      # set back RUNTIME_OUTPUT_DIRECTORY
      caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
      caffe_set_solution_folder(${name} tools)

      # Install
      install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()
endforeach(source)
