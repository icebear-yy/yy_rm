cmake_minimum_required(VERSION 3.8)
project(rune_package)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(video_publisher src/video_publisher.cpp)
ament_target_dependencies(video_publisher
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)
add_executable(dafu_node 
src/dafu_node.cpp
src/image_processor.cpp)
ament_target_dependencies(dafu_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)
target_include_directories(dafu_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include  
)
install(TARGETS
  video_publisher
  dafu_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
ament_package()




