cmake_minimum_required(VERSION 3.8)
project(rune_package)

# 设置 C++17 标准，因为代码中用到了 std::filesystem
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 查找核心 ROS2 和图像处理包 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# --- 查找 OpenCV，并包含 dnn 模块 ---
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn)

# --- 查找 ONNX Runtime (更稳健的方式) ---
# 假设你已将 onnxruntime 安装到 /usr/local
set(ONNXRUNTIME_ROOT_DIR "/usr/local") 
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
             HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_DIR or install it to /usr/local.")
endif()


# =================== 定义可执行文件和依赖 ===================

# --- onnx_detect 节点 ---
add_executable(onnx_detect src/onnx_detect.cpp)
ament_target_dependencies(onnx_detect 
rclcpp
sensor_msgs
cv_bridge
)

target_include_directories(onnx_detect PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(onnx_detect
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARY}
)

# --- video_publisher 节点 ---
add_executable(video_publisher src/video_publisher.cpp)
ament_target_dependencies(video_publisher
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

# --- dafu_node 节点 ---
add_executable(dafu_node src/dafu_node.cpp src/image_processor.cpp)
ament_target_dependencies(dafu_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)
target_include_directories(dafu_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# =================== 安装规则 ===================
install(TARGETS
  video_publisher
  dafu_node
  onnx_detect
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()




