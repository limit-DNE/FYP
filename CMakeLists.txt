cmake_minimum_required(VERSION 2.8)
project(livox_camera_calib)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV 4.4)
find_package(Threads)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(OpenCV_DIR "~/opencv-3.4.1/build")

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(lidar_camera_calib src/lidar_camera_calib.cpp include/lidar_camera_calib.hpp)
add_executable(lidar_camera_multi_calib src/lidar_camera_multi_calib.cpp include/lidar_camera_calib.hpp)
add_executable(bag_to_pcd src/bag_to_pcd.cpp)
add_executable(generate_rgb_pc src/generate_rgb_pc.cpp)

target_link_libraries(lidar_camera_calib 
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
)


target_link_libraries(lidar_camera_multi_calib 
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
)

target_link_libraries(bag_to_pcd 
  ${catkin_LIBRARIES}  ${PCL_LIBRARIES}
)

target_link_libraries(generate_rgb_pc
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES}
)
