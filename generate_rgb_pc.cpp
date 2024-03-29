#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <thread>
#include <filesystem>

using namespace std;
namespace fs = boost::filesystem;

// Overloaded function for visualizing pcl::PointXYZ point clouds
void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& windowTitle) {
    pcl::visualization::PCLVisualizer::Ptr visualizer (new pcl::visualization::PCLVisualizer(windowTitle));
    visualizer->setBackgroundColor(0, 0, 0); // Set background to black

    // Use a default color handler for single color
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor(cloud, 0, 255, 0); // Green

    visualizer->addPointCloud<pcl::PointXYZ>(cloud, singleColor, "sample cloud");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    visualizer->addCoordinateSystem(1.0);
    visualizer->initCameraParameters();

    while (!visualizer->wasStopped()) {
        visualizer->spinOnce(100); // Update the visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& windowTitle) {
    pcl::visualization::PCLVisualizer::Ptr visualizer (new pcl::visualization::PCLVisualizer(windowTitle));
    visualizer->setBackgroundColor(0, 0, 0); // Set background to black
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    visualizer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    visualizer->addCoordinateSystem(1.0);
    visualizer->initCameraParameters();

    while (!visualizer->wasStopped()) {
        visualizer->spinOnce(100); // Update the visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void filterPointsByFOV(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, float hFOV, float vFOV, const Eigen::Vector3f& cameraPos, const Eigen::Vector3f& viewingDirection) {
    for (auto& point : *cloud) {
        Eigen::Vector3f pointVec(point.x - cameraPos[0], point.y - cameraPos[1], point.z - cameraPos[2]);

        float angleHorizontal = acos(pointVec.dot(viewingDirection) / (pointVec.norm() * viewingDirection.norm()));
        float angleVertical = acos(pointVec.dot(viewingDirection) / (pointVec.norm() * viewingDirection.norm()));

        // Convert FOV angles from degrees to radians for comparison
        float hFOVRad = hFOV * M_PI / 180.0;
        float vFOVRad = vFOV * M_PI / 180.0;

        // Check if the point is within the FOV
        if (abs(angleHorizontal) <= hFOVRad / 2 && abs(angleVertical) <= vFOVRad / 2) {
            filtered_cloud->points.push_back(point);
        }
    }
}

void generateRGBPointCloud(const string& pcd_file, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read pcl file \n");
        return;
    }

    // Define camera FOV
    // Values based on Intel RealSense
    float horizontalFOV = 86; // degrees, example value for Intel RealSense D455
    float verticalFOV = 57;

    // Camera position and viewing direction
    Eigen::Vector3f cameraPosition(0.0, 0.0, 0.0);
    Eigen::Vector3f viewingDirection(0.0, 0.0, 1.0);

    filterPointsByFOV(cloud, filtered_cloud, horizontalFOV, verticalFOV, cameraPosition, viewingDirection);

    // Camera intrinsic parameters
    // Camera intrinsic parameters based on provided camera_matrix values
    float fx = 1364.45f;  // Focal length in x direction
    float fy = 1366.46f;  // Focal length in y direction
    float cx = 958.327f;  // Optical center x-coordinate
    float cy = 535.074f;  // Optical center y-coordinate

    // Extrinsic parameters: conversion from rotation matrix to quaternion
    Eigen::Matrix3d R;
    R << -0.0438259, -0.979398, -0.197128,
            0.0457862,  0.195141, -0.979706,
            0.997989,  -0.0519623, 0.0362906;
    Eigen::Quaterniond q(R);  // Eigen automatically converts the rotation matrix to a quaternion

    Eigen::Vector3d transation(0.216124, -0.218298, -0.0310492);

    // Combine rotation and translation into a 4x4 transformation matrix
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    transformation_matrix.block<3, 3>(0, 0) = R.cast<float>();  // Convert double to float
    transformation_matrix.block<3, 1>(0, 3) = transation.cast<float>();  // Convert double to float

    // Transform the point cloud to the camera coordinate system
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_matrix);

    // Assume image_width and image_height are defined somewhere in your code
    int image_width = 640;  // Example value, replace with your actual image width
    int image_height = 480; // Example value, replace with your actual image height

    for (const auto& point : transformed_cloud->points) {
        // Project the point onto the camera's image plane
        float u = fx * point.x / point.z + cx;
        float v = fy * point.y / point.z + cy;

        // Check if the point is within the image boundaries
        if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
            // The point is within the camera's FOV
            // Associate this point with its corresponding pixel in the RGB image
            // For demonstration, we assign a placeholder color (white) to the point
            pcl::PointXYZRGB color_point;
            color_point.x = point.x;
            color_point.y = point.y;
            color_point.z = point.z;
            color_point.r = 255;
            color_point.g = 255;
            color_point.b = 255;

            rgb_cloud->points.push_back(color_point);
        }
    }

    // Set the width, height, and is_dense properties of the rgb_cloud
    rgb_cloud->width = rgb_cloud->points.size();
    rgb_cloud->height = 1;
    rgb_cloud->is_dense = false;
}

void processPointCloudFile(const std::string& pcd_file)
{
    // Load the original point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *original_cloud) == -1) {
        PCL_ERROR("Couldn't read file %s\n", pcd_file.c_str());
        return; // Skip this file if it can't be read
    }

    // Optionally visualize the original point cloud
    // visualizePointCloud(original_cloud, "Original Point Cloud: ");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    generateRGBPointCloud(pcd_file, rgb_cloud);

    // Visualize the resulting RGB point cloud
    visualizePointCloud(rgb_cloud, "RGB Point Cloud Visualization: " + fs::path(pcd_file).filename().string());

    // Optionally save the processed point cloud to a file
    // pcl::io::savePCDFileASCII("processed_" + fs::path(pcd_file).filename().string(), *rgb_cloud);
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_pcd_file>" << std::endl;
        return -1;
    }

    fs::path directory_path(argv[1]);
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Provided path is not a directory." << std::endl;
        return -1;
    }

    fs::directory_iterator end_itr; // Default construction yields past-the-end
    for (fs::directory_iterator itr(directory_path); itr != end_itr; ++itr) {
        if (fs::is_regular_file(itr->status()) && itr->path().extension() == ".pcd") {
            processPointCloudFile(itr->path().string());
        }
    }
    return 0;
}
