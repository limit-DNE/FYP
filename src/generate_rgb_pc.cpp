#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <string>
#include <Eigen/Geometry>

using namespace std;

void generateRGBPointCloud(const string& pcd_file, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read pcl file \n");
        return;
    }

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
            color_point.r = 255;  // Red component
            color_point.g = 255;  // Green component
            color_point.b = 255;  // Blue component

            rgb_cloud->points.push_back(color_point);
        }
    }

    // Set the width, height, and is_dense properties of the rgb_cloud
    rgb_cloud->width = rgb_cloud->points.size();
    rgb_cloud->height = 1;
    rgb_cloud->is_dense = false;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_pcd_file>" << std::endl;
        return -1;
    }

    std::string pcd_file = argv[1];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    generateRGBPointCloud(pcd_file, rgb_cloud);

    // Optional: Save the result or visualize it
    // pcl::io::savePCDFileASCII("output_rgb.pcd", *rgb_cloud);

    return 0;
}