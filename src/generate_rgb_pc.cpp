#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <thread>
#include <filesystem>

using namespace std;
namespace fs = boost::filesystem;

// Overloaded function for visualizing pcl::PointXYZ point clouds
/*void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& windowTitle) {
    pcl::visualization::PCLVisualizer visualizer(windowTitle);
    visualizer.setBackgroundColor(0, 0, 0); // Set background to black

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor(cloud, 0, 255, 0); // Green
    visualizer.addPointCloud<pcl::PointXYZ>(cloud, singleColor, "sample cloud");
    visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    visualizer.addCoordinateSystem(1.0);
    visualizer.initCameraParameters();

    while (!visualizer.wasStopped()) {
        visualizer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& windowTitle) {
    pcl::visualization::PCLVisualizer visualizer(windowTitle);
    visualizer.setBackgroundColor(0, 0, 0); // Set background to black
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    visualizer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
    visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    visualizer.addCoordinateSystem(1.0);
    visualizer.initCameraParameters();

    while (!visualizer.wasStopped()) {
        visualizer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}*/

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

void generateRGBPointCloud(const string& pcd_file, const std::string& image_file, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read pcl file \n");
        return;
    }

    // Load the PNG image
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image from " << image_file << std::endl;
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

    // Obtain image width and height from the cv::Mat object
    int image_width = image.cols;
    int image_height = image.rows;

    for (const auto& point : transformed_cloud->points) {
        // Project the point onto the camera's image plane
        float u = fx * point.x / point.z + cx;
        float v = fy * point.y / point.z + cy;

        // Check if the point is within the image boundaries
        if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
            // The point is within the camera's FOV
            // Associate this point with its corresponding pixel in the RGB image
            // Get the color from the image
            cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(u, v));
            // Add the point with color to the new cloud
            pcl::PointXYZRGB point_rgb;
            point_rgb.x = point.x;
            point_rgb.y = point.y;
            point_rgb.z = point.z;
            point_rgb.r = color[2];
            point_rgb.g = color[1];
            point_rgb.b = color[0];
            rgb_cloud->points.push_back(point_rgb);
        }
    }

    // Set the width, height, and is_dense properties of the rgb_cloud
    rgb_cloud->width = rgb_cloud->points.size();
    rgb_cloud->height = 1;
    rgb_cloud->is_dense = false;
}

void overlayPointCloudOnImage(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, cv::Mat& image, float fx, float fy, float cx, float cy) {
    // Sort the points by Z (depth) in descending order to handle occlusions
    std::sort(cloud->points.begin(), cloud->points.end(), [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) {
        return a.z > b.z;
    });

    for (const auto& point : cloud->points) {
        int u = static_cast<int>((fx * point.x / point.z) + cx);
        int v = static_cast<int>((fy * point.y / point.z) + cy);

        // Check if the projected point is within the image bounds
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
            // Color the pixel at (u, v) with the point's color
            std::cout << "Within image bounds.." << std::endl;
            cv::Vec3b& pixel = image.at<cv::Vec3b>(v, u);
            pixel[0] = point.b;
            pixel[1] = point.g;
            pixel[2] = point.r;
        }
    }
}

void processPointCloudFile(const std::string& pcd_file, const std::string& image_file) {
    // Load the original point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *original_cloud) == -1) {
        PCL_ERROR("Couldn't read file %s\n", pcd_file.c_str());
        return; // Skip this file if it can't be read
    }

    // Optionally visualize the original point cloud
    // visualizePointCloud(original_cloud, "Original Point Cloud");

    std::cout << "Generating RGB point cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    generateRGBPointCloud(pcd_file, image_file, rgb_cloud);
    std::cout << "RGB point cloud generated." << std::endl;

    // Save the processed point cloud to a file
    std::string output_pcd_file = "processed_" + fs::path(pcd_file).filename().string();
    pcl::io::savePCDFileASCII(output_pcd_file, *rgb_cloud);
    std::cout << "Saved processed point cloud to: " << fs::absolute(output_pcd_file) << std::endl;

    // Visualize the resulting RGB point cloud
   // visualizePointCloud(rgb_cloud, "RGB Point Cloud Visualization: " + fs::path(pcd_file).filename().string());

    // Optionally save the processed point cloud to a file
    // pcl::io::savePCDFileASCII("processed_" + fs::path(pcd_file).filename().string(), *rgb_cloud);

    std::cout << "Overlaying point cloud on image..." << std::endl;
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_file << std::endl;
        return;
    }

    // Call overlayPointCloudOnImage here
    // You need to provide the camera's intrinsic parameters: fx, fy, cx, cy
    // These should be known or calibrated values specific to your camera setup
    float fx = 1364.45f;  // Example value, replace with your actual camera's focal length in x direction
    float fy = 1366.46f;  // Example value, replace with your actual camera's focal length in y direction
    float cx = 958.327f;  // Example value, replace with your actual camera's principal point x-coordinate
    float cy = 535.074f;  // Example value, replace with your actual camera's principal point y-coordinate

    overlayPointCloudOnImage(rgb_cloud, image, fx, fy, cx, cy);
    std::cout << "Overlay completed." << std::endl;

    // Save the overlayed image
    std::string output_image_file = "overlayed_" + fs::path(image_file).filename().string();
    cv::imwrite(output_image_file, image);
    std::cout << "Saved overlayed image to: " << fs::absolute(output_image_file) << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Program started" << std::endl;

    // Check for three arguments: the executable, the PCD directory, and the PNG directory
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pcd_directory> <png_directory>" << std::endl;
        return -1;
    }

    fs::path pcd_directory(argv[1]);
    fs::path png_directory(argv[2]);

    // Ensure both provided paths are directories
    if (!fs::is_directory(pcd_directory) || !fs::is_directory(png_directory)) {
        std::cerr << "Both arguments must be valid directories." << std::endl;
        return -1;
    }

    // Iterate over the PCD directory
    fs::directory_iterator end_itr; // Default construction yields past-the-end
    for (fs::directory_iterator pcd_itr(pcd_directory); pcd_itr != end_itr; ++pcd_itr) {
        if (fs::is_regular_file(pcd_itr->status()) && pcd_itr->path().extension() == ".pcd") {
            // Construct the expected PNG filename by changing the extension and directory
            fs::path expected_png_path = png_directory / pcd_itr->path().filename().replace_extension(".png");

            // Check if the expected PNG file exists
            if (fs::exists(expected_png_path)) {
                // Process the PCD and PNG pair
                std::cout << "Processing: " << pcd_itr->path() << " and " << expected_png_path << std::endl;
                processPointCloudFile(pcd_itr->path().string(), expected_png_path.string());

                // After processing, overlay the point cloud on the image if required
                // Overlay function should be defined elsewhere in your code
                // overlayPointCloudOnImage(pcd_itr->path().string(), expected_png_path.string());
            } else {
                std::cerr << "Matching PNG file not found for: " << pcd_itr->path() << std::endl;
            }
        }
    }

    return 0;
}
