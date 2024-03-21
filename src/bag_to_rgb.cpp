#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <livox_ros_driver2/CustomMsg.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// Camera intrinsic parameters and distortion coefficients
// Define the camera matrix based on the values extracted from your configuration file
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 645.252100829746, 0.0, 646.8185880049397,
                                                    0.0, 642.9685575891763, 353.62380523406574,
                                                    0.0, 0.0, 1.0);

cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << -0.051089818530300236, 0.03547486314704304,
                                                 0.0009051276804264704, 0.0012607789620934912, 0.0);
cv_bridge::CvImagePtr cv_ptr;
ros::Publisher pub;  // Define the publisher globally


Eigen::Matrix4f extrinsic_matrix; 

// Transformation from LiDAR frame to camera frame (4x4 matrix)
Eigen::Matrix4f lidar_to_camera;

// Conversion function from Livox custom message to PCL PointXYZ
void convertCustomMsgToPCLPointCloud(const livox_ros_driver2::CustomMsg& custom_msg, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
    pcl_cloud->clear();  // Clear existing points in the cloud

    // Iterate through each point in the custom message and convert it to PCL PointXYZ format
    for (const auto& custom_point : custom_msg.points) {
        // Assuming the custom message contains x, y, z fields for each point
        pcl::PointXYZ pcl_point;
        pcl_point.x = custom_point.x;
        pcl_point.y = custom_point.y;
        pcl_point.z = custom_point.z;

        // Add the converted point to the PCL cloud
        pcl_cloud->push_back(pcl_point);
    }
}


void colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                        const cv::Mat& image,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colorized_cloud) {
    colorized_cloud->clear();

    int colorized_points = 0;
    for (const auto& point : cloud->points) {
        if (point.z <= 0) continue;  // Skip points with non-positive z value to avoid division by zero


        float u = (camera_matrix.at<double>(0, 0) * point.x / point.z) + camera_matrix.at<double>(0, 2);
        float v = (camera_matrix.at<double>(1, 1) * point.y / point.z) + camera_matrix.at<double>(1, 2);

        // Check if the projected point is within the image bounds
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
            // Extract color from the image at the pixel location
            cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(u, v));

            // Assign color to the point in the colorized point cloud
            pcl::PointXYZRGB colorized_point;
            colorized_point.x = point.x;
            colorized_point.y = point.y;
            colorized_point.z = point.z;
            colorized_point.r = color[2];  
            colorized_point.g = color[1];  
            colorized_point.b = color[0];  
            colorized_cloud->points.push_back(colorized_point);
            colorized_points++;
        }
    }

    ROS_INFO("Processed %lu points. Colorized %d points.", cloud->points.size(), colorized_points);
}



void colorizePointCloudFromBag(const std::string& bag_file_path) {
    rosbag::Bag bag;
    bag.open(bag_file_path, rosbag::bagmode::Read);

    std::vector<std::string> topics = {"/livox/lidar", "/cam_1/color/image_raw"};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Temporary storage for the latest image
    cv_bridge::CvImagePtr cv_ptr_latest;

    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == "/livox/lidar" || ("/" + m.getTopic() == "/livox/lidar")) {
            livox_ros_driver2::CustomMsg::ConstPtr custom_msg = m.instantiate<livox_ros_driver2::CustomMsg>();
            if (custom_msg != nullptr && cv_ptr_latest) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
                convertCustomMsgToPCLPointCloud(*custom_msg, cloud);

                // Assuming you have a function to convert and colorize the point cloud:
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                colorizePointCloud(cloud, cv_ptr_latest->image, colorized_cloud);  // Implement this function

                // After colorizing the point cloud, publish it
                sensor_msgs::PointCloud2 output;
                pcl::toROSMsg(*colorized_cloud, output);
                output.header.frame_id = custom_msg->header.frame_id;
                output.header.stamp = custom_msg->header.stamp;
                pub.publish(output);
                ROS_INFO("Published colorized point cloud.");
            }
        } else if (m.getTopic() == "/cam_1/color/image_raw" || ("/" + m.getTopic() == "/cam_1/color/image_raw")) {
            sensor_msgs::Image::ConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
            if (image_msg != nullptr) {
                try {
                    cv_ptr_latest = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                } catch (const cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                }
            }
        }
    }

    bag.close();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "rgb_cloud_publisher");
    ros::NodeHandle nh;

    pub = nh.advertise<sensor_msgs::PointCloud2>("rgb_cloud", 1);

    if (argc < 2) {
        ROS_ERROR("You must specify the bag file path as a command-line argument.");
        return 1;
    }

    // Allow the publisher some time to register with the ROS master
    ros::Rate rate(1); // 1 second should be more than enough
    rate.sleep();

    colorizePointCloudFromBag(argv[1]);

    // No ros::spin() needed; colorizePointCloudFromBag does all the processing and publishing

    ROS_INFO("Finished processing and publishing.");

    return 0;
}
