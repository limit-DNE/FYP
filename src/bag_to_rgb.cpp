#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <livox_ros_driver2/CustomMsg.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath> // For std::atan2 and std::hypot
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/core/affine.hpp> // For cv::Affine3d



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


// Transformation from LiDAR frame to camera frame (4x4 matrix)
Eigen::Matrix4f lidar_to_camera;

Eigen::Matrix4f extrinsic_matrix; 

// Global variable to hold the latest camera pose
geometry_msgs::PoseStamped latest_camera_pose;

void initializeExtrinsicMatrix() {
    extrinsic_matrix << -0.0120393, -0.999923,  0.00311385, -0.173498,
                         0.418446,   -0.00786641, -0.908208, -0.173128,
                         0.908162,   -0.00963118, 0.418509,  0.00790568,
                         0,          0,          0,         1;
}


// Callback function to update the camera pose
void cameraPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    latest_camera_pose = *msg;
}

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


/*void colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const cv::Mat& image, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colorized_cloud) {
    colorized_cloud->clear();

    // Convert ROS pose to OpenCV Affine transformation
    cv::Affine3d camera_pose(
        cv::Affine3d::Mat3(
            latest_camera_pose.pose.orientation.w,
            -latest_camera_pose.pose.orientation.z,
            latest_camera_pose.pose.orientation.y,
            latest_camera_pose.pose.orientation.z,
            latest_camera_pose.pose.orientation.w,
            -latest_camera_pose.pose.orientation.x,
            -latest_camera_pose.pose.orientation.y,
            latest_camera_pose.pose.orientation.x,
            latest_camera_pose.pose.orientation.w
        ),
        cv::Affine3d::Vec3(
            latest_camera_pose.pose.position.x,
            latest_camera_pose.pose.position.y,
            latest_camera_pose.pose.position.z
        )
    );

    for (const auto& point : cloud->points) {
        if (point.z <= 0) continue;  // Skip points behind the camera

        // Transform the point from point cloud frame to camera frame
        cv::Vec3d transformed_point = camera_pose * cv::Vec3d(point.x, point.y, point.z);

        std::vector<cv::Point3f> object_points{ {static_cast<float>(transformed_point[0]), static_cast<float>(transformed_point[1]), static_cast<float>(transformed_point[2])} };
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), camera_matrix, dist_coeffs, image_points);

        auto& ip = image_points[0];
        if (ip.x >= 0 && ip.x < image.cols && ip.y >= 0 && ip.y < image.rows) {
            cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(ip.x, ip.y));
            pcl::PointXYZRGB colorized_point;
            colorized_point.x = point.x;
            colorized_point.y = point.y;
            colorized_point.z = point.z;
            colorized_point.r = color[2];
            colorized_point.g = color[1];
            colorized_point.b = color[0];
            colorized_cloud->points.push_back(colorized_point);
        }
    }
    ROS_INFO("Processed %lu points. Colorized %d points.", cloud->points.size(), colorized_cloud->points.size());
} */

void colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const cv::Mat& image, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& colorized_cloud) {
    colorized_cloud->clear();

    Eigen::Matrix4f transformation_matrix = extrinsic_matrix; // Assuming extrinsic_matrix represents the transformation from point cloud frame to camera frame

    for (const auto& point : cloud->points) {
        // Apply the extrinsic transformation
        Eigen::Vector4f transformed_point = transformation_matrix * Eigen::Vector4f(point.x, point.y, point.z, 1.0f);

        // Project the transformed point to the image plane
        std::vector<cv::Point3f> object_points{ {transformed_point(0), transformed_point(1), transformed_point(2)} };
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), camera_matrix, dist_coeffs, image_points);

        auto& ip = image_points[0];
        if (ip.x >= 0 && ip.x < image.cols && ip.y >= 0 && ip.y < image.rows) {
            cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(ip.x, ip.y));
            pcl::PointXYZRGB colorized_point;
            colorized_point.x = transformed_point(0);
            colorized_point.y = transformed_point(1);
            colorized_point.z = transformed_point(2);
            colorized_point.r = color[2];
            colorized_point.g = color[1];
            colorized_point.b = color[0];
            colorized_cloud->points.push_back(colorized_point);
        }
    }
    ROS_INFO("Processed %lu points. Colorized %d points.", cloud->points.size(), colorized_cloud->points.size());
}


void colorizePointCloudFromBag(const std::string& bag_file_path, ros::NodeHandle& nh) {
    rosbag::Bag bag;
    bag.open(bag_file_path, rosbag::bagmode::Read);

    std::vector<std::string> topics = {"/livox/lidar", "/cam_1/color/image_raw"};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Temporary storage for the latest image and point cloud
    cv_bridge::CvImagePtr cv_ptr_latest;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == "/livox/lidar" || ("/" + m.getTopic() == "/livox/lidar")) {
            livox_ros_driver2::CustomMsg::ConstPtr custom_msg = m.instantiate<livox_ros_driver2::CustomMsg>();
            if (custom_msg != nullptr) {
                convertCustomMsgToPCLPointCloud(*custom_msg, cloud);
            }
        } else if (m.getTopic() == "/cam_1/color/image_raw" || ("/" + m.getTopic() == "/cam_1/color/image_raw")) {
            sensor_msgs::Image::ConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
            if (image_msg != nullptr) {
                try {
                    // Convert ROS image to OpenCV image
                    cv_ptr_latest = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                    
                    // Create a new Mat for the undistorted image
                    cv::Mat undistorted_image;
                    // Apply distortion correction
                    cv::undistort(cv_ptr_latest->image, undistorted_image, camera_matrix, dist_coeffs);
                    // Assign the undistorted image back to cv_ptr_latest
                    cv_ptr_latest->image = undistorted_image;
                } catch (const cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                }
            }
        }
    }

    bag.close();

    // Publishers
    pub = nh.advertise<sensor_msgs::PointCloud2>("rgb_cloud", 1);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher image_pub = it.advertise("camera_image", 1);

    // After loading the data, enter an interactive loop for publishing
    while (ros::ok()) {
        if (cloud && cv_ptr_latest) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            colorizePointCloud(cloud, cv_ptr_latest->image, colorized_cloud);

            sensor_msgs::PointCloud2 output;
            pcl::toROSMsg(*colorized_cloud, output);
            output.header.frame_id = "livox";
            output.header.stamp = ros::Time::now();
            pub.publish(output);
            ROS_INFO("Published colorized point cloud.");
        }

        // Publish the latest undistorted image
        if (cv_ptr_latest) {
            sensor_msgs::ImagePtr img_msg = cv_ptr_latest->toImageMsg();
            image_pub.publish(img_msg);
        }

        // Wait for user input to continue
        std::cout << "Press Enter to publish again" << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "bag_to_rgb");
    ros::NodeHandle nh;
     // Initialize extrinsic matrix
    initializeExtrinsicMatrix();


    pub = nh.advertise<sensor_msgs::PointCloud2>("rgb_cloud", 1);
    
    // Subscribe to the camera pose topic
    ros::Subscriber pose_sub = nh.subscribe("/camera_pose", 1, cameraPoseCallback);

    if (argc < 2) {
        ROS_ERROR("You must specify the bag file path as a command-line argument.");
        return 1;
    }

    // Allow the publisher some time to register with the ROS master
    ros::Rate rate(1); // Define rate here and use it throughout the function

    rate.sleep(); // Sleep for a moment to give time for publisher registration

    colorizePointCloudFromBag(argv[1], nh);

    // Keep the node alive to listen to callbacks and to keep publishing if necessary
    while (ros::ok()) {
        ros::spinOnce(); // Handle ROS callbacks and check for new messages
        rate.sleep(); // Sleep to maintain the loop rate
    }

    ROS_INFO("Finished processing and publishing.");

    return 0;
}
