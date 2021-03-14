#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <dvs_avoidance/point2i.h>
#include <dvs_avoidance/pointarray.h>
#include <dvs_avoidance/point2darray.h>
#include <iostream>
#include "cluster.h"
#include "egomotion.h"
#include <string.h>


//global param
cv_bridge::CvImagePtr cv_ptr2v;
dvs_msgs::EventArray eventarray; //same as vecotr<event>
sensor_msgs::Imu imu;
/* imu
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Quaternion orientation
  float64 x
  float64 y
  float64 z
  float64 w
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
  float64 x
  float64 y
  float64 z
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
  float64 x
  float64 y
  float64 z
float64[9] linear_acceleration_covariance
*/


//callback
//copy the msgs
void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg){
  eventarray = dvs_msgs::EventArray(*msg);
}

void eventimgCallback(const sensor_msgs::Image::ConstPtr& msg){
cv_ptr2v = cv_bridge::toCvCopy(msg);
}

//copy imu
void imuCallback(const sensor_msgs::Imu::ConstPtr& msg){
  imu = sensor_msgs::Imu(*msg);
}

//main
int main(int argc, char* argv[])
{
  ros::init(argc, argv, "cluster");
  ros::NodeHandle nh_;
  ros::Subscriber event_sub_ = nh_.subscribe("/dvs/events", 1, &eventsCallback);
  ros::Subscriber event_2value_img_ = nh_.subscribe("dvs_redblue", 1, &eventimgCallback);
  ros::Subscriber camerainfo_ = nh_.subscribe("/dvs/imu", 1, &imuCallback);
  ros::Publisher cluster_pub= nh_.advertise<dvs_avoidance::point2darray>("/cluster_point", 1);
  //test
  image_transport::ImageTransport it(nh_);
  image_transport::Publisher output_pub = it.advertise("/output_img", 1);
  //ros::NodeHandle nh;
  ros::Rate loop_rate(100);

  EgoMotion egomotion = EgoMotion(2,346,260);
  Cluster cluster = Cluster(40,20);
  
  while(nh_.ok()){
    if(eventarray.events.size() == 0)
    {
      ros::spinOnce();
      loop_rate.sleep();
      continue;
    }
    //std::cout<<eventarray.events.size()<<std::endl;
    cv::Mat events( 3, eventarray.events.size(), CV_64FC1);
    for(int i = 0; i < eventarray.events.size();i++){
        events.at<double>(0,i) = eventarray.events[i].x;
        events.at<double>(1,i) = eventarray.events[i].y;
        events.at<double>(2,i) = eventarray.events[i].ts.toSec();
    }
    cv::Vec3d omegas;
    omegas[0] = imu.angular_velocity.x;
    omegas[1] = imu.angular_velocity.y;
    omegas[2] = imu.angular_velocity.z;
    std::vector<std::vector<cv::Point>> cluster_point;

    egomotion.ego_motion(events,omegas);
    cluster_point = cluster.cluster(egomotion.Normalized_time_image,egomotion.Untreshold_normalized_time_image, output_pub);
    //using the global params to continue the task
    //std::cout<<imu.angular_velocity.x<<std::endl;
    
    dvs_avoidance::point2darray cluster_msg;
    if (cluster_point.size()>0){
      for (int i = 0;i < cluster_point.size();i++){
        dvs_avoidance::pointarray array;
        for (int j = 0;j < cluster_point[i].size();j++){
          dvs_avoidance::point2i point;
          point.x = cluster_point[i][j].x;
          point.y=cluster_point[i][j].y;
            array.pointarray.push_back(point);
        }
        cluster_msg.point2darray.push_back(array);
      }
    }
    cluster_pub.publish(cluster_msg);

    //
    ros::spinOnce();
    loop_rate.sleep();
    
  }

  return 0;
}