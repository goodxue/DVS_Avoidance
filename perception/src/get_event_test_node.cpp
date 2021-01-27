#include "get_event_img.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "dvs_renderer");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  int SampleRate = 100;
  int ros_loop_rate = 100;
  ros::Rate loop_rate(ros_loop_rate);
  Renderer renderer(nh, nh_private, int(ros_loop_rate/SampleRate-1));
  while(nh.ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}