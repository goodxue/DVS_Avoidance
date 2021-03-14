#ifndef EGOMOTION_H
#define EGOMOTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>
#include <ros/ros.h>
class EgoMotion
{
public:
    EgoMotion(int erode_kernal_size, int W, int H);
    virtual ~EgoMotion();

    cv::Mat Count_image, Time_image, Normalized_time_image, Untreshold_normalized_time_image;
    std::vector<cv::Vec3d> events;
    //events 3*N
    void ego_motion(cv::Mat events, cv::Vec3d omegas);

private:
    int kernal_size, W, H;
    double timelen;
    std::vector<cv::Vec3d> warp(cv::Mat events,cv::Vec3d omegas);
    void event_counter(std::vector<cv::Vec3d> events, int W, int H);
    void deal_time_image();
};

#endif // EGOMOTION_H
