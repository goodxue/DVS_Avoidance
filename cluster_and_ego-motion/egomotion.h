#ifndef EGOMOTION_H
#define EGOMOTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>

class EgoMotion
{
public:
    EgoMotion(int kernal_size);
    virtual ~EgoMotion();

    cv::Mat Count_image, Time_image, Normalized_time_image, Untreshold_normalized_time_image;
    //events 3*N
    void ego_motion(cv::Mat events, std::vector<cv::Vec3f> omegas, int W, int H);

private:
    int kernal_size;
    float timelen;
    std::vector<cv::Vec3f> warp(cv::Mat events, std::vector<cv::Vec3f> omegas, int W, int H);
    void event_counter(std::vector<cv::Vec3f> events, int W, int H);
    void deal_time_image();
};

#endif // EGOMOTION_H
