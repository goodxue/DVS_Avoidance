#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <stack>
#include "opencv2/opencv.hpp"
#include "dbscan.h"

//test
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

class PreCluster
{
public:
    cv::Point2f center;
    float radius;
    float operator [](int i) const
    {
        if(i==0)
            return center.y;
        else
            return center.x;
    }

};

/*
 * optical flow
class Vec2f : public cv::Vec2f{
public:
    Vec2f(float a, float b) : cv::Vec2f(a,b){}
    bool operator < (const cv::Vec2f& p) const{
        if ((*this)[0] < p[0])return true;
        if ((*this)[0] > p[0])return false;
        if ((*this)[1] < p[1])return true;
        return false;
    }
};
*/


class Cluster
{
    public:
        Cluster(float threshold, int MinPts);
        virtual ~Cluster();
        std::vector<std::vector<cv::Point> > cluster(cv::Mat &Normalized_time_image, cv::Mat& Time_image, image_transport::Publisher output_pub);


    protected:

    private:
        int MinPts;
        float threshold;
        cv::Mat OpticalFlow;
        cv::Mat Timeimage;
        std::vector<std::vector<cv::Point> > img2point(cv::Mat &img);
        //float distance(PreCluster a, PreCluster b);
        //cv::Mat optical_flow(cv::Mat prev, cv::Mat next, std::vector<cv::Point2f> prev_events, std::vector<cv::Point2f> next_events, int window_size);
};

#endif // CLUSTER_H
