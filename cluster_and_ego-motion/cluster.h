#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <stack>
#include "opencv2/opencv.hpp"

class point{
public:
    float x;
    float y;
    int cluster=0;
    int pointType=1;//1 noise 2 border 3 core
    int pts=0;//points in MinPts
    std::vector<int> corepts;
    int visited = 0;
    point (){}
    point (float a,float b,int c){
        x = a;
        y = b;
        cluster = c;
    }
};

class Cluster
{
    public:
        Cluster(float threshold, cv::Mat Optical_flow, cv::Mat Time_image, float wp, float wv, float wrho);
        virtual ~Cluster();
        std::vector<std::vector<cv::Point> > cluster(cv::Mat img, float threshold, int MinPts);

    protected:

    private:
        float threshold;
        float wp, wv, wrho;
        cv::Mat OpticalFlow;
        cv::Mat Timeimage;
        std::vector<point> img2point(cv::Mat img);
        float Cost(point a, point b);
};

#endif // CLUSTER_H
