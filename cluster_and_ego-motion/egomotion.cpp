#include "egomotion.h"

EgoMotion::EgoMotion(int kernal_size):
    kernal_size(kernal_size)
{

}

EgoMotion::~EgoMotion()
{
    //dtor
}

void EgoMotion::ego_motion(cv::Mat events, std::vector<cv::Vec3f> omegas, int W, int H)
{
    //warp
    std::vector<cv::Vec3f> event_new = warp(events, omegas, W, H);
    //count image
    event_counter(event_new, W, H);
    deal_time_image();
    cv::Mat kernal = cv::Mat::ones(this->kernal_size, this->kernal_size, CV_8UC1);
    cv::erode(this->Normalized_time_image, this->Normalized_time_image, kernal);
}

std::vector<cv::Vec3f> EgoMotion::warp(cv::Mat events, std::vector<cv::Vec3f> omegas, int W, int H)
{
    //warp
    cv::Vec3f omega_bar = std::accumulate(omegas.begin(),omegas.end(),cv::Vec3f(0,0,0)) / (int)omegas.size();
    omega_bar = omega_bar*3.14/180.0;
    float Rinit[] = {1, -omega_bar[2], omega_bar[1], omega_bar[2], 1, -omega_bar[0], 0, 0, 1};
    cv::Mat R = cv::Mat(3, 3, CV_32FC1, Rinit);
    float t0 = events.at<float>(2,0);
    cv::Mat event_new = R*events;
    std::vector<cv::Vec3f> result;
    for(int i=0;i<event_new.cols;i++)
    {
        float y = event_new.at<float>(0,i);
        float x = event_new.at<float>(1,i);
        if(x<H && x>=0 && y<W && y>0)
            result.push_back(cv::Vec3f(round(x), round(y), event_new.at<float>(2,i)-t0));
    }
    //calculate timelen
    this->timelen = (*(result.end() - 1))[2];

    return result;
}

void EgoMotion::event_counter(std::vector<cv::Vec3f> events, int W, int H)
{
    cv::Mat count_image = cv::Mat::zeros(H, W, CV_8UC1);
    cv::Mat time_image = cv::Mat::zeros(H, W, CV_8UC1);
    for(std::vector<cv::Vec3f>::iterator iter = events.begin(); iter != events.end(); iter++)
    {
        count_image.at<uchar>((*iter)[0], (*iter)[1]) += 1;
        time_image.at<uchar>((*iter)[0], (*iter)[1]) += (*iter)[2];
    }
    for(int r = 0; r < H; r++)
        for(int c = 0; c < W; c++)
            if(count_image.at<uchar>(r,c) != 0)
                time_image.at<uchar>(r,c) /= count_image.at<uchar>(r,c);
    this->Count_image = count_image.clone();
    this->Time_image = time_image.clone();
}

void EgoMotion::deal_time_image()
{
    double minv, maxv;
    cv::Point minl, maxl;
    cv::minMaxLoc(this->Time_image, &minv, &maxv, &minl, &maxl);
    float miu = cv::mean(this->Time_image)[0];
    this->Untreshold_normalized_time_image = (this->Time_image - miu)/this->timelen;

    cv::Scalar miu_new, std;
    cv::meanStdDev(this->Untreshold_normalized_time_image, miu_new, std);
    float threshold = miu_new[0] + 2 * std[0];
    this->Normalized_time_image = this->Untreshold_normalized_time_image.clone();
    for(int r = 0; r < this->Untreshold_normalized_time_image.rows; r++)
        for(int c = 0; c < this->Untreshold_normalized_time_image.cols; c++)
        {
            if(this->Untreshold_normalized_time_image.at<uchar>(r,c) <= threshold)
                this->Normalized_time_image.at<uchar>(r,c) = 0;
            else
                this->Normalized_time_image.at<uchar>(r,c) = 1;
        }
}
