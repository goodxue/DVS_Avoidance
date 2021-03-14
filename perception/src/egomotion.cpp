#include "egomotion.h"

EgoMotion::EgoMotion(int erode_kernal_size, int W, int H):
    kernal_size(erode_kernal_size),
    W(W),
    H(H)
{
    Normalized_time_image = cv::Mat(H, W, CV_8UC1);
}

EgoMotion::~EgoMotion()
{
    //dtor
}

void EgoMotion::ego_motion(cv::Mat events, cv::Vec3d omegas)
{
    //warp
    std::vector<cv::Vec3d> event_new = warp(events, omegas);
    this->events = event_new;
    //count image
    event_counter(event_new, W, H);
    deal_time_image();
    cv::Mat kernal = cv::Mat::ones(this->kernal_size, this->kernal_size, CV_8UC1);
    cv::erode(this->Normalized_time_image, this->Normalized_time_image, kernal);
    cv::erode(this->Untreshold_normalized_time_image, this->Untreshold_normalized_time_image, kernal);
    cv::erode(this->Count_image, this->Count_image, kernal);
}

cv::Mat eulerAnglesToRotationMatrix(cv::Vec3d theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);


    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;

}


std::vector<cv::Vec3d> EgoMotion::warp(cv::Mat events, cv::Vec3d omegas)
{
    //warp
    double t0 = events.at<double>(2,0);
    /*
    //cv::Vec3d omega_bar = std::accumulate(omegas.begin(),omegas.end(),cv::Vec3d(0,0,0)) / (int)omegas.size();
    cv::Vec3d omega_bar = omegas*3.14/180.0;
    cv::Mat R;
    R = eulerAnglesToRotationMatrix(omega_bar * (events.at<double>(2,events.cols-1) - t0));
    R.at<double>(0,2) = 0;
    R.at<double>(1,2) = 0;
    R.at<double>(2,0) = 0;
    R.at<double>(2,1) = 0;
    R.at<double>(2,2) = 1;
    */
    cv::Mat event_new = events;
    std::vector<cv::Vec3d> result;
    for(int i=0;i<events.cols;i++)
    {
        events.at<double>(2,i) = events.at<double>(2,i)-t0;

        double y = event_new.at<double>(0,i);
        double x = event_new.at<double>(1,i);
        if(x<H && x>=0 && y<W && y>=0)
            result.push_back(cv::Vec3d(round(x), round(y), events.at<double>(2,i)));
    }
    //calculate timelen
    this->timelen = (*(result.end() - 1))[2];

    return result;
}

void EgoMotion::event_counter(std::vector<cv::Vec3d> events, int W, int H)
{
    cv::Mat count_image = cv::Mat::zeros(H, W, CV_64FC1);
    cv::Mat time_image = cv::Mat::zeros(H, W, CV_64FC1);
    for(std::vector<cv::Vec3d>::iterator iter = events.begin(); iter != events.end(); iter++)
    {
        count_image.at<double>((int)(*iter)[0], (int)(*iter)[1]) += 1;
        time_image.at<double>((int)(*iter)[0], (int)(*iter)[1]) += (*iter)[2];
    }
    for(int r = 0; r < H; r++)
        for(int c = 0; c < W; c++)
            if(count_image.at<double>(r,c) > 1e-6)
                time_image.at<double>(r,c) /= count_image.at<double>(r,c);
    this->Count_image = count_image.clone();
    this->Time_image = time_image.clone();
}

void EgoMotion::deal_time_image()
{
    double minv, maxv;
    cv::minMaxLoc(this->Time_image, &minv, &maxv, 0, 0);
    //double miu = cv::sum(this->Time_image)[0]/this->events.size();
    this->Untreshold_normalized_time_image = (this->Time_image)/maxv;

    cv::Scalar miu_new, std;
    cv::meanStdDev(this->Untreshold_normalized_time_image, miu_new, std);
    double threshold = miu_new[0] + 2 * std[0];
    for(int r = 0; r < this->Untreshold_normalized_time_image.rows; r++)
        for(int c = 0; c < this->Untreshold_normalized_time_image.cols; c++)
        {
            if(this->Untreshold_normalized_time_image.at<double>(r,c) <= threshold)
                this->Normalized_time_image.at<uchar>(r,c) = 0;
            else
                this->Normalized_time_image.at<uchar>(r,c) = 1;
        }
}

