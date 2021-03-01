#include "Cluster.h"

Cluster::Cluster(float threshold, int MinPts):
    threshold(threshold),
    MinPts(MinPts)
{
    //ctor
}

Cluster::~Cluster()
{
    //dtor
}

std::vector< std::vector<cv::Point> > Cluster::img2point(cv::Mat &img)
{
    std::vector< std::vector<cv::Point> > result;
    for(int r=0;r<img.rows;r++)
    {
        for(int c=0;c<img.cols;c++)
        {
            int id = img.at<uchar>(r,c);
            if(id > result.size())
            {
                for(int n=result.size(); n<id;n++)
                {
                    std::vector<cv::Point> newclass;
                    result.push_back(newclass);
                }
                result[id-1].push_back(cv::Point(c, r));
            }
            else if(id>0 && id<=result.size())
                result[id-1].push_back(cv::Point(c, r));
        }
    }
    return result;
}

float distance(PreCluster a, PreCluster b)
{
    double dx = a.center.x - b.center.x;
    double dy = a.center.y - b.center.y;
    return (float)(sqrt(dx*dx+dy*dy) - a.radius - b.radius);
}
               

std::vector< std::vector<cv::Point> > Cluster::cluster(cv::Mat &Normalized_time_image, cv::Mat& Time_image)
{
    std::vector< std::vector<unsigned int> > output_id;
    std::vector< std::vector<cv::Point> > pre_point;
    std::vector< std::vector<cv::Point> > output;
    //std::vector<Point> dataset;
    cv::Mat pre_ids;
    cv::connectedComponents(Normalized_time_image, pre_ids);
    pre_ids.convertTo(pre_ids, CV_8U);

    //this->Timeimage = Time_image;
    pre_point = this->img2point(pre_ids);
    std::vector<PreCluster> clusters;
    for(int i=0; i<pre_point.size(); i++)
    {
        PreCluster pre_cluster;
        cv::minEnclosingCircle(pre_point[i], pre_cluster.center, pre_cluster.radius);
        clusters.push_back(pre_cluster);
    }

    auto dbscan = DBSCAN<PreCluster, float>();
    dbscan.Run(&clusters, 2, 200.0f, 5, distance);
    output_id = dbscan.Clusters;
    for(std::vector< std::vector<unsigned int> >::iterator iter = output_id.begin(); iter!=output_id.end(); iter++)
    {
        std::vector<cv::Point> newclass;
        for(std::vector<unsigned int>::iterator it = (*iter).begin(); it!=(*iter).end(); it++)
            newclass.insert(newclass.end(), pre_point[*it].begin(), pre_point[*it].end());
        output.push_back(newclass);
    }

    /* save
    std::ofstream fs;
    fs.open("D://1.txt",std::ios::in|std::ios::ate);
    int maxx=0, maxy=0, minx=500, miny=500;

    if(output.size()==0)
        fs<<std::endl;
    else
    {
        for(std::vector<unsigned int>::iterator iter = output[0].begin(); iter!=output[0].end(); iter++)
        {
            if(maxx<(int)eventpoint[*iter][0])
                maxx = (int)eventpoint[*iter][0];
            if(maxy<(int)eventpoint[*iter][1])
                maxy = (int)eventpoint[*iter][1];
            if(minx>(int)eventpoint[*iter][0])
                minx = (int)eventpoint[*iter][0];
            if(miny>(int)eventpoint[*iter][1])
                miny = (int)eventpoint[*iter][1];
        }
        //fs<<maxy-miny<<" "<<maxx-minx<<" "<<miny<<" "<<minx<<std::endl;
    }
    cv::rectangle(Time_image, cv::Point(miny, minx), cv::Point(maxy, maxx), cv::Scalar(1), 5);
*/
    //cv::namedWindow("w");
    //cv::imshow("w", Time_image);
    //cv::waitKey(1);
    //std::cout<<output.size()<<std::endl;

    return output;
}

cv::Mat Cluster::optical_flow(cv::Mat prev, cv::Mat next, std::vector<cv::Point2f> prev_events, std::vector<cv::Point2f> next_events, int window_size)
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    prev*=255;next*=255;
    prev.convertTo(prev, CV_8UC1);
    next.convertTo(next, CV_8UC1);

    cv::calcOpticalFlowPyrLK(prev, next, prev_events, next_events, status, err, cv::Size(window_size, window_size),
                            3, termcrit, 0, 0.001);
    cv::Mat opticalflow(next.rows, next.cols, CV_64FC2);
    std::vector<cv::Point2f>::iterator iterprev = prev_events.begin();
    std::vector<cv::Point2f>::iterator iternext = next_events.begin();
    std::vector<uchar>::iterator iterstatus = status.begin();

    while(iterprev != prev_events.end())
    {
        if(*(iterstatus) != '\0')
            opticalflow.at<cv::Vec2d>(round((*iterprev).y), round((*iterprev).x)) = \
                    cv::Vec2d((double)((*iternext).x - (*iterprev).x), \
                              (double)((*iternext).y - (*iterprev).y));
        iterprev++;
        iternext++;
        iterstatus++;
    }
    return opticalflow;
}
