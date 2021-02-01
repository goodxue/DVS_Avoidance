#include "Cluster.h"

Cluster::Cluster(float threshold, cv::Mat Optical_flow, cv::Mat Time_image, float wp, float wv, float wrho):
    threshold(threshold),
    OpticalFlow(Optical_flow),
    Timeimage(Time_image),
    wp(wp),
    wv(wv),
    wrho(wrho)
{
    //ctor
}

Cluster::~Cluster()
{
    //dtor
}

float Cluster::Cost(point a,point b){
    float p, v, t;
    cv::Mat OpticalFlow = this->OpticalFlow;
    cv::Mat Timeimage = this->Timeimage;
    p = sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
    cv::Vec2f ofa = OpticalFlow.at<cv::Vec2f>(a.x,a.y);
    cv::Vec2f ofb = OpticalFlow.at<cv::Vec2f>(b.x,b.y);
    v = sqrt((ofa[0]-ofb[0])*(ofa[0]-ofb[0])+(ofa[1]-ofb[1])*(ofa[1]-ofb[1]));
    t = std::fabs(Timeimage.at<float>(a.x,a.y) - Timeimage.at<float>(b.x,b.y));
    return this->wp*p+this->wv*v+this->wrho*t;
}

std::vector<point> Cluster::img2point(cv::Mat img){
    std::vector<point> points;
    int i = 1;
    for(int r=0;r<img.rows;r++){
        for(int c=0;c<img.cols;c++){
            if(img.at<float>(r,c)>1e-6)
                points.push_back(point(r,c,i++));
        }
    }
    return points;
}

std::vector< std::vector<cv::Point> > Cluster::cluster(cv::Mat img, float threshold, int MinPts){
    std::vector<point> dataset;
    dataset = img2point(img);
    int len = dataset.size();
    //calculate pts
    std::cout<<"calculate pts"<<std::endl;
    for(int i=0;i<len;i++){
        for(int j=i+1;j<len;j++){
            if(Cost(dataset[i],dataset[j])<threshold)
                dataset[i].pts++;
                dataset[j].pts++;
        }
    }
    //core point
    std::cout<<"core point "<<std::endl;
    std::vector<point> corePoint;
    for(int i=0;i<len;i++){
        if(dataset[i].pts>=MinPts) {
            dataset[i].pointType = 3;
            corePoint.push_back(dataset[i]);
        }
    }
    std::cout<<"joint core point"<<std::endl;
    //joint core point
    for(int i=0;i<corePoint.size();i++){
        for(int j=i+1;j<corePoint.size();j++){
            if(Cost(corePoint[i],corePoint[j])<threshold){
                corePoint[i].corepts.push_back(j);
                corePoint[j].corepts.push_back(i);
            }
        }
    }
    for(int i=0;i<corePoint.size();i++){
        std::stack<point*> ps;
        if(corePoint[i].visited == 1) continue;
        ps.push(&corePoint[i]);
        point *v;
        while(!ps.empty()){
            v = ps.top();
            v->visited = 1;
            ps.pop();
            for(int j=0;j<v->corepts.size();j++){
                if(corePoint[v->corepts[j]].visited==1) continue;
                corePoint[v->corepts[j]].cluster = corePoint[i].cluster;
                corePoint[v->corepts[j]].visited = 1;
                ps.push(&corePoint[v->corepts[j]]);
            }
        }
    }
    std::cout<<"border point,joint border point to core point"<<std::endl;
    //border point,joint border point to core point
    for(int i=0;i<len;i++){
        if(dataset[i].pointType==3) continue;
        for(int j=0;j<corePoint.size();j++){
            if(Cost(dataset[i],corePoint[j])<threshold) {
                dataset[i].pointType = 2;
                dataset[i].cluster = corePoint[j].cluster;
                break;
            }
        }
    }
    std::cout<<"output"<<std::endl;
    //output
    std::vector< std::vector<cv::Point> > output;
    for(int i=0;i<len;i++){
        if(dataset[i].pointType == 2){
            if(dataset[i].cluster > output.size()){
                std::vector<cv::Point> newvec;
                newvec.push_back(cv::Point(dataset[i].x,dataset[i].y));
                output.push_back(newvec);
            }
            else
                output[dataset[i].cluster].push_back(cv::Point(dataset[i].x,dataset[i].y));
        }
    }
    for(int i=0;i<corePoint.size();i++){
        if(corePoint[i].cluster > output.size()){
            std::vector<cv::Point> newvec;
            newvec.push_back(cv::Point(corePoint[i].x,corePoint[i].y));
            output.push_back(newvec);
        }
        else
            output[corePoint[i].cluster].push_back(cv::Point(corePoint[i].x,corePoint[i].y));
    }
}
